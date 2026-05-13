#include "gwn3d/gwn.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include <omp.h>

#include "boundary_processing.h"
#include "curve_net_parser.h"
#include "curve_net_wn.h"
#include "intersections.h"
#include "math_util.h"
#include "patch.h"
#include "region_splitting.h"

namespace gwn3d {

namespace {

// For prefers_global_intersection surfaces, per_patch has a single entry that
// represents the whole geometry; otherwise it is one entry per boundary patch.
struct RayIntersections {
    std::vector<all_intersections_with_normals_result> per_patch;
    bool single_global = false;
};

void render_progress(const char* label, int done, int total, int width = 30) {
    if (total <= 0) return;
    double frac = std::min(1.0, double(done) / double(total));
    int filled = int(frac * width);
    std::fprintf(stderr, "\r%-12s [", label);
    for (int i = 0; i < width; ++i) std::fputc(i < filled ? '#' : '-', stderr);
    std::fprintf(stderr, "] %4d/%-4d (%3d%%)", done, total, int(frac * 100));
    std::fflush(stderr);
    if (done >= total) std::fputc('\n', stderr);
}

} // namespace

GwnResult compute_gwn(
    Surface& surface,
    const Eigen::MatrixXd& query_points,
    const std::vector<std::vector<int>>& rays_input,
    const GwnOptions& opts)
{
    if (rays_input.empty() || rays_input[0].empty()) {
        throw std::runtime_error("compute_gwn: rays decomposition must be non-empty");
    }

    GwnResult result;
    result.resolution = opts.resolution;
    GwnTimingInfo& timing = result.timing;

    std::vector<patch_t> patches = surface.boundary_patches();
    if (!opts.flip_patch_orientation.empty()) {
        for (size_t i = 0; i < patches.size(); ++i)
            if (opts.flip_patch_orientation[i])
                patches[i] = flip_patch(patches[i]);
    }
    std::vector<space_curve_t> closed_patches = get_closed_patches_space_curves(patches);

    timing.num_patches = static_cast<int>(patches.size());
    timing.total_rays = static_cast<int>(rays_input.size());
    timing.ray_length = static_cast<int>(rays_input[0].size());
    timing.total_patch_samples = 0;
    for (const auto& p : patches)
        timing.total_patch_samples += static_cast<int>(p.curve.rows());

    std::cout << "Running on " << query_points.rows()
              << " query points on " << patches.size() << " patches" << std::endl;
    std::cout << "Shooting " << rays_input.size()
              << " rays of size " << rays_input[0].size() << std::endl;
    std::cout << "Patches have a total of " << timing.total_patch_samples << std::endl;

    auto precompute_tic = std::chrono::high_resolution_clock::now();
    surface.prepare(opts.compute_threads);
    timing.precompute_wall = std::chrono::high_resolution_clock::now() - precompute_tic;
    std::cout << "precomputed done" << std::endl;

    const int n_threads = opts.compute_threads;
    omp_set_num_threads(n_threads);

    std::vector<double> winding_numbers(query_points.rows(), 0.0);
    std::vector<std::pair<int, int>> ignored_rays;

    // Shuffle so OpenMP threads see a balanced mix of expensive and cheap rays.
    std::vector<std::vector<int>> rays = rays_input;
    std::vector<int> rays_ordering(rays.size());
    std::iota(rays_ordering.begin(), rays_ordering.end(), 0);
    rays = random_shuffle<std::vector<int>>(rays, rays_ordering);

    std::vector<RayIntersections> intersections(rays.size());
    for (auto& ri : intersections) {
        ri.per_patch.assign(closed_patches.size(), {});
        for (auto& r : ri.per_patch) r.valid_ray = false;
    }

    // Phase 1: shoot one ray per row of the grid decomposition.
    auto ray_shot_tic = std::chrono::high_resolution_clock::now();

    const bool surface_is_global = surface.prefers_global_intersection();

    std::atomic<int> rays_done{0};
    const int progress_step = std::max(1, static_cast<int>(rays.size()) / 50);

#pragma omp parallel for num_threads(n_threads)
    for (int ray_index = 0; ray_index < static_cast<int>(rays.size()); ++ray_index) {
        const int first_idx = rays[ray_index][0];
        const int second_idx = rays[ray_index][1];
        Eigen::Vector3d origin = query_points.row(first_idx);
        Eigen::Vector3d direction = (query_points.row(second_idx) - query_points.row(first_idx)).normalized();
        int thread_id = omp_get_thread_num();

        if (surface_is_global) {
            auto hit = surface.intersect_ray(origin, direction, 0, opts.parametric_max_t, thread_id);
            intersections[ray_index].per_patch = { std::move(hit) };
            intersections[ray_index].single_global = true;
            continue;
        }

        for (size_t patch_index = 0; patch_index < closed_patches.size(); ++patch_index) {
            region_weighted_rays_info region_weighted_rays =
                get_weighted_rays(closed_patches[patch_index], origin);
            if (region_weighted_rays.polygonal_regions.empty()) break;

            auto hit = surface.intersect_ray(origin, direction,
                                             static_cast<int>(patch_index),
                                             opts.parametric_max_t, thread_id);

            // Spherical-region segments grazing the ray make the row-wise
            // result unreliable; punt to the per-point one-shot path.
            if (hit.valid_ray) {
                const double threshold = 0.006;
                double min_dist = find_minimum_distance_to_segment_all_polygons(
                    region_weighted_rays.polygonal_regions, direction);
                if (min_dist < threshold) hit.valid_ray = false;
            }

            intersections[ray_index].per_patch[patch_index] = std::move(hit);
        }

        int done = rays_done.fetch_add(1) + 1;
        if (done % progress_step == 0 || done == static_cast<int>(rays.size())) {
#pragma omp critical(progress)
            render_progress("ray-shoot", done, static_cast<int>(rays.size()));
        }
    }

    auto ray_shot_toc = std::chrono::high_resolution_clock::now();
    timing.ray_shooting += ray_shot_toc - ray_shot_tic;

    rays = unshuffle_from_random(rays, rays_ordering);
    intersections = unshuffle_from_random(intersections, rays_ordering);
    std::cout << "Ray shooting done" << std::endl;

    // Phase 2: distribute each ray's intersections to the points along it.
    auto boundary_processing_start = std::chrono::high_resolution_clock::now();
    std::atomic<int> bp_done{0};

#pragma omp parallel for num_threads(n_threads)
    for (int ray_index = 0; ray_index < static_cast<int>(rays.size()); ++ray_index) {
        for (size_t patch_index = 0; patch_index < closed_patches.size(); ++patch_index) {
            const auto& ri = intersections[ray_index];
            const auto& patch_hits = ri.single_global
                                       ? ri.per_patch[0]
                                       : ri.per_patch[patch_index];

            if (!patch_hits.valid_ray) {
                if (!ri.single_global) {
#pragma omp critical
                    ignored_rays.emplace_back(ray_index, static_cast<int>(patch_index));
                }
                continue;
            }

            std::vector<region_weighted_rays_info> regions_infos(rays[ray_index].size());
            for (size_t along = 0; along < rays[ray_index].size(); ++along) {
                int q = rays[ray_index][along];
                regions_infos[along] = get_weighted_rays(closed_patches[patch_index],
                                                        query_points.row(q),
                                                        opts.perform_self_intersections);
            }

            std::vector<double> wn_along_ray = winding_numbers_along_ray(
                patch_hits.all_intersections, regions_infos, rays[ray_index], query_points);

#pragma omp critical
            {
                for (size_t along = 0; along < rays[ray_index].size(); ++along)
                    winding_numbers[rays[ray_index][along]] += wn_along_ray[along];
            }
        }

        int done = bp_done.fetch_add(1) + 1;
        if (done % progress_step == 0 || done == static_cast<int>(rays.size())) {
#pragma omp critical(progress)
            render_progress("bndy-proc", done, static_cast<int>(rays.size()));
        }
    }

    auto boundary_processing_stop = std::chrono::high_resolution_clock::now();
    timing.boundary_processing += boundary_processing_stop - boundary_processing_start;
    std::cout << "Boundary processing done" << std::endl;
    std::cout << "time: " << (std::chrono::high_resolution_clock::now() - ray_shot_tic).count() << std::endl;

    // Phase 3: one-shot fallback for (ray, patch) pairs flagged as unreliable.
    std::vector<std::pair<int, int>> leftovers;
    leftovers.reserve(ignored_rays.size() * rays[0].size());
    for (const auto& ir : ignored_rays) {
        for (int along : rays[ir.first]) leftovers.emplace_back(along, ir.second);
    }

    std::cout << "Leftover: " << leftovers.size() << std::endl;

    if (!leftovers.empty()) {
        std::vector<Eigen::Vector3d> leftover_ray_dir(leftovers.size(), Eigen::Vector3d::Zero());
        std::vector<std::vector<int>> leftover_rel_wn(leftovers.size());
        std::vector<std::vector<double>> leftover_areas(leftovers.size());

#pragma omp parallel for num_threads(n_threads)
        for (int li = 0; li < static_cast<int>(leftovers.size()); ++li) {
            int qi = leftovers[li].first;
            int pi = leftovers[li].second;
            region_weighted_rays_info wr = get_weighted_rays(closed_patches[pi], query_points.row(qi));
            if (wr.polygonal_regions.empty()) continue;
            if (!wr.rays.empty()) {
                leftover_ray_dir[li] = wr.rays[0];
                leftover_rel_wn[li] = wr.relative_wn;
                leftover_areas[li] = wr.areas;
            }
        }

#pragma omp parallel for num_threads(n_threads)
        for (int li = 0; li < static_cast<int>(leftovers.size()); ++li) {
            int qi = leftovers[li].first;
            int pi = leftovers[li].second;
            Eigen::Vector3d origin = query_points.row(qi);
            Eigen::Vector3d direction = leftover_ray_dir[li];
            const std::vector<int>& rel_wn = leftover_rel_wn[li];
            const std::vector<double>& areas = leftover_areas[li];
            bool has_no_rays = direction.isZero();
            int thread_id = omp_get_thread_num();

            int chi = 0;
            if (!has_no_rays && areas.size() != 1) {
                auto hit = surface.intersect_ray(origin, direction, pi,
                                                 opts.parametric_max_t, thread_id);
                if (hit.valid_ray) {
                    for (const auto& h : hit.all_intersections) chi += h.second;
                }
            }

            for (size_t k = 0; k < areas.size(); ++k) {
                int integer_chi = chi + rel_wn[k] - rel_wn[0];
                double final_chi = static_cast<double>(integer_chi);
#pragma omp critical
                {
                    winding_numbers[qi] += areas[k] * final_chi;
                }
            }
        }
    }

    // Phase 4: optional multi-boundary remap (see GwnOptions::symmetric_patch_adjustment).
    if (opts.symmetric_patch_adjustment) {
        for (auto& w : winding_numbers) {
            if (w > 0) w = 1.0 - 2.0 * (1.0 - w);
            else w *= 2.0;
        }
    }

    result.gwn = std::move(winding_numbers);
    return result;
}

double compute_gwn(
    Surface& surface,
    const Eigen::Vector3d& point,
    const GwnOptions& opts)
{
    // The rays-overload requires at least two points per ray to derive a
    // direction. Synthesize a tiny +x companion so the batching path applies.
    Eigen::MatrixXd qp(2, 3);
    qp.row(0) = point;
    qp.row(1) = point + Eigen::Vector3d(1e-3, 0.0, 0.0);
    const std::vector<std::vector<int>> rays = { { 0, 1 } };
    return compute_gwn(surface, qp, rays, opts).gwn[0];
}

GwnResult compute_gwn(
    Surface& surface,
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& p1,
    int width, int height,
    const GwnOptions& opts)
{
    if (width <= 0 || height <= 0)
        throw std::runtime_error("compute_gwn: width and height must be positive");

    Eigen::Vector3d diff = p1 - p0;
    const int normal_axis = [&]() {
        int n = 0;
        double smallest = std::abs(diff(0));
        for (int i = 1; i < 3; ++i) {
            if (std::abs(diff(i)) < smallest) { smallest = std::abs(diff(i)); n = i; }
        }
        return n;
    }();
    int a = (normal_axis + 1) % 3;
    int b = (normal_axis + 2) % 3;
    if (a > b) std::swap(a, b); // keep world ordering: a is the first remaining axis

    const double normal_value = 0.5 * (p0(normal_axis) + p1(normal_axis));

    // Map width -> axis a, height -> axis b. Rays sweep whichever has more
    // cells so write_gwn_ppm's layout (slow * n_fast + fast) is honored.
    const int n_a = width;
    const int n_b = height;
    const int n_slow = std::max(n_a, n_b);
    const int n_fast = std::min(n_a, n_b);
    const bool a_is_slow = (n_a >= n_b);
    const int slow_axis = a_is_slow ? a : b;
    const int fast_axis = a_is_slow ? b : a;
    const double slow_lo = p0(slow_axis);
    const double slow_hi = p1(slow_axis);
    const double fast_lo = p0(fast_axis);
    const double fast_hi = p1(fast_axis);

    Eigen::MatrixXd qp(n_slow * n_fast, 3);
    for (int slow = 0; slow < n_slow; ++slow) {
        double s = (n_slow == 1) ? 0.5 : double(slow) / double(n_slow - 1);
        double slow_coord = slow_lo + s * (slow_hi - slow_lo);
        for (int fast = 0; fast < n_fast; ++fast) {
            double f = (n_fast == 1) ? 0.5 : double(fast) / double(n_fast - 1);
            double fast_coord = fast_lo + f * (fast_hi - fast_lo);
            Eigen::Vector3d row;
            row(normal_axis) = normal_value;
            row(slow_axis)   = slow_coord;
            row(fast_axis)   = fast_coord;
            qp.row(slow * n_fast + fast) = row;
        }
    }
    // slice_to_rays(n_fast, n_slow, 0) yields n_fast rays of length n_slow,
    // with ray i indexing [i, i + n_fast, i + 2*n_fast, ...]. That walks
    // qp's slow axis, which is exactly what the ray-batching expects.
    std::vector<std::vector<int>> rays = slice_to_rays(n_fast, n_slow, 0);

    GwnOptions o = opts;
    o.resolution = { width, height };
    return compute_gwn(surface, qp, rays, o);
}

void write_gwn_ppm(const GwnResult& result, const std::string& filename) {
    // dimension_to_rays makes the larger element of opts.resolution the ray
    // length (slow axis in memory) and the smaller one the ray count (fast
    // axis). slice_to_rays then lays out query points as
    //   gwn[fast_idx + n_fast * slow_idx]
    // so the image is n_fast cols * n_slow rows.
    const int n_slow = std::max(result.resolution[0], result.resolution[1]);
    const int n_fast = std::min(result.resolution[0], result.resolution[1]);
    if (n_slow <= 0 || n_fast <= 0)
        throw std::runtime_error("write_gwn_ppm: resolution must be positive");
    if (static_cast<int>(result.gwn.size()) != n_slow * n_fast)
        throw std::runtime_error("write_gwn_ppm: result.gwn size mismatch with resolution");

    std::ofstream f(filename, std::ios::binary);
    if (!f) throw std::runtime_error("write_gwn_ppm: failed to open " + filename);
    f << "P6\n" << n_fast << " " << n_slow << "\n255\n";

    std::vector<unsigned char> buf(n_slow * n_fast * 3);
    for (int row = 0; row < n_slow; ++row) {
        for (int col = 0; col < n_fast; ++col) {
            double v = std::max(-1.0, std::min(1.0, result.gwn[row * n_fast + col]));
            unsigned char r = v < 0.0 ? static_cast<unsigned char>(std::round(-v * 255.0)) : 0;
            unsigned char g = v > 0.0 ? static_cast<unsigned char>(std::round( v * 255.0)) : 0;
            const int dst = (row * n_fast + col) * 3;
            buf[dst + 0] = r;
            buf[dst + 1] = g;
            buf[dst + 2] = 0;
        }
    }
    f.write(reinterpret_cast<const char*>(buf.data()), buf.size());
}

} // namespace gwn3d
