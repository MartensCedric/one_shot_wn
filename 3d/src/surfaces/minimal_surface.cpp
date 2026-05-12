#include "gwn3d/minimal_surface.h"

#include <gsl/gsl_multiroots.h>

#include "bem_solver.h"
#include "curve_net_parser.h"

namespace gwn3d {

struct MinimalSurface::ThreadCache {
    gsl_multiroot_fsolver* solver = nullptr;
    gsl_multiroot_function F{};

    ThreadCache() {
        const gsl_multiroot_fsolver_type* T = gsl_multiroot_fsolver_hybrids;
        F.f = &f_func;
        F.n = 3;
        solver = gsl_multiroot_fsolver_alloc(T, F.n);
        solver->x = gsl_vector_alloc(F.n);
        solver->function = &F;
    }
    ~ThreadCache() {
        if (solver) gsl_multiroot_fsolver_free(solver);
    }
};

MinimalSurface::MinimalSurface(std::vector<patch_t> boundary_patches)
    : boundary_patches_(std::move(boundary_patches)) {}

MinimalSurface::~MinimalSurface() {
    if (prepared_) free_precompute(precompute_);
}

void MinimalSurface::prepare(int n_threads) {
    if (!prepared_) {
        std::vector<patch_t> closed = get_closed_patches(boundary_patches_);
        std::vector<space_curve_t> closed_curves;
        closed_curves.reserve(closed.size());
        for (const auto& p : closed) closed_curves.push_back(p.curve);

        std::vector<double> insideness(closed_curves.size(), 1.0);
        precompute_ = precompute_patches(closed_curves, insideness);
        prepared_ = true;
    }

    thread_caches_.clear();
    thread_caches_.reserve(n_threads);
    for (int i = 0; i < n_threads; ++i)
        thread_caches_.emplace_back(std::make_unique<ThreadCache>());
}

all_intersections_with_normals_result MinimalSurface::intersect_ray(
    const Eigen::Vector3d& origin,
    const Eigen::Vector3d& direction,
    int patch_idx,
    double /*t_max*/,
    int thread_id) const
{
    bem_solver_params params{
        precompute_.int_params[patch_idx],
        precompute_.full_patches[patch_idx],
        precompute_.df_dns[patch_idx],
        origin,
        direction,
    };

    auto& cache = *thread_caches_.at(thread_id);
    cache.F.params = &params;

    const box& bb = precompute_.bounding_boxes[patch_idx];
    double max_diag = Eigen::Vector3d(
        bb.x_max - bb.x_min,
        bb.y_max - bb.y_min,
        bb.z_max - bb.z_min).norm();

    return find_all_intersections_bem(bb, params, cache.solver, max_diag);
}

} // namespace gwn3d
