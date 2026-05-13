#include "gwn3d/coons_surface.h"

#include "parametric.h"
#include "gsl_solver_pool.h"

namespace gwn3d {

struct CoonsSurface::ThreadCache : GslHybridsSolver3 {
    ThreadCache() : GslHybridsSolver3(&f_func_parametric) {}
};

CoonsSurface::CoonsSurface(
    std::vector<coons_patch> patches,
    std::vector<patch_t> boundary_patches,
    std::vector<bool> flip_normals)
    : patches_(std::move(patches)),
      boundary_patches_(std::move(boundary_patches)),
      flip_normals_(std::move(flip_normals))
{}

CoonsSurface::~CoonsSurface() = default;

CoonsSurface CoonsSurface::from_folder(
    const std::string& folder,
    const std::string& base_name,
    int num_patches)
{
    std::vector<patch_t> bps;
    std::vector<coons_patch> cps;
    load_coons_patches_from_objs(folder, base_name, num_patches, bps, cps);
    return CoonsSurface(std::move(cps), std::move(bps));
}

void CoonsSurface::prepare(int n_threads) {
    thread_caches_.clear();
    thread_caches_.reserve(n_threads);
    for (int i = 0; i < n_threads; ++i)
        thread_caches_.emplace_back(std::make_unique<ThreadCache>());
}

all_intersections_with_normals_result CoonsSurface::intersect_ray(
    const Eigen::Vector3d& origin,
    const Eigen::Vector3d& direction,
    int patch_idx,
    double t_max,
    int thread_id) const
{
    const auto& patch_func = patches_.at(patch_idx).func;
    auto domain = [](Eigen::Vector2d uv) {
        return uv(0) >= 0.0 && uv(0) <= 1.0 && uv(1) >= 0.0 && uv(1) <= 1.0;
    };
    parametric_solver_params params{
        patch_func, fd_jacobian(patch_func, 1e-8),
        origin, direction, domain,
    };
    auto& cache = *thread_caches_.at(thread_id);
    cache.F.params = &params;
    auto result = find_all_intersections_parametric(cache.solver, params, 0.0, t_max);

    if (!flip_normals_.empty() && flip_normals_.at(patch_idx)) {
        for (auto& hit : result.all_intersections)
            hit.second = -hit.second;
    }
    return result;
}

} // namespace gwn3d
