#include "gwn3d/parametric_surface.h"

#include <gsl/gsl_multiroots.h>

namespace gwn3d {

struct ParametricSurface::ThreadCache {
    gsl_multiroot_fsolver* solver = nullptr;
    gsl_multiroot_function F{};

    ThreadCache() {
        const gsl_multiroot_fsolver_type* T = gsl_multiroot_fsolver_hybrids;
        F.f = &f_func_parametric;
        F.n = 3;
        solver = gsl_multiroot_fsolver_alloc(T, F.n);
        solver->x = gsl_vector_alloc(F.n);
        solver->function = &F;
    }

    ~ThreadCache() {
        if (solver) gsl_multiroot_fsolver_free(solver);  // also frees s->x
    }
};

ParametricSurface::ParametricSurface(
    std::function<Eigen::Vector3d(Eigen::Vector2d)> func,
    std::vector<patch_t> boundary_patches,
    std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> jac,
    std::function<bool(Eigen::Vector2d)> is_in_domain,
    SamplingStrategy strategy)
    : func_(std::move(func)),
      jac_(std::move(jac)),
      is_in_domain_(std::move(is_in_domain)),
      boundary_patches_(std::move(boundary_patches)),
      strategy_(strategy)
{
    if (!jac_) jac_ = fd_jacobian(func_, 1e-8);
    if (!is_in_domain_) is_in_domain_ = [](Eigen::Vector2d uv) {
        return uv(0) >= 0.0 && uv(0) <= 1.0 && uv(1) >= 0.0 && uv(1) <= 1.0;
    };
}

ParametricSurface::~ParametricSurface() = default;

void ParametricSurface::prepare(int n_threads) {
    thread_caches_.clear();
    thread_caches_.reserve(n_threads);
    for (int i = 0; i < n_threads; ++i)
        thread_caches_.emplace_back(std::make_unique<ThreadCache>());
}

parametric_solver_params ParametricSurface::make_solver_params(
    const Eigen::Vector3d& origin,
    const Eigen::Vector3d& direction,
    int /*patch_idx*/) const
{
    return parametric_solver_params{
        func_, jac_, origin, direction, is_in_domain_,
    };
}

all_intersections_with_normals_result ParametricSurface::intersect_ray(
    const Eigen::Vector3d& origin,
    const Eigen::Vector3d& direction,
    int patch_idx,
    double t_max,
    int thread_id) const
{
    parametric_solver_params params = make_solver_params(origin, direction, patch_idx);
    auto& cache = *thread_caches_.at(thread_id);
    cache.F.params = &params;
    if (strategy_ == SamplingStrategy::Coarse)
        return find_all_intersections_parametric_gears(cache.solver, params, 0.0, t_max);
    return find_all_intersections_parametric(cache.solver, params, 0.0, t_max);
}

} // namespace gwn3d
