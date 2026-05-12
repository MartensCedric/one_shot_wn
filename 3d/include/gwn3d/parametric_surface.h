#pragma once

#include <Eigen/Core>
#include <functional>
#include <memory>
#include <vector>

#include "gwn3d/surface.h"
#include "parametric.h"

namespace gwn3d {

/// Surface defined by a parametric map f: [0,1]^2 -> R^3.
///
/// If no analytic Jacobian is supplied, a finite-difference Jacobian
/// (eps = 1e-8) is used instead.
class ParametricSurface : public Surface {
public:
    /// Initial (u, v) seed grid for the GSL multiroot solver.
    /// Standard is a 20 x 10 grid; Coarse is a 500 x 1 strip at v = 0.5
    /// (legacy gears-style sampling).
    enum class SamplingStrategy { Standard, Coarse };

    ParametricSurface(
        std::function<Eigen::Vector3d(Eigen::Vector2d)> func,
        std::vector<patch_t> boundary_patches,
        std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> jac = {},
        std::function<bool(Eigen::Vector2d)> is_in_domain = {},
        SamplingStrategy strategy = SamplingStrategy::Standard);

    ~ParametricSurface() override;

    std::vector<patch_t> boundary_patches() const override { return boundary_patches_; }

    void prepare(int n_threads) override;

    all_intersections_with_normals_result intersect_ray(
        const Eigen::Vector3d& origin,
        const Eigen::Vector3d& direction,
        int patch_idx,
        double t_max,
        int thread_id) const override;

protected:
    parametric_solver_params make_solver_params(
        const Eigen::Vector3d& origin,
        const Eigen::Vector3d& direction,
        int patch_idx) const;

    std::function<Eigen::Vector3d(Eigen::Vector2d)> func_;
    std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> jac_;
    std::function<bool(Eigen::Vector2d)> is_in_domain_;
    std::vector<patch_t> boundary_patches_;
    SamplingStrategy strategy_;

    struct ThreadCache;
    mutable std::vector<std::unique_ptr<ThreadCache>> thread_caches_;
};

} // namespace gwn3d
