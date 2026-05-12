#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "gwn3d/surface.h"
#include "curve_net.h"
#include "curve_net_wn.h"

namespace gwn3d {

/// Minimal surface defined implicitly by closed boundary curves, solved
/// via BEM. prepare() runs the BEM precompute. Kept in the library to keep
/// the legacy MINIMAL inputs building; not on the must-work runtime path.
class MinimalSurface : public Surface {
public:
    explicit MinimalSurface(std::vector<patch_t> boundary_patches);

    ~MinimalSurface() override;

    std::vector<patch_t> boundary_patches() const override { return boundary_patches_; }

    void prepare(int n_threads) override;

    all_intersections_with_normals_result intersect_ray(
        const Eigen::Vector3d& origin,
        const Eigen::Vector3d& direction,
        int patch_idx,
        double t_max,
        int thread_id) const override;

    const precomputed_curve_data& precomputed() const { return precompute_; }

private:
    std::vector<patch_t> boundary_patches_;
    precomputed_curve_data precompute_{};
    bool prepared_ = false;

    struct ThreadCache;
    mutable std::vector<std::unique_ptr<ThreadCache>> thread_caches_;
};

} // namespace gwn3d
