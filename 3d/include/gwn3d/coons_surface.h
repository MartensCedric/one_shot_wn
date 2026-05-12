#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "gwn3d/surface.h"
#include "coons.h"

namespace gwn3d {

/// Collection of Coons patches loaded from .coon_in files. Each patch
/// carries its own (u, v) -> R^3 function. Kept in the library to keep
/// the legacy Coons inputs building; not on the must-work runtime path.
class CoonsSurface : public Surface {
public:
    CoonsSurface(
        std::vector<coons_patch> patches,
        std::vector<patch_t> boundary_patches,
        std::vector<bool> flip_normals = {});

    ~CoonsSurface() override;

    static CoonsSurface from_folder(
        const std::string& folder,
        const std::string& base_name,
        int num_patches);

    std::vector<patch_t> boundary_patches() const override { return boundary_patches_; }

    void prepare(int n_threads) override;

    all_intersections_with_normals_result intersect_ray(
        const Eigen::Vector3d& origin,
        const Eigen::Vector3d& direction,
        int patch_idx,
        double t_max,
        int thread_id) const override;

private:
    std::vector<coons_patch> patches_;
    std::vector<patch_t> boundary_patches_;
    std::vector<bool> flip_normals_;

    struct ThreadCache;
    mutable std::vector<std::unique_ptr<ThreadCache>> thread_caches_;
};

} // namespace gwn3d
