#pragma once

#include <Eigen/Core>
#include <string>

#include "gwn3d/surface.h"

namespace gwn3d {

/// Triangle mesh surface. Boundary loops are extracted via libigl's
/// boundary_loop; ray intersections use libigl's ray-triangle test.
class MeshSurface : public Surface {
public:
    MeshSurface(Eigen::MatrixXd V, Eigen::MatrixXi F);

    /// Load from a .obj or .off file.
    static MeshSurface from_file(const std::string& path);

    std::vector<patch_t> boundary_patches() const override;

    all_intersections_with_normals_result intersect_ray(
        const Eigen::Vector3d& origin,
        const Eigen::Vector3d& direction,
        int patch_idx,
        double t_max,
        int thread_id) const override;

    bool prefers_global_intersection() const override { return true; }

    const Eigen::MatrixXd& vertices() const { return vertices_; }
    const Eigen::MatrixXi& faces() const { return faces_; }

private:
    Eigen::MatrixXd vertices_;
    Eigen::MatrixXi faces_;
};

} // namespace gwn3d
