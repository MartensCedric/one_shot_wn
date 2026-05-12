#include "gwn3d/mesh_surface.h"

#include <stdexcept>

#include "mesh.h"

namespace gwn3d {

MeshSurface::MeshSurface(Eigen::MatrixXd V, Eigen::MatrixXi F)
    : vertices_(std::move(V)), faces_(std::move(F)) {}

MeshSurface MeshSurface::from_file(const std::string& path) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".off") == 0) {
        read_off(path, V, F);
    } else if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".obj") == 0) {
        read_obj(path, V, F);
    } else {
        throw std::runtime_error("MeshSurface::from_file: unsupported extension (expected .obj or .off): " + path);
    }
    return MeshSurface(std::move(V), std::move(F));
}

std::vector<patch_t> MeshSurface::boundary_patches() const {
    return extract_mesh_boundary(vertices_, faces_);
}

all_intersections_with_normals_result MeshSurface::intersect_ray(
    const Eigen::Vector3d& origin,
    const Eigen::Vector3d& direction,
    int /*patch_idx*/,
    double /*t_max*/,
    int /*thread_id*/) const
{
    return find_all_intersections_mesh(origin, direction, vertices_, faces_);
}

} // namespace gwn3d
