// Mesh demo: load a triangle mesh and render the GWN field on the xy slice
// through the middle of the mesh's bounding box. Uses the grid overload of
// compute_gwn (two 3D corner points + resolution).

#include <iostream>
#include <string>

#include "gwn3d/gwn.h"
#include "gwn3d/mesh_surface.h"

int main(int argc, char** argv) {
    std::string mesh = argc > 1 ? argv[1] : "../../../inputs/camelhead.obj";
    std::string name = argc > 2 ? argv[2] : "camel";

    auto surface = gwn3d::MeshSurface::from_file(mesh);

    const auto& V = surface.vertices();
    Eigen::Vector3d lo = V.colwise().minCoeff();
    Eigen::Vector3d hi = V.colwise().maxCoeff();
    double mid_z = 0.5 * (lo.z() + hi.z());

    Eigen::Vector3d p0(lo.x(), lo.y(), mid_z);
    Eigen::Vector3d p1(hi.x(), hi.y(), mid_z);

    auto res = gwn3d::compute_gwn(surface, p0, p1, 500, 300);
    gwn3d::write_gwn_ppm(res, name + ".ppm");
    std::cout << "wrote " << name << ".ppm\n";
    return 0;
}
