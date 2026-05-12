// Mesh demo: load a triangle mesh, run compute_gwn, write a PPM.
//
//   mesh_demo <mesh.obj|.off> <queries.points> <output_name>
//
// Defaults reproduce the camel example bundled in inputs/.

#include <iostream>
#include <string>

#include "gwn3d/gwn.h"
#include "gwn3d/mesh_surface.h"
#include "curve_net_parser.h"

int main(int argc, char** argv) {
    std::string mesh  = argc > 1 ? argv[1] : "../../../inputs/camelhead.obj";
    std::string qfile = argc > 2 ? argv[2] : "../../../inputs/camelhead_500_300.points";
    std::string name  = argc > 3 ? argv[3] : "camel";

    auto surface = gwn3d::MeshSurface::from_file(mesh);
    Eigen::MatrixXd qp = load_query_points(qfile);

    gwn3d::GwnOptions opts;
    opts.resolution = { 500, 300 };
    auto rays = dimension_to_rays({ 500, 300 });

    auto res = gwn3d::compute_gwn(surface, qp, rays, opts);
    gwn3d::write_gwn_ppm(res, name + ".ppm");
    std::cout << "wrote " << name << ".ppm\n";
    return 0;
}
