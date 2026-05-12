// Sanity check: compare gwn3d::compute_gwn against igl::winding_number on the
// same triangle mesh and query points.
//
//   compare_with_libigl <mesh.obj|.off> <queries.points> [resolution_w] [resolution_h]
//
// Defaults to the camel example. Both methods compute the generalized winding
// number for an open mesh; for a closed manifold they should agree to within
// floating-point noise. The PPMs are written for visual inspection.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

#include <igl/winding_number.h>

#include "gwn3d/gwn.h"
#include "gwn3d/mesh_surface.h"
#include "curve_net_parser.h"

int main(int argc, char** argv) {
    std::string mesh  = argc > 1 ? argv[1] : "../../../inputs/camelhead.obj";
    std::string qfile = argc > 2 ? argv[2] : "../../../inputs/camelhead_500_300.points";
    int W = argc > 3 ? std::stoi(argv[3]) : 500;
    int H = argc > 4 ? std::stoi(argv[4]) : 300;

    auto surface = gwn3d::MeshSurface::from_file(mesh);
    Eigen::MatrixXd qp = load_query_points(qfile);

    std::cout << "loaded " << surface.vertices().rows() << " vertices, "
              << surface.faces().rows() << " faces, "
              << qp.rows() << " query points\n";

    Eigen::VectorXd ref;
    igl::winding_number(surface.vertices(), surface.faces(), qp, ref);

    gwn3d::GwnOptions opts;
    opts.resolution = { W, H };
    auto rays = dimension_to_rays({ W, H });
    auto res = gwn3d::compute_gwn(surface, qp, rays, opts);

    double max_abs_diff = 0.0;
    double sum_abs_diff = 0.0;
    int worst_idx = 0;
    for (int i = 0; i < qp.rows(); ++i) {
        double d = std::abs(res.gwn[i] - ref(i));
        sum_abs_diff += d;
        if (d > max_abs_diff) { max_abs_diff = d; worst_idx = i; }
    }

    std::printf("\nigl::winding_number range : [%g, %g]\n", ref.minCoeff(), ref.maxCoeff());
    std::printf("gwn3d::compute_gwn range  : [%g, %g]\n",
                *std::min_element(res.gwn.begin(), res.gwn.end()),
                *std::max_element(res.gwn.begin(), res.gwn.end()));
    std::printf("max |gwn3d - libigl| = %g (at index %d: gwn3d=%g, libigl=%g)\n",
                max_abs_diff, worst_idx, res.gwn[worst_idx], ref(worst_idx));
    std::printf("mean |gwn3d - libigl| = %g\n", sum_abs_diff / qp.rows());

    // Save both as PPM for visual inspection.
    gwn3d::write_gwn_ppm(res, "gwn3d.ppm");
    gwn3d::GwnResult ref_res;
    ref_res.gwn.assign(ref.data(), ref.data() + ref.size());
    ref_res.resolution = { W, H };
    gwn3d::write_gwn_ppm(ref_res, "libigl.ppm");

    std::cout << "wrote gwn3d.ppm and libigl.ppm\n";
    return 0;
}
