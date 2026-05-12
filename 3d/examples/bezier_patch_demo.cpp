// BezierPatchSurface API demo.
//
// Cubic Bezier tensor-product patches go through the same GSL multiroot path
// as ParametricSurface but with a heavier per-evaluation cost, so even small
// grids take minutes. This demo skips the full GWN grid and just shows the
// surface constructs and that intersect_ray runs end-to-end.

#include <iostream>

#include "gwn3d/bezier_patch_surface.h"

int main() {
    // 4x4 control grid for a bicubic paraboloid bowl.
    Eigen::MatrixXd cp(16, 3);
    int r = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double u = double(j) / 3.0;
            double v = double(i) / 3.0;
            double x = 2.0 * u - 1.0;
            double y = 2.0 * v - 1.0;
            cp.row(r++) << x, y, x * x + y * y;
        }
    }

    gwn3d::BezierPatchSurface surface(cp, /*num_boundary_points_per_edge=*/16);
    auto patches = surface.boundary_patches();
    std::cout << "surface has " << patches.size() << " boundary patch(es), "
              << patches[0].curve.rows() << " boundary points\n";

    surface.prepare(1);
    auto hits = surface.intersect_ray(
        Eigen::Vector3d(0.0, 0.0, -1.0),
        Eigen::Vector3d(0.0, 0.0,  1.0),
        /*patch_idx=*/0, /*t_max=*/3.0, /*thread_id=*/0);

    std::cout << "intersect_ray valid=" << hits.valid_ray
              << ", hits=" << hits.all_intersections.size() << "\n";
    for (const auto& h : hits.all_intersections)
        std::cout << "  t=" << h.first << ", sign=" << h.second << "\n";
    return 0;
}
