// ParametricSurface demo: a paraboloid bowl evaluated on a 2D slice in 3D space.

#include <cmath>
#include <iostream>
#include <vector>

#include "gwn3d/gwn.h"
#include "gwn3d/parametric_surface.h"
#include "curve_net_parser.h"

static Eigen::Vector3d paraboloid(Eigen::Vector2d uv) {
    double x = 2.0 * uv(0) - 1.0;
    double y = 2.0 * uv(1) - 1.0;
    return Eigen::Vector3d(x, y, x * x + y * y);
}

// Boundary loop: the four edges of [0,1]^2 in UV, lifted through paraboloid.
static std::vector<patch_t> paraboloid_boundary(int n_per_edge) {
    patch_t patch;
    patch.is_open = false;
    patch.curve.resize(4 * n_per_edge, 3);
    int row = 0;
    auto push = [&](double u, double v) {
        patch.curve.row(row++) = paraboloid(Eigen::Vector2d(u, v));
    };
    for (int i = 0; i < n_per_edge; ++i) push(double(i) / n_per_edge, 0.0);
    for (int i = 0; i < n_per_edge; ++i) push(1.0, double(i) / n_per_edge);
    for (int i = 0; i < n_per_edge; ++i) push(1.0 - double(i) / n_per_edge, 1.0);
    for (int i = 0; i < n_per_edge; ++i) push(0.0, 1.0 - double(i) / n_per_edge);
    return { patch };
}

int main() {
    const int W = 120, H = 80;
    gwn3d::ParametricSurface surface(paraboloid, paraboloid_boundary(64));

    Eigen::MatrixXd qp(W * H, 3);
    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            double x = -1.5 + 3.0 * i / (W - 1);
            double y = -1.5 + 3.0 * j / (H - 1);
            qp.row(j * W + i) << x, y, 1.0;
        }
    }

    gwn3d::GwnOptions opts;
    opts.resolution = { W, H };
    auto rays = dimension_to_rays({ W, H });

    auto res = gwn3d::compute_gwn(surface, qp, rays, opts);
    gwn3d::write_gwn_ppm(res, "paraboloid.ppm");
    std::cout << "wrote paraboloid.ppm\n";
    return 0;
}
