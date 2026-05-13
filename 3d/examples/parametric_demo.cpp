// ParametricSurface demo: a paraboloid bowl evaluated on a 2D slice in 3D
// space using the grid overload of compute_gwn.

#include <cmath>
#include <iostream>
#include <vector>

#include "gwn3d/gwn.h"
#include "gwn3d/parametric_surface.h"

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
    gwn3d::ParametricSurface surface(paraboloid, paraboloid_boundary(64));

    // Render a 2D slice at z = 1 (just above the paraboloid's apex).
    Eigen::Vector3d p0(-1.5, -1.5, 1.0);
    Eigen::Vector3d p1( 1.5,  1.5, 1.0);

    auto res = gwn3d::compute_gwn(surface, p0, p1, 120, 80);
    gwn3d::write_gwn_ppm(res, "paraboloid.ppm");
    std::cout << "wrote paraboloid.ppm\n";
    return 0;
}
