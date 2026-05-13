// MinimalSurface demo: a circular boundary in 3D bounds a disk-like minimal
// surface, solved via BEM. The GWN field is rendered on a planar slice
// using the grid overload of compute_gwn.

#include <cmath>
#include <iostream>
#include <vector>

#include "gwn3d/gwn.h"
#include "gwn3d/minimal_surface.h"

// Closed unit circle in the z = 0 plane.
static std::vector<patch_t> circle_boundary(int n) {
    patch_t patch;
    patch.is_open = false;
    patch.curve.resize(n, 3);
    for (int i = 0; i < n; ++i) {
        double t = 2.0 * M_PI * double(i) / n;
        patch.curve.row(i) << std::cos(t), std::sin(t), 0.0;
    }
    return { patch };
}

int main() {
    gwn3d::MinimalSurface surface(circle_boundary(200));

    // Render a slice at z = 0.05, just off the minimal surface itself.
    Eigen::Vector3d p0(-1.5, -1.5, 0.05);
    Eigen::Vector3d p1( 1.5,  1.5, 0.05);

    auto res = gwn3d::compute_gwn(surface, p0, p1, 160, 160);
    gwn3d::write_gwn_ppm(res, "minimal_surface.ppm");
    std::cout << "wrote minimal_surface.ppm\n";
    return 0;
}
