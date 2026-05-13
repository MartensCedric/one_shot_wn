// CoonsSurface demo: build a single Coons patch programmatically from four
// boundary curves (a flat unit square in xy lifted by a sin bump along the
// top edge), then render the GWN field on a planar slice using the grid
// overload of compute_gwn.

#include <cmath>
#include <iostream>
#include <vector>

#include "gwn3d/gwn.h"
#include "gwn3d/coons_surface.h"

static Eigen::Vector3d c0(double s) { return Eigen::Vector3d(s, 0.0, 0.0); }
static Eigen::Vector3d c1(double s) { return Eigen::Vector3d(s, 1.0, 0.4 * std::sin(M_PI * s)); }
static Eigen::Vector3d d0(double t) { return Eigen::Vector3d(0.0, t, 0.0); }
static Eigen::Vector3d d1(double t) { return Eigen::Vector3d(1.0, t, 0.0); }

static Eigen::Vector3d coons_eval(Eigen::Vector2d uv) {
    double s = uv(0), t = uv(1);
    Eigen::Vector3d Lc = (1.0 - t) * c0(s) + t * c1(s);
    Eigen::Vector3d Ld = (1.0 - s) * d0(t) + s * d1(t);
    Eigen::Vector3d Lb = c0(0.0) * (1.0 - s) * (1.0 - t)
                       + c0(1.0) * s * (1.0 - t)
                       + c1(0.0) * (1.0 - s) * t
                       + c1(1.0) * s * t;
    return Lc + Ld - Lb;
}

static patch_t coons_boundary(int n_per_edge) {
    patch_t patch;
    patch.is_open = false;
    patch.curve.resize(4 * n_per_edge, 3);
    int row = 0;
    for (int i = 0; i < n_per_edge; ++i) patch.curve.row(row++) = c0(double(i) / n_per_edge);
    for (int i = 0; i < n_per_edge; ++i) patch.curve.row(row++) = d1(double(i) / n_per_edge);
    for (int i = 0; i < n_per_edge; ++i) patch.curve.row(row++) = c1(1.0 - double(i) / n_per_edge);
    for (int i = 0; i < n_per_edge; ++i) patch.curve.row(row++) = d0(1.0 - double(i) / n_per_edge);
    return patch;
}

int main() {
    coons_patch cp;
    cp.func = coons_eval;
    cp.is_closed = true;

    std::vector<coons_patch> patches = { cp };
    std::vector<patch_t> boundaries = { coons_boundary(64) };
    gwn3d::CoonsSurface surface(patches, boundaries);

    // Render a slice at z = 0.05, just above the patch's base plane.
    Eigen::Vector3d p0(-0.5, -0.5, 0.05);
    Eigen::Vector3d p1( 1.5,  1.5, 0.05);

    auto res = gwn3d::compute_gwn(surface, p0, p1, 160, 160);
    gwn3d::write_gwn_ppm(res, "coons.ppm");
    std::cout << "wrote coons.ppm\n";
    return 0;
}
