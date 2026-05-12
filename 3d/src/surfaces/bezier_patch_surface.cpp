#include "gwn3d/bezier_patch_surface.h"

#include "bezier.h"

namespace gwn3d {

namespace {

// Sample the four edges of [0,1]^2 counter-clockwise and push them through f
// to form one closed boundary loop.
patch_t make_bezier_patch_boundary(
    const std::function<Eigen::Vector3d(Eigen::Vector2d)>& f,
    int num_per_edge)
{
    patch_t patch;
    patch.is_open = false;
    patch.curve.resize(4 * num_per_edge, 3);
    int row = 0;
    for (int i = 0; i < num_per_edge; ++i) {
        double t = double(i) / num_per_edge;
        patch.curve.row(row++) = f(Eigen::Vector2d(t, 0.0));
    }
    for (int i = 0; i < num_per_edge; ++i) {
        double t = double(i) / num_per_edge;
        patch.curve.row(row++) = f(Eigen::Vector2d(1.0, t));
    }
    for (int i = 0; i < num_per_edge; ++i) {
        double t = double(i) / num_per_edge;
        patch.curve.row(row++) = f(Eigen::Vector2d(1.0 - t, 1.0));
    }
    for (int i = 0; i < num_per_edge; ++i) {
        double t = double(i) / num_per_edge;
        patch.curve.row(row++) = f(Eigen::Vector2d(0.0, 1.0 - t));
    }
    return patch;
}

} // namespace

BezierPatchSurface::BezierPatchSurface(
    const Eigen::MatrixXd& control_points,
    int num_boundary_points_per_edge)
    : ParametricSurface(
          cubic_bezier_surface(control_points),
          { make_bezier_patch_boundary(cubic_bezier_surface(control_points), num_boundary_points_per_edge) },
          // Skip the analytic jacobian; the closed-form expression has many
          // pow() calls and stalls the GSL Hybrids solver. FD is faster.
          {},
          {},
          SamplingStrategy::Standard)
{
}

} // namespace gwn3d
