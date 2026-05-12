#pragma once

#include <Eigen/Core>

#include "gwn3d/parametric_surface.h"

namespace gwn3d {

/// Cubic Bezier tensor-product patch. Takes a 16 x 3 matrix of control
/// points (4 x 4 grid, row-major: row i*4 + j is the (i, j) control point,
/// with u along the column index and v along the row index).
///
/// num_boundary_points_per_edge controls boundary discretisation: the
/// four edges of [0,1]^2 are each sampled at that many points, then
/// concatenated into one closed loop.
class BezierPatchSurface : public ParametricSurface {
public:
    BezierPatchSurface(
        const Eigen::MatrixXd& control_points,
        int num_boundary_points_per_edge = 64);
};

} // namespace gwn3d
