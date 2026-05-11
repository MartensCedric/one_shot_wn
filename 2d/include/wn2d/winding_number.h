#pragma once
#include "circle_region.h"
#include <Eigen/Core>
#include <vector>

// Public API for wn2d. Sign convention: CCW curves give positive winding numbers,
// so the interior of a closed CCW loop has winding number +1.

namespace wn2d {

/// Generalized winding number of a single cubic Bezier curve around p.
double winding_number(
    const BezierCurve& curve,
    const Eigen::Vector2d& p,
    double tol = 1e-9);

/// Sum of per-curve winding numbers around p. Curves do not need to form a closed loop.
double winding_number(
    const std::vector<BezierCurve>& curves,
    const Eigen::Vector2d& p,
    double tol = 1e-9);

/// Evaluate winding numbers on a uniform resolution x resolution grid.
/// One ray per curve per row is reused across all columns. Results are
/// bit-identical to calling winding_number at each cell.
/// W(row, col) maps row 0 to y_min and col 0 to x_min.
Eigen::MatrixXd winding_number_grid(
    const std::vector<BezierCurve>& curves,
    double x_min, double x_max,
    double y_min, double y_max,
    int resolution,
    double tol = 1e-9);

/// Same as above, but the bounding box is derived from the curves and padded
/// by margin times the longest side.
Eigen::MatrixXd winding_number_grid(
    const std::vector<BezierCurve>& curves,
    int resolution,
    double margin = 0.1,
    double tol    = 1e-9);

} // namespace wn2d
