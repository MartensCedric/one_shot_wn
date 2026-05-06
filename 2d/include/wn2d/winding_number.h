#pragma once
#include "circle_region.h"
#include <Eigen/Core>
#include <vector>

/// @file winding_number.h
/// @brief Public API for the wn2d generalized winding number library.
///
/// Include this single header to access all winding number functionality.
/// Link against the `wn2d` CMake target (which transitively provides Eigen).
///
/// ### Sign convention
/// Positive winding numbers correspond to CCW curve orientation.
/// For a closed CCW loop the interior has winding number +1.
///
/// ### Degenerate inputs
/// Results are undefined when the query point lies exactly on the curve.
/// Points very close to a curve endpoint may trigger the degenerate path
/// in compute_circle_regions and return a best-effort integer value.

namespace wn2d {

/// Generalized winding number of a single open or closed cubic Bézier curve
/// around the query point p.
///
/// Uses the one-shot two-arc formula: the projection angles of the two
/// endpoints onto the unit circle around p define two arc regions whose
/// winding numbers differ by 1.  A single +x ray determines the absolute
/// value; no angular quadrature is performed.
///
/// For a closed curve (P[3] == P[0]) the result is an integer ± ε where
/// ε is determined by the Bézier approximation error only.
/// For an open curve the result is a real number in (−1, 1).
///
/// @param curve  Cubic Bézier curve.
/// @param p      Query point (must not lie on the curve).
/// @param tol    Ray-intersection convergence tolerance (default 1e-9).
/// @return       Generalized winding number.
double winding_number(
    const BezierCurve& curve,
    const Eigen::Vector2d& p,
    double tol = 1e-9);

/// Generalized winding number of a collection of cubic Bézier curves around p.
///
/// The total winding number is the sum of each curve's independent contribution.
/// Each curve is processed with its own ray intersection; curves do not need to
/// form a closed or connected loop.
///
/// @param curves  Collection of cubic Bézier curves.
/// @param p       Query point (must not lie on any curve).
/// @param tol     Ray-intersection convergence tolerance (default 1e-9).
/// @return        Generalized winding number.
double winding_number(
    const std::vector<BezierCurve>& curves,
    const Eigen::Vector2d& p,
    double tol = 1e-9);

/// Evaluate winding numbers on a uniform resolution×resolution grid over
/// [x_min, x_max] × [y_min, y_max] using one-shot row-wise ray reuse.
///
/// For each row y_i the algorithm shoots exactly K rays (one per curve) and
/// builds per-curve suffix-sum arrays of signed crossings.  Each of the
/// resolution query points in the row then looks up its chi value in O(K log H)
/// time (H = hits per row per curve), with no additional ray shooting.
/// Total ray count: K × resolution  (vs K × resolution² for per-point evaluation).
///
/// The returned matrix W has dimensions (resolution × resolution):
///   W(row, col)  where row 0 corresponds to y_min and col 0 to x_min.
///
/// Grid results are bit-identical to per-point winding_number evaluation.
///
/// @param curves      Collection of cubic Bézier curves.
/// @param x_min       Left boundary of the query grid.
/// @param x_max       Right boundary of the query grid.
/// @param y_min       Bottom boundary of the query grid.
/// @param y_max       Top boundary of the query grid.
/// @param resolution  Number of samples along each axis (grid is resolution²).
/// @param tol         Ray-intersection convergence tolerance (default 1e-9).
/// @return            Matrix of winding numbers, shape (resolution, resolution).
Eigen::MatrixXd winding_number_grid(
    const std::vector<BezierCurve>& curves,
    double x_min, double x_max,
    double y_min, double y_max,
    int resolution,
    double tol = 1e-9);

/// Convenience overload: automatically computes the bounding box from the
/// control polygons of all curves and expands it by a fractional margin.
///
/// @param curves      Collection of cubic Bézier curves.
/// @param resolution  Number of samples along each axis.
/// @param margin      Fractional padding relative to the longest bbox side (default 0.1).
/// @param tol         Ray-intersection convergence tolerance (default 1e-9).
/// @return            Matrix of winding numbers, shape (resolution, resolution).
Eigen::MatrixXd winding_number_grid(
    const std::vector<BezierCurve>& curves,
    int resolution,
    double margin = 0.1,
    double tol    = 1e-9);

} // namespace wn2d
