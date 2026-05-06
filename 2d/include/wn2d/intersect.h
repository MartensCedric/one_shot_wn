#pragma once
#include "bezier.h"
#include <vector>

namespace wn2d {

/// A single intersection of a horizontal ray with a Bézier curve.
struct RayIntersection {
    double t;    ///< Curve parameter in [0, 1] at the intersection.
    double x;    ///< x-coordinate of the intersection point.
    int    sign; ///< +1 if the curve crosses upward (tangent.y > 0),
                 ///<  -1 if downward.  Tangential hits (|tang.y| < 1e-14)
                 ///< are dropped before this struct is returned.
};

/// Find all intersections of the half-line  { (x, y0) : x ≥ x_origin }
/// with the given cubic Bézier curve.
///
/// Uses fat-line (Sederberg–Nishita 1990) clipping for robust, efficient
/// root isolation, falling back to bisection when convergence stalls.
///
/// The results are sorted by x in ascending order.  Tangential crossings
/// (where the curve merely touches y = y0 without crossing) are excluded.
///
/// @param curve      The cubic Bézier curve to intersect.
/// @param y0         y-coordinate of the horizontal ray.
/// @param x_origin   Left endpoint of the ray; intersections with x < x_origin
///                   are ignored.  Pass -1e300 to test the full infinite line.
/// @param tol        Convergence threshold on the y-component flatness (default 1e-9).
/// @param max_depth  Maximum recursion depth for the clipping stack (default 64).
/// @return           Sorted list of intersections with x ≥ x_origin.
std::vector<RayIntersection> ray_bezier_intersect(
    const BezierCurve& curve,
    double y0,
    double x_origin  = -1e300,
    double tol       = 1e-9,
    int    max_depth = 64);

} // namespace wn2d
