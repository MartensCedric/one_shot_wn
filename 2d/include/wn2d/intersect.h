#pragma once
#include "bezier.h"
#include <vector>

namespace wn2d {

/// One intersection of a horizontal ray with a Bezier curve.
/// sign is +1 for upward crossings (tangent.y > 0) and -1 for downward.
/// Tangential touches are dropped.
struct RayIntersection {
    double t;
    double x;
    int    sign;
};

/// Find all intersections of the half-line {(x, y0) : x >= x_origin} with the curve.
/// Results are sorted by x ascending. Pass x_origin = -1e300 for the full line.
std::vector<RayIntersection> ray_bezier_intersect(
    const BezierCurve& curve,
    double y0,
    double x_origin  = -1e300,
    double tol       = 1e-9,
    int    max_depth = 64);

} // namespace wn2d
