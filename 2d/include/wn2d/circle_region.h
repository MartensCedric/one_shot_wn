#pragma once
#include "bezier.h"
#include <vector>

namespace wn2d {

/// One of the two arcs on the unit circle around a query point.
/// Angles are CCW from the +x axis; theta_end may exceed 2*pi when the arc wraps.
struct ArcRegion {
    double theta_start;
    double theta_end;
    double arc_fraction;
    int    chi;
};

/// Two-arc split of the unit circle around a query point for one curve.
/// If degenerate, arcs holds a single placeholder spanning the full circle.
struct CircleRegionInfo {
    std::vector<ArcRegion> arcs;
    Eigen::Vector2d query_point;
    bool degenerate = false;
};

/// Split the unit circle around p into the two arcs induced by the curve's endpoint angles.
/// Sets degenerate = true when p sits on an endpoint or both endpoints share an angle.
CircleRegionInfo compute_circle_regions(
    const BezierCurve& curve,
    const Eigen::Vector2d& p,
    double angle_tol     = 1e-10,
    double intersect_tol = 1e-9);

} // namespace wn2d
