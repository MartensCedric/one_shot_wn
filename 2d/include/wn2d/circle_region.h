#pragma once
#include "bezier.h"
#include <vector>

namespace wn2d {

/// One of the two arc regions on the unit circle around a query point p.
///
/// The two endpoint projection angles of a Bézier curve split the unit circle
/// into exactly two arcs.  Adjacent arcs always differ in winding number by 1,
/// so a single ray suffices to anchor the absolute values.
///
/// Angles are measured CCW from the +x axis and lie in [0, 2π).
/// theta_end > theta_start; if the arc wraps past 2π, theta_end may exceed 2π.
struct ArcRegion {
    double theta_start;    ///< Start angle in [0, 2π).
    double theta_end;      ///< End angle > theta_start (may exceed 2π for the wrap arc).
    double arc_fraction;   ///< Fraction of the full circle: (theta_end − theta_start) / 2π.
    int    winding_number; ///< Absolute winding number (chi) for this arc region.
};

/// Result of splitting the unit circle around p into arc regions for one curve.
struct CircleRegionInfo {
    /// Exactly 2 arcs whose arc_fractions sum to 1.0, sorted by theta_start.
    /// If degenerate == true, contains a single placeholder arc.
    std::vector<ArcRegion> arcs;
    Eigen::Vector2d query_point;
    bool degenerate = false; ///< True when p coincides with a curve endpoint.
};

/// Compute the two-arc decomposition of the unit circle around p induced by curve.
///
/// The start angle θ_A = atan2(P[0]−p) and end angle θ_B = atan2(P[3]−p) partition
/// the circle into an inner arc (CCW from θ_A to θ_B, span δ) and an outer arc
/// (complement, span 2π − δ).  A single +x ray intersection determines the
/// winding number for the arc containing θ = 0; the other arc is that value ± 1.
///
/// Degenerate cases (p on an endpoint, or both endpoints at the same angle)
/// set CircleRegionInfo::degenerate = true.
///
/// @param curve          Cubic Bézier curve.
/// @param p              Query point.
/// @param angle_tol      Reserved for API compatibility (unused).
/// @param intersect_tol  Tolerance forwarded to ray_bezier_intersect.
CircleRegionInfo compute_circle_regions(
    const BezierCurve& curve,
    const Eigen::Vector2d& p,
    double angle_tol     = 1e-10,
    double intersect_tol = 1e-9);

} // namespace wn2d
