#include "wn2d/circle_region.h"
#include "wn2d/intersect.h"
#include <algorithm>
#include <cmath>

namespace wn2d {

namespace {

constexpr double TWO_PI = 2.0 * M_PI;

// Normalize angle to [0, 2π)
inline double norm_angle(double a) {
    a = std::fmod(a, TWO_PI);
    if (a < 0.0) a += TWO_PI;
    return a;
}

} // anonymous namespace

// Single-curve: the two endpoint angles split the circle into exactly 2 arcs.
// One ray (+x direction) determines chi for one arc; the other is chi ± 1.
CircleRegionInfo compute_circle_regions(
    const BezierCurve& curve,
    const Eigen::Vector2d& p,
    double /*angle_tol*/,
    double intersect_tol)
{
    CircleRegionInfo info;
    info.query_point = p;

    Eigen::Vector2d r0 = curve.P[0] - p;
    Eigen::Vector2d r3 = curve.P[3] - p;

    // Degenerate: either endpoint is on top of the query point
    if (r0.squaredNorm() < 1e-14 || r3.squaredNorm() < 1e-14) {
        info.degenerate = true;
        info.arcs.push_back({0.0, TWO_PI, 1.0, 0});
        return info;
    }

    double theta_A = norm_angle(std::atan2(r0.y(), r0.x()));
    double theta_B = norm_angle(std::atan2(r3.y(), r3.x()));

    // CCW angle from A to B = span of the inner arc
    double delta = theta_B - theta_A;
    if (delta <= 0.0) delta += TWO_PI;

    // Degenerate: both endpoints project to essentially the same angle
    if (delta < 1e-10 || TWO_PI - delta < 1e-10) {
        // Collapse to single arc; chi from one ray
        auto hits = ray_bezier_intersect(curve, p.y(), p.x(), intersect_tol);
        int chi = 0;
        for (auto& h : hits) chi += h.sign;
        info.arcs.push_back({0.0, TWO_PI, 1.0, chi});
        return info;
    }

    // Shoot one horizontal (+x) ray to determine chi for the arc containing θ=0
    auto hits = ray_bezier_intersect(curve, p.y(), p.x(), intersect_tol);
    int chi = 0;
    for (auto& h : hits) chi += h.sign;

    // The inner arc goes CCW from theta_A to theta_B (span = delta).
    // The outer arc goes CCW from theta_B to theta_A (span = 2π - delta).
    //
    // θ=0 is in the inner arc iff theta_A > theta_B
    // (the inner arc wraps past 0 only when theta_A > theta_B).
    bool ray_in_inner = (theta_A > theta_B);

    int chi_inner = ray_in_inner ? chi      : chi + 1;
    int chi_outer = ray_in_inner ? chi - 1  : chi;

    double inner_frac = delta / TWO_PI;
    double outer_frac = 1.0 - inner_frac;

    // Build the two arcs in CCW order (sorted by theta_start)
    ArcRegion inner_arc, outer_arc;
    inner_arc.theta_start  = theta_A;
    inner_arc.theta_end    = theta_A + delta;
    inner_arc.arc_fraction = inner_frac;
    inner_arc.winding_number = chi_inner;

    outer_arc.theta_start  = theta_B;
    outer_arc.theta_end    = theta_B + (TWO_PI - delta);
    outer_arc.arc_fraction = outer_frac;
    outer_arc.winding_number = chi_outer;

    // Sort by theta_start so the arcs are in CCW order on the circle
    if (theta_A <= theta_B) {
        info.arcs = {inner_arc, outer_arc};
    } else {
        info.arcs = {outer_arc, inner_arc};
    }

    return info;
}

} // namespace wn2d
