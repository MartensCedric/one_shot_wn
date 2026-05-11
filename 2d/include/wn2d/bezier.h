#pragma once
#include <Eigen/Core>
#include <array>
#include <utility>

namespace wn2d {

/// Axis-aligned bounding box in 2D.
struct BBox2 {
    Eigen::Vector2d lo, hi;

    bool contains(const Eigen::Vector2d& p) const {
        return p.x() >= lo.x() && p.x() <= hi.x() &&
               p.y() >= lo.y() && p.y() <= hi.y();
    }
};

/// Cubic Bezier curve with control points P[0..3]. P[0] and P[3] are the endpoints.
struct BezierCurve {
    std::array<Eigen::Vector2d, 4> P;

    /// Position at parameter t in [0, 1].
    Eigen::Vector2d eval(double t) const;

    /// First derivative at t (not unit length).
    Eigen::Vector2d tangent(double t) const;

    /// De Casteljau split at t. Returns the two halves, each reparametrised to [0, 1].
    std::pair<BezierCurve, BezierCurve> split(double t) const;

    /// Bounding box of the control polygon (a valid bound for the curve itself).
    BBox2 bbox() const;

    /// Maximum perpendicular distance from the inner control points to the chord.
    double flatness() const;
};

} // namespace wn2d
