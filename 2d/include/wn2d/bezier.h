#pragma once
#include <Eigen/Core>
#include <array>
#include <utility>

namespace wn2d {

/// Axis-aligned bounding box in 2D.
struct BBox2 {
    Eigen::Vector2d lo, hi;

    /// Returns true if p is inside or on the boundary of this box.
    bool contains(const Eigen::Vector2d& p) const {
        return p.x() >= lo.x() && p.x() <= hi.x() &&
               p.y() >= lo.y() && p.y() <= hi.y();
    }
};

/// Cubic Bézier curve  γ(t) = Σ_{i=0}^{3} B_{3,i}(t) · P[i],  t ∈ [0, 1].
///
/// P[0] and P[3] are the start and end points; P[1] and P[2] are control points.
/// The curve interpolates the endpoints: γ(0) = P[0], γ(1) = P[3].
struct BezierCurve {
    std::array<Eigen::Vector2d, 4> P;  ///< Control points: P[0]=start, P[3]=end.

    /// Evaluate the curve at parameter t ∈ [0, 1].
    Eigen::Vector2d eval(double t) const;

    /// Evaluate the first derivative (tangent vector, not unit-length) at t.
    Eigen::Vector2d tangent(double t) const;

    /// De Casteljau subdivision at t.
    /// Returns {left, right} where left covers [0, t] and right covers [t, 1],
    /// both reparametrised to [0, 1].
    std::pair<BezierCurve, BezierCurve> split(double t) const;

    /// Axis-aligned bounding box of the four control points.
    /// By the convex-hull property this is a valid outer bound for the curve.
    BBox2 bbox() const;

    /// Maximum perpendicular deviation of P[1] and P[2] from the chord P[0]–P[3].
    /// A curve is "flat" when flatness() < tolerance.
    double flatness() const;
};

} // namespace wn2d
