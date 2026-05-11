#include "wn2d/bezier.h"
#include <algorithm>
#include <cmath>

namespace wn2d {

Eigen::Vector2d BezierCurve::eval(double t) const
{
    const double s = 1.0 - t;
    return s*s*s * P[0]
         + 3.0*s*s*t * P[1]
         + 3.0*s*t*t * P[2]
         + t*t*t     * P[3];
}

Eigen::Vector2d BezierCurve::tangent(double t) const
{
    const double s = 1.0 - t;
    return 3.0 * (s*s * (P[1]-P[0])
                + 2.0*s*t * (P[2]-P[1])
                + t*t     * (P[3]-P[2]));
}

std::pair<BezierCurve, BezierCurve> BezierCurve::split(double t) const
{
    Eigen::Vector2d Q0 = (1-t)*P[0] + t*P[1];
    Eigen::Vector2d Q1 = (1-t)*P[1] + t*P[2];
    Eigen::Vector2d Q2 = (1-t)*P[2] + t*P[3];
    Eigen::Vector2d R0 = (1-t)*Q0   + t*Q1;
    Eigen::Vector2d R1 = (1-t)*Q1   + t*Q2;
    Eigen::Vector2d S0 = (1-t)*R0   + t*R1;

    BezierCurve left,  right;
    left.P  = {P[0], Q0, R0, S0};
    right.P = {S0,   R1, Q2, P[3]};
    return {left, right};
}

BBox2 BezierCurve::bbox() const
{
    Eigen::Vector2d lo = P[0], hi = P[0];
    for (int i = 1; i < 4; ++i) {
        lo = lo.cwiseMin(P[i]);
        hi = hi.cwiseMax(P[i]);
    }
    return {lo, hi};
}

double BezierCurve::flatness() const
{
    Eigen::Vector2d chord = P[3] - P[0];
    double len = chord.norm();
    if (len < 1e-14) {
        // Chord is degenerate, fall back to the largest control-point offset.
        return std::max({(P[1]-P[0]).norm(), (P[2]-P[0]).norm(), (P[3]-P[0]).norm()});
    }
    Eigen::Vector2d n(-chord.y()/len, chord.x()/len);
    return std::max(std::abs(n.dot(P[1] - P[0])),
                    std::abs(n.dot(P[2] - P[0])));
}

} // namespace wn2d
