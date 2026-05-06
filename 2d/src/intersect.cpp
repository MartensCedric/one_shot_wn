#include "wn2d/intersect.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace wn2d {

namespace {

// Clip [0,1] using the convex hull of {(u[i], d[i])} against d=0.
// u values are fixed: 0, 1/3, 2/3, 1.
// Returns false if no crossing; otherwise sets lo,hi.
bool bezier_clip_1d(const double d[4], double& lo, double& hi)
{
    static const double u[4] = {0.0, 1.0/3.0, 2.0/3.0, 1.0};

    lo =  1.0;
    hi = -1.0;

    for (int i = 0; i < 4; ++i) {
        for (int j = i+1; j < 4; ++j) {
            double d0 = d[i], d1 = d[j];
            double u0 = u[i], u1 = u[j];
            if (d0 == d1) {
                if (std::abs(d0) < 1e-14) { lo = std::min(lo, u0); hi = std::max(hi, u1); }
                continue;
            }
            if ((d0 > 0.0) == (d1 > 0.0) && d0 != 0.0 && d1 != 0.0) continue;
            double t_cross = u0 + (u1-u0) * (-d0) / (d1-d0);
            lo = std::min(lo, t_cross);
            hi = std::max(hi, t_cross);
        }
    }

    if (hi < lo) return false;
    lo = std::max(lo, 0.0);
    hi = std::min(hi, 1.0);
    return lo <= hi;
}

struct ClipTask {
    BezierCurve seg;
    double t_lo, t_hi;
    int    depth;
};

} // anonymous namespace

std::vector<RayIntersection> ray_bezier_intersect(
    const BezierCurve& curve,
    double y0,
    double x_origin,
    double tol,
    int max_depth)
{
    std::vector<RayIntersection> results;
    std::vector<ClipTask> stack;
    stack.reserve(64);
    stack.push_back({curve, 0.0, 1.0, 0});

    // sqrt(tol) threshold: convergence fires before rounding pushes d[0]→negative
    const double tol_sqrt = std::sqrt(tol);

    while (!stack.empty()) {
        auto [seg, t_lo, t_hi, depth] = stack.back();
        stack.pop_back();

        double d[4];
        for (int i = 0; i < 4; ++i) d[i] = seg.P[i].y() - y0;

        double max_abs_d = std::max({std::abs(d[0]),std::abs(d[1]),
                                     std::abs(d[2]),std::abs(d[3])});
        double t_span    = t_hi - t_lo;

        // Convergence check BEFORE sign-consistency reject: when max_abs_d is small
        // (below sqrt(tol)) the clipping can push the interval just past the root,
        // producing an all-negative segment that would be incorrectly rejected.
        // ALWAYS use the ORIGINAL curve for evaluation — clipped segments are degenerate.
        if (max_abs_d < tol_sqrt || t_span < tol || depth >= max_depth) {
            double t_mid = 0.5 * (t_lo + t_hi);
            Eigen::Vector2d pos  = curve.eval(t_mid);
            Eigen::Vector2d tang = curve.tangent(t_mid);
            if (pos.x() >= x_origin && std::abs(tang.y()) > 1e-14) {
                results.push_back({t_mid, pos.x(), (tang.y() > 0) ? +1 : -1});
            }
            continue;
        }

        // Quick sign-consistency reject (only after convergence is ruled out)
        {
            bool all_pos = d[0]>0&&d[1]>0&&d[2]>0&&d[3]>0;
            bool all_neg = d[0]<0&&d[1]<0&&d[2]<0&&d[3]<0;
            if (all_pos || all_neg) continue;
        }

        double u_lo, u_hi;
        bool has_crossing = bezier_clip_1d(d, u_lo, u_hi);
        if (!has_crossing) continue;

        double new_span = u_hi - u_lo;

        // If new_span is negligible, converge immediately
        if (new_span < 1e-10) {
            // Skip right-endpoint roots: the adjacent right interval will record them
            // when they appear as u=0 at its left end, preventing double-counting.
            if (u_hi >= 1.0 - 1e-10) continue;
            double t_hit = t_lo + 0.5*(u_lo + u_hi) * t_span;
            Eigen::Vector2d pos  = curve.eval(t_hit);
            // Guard against spurious hits where a control point has d=0 but the
            // actual curve doesn't cross y=y0 (clipping snaps lo=hi to the cp index).
            if (std::abs(pos.y() - y0) > tol_sqrt) continue;
            Eigen::Vector2d tang = curve.tangent(t_hit);
            if (pos.x() >= x_origin && std::abs(tang.y()) > 1e-14) {
                results.push_back({t_hit, pos.x(), (tang.y() > 0) ? +1 : -1});
            }
            continue;
        }

        // If clipping removed < 20% or past half max_depth → bisect
        if (new_span > 0.8 || depth >= max_depth / 2) {
            auto [left, right] = seg.split(0.5);
            double t_mid_g = t_lo + 0.5 * t_span;
            stack.push_back({right, t_mid_g, t_hi,   depth+1});
            stack.push_back({left,  t_lo,    t_mid_g, depth+1});
        } else {
            // Reparametrise to [u_lo, u_hi]
            BezierCurve clipped;
            if (u_lo > 1e-10) {
                auto [lft, rgt] = seg.split(u_lo);
                double t2 = (u_hi - u_lo) / (1.0 - u_lo);
                auto [cl, cr] = rgt.split(t2);
                clipped = cl;
            } else {
                auto [cl, cr] = seg.split(u_hi);
                clipped = cl;
            }
            double new_t_lo = t_lo + u_lo * t_span;
            double new_t_hi = t_lo + u_hi * t_span;
            stack.push_back({clipped, new_t_lo, new_t_hi, depth+1});
        }
    }

    // Sort by x, then merge duplicates within tol
    std::sort(results.begin(), results.end(),
              [](const RayIntersection& a, const RayIntersection& b){ return a.x < b.x; });

    std::vector<RayIntersection> unique;
    unique.reserve(results.size());
    for (auto& r : results) {
        if (!unique.empty() && std::abs(r.x - unique.back().x) < tol * 10.0) {
            unique.back().sign += r.sign;
        } else {
            unique.push_back(r);
        }
    }
    unique.erase(std::remove_if(unique.begin(), unique.end(),
                                [](const RayIntersection& r){ return r.sign == 0; }),
                 unique.end());

    return unique;
}

} // namespace wn2d
