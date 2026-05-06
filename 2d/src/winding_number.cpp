#include "wn2d/winding_number.h"
#include "wn2d/intersect.h"
#include <algorithm>
#include <cmath>

namespace wn2d {

namespace {

constexpr double TWO_PI = 2.0 * M_PI;

inline double norm_angle(double a) {
    a = std::fmod(a, TWO_PI);
    if (a < 0.0) a += TWO_PI;
    return a;
}

// Fractional winding number contribution of a single Bezier curve.
// chi = signed crossing count of the +x ray from p through this curve only.
//
// The two endpoint angles split the circle into an "inner" arc (CCW from
// theta_A to theta_B, span = delta) and an "outer" arc (complement).
// Chi tells us which arc the +x direction falls in:
//   theta_A < theta_B → +x is in the outer arc → chi_outer = chi
//   theta_A > theta_B → +x is in the inner arc → chi_inner = chi
// wn = chi_outer * outer_frac + chi_inner * inner_frac
//    = chi_outer + inner_frac       (since chi_inner = chi_outer + 1)
double curve_contribution(const BezierCurve& curve,
                          const Eigen::Vector2d& p,
                          int chi)
{
    Eigen::Vector2d r0 = curve.P[0] - p;
    Eigen::Vector2d r3 = curve.P[3] - p;

    if (r0.squaredNorm() < 1e-14 || r3.squaredNorm() < 1e-14)
        return static_cast<double>(chi); // degenerate: full-circle contribution

    double theta_A = norm_angle(std::atan2(r0.y(), r0.x()));
    double theta_B = norm_angle(std::atan2(r3.y(), r3.x()));

    double delta = theta_B - theta_A;
    if (delta <= 0.0) delta += TWO_PI;

    // Degenerate: both endpoints at essentially the same angle
    if (delta < 1e-10 || TWO_PI - delta < 1e-10)
        return static_cast<double>(chi);

    double inner_frac = delta / TWO_PI;
    int chi_outer = (theta_A < theta_B) ? chi : chi - 1;
    return static_cast<double>(chi_outer) + inner_frac;
}

} // anonymous namespace

double winding_number(const BezierCurve& curve, const Eigen::Vector2d& p, double tol)
{
    auto hits = ray_bezier_intersect(curve, p.y(), p.x(), tol);
    int chi = 0;
    for (auto& h : hits) chi += h.sign;
    return curve_contribution(curve, p, chi);
}

double winding_number(const std::vector<BezierCurve>& curves,
                      const Eigen::Vector2d& p, double tol)
{
    double wn = 0.0;
    for (const auto& c : curves) {
        auto hits = ray_bezier_intersect(c, p.y(), p.x(), tol);
        int chi = 0;
        for (auto& h : hits) chi += h.sign;
        wn += curve_contribution(c, p, chi);
    }
    return wn;
}

// Grid evaluation: for each row, shoot one ray per curve and build a per-curve
// suffix-sum of chi.  Each query point does O(K log H) lookups (K curves,
// H hits per row) plus K calls to curve_contribution — no per-point ray shooting.
Eigen::MatrixXd winding_number_grid(
    const std::vector<BezierCurve>& curves,
    double x_min, double x_max,
    double y_min, double y_max,
    int resolution,
    double tol)
{
    Eigen::MatrixXd W(resolution, resolution);
    W.setZero();
    if (curves.empty()) return W;

    double dx = (x_max - x_min) / (resolution - 1);
    double dy = (y_max - y_min) / (resolution - 1);

    struct Hit { double x; int sign; };

    // Per-curve storage reused across rows
    int K = static_cast<int>(curves.size());
    std::vector<std::vector<Hit>> row_hits(K);
    std::vector<std::vector<int>> chi_suffix(K);

    for (int row = 0; row < resolution; ++row) {
        double y = y_min + row * dy;

        // Shoot one ray per curve for this row
        for (int k = 0; k < K; ++k) {
            auto isects = ray_bezier_intersect(curves[k], y, -1e300, tol);
            auto& hk = row_hits[k];
            hk.clear();
            for (auto& h : isects) hk.push_back({h.x, h.sign});
            // isects are already sorted by x

            // Build suffix-sum: chi_suffix[k][j] = sum of signs for hits[j..n-1]
            int n = static_cast<int>(hk.size());
            chi_suffix[k].assign(n + 1, 0);
            for (int j = n - 1; j >= 0; --j)
                chi_suffix[k][j] = chi_suffix[k][j + 1] + hk[j].sign;
        }

        for (int col = 0; col < resolution; ++col) {
            double x = x_min + col * dx;
            Eigen::Vector2d p(x, y);

            double wn = 0.0;
            for (int k = 0; k < K; ++k) {
                // Binary search: first hit strictly to the right of x
                const auto& hk = row_hits[k];
                int j = static_cast<int>(
                    std::upper_bound(hk.begin(), hk.end(), x,
                        [](double val, const Hit& h){ return val < h.x; })
                    - hk.begin());
                int chi = chi_suffix[k][j];
                wn += curve_contribution(curves[k], p, chi);
            }
            W(row, col) = wn;
        }
    }

    return W;
}

Eigen::MatrixXd winding_number_grid(
    const std::vector<BezierCurve>& curves,
    int resolution,
    double margin,
    double tol)
{
    if (curves.empty()) return Eigen::MatrixXd::Zero(resolution, resolution);

    Eigen::Vector2d lo = curves[0].P[0], hi = curves[0].P[0];
    for (const auto& c : curves)
        for (const auto& pt : c.P) { lo = lo.cwiseMin(pt); hi = hi.cwiseMax(pt); }

    Eigen::Vector2d extent = hi - lo;
    double m = margin * std::max(extent.x(), extent.y());
    return winding_number_grid(curves,
                               lo.x()-m, hi.x()+m,
                               lo.y()-m, hi.y()+m,
                               resolution, tol);
}

} // namespace wn2d
