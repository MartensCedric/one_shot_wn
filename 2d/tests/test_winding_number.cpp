#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "wn2d/winding_number.h"
#include <cmath>

using namespace wn2d;
using Catch::Matchers::WithinAbs;

static std::vector<BezierCurve> unit_circle_bezier()
{
    const double k = 0.5522847498;
    return {
        {{{ {1.0, 0.0}, {1.0, k}, {k, 1.0}, {0.0, 1.0} }}},
        {{{ {0.0, 1.0}, {-k, 1.0}, {-1.0, k}, {-1.0, 0.0} }}},
        {{{ {-1.0, 0.0}, {-1.0, -k}, {-k, -1.0}, {0.0, -1.0} }}},
        {{{ {0.0, -1.0}, {k, -1.0}, {1.0, -k}, {1.0, 0.0} }}}
    };
}

// Numerical winding number via dense polyline approximation (reference)
static double winding_number_quadrature(const BezierCurve& c,
                                         const Eigen::Vector2d& p,
                                         int n = 10000)
{
    double angle = 0.0;
    Eigen::Vector2d prev = c.eval(0.0) - p;
    for (int i = 1; i <= n; ++i) {
        double t = static_cast<double>(i) / n;
        Eigen::Vector2d curr = c.eval(t) - p;
        double cross = prev.x()*curr.y() - prev.y()*curr.x();
        double dot   = prev.x()*curr.x() + prev.y()*curr.y();
        angle += std::atan2(cross, dot);
        prev = curr;
    }
    return angle / (2.0 * M_PI);
}

TEST_CASE("Unit circle: inside query has wn ≈ 1", "[winding_number]") {
    auto circle = unit_circle_bezier();
    double wn = winding_number(circle, {0.0, 0.0});
    REQUIRE_THAT(wn, WithinAbs(1.0, 0.02));
}

TEST_CASE("Unit circle: outside query has wn ≈ 0", "[winding_number]") {
    auto circle = unit_circle_bezier();
    REQUIRE_THAT(winding_number(circle, {2.0, 0.0}), WithinAbs(0.0, 0.02));
}

TEST_CASE("Unit circle: multiple exterior points", "[winding_number]") {
    auto circle = unit_circle_bezier();
    for (auto [px,py] : std::vector<std::pair<double,double>>{
            {3.0,0.0},{0.0,3.0},{-2.0,0.0},{1.5,1.5}}) {
        REQUIRE_THAT(winding_number(circle, {px, py}), WithinAbs(0.0, 0.05));
    }
}

TEST_CASE("Open C-curve agrees with quadrature", "[winding_number]") {
    BezierCurve c;
    c.P = {{ {0.0,0.0}, {2.0,0.0}, {2.0,1.0}, {0.0,1.0} }};
    Eigen::Vector2d p(1.0, 0.5);
    double ref = winding_number_quadrature(c, p, 10000);
    double wn  = winding_number(c, p);
    REQUIRE_THAT(wn, WithinAbs(ref, 1e-3));
}

TEST_CASE("Straight arc (-1,0)→(1,0): wn=0.25 at (0,1)", "[winding_number]") {
    // The two endpoints project to angles 5π/4 and 7π/4 from (0,1).
    // Inner arc spans π/2 → inner_frac=0.25. Chi=0 (ray at y=1 misses the y=0 line).
    // wn = chi_outer + inner_frac = 0 + 0.25 = 0.25.
    BezierCurve c;
    c.P = {{ {-1.0,0.0}, {-1.0/3,0.0}, {1.0/3,0.0}, {1.0,0.0} }};
    Eigen::Vector2d p(0.0, 1.0);
    double ref = winding_number_quadrature(c, p, 10000);
    double wn  = winding_number(c, p);
    REQUIRE_THAT(ref, WithinAbs(0.25, 1e-4));
    REQUIRE_THAT(wn,  WithinAbs(0.25, 1e-6));
}

TEST_CASE("Grid matches point-by-point evaluation", "[winding_number]") {
    BezierCurve c;
    c.P = {{ {0.0,0.0}, {2.0,0.0}, {2.0,1.0}, {0.0,1.0} }};
    std::vector<BezierCurve> curves = {c};

    const int res = 10;
    const double x0 = -0.5, x1 = 2.5, y0 = -0.5, y1 = 1.5;
    double dx = (x1-x0)/(res-1), dy = (y1-y0)/(res-1);

    auto W = winding_number_grid(curves, x0, x1, y0, y1, res);

    for (int row = 0; row < res; ++row) {
        for (int col = 0; col < res; ++col) {
            double x = x0 + col * dx;
            double y = y0 + row * dy;
            double wn_pt = winding_number(curves, {x, y});
            REQUIRE_THAT(W(row, col), WithinAbs(wn_pt, 1e-9));
        }
    }
}

TEST_CASE("Grid reuses rays: same result as per-point for multi-curve", "[winding_number]") {
    // Unit circle (4 curves) on a 5×5 grid
    auto circle = unit_circle_bezier();
    const int res = 5;
    const double r = 1.5;
    auto W = winding_number_grid(circle, -r, r, -r, r, res);
    double dx = (2*r)/(res-1), dy = (2*r)/(res-1);
    for (int row = 0; row < res; ++row) {
        for (int col = 0; col < res; ++col) {
            double x = -r + col * dx;
            double y = -r + row * dy;
            double wn_pt = winding_number(circle, {x, y});
            REQUIRE_THAT(W(row, col), WithinAbs(wn_pt, 1e-9));
        }
    }
}

TEST_CASE("Single curve: per-point and grid agree exactly", "[winding_number]") {
    BezierCurve c;
    c.P = {{ {0.0,0.0}, {1.0,2.0}, {2.0,-1.0}, {3.0,1.0} }};
    std::vector<BezierCurve> curves = {c};
    const int res = 8;
    auto W = winding_number_grid(curves, -0.5, 3.5, -1.5, 2.5, res);
    double dx = 4.0/(res-1), dy = 4.0/(res-1);
    for (int row = 0; row < res; ++row) {
        for (int col = 0; col < res; ++col) {
            Eigen::Vector2d p(-0.5 + col*dx, -1.5 + row*dy);
            REQUIRE_THAT(W(row, col), WithinAbs(winding_number(c, p), 1e-9));
        }
    }
}

TEST_CASE("Open arc quadrature vs one-shot", "[winding_number]") {
    BezierCurve c;
    c.P = {{ {0.0,0.0}, {1.0,2.0}, {2.0,-1.0}, {3.0,1.0} }};
    // Note: avoid query points where a curve endpoint lies exactly on the +x ray
    // (i.e. p.y == P[0].y or p.y == P[3].y), as that creates a degenerate
    // boundary condition not covered by the 2-arc formula.
    for (auto& p : std::vector<Eigen::Vector2d>{{0.5,0.5},{1.5,0.3},{2.5,0.5},{-1.0,-0.5}}) {
        double ref = winding_number_quadrature(c, p, 20000);
        double wn  = winding_number(c, p);
        REQUIRE_THAT(wn, WithinAbs(ref, 1e-2));
    }
}
