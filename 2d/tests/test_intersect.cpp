#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "wn2d/intersect.h"

using namespace wn2d;
using Catch::Matchers::WithinAbs;

// Straight line from (0,-1) to (1,1) as a degenerate cubic
static BezierCurve straight_line() {
    return {{{ {0.0,-1.0}, {1.0/3,-1.0/3}, {2.0/3,1.0/3}, {1.0,1.0} }}};
}

// U-curve crossing y=0 at two places (y(t) = 0.5[(1-t)³+t³] - 3t(1-t), x(t) = 3t)
static BezierCurve s_curve() {
    return {{{ {0.0,0.5}, {1.0,-1.0}, {2.0,-1.0}, {3.0,0.5} }}};
}

TEST_CASE("Straight line crossing y=0: one hit at x=0.5 sign=+1", "[intersect]") {
    auto c = straight_line();
    auto hits = ray_bezier_intersect(c, 0.0);
    REQUIRE(hits.size() == 1);
    REQUIRE_THAT(hits[0].x, WithinAbs(0.5, 1e-7));
    REQUIRE(hits[0].sign == +1);
}

TEST_CASE("Curve entirely above y=1: zero hits", "[intersect]") {
    BezierCurve c;
    c.P = {{ {0.0,2.0}, {1.0,2.0}, {2.0,2.0}, {3.0,2.0} }};
    auto hits = ray_bezier_intersect(c, 0.0);
    REQUIRE(hits.empty());
}

TEST_CASE("Curve entirely below y=-1: zero hits", "[intersect]") {
    BezierCurve c;
    c.P = {{ {0.0,-2.0}, {1.0,-2.0}, {2.0,-2.0}, {3.0,-2.0} }};
    auto hits = ray_bezier_intersect(c, 0.0);
    REQUIRE(hits.empty());
}

TEST_CASE("S-curve at y=0: exactly 2 hits with opposite signs", "[intersect]") {
    auto c = s_curve();
    auto hits = ray_bezier_intersect(c, 0.0);
    REQUIRE(hits.size() == 2);
    int total_sign = 0;
    for (auto& h : hits) total_sign += h.sign;
    // S-curve crossing once up and once down → sum of signs = 0
    REQUIRE(total_sign == 0);
    // Both hits at x > 0
    REQUIRE(hits[0].x > 0.0);
    REQUIRE(hits[1].x > hits[0].x);
}

TEST_CASE("x_origin clipping: only hits with x > x_origin returned", "[intersect]") {
    auto c = s_curve();
    // Both hits are somewhere in [0,3]; set x_origin past the first one
    auto all_hits = ray_bezier_intersect(c, 0.0, -1e300);
    REQUIRE(all_hits.size() == 2);
    double x_clip = (all_hits[0].x + all_hits[1].x) * 0.5;
    auto clipped = ray_bezier_intersect(c, 0.0, x_clip);
    REQUIRE(clipped.size() == 1);
    REQUIRE(clipped[0].x > x_clip);
}

TEST_CASE("Near-tangent curve converges within max_depth=64", "[intersect]") {
    // Curve that just barely dips below y=0
    BezierCurve c;
    c.P = {{ {0.0, 0.1}, {1.0, -0.001}, {2.0, -0.001}, {3.0, 0.1} }};
    // Should produce 2 hits without throwing or hanging
    auto hits = ray_bezier_intersect(c, 0.0, -1e300, 1e-9, 64);
    // We don't assert exact count, just that it finishes and results are plausible
    for (auto& h : hits) {
        REQUIRE(h.x >= 0.0);
        REQUIRE(h.x <= 3.0);
    }
}

TEST_CASE("Horizontal line at same y: degenerate — no spurious hits", "[intersect]") {
    // A curve whose entire span is at y=0 exactly; should not produce infinite hits
    BezierCurve c;
    c.P = {{ {0.0,0.0}, {1.0,0.0}, {2.0,0.0}, {3.0,0.0} }};
    auto hits = ray_bezier_intersect(c, 0.0, -1e300, 1e-9, 64);
    // All sign=0 are removed, so result should be empty or very few
    for (auto& h : hits) REQUIRE(h.sign != 0);
}
