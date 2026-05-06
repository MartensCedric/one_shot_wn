#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "wn2d/bezier.h"

using namespace wn2d;
using Catch::Matchers::WithinAbs;

static BezierCurve quarter_arc() {
    // Standard quarter-arc approximation of the unit circle
    return {{{ {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0} }}};
}

TEST_CASE("BezierCurve eval at endpoints and midpoint", "[bezier]") {
    auto c = quarter_arc();

    auto p0 = c.eval(0.0);
    REQUIRE_THAT(p0.x(), WithinAbs(0.0, 1e-14));
    REQUIRE_THAT(p0.y(), WithinAbs(0.0, 1e-14));

    auto p1 = c.eval(1.0);
    REQUIRE_THAT(p1.x(), WithinAbs(0.0, 1e-14));
    REQUIRE_THAT(p1.y(), WithinAbs(1.0, 1e-14));

    // Analytic midpoint: 3/8*(1,0) + 3/8*(1,1) + 0 = (3/4, 3/8+1/8)
    // = 1/8*(0,0) + 3/8*(1,0) + 3/8*(1,1) + 1/8*(0,1)
    // x: 0 + 3/8 + 3/8 + 0 = 3/4
    // y: 0 + 0   + 3/8 + 1/8 = 1/2
    auto pm = c.eval(0.5);
    REQUIRE_THAT(pm.x(), WithinAbs(0.75, 1e-14));
    REQUIRE_THAT(pm.y(), WithinAbs(0.5,  1e-14));
}

TEST_CASE("BezierCurve split consistency", "[bezier]") {
    auto c = quarter_arc();
    const double T = 0.3;
    auto [left, right] = c.split(T);

    // Endpoints of the split pieces must match the original curve
    auto junction = c.eval(T);
    REQUIRE_THAT((left.eval(1.0)  - junction).norm(), WithinAbs(0.0, 1e-13));
    REQUIRE_THAT((right.eval(0.0) - junction).norm(), WithinAbs(0.0, 1e-13));
    REQUIRE_THAT((left.eval(0.0)  - c.eval(0.0)).norm(), WithinAbs(0.0, 1e-13));
    REQUIRE_THAT((right.eval(1.0) - c.eval(1.0)).norm(), WithinAbs(0.0, 1e-13));

    // Interior values must agree with the original
    REQUIRE_THAT((left.eval(0.5)  - c.eval(T * 0.5)).norm(),          WithinAbs(0.0, 1e-12));
    REQUIRE_THAT((right.eval(0.5) - c.eval(T + (1.0-T)*0.5)).norm(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("BezierCurve bbox contains all curve points", "[bezier]") {
    auto c = quarter_arc();
    BBox2 bb = c.bbox();

    for (int i = 0; i <= 1000; ++i) {
        double t = static_cast<double>(i) / 1000.0;
        auto pt = c.eval(t);
        REQUIRE(bb.contains(pt));
    }
}

TEST_CASE("BezierCurve tangent at t=0 matches finite difference", "[bezier]") {
    auto c = quarter_arc();
    auto tang = c.tangent(0.0);
    const double h = 1e-7;
    auto fd = (c.eval(h) - c.eval(0.0)) / h;
    REQUIRE_THAT((tang - fd).norm(), WithinAbs(0.0, 1e-5));
}

TEST_CASE("BezierCurve tangent at t=1 matches finite difference", "[bezier]") {
    auto c = quarter_arc();
    auto tang = c.tangent(1.0);
    const double h = 1e-7;
    auto fd = (c.eval(1.0) - c.eval(1.0 - h)) / h;
    REQUIRE_THAT((tang - fd).norm(), WithinAbs(0.0, 1e-5));
}

TEST_CASE("BezierCurve flatness of straight line is zero", "[bezier]") {
    // Collinear control points -> flatness == 0
    BezierCurve line;
    line.P = {{ {0.0,0.0}, {1.0/3,0.0}, {2.0/3,0.0}, {1.0,0.0} }};
    REQUIRE_THAT(line.flatness(), WithinAbs(0.0, 1e-14));
}

TEST_CASE("BezierCurve flatness is positive for non-linear curve", "[bezier]") {
    auto c = quarter_arc();
    REQUIRE(c.flatness() > 0.0);
}
