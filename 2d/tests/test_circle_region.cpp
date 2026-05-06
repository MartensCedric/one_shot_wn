#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "wn2d/circle_region.h"
#include <cmath>
#include <numeric>

using namespace wn2d;
using Catch::Matchers::WithinAbs;

static BezierCurve horiz_line() {
    return {{{ {-1.0,0.0}, {-1.0/3,0.0}, {1.0/3,0.0}, {1.0,0.0} }}};
}

TEST_CASE("Always exactly 2 arcs for non-degenerate query", "[circle_region]") {
    BezierCurve c;
    c.P = {{ {0.0,0.0}, {1.0,0.0}, {1.0,1.0}, {0.0,1.0} }};
    Eigen::Vector2d p(0.5, 0.5);
    auto info = compute_circle_regions(c, p);
    REQUIRE(info.arcs.size() == 2);
}

TEST_CASE("Arc fractions sum to 1", "[circle_region]") {
    BezierCurve c;
    c.P = {{ {0.0,0.0}, {2.0,0.0}, {2.0,1.0}, {0.0,1.0} }};
    for (double px : {-1.0, 0.5, 1.0, 3.0}) {
        for (double py : {-1.0, 0.5, 2.0}) {
            Eigen::Vector2d p(px, py);
            auto info = compute_circle_regions(c, p);
            double sum = 0.0;
            for (auto& a : info.arcs) sum += a.arc_fraction;
            REQUIRE_THAT(sum, WithinAbs(1.0, 1e-9));
        }
    }
}

TEST_CASE("All arc fractions are positive", "[circle_region]") {
    BezierCurve c;
    c.P = {{ {-1.0,0.5}, {0.0,2.0}, {2.0,2.0}, {1.5,0.5} }};
    Eigen::Vector2d p(0.0, 0.0);
    auto info = compute_circle_regions(c, p);
    for (auto& a : info.arcs)
        REQUIRE(a.arc_fraction > 0.0);
}

TEST_CASE("Adjacent arcs differ in chi by exactly 1", "[circle_region]") {
    BezierCurve c;
    c.P = {{ {0.0,0.0}, {2.0,0.0}, {2.0,1.0}, {0.0,1.0} }};
    Eigen::Vector2d p(1.0, 0.5);
    auto info = compute_circle_regions(c, p);
    REQUIRE(info.arcs.size() == 2);
    int diff = std::abs(info.arcs[0].winding_number - info.arcs[1].winding_number);
    REQUIRE(diff == 1);
}

TEST_CASE("Straight horizontal line from (0,1): fractions 0.25 and 0.75", "[circle_region]") {
    // From p=(0,1):
    //   P[0]=(-1,0): r=(-1,-1) → angle 5π/4 ≈ 3.927
    //   P[3]=(1,0):  r=(1,-1)  → angle 7π/4 ≈ 5.498
    //   CCW span from 3.927 to 5.498 = π/2 → inner_frac = 0.25
    auto c = horiz_line();
    Eigen::Vector2d p(0.0, 1.0);
    auto info = compute_circle_regions(c, p);
    REQUIRE(info.arcs.size() == 2);
    double small_f = std::min(info.arcs[0].arc_fraction, info.arcs[1].arc_fraction);
    double large_f = std::max(info.arcs[0].arc_fraction, info.arcs[1].arc_fraction);
    REQUIRE_THAT(small_f, WithinAbs(0.25, 1e-6));
    REQUIRE_THAT(large_f, WithinAbs(0.75, 1e-6));
    int diff = std::abs(info.arcs[0].winding_number - info.arcs[1].winding_number);
    REQUIRE(diff == 1);
}

TEST_CASE("Degenerate: query point on an endpoint returns degenerate flag", "[circle_region]") {
    BezierCurve c;
    c.P = {{ {0.0,0.0}, {1.0,0.0}, {2.0,0.0}, {3.0,0.0} }};
    Eigen::Vector2d p(0.0, 0.0); // exactly on P[0]
    auto info = compute_circle_regions(c, p);
    REQUIRE(info.degenerate == true);
}

TEST_CASE("winding_number via arcs matches chi formula: chi_outer + inner_frac", "[circle_region]") {
    // Use the straight horizontal line from (0,1):
    //   +x ray at y=1 doesn't hit the line at y=0 → chi=0
    //   theta_A=5π/4, theta_B=7π/4  (theta_A < theta_B) → ray in outer arc → chi_outer=0
    //   inner_frac=0.25
    //   wn = 0 + 0.25 = 0.25
    auto c = horiz_line();
    Eigen::Vector2d p(0.0, 1.0);
    auto info = compute_circle_regions(c, p);
    double wn = 0.0;
    for (auto& a : info.arcs)
        wn += a.arc_fraction * static_cast<double>(a.winding_number);
    REQUIRE_THAT(wn, WithinAbs(0.25, 1e-6));
}
