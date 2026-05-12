#include <catch2/catch_test_macros.hpp>

#include "curve_net_parser.h"

TEST_CASE("dimension_to_rays of {500, 300} -> 300 rays of length 500", "[grid]") {
    auto rays = dimension_to_rays({ 500, 300 });
    REQUIRE(rays.size() == 300);
    for (const auto& r : rays) REQUIRE(r.size() == 500);
}

TEST_CASE("slice_to_rays orientation matches axis", "[grid]") {
    auto rays = slice_to_rays(4, 3, 0);
    REQUIRE(rays.size() == 4);
    REQUIRE(rays[0].size() == 3);
}
