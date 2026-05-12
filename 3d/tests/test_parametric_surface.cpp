#include <catch2/catch_test_macros.hpp>

#include "gwn3d/parametric_surface.h"

static Eigen::Vector3d flat(Eigen::Vector2d uv) {
    return Eigen::Vector3d(uv(0), uv(1), 0.0);
}

TEST_CASE("ParametricSurface keeps the boundary it was given", "[parametric][smoke]") {
    patch_t border;
    border.is_open = false;
    border.curve.resize(4, 3);
    border.curve << 0, 0, 0,
                    1, 0, 0,
                    1, 1, 0,
                    0, 1, 0;

    gwn3d::ParametricSurface s(flat, { border });
    auto patches = s.boundary_patches();
    REQUIRE(patches.size() == 1);
    REQUIRE(patches[0].curve.rows() == 4);
}
