#include <catch2/catch_test_macros.hpp>

#include "gwn3d/mesh_surface.h"

TEST_CASE("MeshSurface extracts a boundary loop from an open mesh", "[mesh][smoke]") {
    // A pyramid roof over a square base: four triangles meeting at an apex
    // leave one 4-vertex boundary loop (the base square).
    Eigen::MatrixXd V(5, 3);
    V << 0, 0, 0,
         1, 0, 0,
         1, 1, 0,
         0, 1, 0,
         0.5, 0.5, 1;
    Eigen::MatrixXi F(4, 3);
    F << 0, 1, 4,
         1, 2, 4,
         2, 3, 4,
         3, 0, 4;

    gwn3d::MeshSurface s(V, F);
    auto patches = s.boundary_patches();
    REQUIRE(patches.size() == 1);
    REQUIRE(patches[0].curve.rows() == 4);
}
