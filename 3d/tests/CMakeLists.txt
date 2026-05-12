add_executable(gwn3d_tests
  test_grid.cpp
  test_mesh_surface.cpp
  test_parametric_surface.cpp
)
target_link_libraries(gwn3d_tests PRIVATE gwn3d Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(gwn3d_tests)
