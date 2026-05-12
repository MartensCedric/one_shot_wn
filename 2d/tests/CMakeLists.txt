add_executable(wn2d_tests
  test_bezier.cpp
  test_intersect.cpp
  test_circle_region.cpp
  test_winding_number.cpp
)

target_link_libraries(wn2d_tests PRIVATE wn2d Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(wn2d_tests)
