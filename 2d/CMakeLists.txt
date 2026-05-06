cmake_minimum_required(VERSION 3.20)
project(wn2d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(wn2d STATIC
  src/bezier.cpp
  src/intersect.cpp
  src/circle_region.cpp
  src/winding_number.cpp
)

target_include_directories(wn2d PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(MSVC)
  target_compile_options(wn2d PRIVATE /W4)
else()
  target_compile_options(wn2d PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Reuse Eigen3::Eigen already configured by the 3d libigl step; fall back to find_package.
if(NOT TARGET Eigen3::Eigen)
  find_package(Eigen3 REQUIRED NO_MODULE)
endif()
target_link_libraries(wn2d PUBLIC Eigen3::Eigen)

add_subdirectory(tests)
add_subdirectory(examples)
