cmake_minimum_required(VERSION 3.20)

set(LIB_NAME   gwn3d)
set(EXEC_NAME  one_shot_wn_3d)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

find_package(OpenMP)

include(FetchContent)
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v3.2.0
)
FetchContent_MakeAvailable(cxxopts)

# Expecting: <repo_root>/extern/libigl
set(LIBIGL_DIR "${CMAKE_SOURCE_DIR}/extern/libigl")
if(NOT EXISTS "${LIBIGL_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "libigl submodule not found at: ${LIBIGL_DIR}\n"
    "Run: git submodule update --init --recursive\n"
  )
endif()
set(LIBIGL_PREDICATES ON CACHE BOOL "" FORCE)
set(LIBIGL_COPYLEFT_CGAL ON CACHE BOOL "" FORCE)
set(LIBIGL_USE_STATIC_LIBRARY OFF CACHE BOOL "" FORCE)
if(NOT TARGET igl::core)
  add_subdirectory(${LIBIGL_DIR} ${CMAKE_BINARY_DIR}/_deps/libigl)
endif()

# ---------------------------------------------------------------------------
# gwn3d library: core GWN computation + surface abstractions
# ---------------------------------------------------------------------------

file(GLOB GWN3D_SRC CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/src/surfaces/*.cpp"
)

add_library(${LIB_NAME} STATIC ${GWN3D_SRC})

target_include_directories(${LIB_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
)

set_target_properties(${LIB_NAME} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

if(MSVC)
  target_compile_options(${LIB_NAME} PRIVATE /W4)
else()
  target_compile_options(${LIB_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_link_libraries(${LIB_NAME} PUBLIC
  igl::core
  igl_predicates
  igl_copyleft::cgal
  gsl gslcblas m
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(${LIB_NAME} PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(${LIB_NAME} PUBLIC ONE_SHOT_WN_HAS_OPENMP=1)
else()
  message(WARNING "OpenMP not found. Building gwn3d without OpenMP.")
endif()

# ---------------------------------------------------------------------------
# one_shot_wn_3d executable: thin CLI
# ---------------------------------------------------------------------------

add_executable(${EXEC_NAME} app/main.cpp)
target_link_libraries(${EXEC_NAME} PRIVATE ${LIB_NAME} cxxopts)

# ---------------------------------------------------------------------------
# Examples + tests (off by default while skeleton is in place)
# ---------------------------------------------------------------------------

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/examples/CMakeLists.txt")
  add_subdirectory(examples)
endif()
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()
