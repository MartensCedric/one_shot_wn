cmake_minimum_required(VERSION 3.20)

set(TARGET_NAME one_shot_wn_3d)

file(GLOB SRC CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
)

add_executable(${TARGET_NAME} ${SRC})

target_include_directories(${TARGET_NAME} PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}
)

if(MSVC)
  target_compile_options(${TARGET_NAME} PRIVATE /W4)
else()
  target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(${TARGET_NAME} PRIVATE ONE_SHOT_WN_HAS_OPENMP=1)
else()
  message(WARNING "OpenMP not found. Building without OpenMP.")
endif()

# Expecting: <repo_root>/extern/libigl
set(LIBIGL_DIR "${CMAKE_SOURCE_DIR}/extern/libigl")

if(NOT EXISTS "${LIBIGL_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "libigl submodule not found at: ${LIBIGL_DIR}\n"
    "Run:\n"
    "  git submodule update --init --recursive\n"
  )
endif()

set(LIBIGL_PREDICATES ON CACHE BOOL "" FORCE)
set(LIBIGL_COPYLEFT_CGAL ON CACHE BOOL "" FORCE)


set(LIBIGL_USE_STATIC_LIBRARY OFF CACHE BOOL "" FORCE)

add_subdirectory(${LIBIGL_DIR} ${CMAKE_BINARY_DIR}/_deps/libigl)

target_link_libraries(${TARGET_NAME} PRIVATE
  igl::core
  igl_predicates
  igl_copyleft::cgal
  gsl gslcblas m
)
