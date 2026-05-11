# One-Shot Method for Computing Generalized Winding Numbers

C++ implementation of [*One-Shot Method for Computing Generalized Winding Numbers*](https://martenscedric.github.io/academic-page/publications/1s_wn.html)
(Martens & Bessmeltsev, CGF 2025).

This repository hosts a reference implementation of both the 3D and 2D code.

## Citation

```bibtex
@article{Martens2025WindingNumberOneShot,
  title   = {One-Shot Method for Computing Generalized Winding Numbers},
  author  = {Martens, Cedric and Bessmeltsev, Mikhail},
  journal = {Computer Graphics Forum},
  doi     = {10.1111/cgf.70194},
  volume  = {44},
  number  = {5},
  year    = {2025},
}
```

# 2D code
## How does it work? 

The 2D one-shot formula: the two endpoints of each curve project onto the unit
circle around **p**, splitting it into two arcs whose `chi` differ by
exactly 1. A single ray interesection is needed to obtain both values. The GWN is obtained by weighing these chis with each region's arc length. 
subdivision. For dense grid queries, rays are reused row-by-row, meaning that less than a ray intersection is required on average!

## Build

Needs a C++17 compiler, CMake >= 3.20, and Eigen 3.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target wn2d wn2d_tests wn2d_benchmark winding_field_plot
```

Run the tests and the examples:

```bash
ctest --test-dir build -R wn2d --output-on-failure
./build/2d/examples/wn2d_benchmark
./build/2d/examples/winding_field_plot   # writes a PPM image
```

## Using it from CMake

```cmake
add_subdirectory(path/to/one_shot_wn/2d)
target_link_libraries(my_app PRIVATE wn2d)
```

Or pull it in with `FetchContent` and `SOURCE_SUBDIR 2d`.

## Example

```cpp
#include <wn2d/winding_number.h>

const double k = 0.5522847498;   // cubic-Bézier circle constant
std::vector<wn2d::BezierCurve> circle = {
    {{{ {1,0}, {1,k}, {k,1}, {0,1} }}},
    {{{ {0,1}, {-k,1}, {-1,k}, {-1,0} }}},
    {{{ {-1,0}, {-1,-k}, {-k,-1}, {0,-1} }}},
    {{{ {0,-1}, {k,-1}, {1,-k}, {1,0} }}}
};

double w_in  = wn2d::winding_number(circle, {0.0, 0.0});
double w_out = wn2d::winding_number(circle, {2.0, 0.0});

Eigen::MatrixXd W = wn2d::winding_number_grid(
    circle, -1.5, 1.5, -1.5, 1.5, /*resolution*/ 256);
```

The full API is in `2d/include/wn2d/winding_number.h`.

# 3D code

I am in the process of improving the code quality and adding examples. 

Email me at firstname.lastname@umontreal.ca for expedited assistance. 
