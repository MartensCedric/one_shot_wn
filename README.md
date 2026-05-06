# One-Shot Method for Computing Generalized Winding Numbers

[Project Page](https://martenscedric.github.io/academic-page/publications/1s_wn.html)

# Citing

```bibtex
@article{Martens2025WindingNumberOneShot,
  title  = {One-Shot Method for Computing Generalized Winding Numbers},
  author = {Martens, Cedric and Bessmeltsev, Mikhail},
  journal = {Computer Graphics Forum},
  doi    = {10.1111/cgf.70194},
  volume = {44},
  number = {5},
  year   = {2025},
}
```

---

## 2D Library — `wn2d`

A self-contained C++17 library for computing generalized winding numbers of
planar cubic Bézier curve networks.  Single header to include, one CMake
target to link, no dependencies beyond Eigen.

### What it computes

For a query point **p** and a collection of cubic Bézier curves, `wn2d` returns
the **generalized winding number** — a real-valued extension of the classical
integer winding number that is well-defined even for open curves.

- For a closed CCW loop the interior has winding number **+1**.
- For an open curve the result is a real value in (−1, 1) representing the
  fractional solid angle subtended by the curve.
- Multiple curves are summed independently; the curves do not need to form a
  connected or closed network.

The algorithm is the 2D analogue of the one-shot method from the paper: the two
endpoints of each curve are projected onto the unit circle around **p**, splitting
it into two arc regions whose winding numbers differ by exactly 1.  A single ray
intersection anchors the absolute value — no angular quadrature or subdivision is
needed.

For grid evaluation an additional **ray-reuse** optimisation reduces the total
ray count from K × N² to K × N for a K-curve scene on an N × N grid.

### Dependencies

| Dependency | Version | Notes |
|-----------|---------|-------|
| **C++ compiler** | C++17 or later | GCC ≥ 9, Clang ≥ 9, MSVC ≥ 19.14 |
| **CMake** | ≥ 3.20 | |
| **Eigen** | ≥ 3.3 | Header-only; detected automatically via `find_package(Eigen3)` |

Catch2 v3 is fetched automatically by CMake when building the tests.

### Building

```bash
# Clone (and init submodules if building the 3D part too)
git clone https://github.com/martenscedric/one_shot_wn.git
cd one_shot_wn

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target wn2d wn2d_tests wn2d_benchmark winding_field_plot

# Run tests
ctest --test-dir build -R wn2d --output-on-failure

# Run the benchmark (500×500 grid, polyline N=100)
./build/2d/examples/wn2d_benchmark

# Render a winding-number field image (outputs arch_wn.ppm)
./build/2d/examples/winding_field_plot
```

### Integrating into your CMake project

**Option A — `add_subdirectory`** (if you vendored the `2d/` folder):

```cmake
add_subdirectory(path/to/wn2d/2d)   # exposes the `wn2d` target
target_link_libraries(my_app PRIVATE wn2d)
```

**Option B — FetchContent**:

```cmake
include(FetchContent)
FetchContent_Declare(
  wn2d
  GIT_REPOSITORY https://github.com/martenscedric/one_shot_wn.git
  GIT_TAG        main
  SOURCE_SUBDIR  2d
)
FetchContent_MakeAvailable(wn2d)
target_link_libraries(my_app PRIVATE wn2d)
```

### Quick-start example

```cpp
#include <wn2d/winding_number.h>

// Build a unit-circle approximation from four cubic Bézier arcs.
const double k = 0.5522847498;   // standard cubic Bézier circle magic number
std::vector<wn2d::BezierCurve> circle = {
    {{{ {1,0}, {1,k}, {k,1}, {0,1} }}},
    {{{ {0,1}, {-k,1}, {-1,k}, {-1,0} }}},
    {{{ {-1,0}, {-1,-k}, {-k,-1}, {0,-1} }}},
    {{{ {0,-1}, {k,-1}, {1,-k}, {1,0} }}}
};

// Per-point query
double wn_inside  = wn2d::winding_number(circle, {0.0, 0.0});   // ≈ +1
double wn_outside = wn2d::winding_number(circle, {2.0, 0.0});   // ≈  0

// Grid query with ray reuse (fast path for dense evaluation)
// Returns a 256×256 Eigen matrix; W(row, col), row 0 = bottom.
Eigen::MatrixXd W = wn2d::winding_number_grid(
    circle,
    /*x_min*/ -1.5, /*x_max*/ 1.5,
    /*y_min*/ -1.5, /*y_max*/ 1.5,
    /*resolution*/ 256);
```

### API reference

All public symbols live in the `wn2d` namespace.  Include `<wn2d/winding_number.h>`.

#### Core types (`<wn2d/bezier.h>`)

```cpp
struct BezierCurve {
    std::array<Eigen::Vector2d, 4> P;   // P[0]=start, P[3]=end
    Eigen::Vector2d eval(double t) const;
    Eigen::Vector2d tangent(double t) const;
    std::pair<BezierCurve, BezierCurve> split(double t) const;
    BBox2  bbox() const;
    double flatness() const;
};
```

#### Winding number (`<wn2d/winding_number.h>`)

| Function | Description |
|---------|-------------|
| `winding_number(curve, p)` | Single curve, single query point. |
| `winding_number(curves, p)` | Multiple curves, single query point (summed). |
| `winding_number_grid(curves, x_min, x_max, y_min, y_max, res)` | N×N grid with ray reuse. Returns `Eigen::MatrixXd`. |
| `winding_number_grid(curves, res, margin)` | Same, auto-computes bbox from control polygons. |

All functions accept an optional `tol` parameter (default `1e-9`) that controls
the ray–Bézier intersection convergence threshold.

#### Ray intersection (`<wn2d/intersect.h>`)

```cpp
std::vector<RayIntersection> ray_bezier_intersect(
    const BezierCurve& curve,
    double y0,
    double x_origin = -1e300,
    double tol      = 1e-9,
    int    max_depth = 64);
```

Exposes the fat-line clipping intersection primitive for downstream use.

---

## 3D Reference Implementation

### Building

```bash
git submodule update --init --recursive   # libigl lives in extern/libigl
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Required (install separately): `GSL`, `Boost`, `CGAL`, `OpenMP` (optional).

### Running

```bash
cd build/3d/
./one_shot_wn_3d -n camel -m ../../inputs/camelhead.obj -q ../../inputs/camelhead_500_300.points
cd ../../
matlab -nodesktop -nosplash -nojvm -softwareopengl -batch "name='camel'; run('3d/visualize_results.m');"
```

