// winding_field_plot.cpp
// Evaluates the one-shot winding number field of an arch Bezier curve on a 200×200
// grid and writes a PPM image (readable by any image viewer, no dependencies).
//
// Usage:
//   ./winding_field_plot
// Output:
//   arch_wn.ppm   — false-colour image of the winding number field

#include "wn2d/winding_number.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

namespace {

// Map a scalar to RGB: red (-1) → olive (0) → green (+1), clamped.
void value_to_rgb(double v,
                  unsigned char& r, unsigned char& g, unsigned char& b)
{
    double t = std::max(-1.0, std::min(1.0, v));  // clamp to [-1, 1]
    double s = (t + 1.0) * 0.5;                   // remap to [0, 1]
    // Linear: red (1,0,0) at s=0, green (0,1,0) at s=1
    r = static_cast<unsigned char>((1.0 - s) * 255);
    g = static_cast<unsigned char>(s * 255);
    b = 0;
}

void write_ppm(const std::string& path, const Eigen::MatrixXd& W)
{
    const int rows = static_cast<int>(W.rows());
    const int cols = static_cast<int>(W.cols());

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) { std::cerr << "Cannot open " << path << "\n"; return; }

    out << "P6\n" << cols << " " << rows << "\n255\n";
    // Row 0 = y_min (bottom). PPM origin is top-left, so flip vertically.
    for (int row = rows - 1; row >= 0; --row) {
        for (int col = 0; col < cols; ++col) {
            unsigned char r, g, b;
            value_to_rgb(W(row, col), r, g, b);
            out.put(static_cast<char>(r));
            out.put(static_cast<char>(g));
            out.put(static_cast<char>(b));
        }
    }
    std::cout << "Wrote " << path << " (" << cols << "x" << rows << " PPM)\n";
}

} // anonymous namespace

int main()
{
    using namespace wn2d;

    // Symmetric arch: P0=(-1,0) P1=(-1,1.5) P2=(1,1.5) P3=(1,0)
    std::vector<BezierCurve> curves = {
        {{{ {-1.0, 0.0}, {-1.0, 1.5}, {1.0, 1.5}, {1.0, 0.0} }}}
    };

    const int resolution = 200;
    const double x_min = -1.5, x_max = 1.5;
    const double y_min = -0.5, y_max = 2.0;

    std::cout << "Evaluating winding number field ("
              << resolution << "×" << resolution << " grid)...\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    auto W  = winding_number_grid(curves, x_min, x_max, y_min, y_max, resolution);
    auto t1 = std::chrono::high_resolution_clock::now();
    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();

    std::cout << "Done in " << ms << " ms.\n";

    // Print range
    double wmin = W.minCoeff(), wmax = W.maxCoeff();
    std::cout << "Winding number range: [" << wmin << ", " << wmax << "]\n";

    write_ppm("arch_wn.ppm", W);
    return 0;
}
