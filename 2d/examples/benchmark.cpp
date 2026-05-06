// benchmark.cpp
// Compares timing of three methods evaluated on an N×N query grid:
//
//   A) per-point:   winding_number(curves, p) called N² times independently
//   B) grid (ray-reuse): winding_number_grid — shoots K rays per row (K curves),
//                        reuses suffix-sum chi for all N points in that row.
//                        Total rays: K × N  (vs K × N² for per-point).
//   C) polyline baseline: subdivide each Bezier into n_seg segments, compute
//                         the angular sum per query point.  O(K × n_seg × N²).
//
// Usage:  ./wn2d_benchmark [grid_resolution] [polyline_N]
//   grid_resolution  — query grid side length (default 500)
//   polyline_N       — segments per Bezier curve for baseline (default 100)

#include "wn2d/winding_number.h"
#include "wn2d/bezier.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

// Per-point evaluation (no ray reuse)
Eigen::MatrixXd per_point_grid(const std::vector<wn2d::BezierCurve>& curves,
                                double x_min, double x_max,
                                double y_min, double y_max,
                                int resolution)
{
    Eigen::MatrixXd W(resolution, resolution);
    double dx = (x_max - x_min) / (resolution - 1);
    double dy = (y_max - y_min) / (resolution - 1);
    for (int row = 0; row < resolution; ++row) {
        double y = y_min + row * dy;
        for (int col = 0; col < resolution; ++col) {
            double x = x_min + col * dx;
            W(row, col) = wn2d::winding_number(curves, {x, y});
        }
    }
    return W;
}

// Polyline subdivision baseline
double polyline_winding_number(const std::vector<wn2d::BezierCurve>& curves,
                                const Eigen::Vector2d& p, int n_seg)
{
    double angle = 0.0;
    for (const auto& c : curves) {
        Eigen::Vector2d prev = c.eval(0.0) - p;
        for (int i = 1; i <= n_seg; ++i) {
            double t = static_cast<double>(i) / n_seg;
            Eigen::Vector2d curr = c.eval(t) - p;
            double cross = prev.x()*curr.y() - prev.y()*curr.x();
            double dot   = prev.dot(curr);
            angle += std::atan2(cross, dot);
            prev = curr;
        }
    }
    return angle / (2.0 * M_PI);
}

Eigen::MatrixXd polyline_winding_grid(const std::vector<wn2d::BezierCurve>& curves,
                                       double x_min, double x_max,
                                       double y_min, double y_max,
                                       int resolution, int n_seg)
{
    Eigen::MatrixXd W(resolution, resolution);
    double dx = (x_max - x_min) / (resolution - 1);
    double dy = (y_max - y_min) / (resolution - 1);
    for (int row = 0; row < resolution; ++row) {
        double y = y_min + row * dy;
        for (int col = 0; col < resolution; ++col) {
            double x = x_min + col * dx;
            W(row, col) = polyline_winding_number(curves, {x, y}, n_seg);
        }
    }
    return W;
}

template<typename F>
long long time_ms(F&& f)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    f();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
}

void run_benchmark(const char* label,
                   const std::vector<wn2d::BezierCurve>& curves,
                   double x_min, double x_max,
                   double y_min, double y_max,
                   int resolution, int n_seg)
{
    std::cout << "=== " << label << " ===\n";
    std::cout << "Grid: " << resolution << "×" << resolution
              << "  |  K=" << curves.size() << " curves"
              << "  |  Polyline N=" << n_seg << " per curve\n\n";

    Eigen::MatrixXd W_pp, W_gr, W_pl;

    long long t_pp = time_ms([&]{
        W_pp = per_point_grid(curves, x_min, x_max, y_min, y_max, resolution);
    });

    long long t_gr = time_ms([&]{
        W_gr = wn2d::winding_number_grid(curves, x_min, x_max, y_min, y_max, resolution);
    });

    long long t_pl = time_ms([&]{
        W_pl = polyline_winding_grid(curves, x_min, x_max, y_min, y_max, resolution, n_seg);
    });

    double gr_err = (W_gr - W_pp).cwiseAbs().maxCoeff();
    double pl_err = (W_pl - W_pp).cwiseAbs().maxCoeff();

    std::cout << "Per-point (no reuse): " << t_pp << " ms\n";
    std::cout << "Grid (ray reuse):     " << t_gr << " ms";
    if (t_gr > 0)
        std::cout << "  (" << static_cast<double>(t_pp)/t_gr << "× faster than per-point)";
    std::cout << "\n";
    std::cout << "Polyline (N=" << n_seg << "):    " << t_pl << " ms";
    if (t_pl > 0 && t_gr > 0)
        std::cout << "  (" << static_cast<double>(t_pl)/t_gr << "× slower than grid)";
    std::cout << "\n";
    std::cout << "Grid vs per-point max|err|:  " << gr_err << "\n";
    std::cout << "Polyline vs per-point max|err|: " << pl_err << "\n\n";
}

} // anonymous namespace

int main(int argc, char** argv)
{
    int resolution = (argc > 1) ? std::atoi(argv[1]) : 500;
    int n_seg      = (argc > 2) ? std::atoi(argv[2]) : 100;

    using namespace wn2d;

    // Single arch Bezier
    std::vector<BezierCurve> arch = {
        {{{ {-1.0, 0.0}, {-1.0, 1.5}, {1.0, 1.5}, {1.0, 0.0} }}}
    };
    run_benchmark("Single arch Bezier (K=1)",
                  arch, -1.5, 1.5, -0.5, 2.0, resolution, n_seg);

    // Unit circle (4 arcs)
    const double k = 0.5522847498;
    std::vector<BezierCurve> circle = {
        {{{ {1.0,0.0}, {1.0,k}, {k,1.0}, {0.0,1.0} }}},
        {{{ {0.0,1.0}, {-k,1.0}, {-1.0,k}, {-1.0,0.0} }}},
        {{{ {-1.0,0.0}, {-1.0,-k}, {-k,-1.0}, {0.0,-1.0} }}},
        {{{ {0.0,-1.0}, {k,-1.0}, {1.0,-k}, {1.0,0.0} }}}
    };
    run_benchmark("Unit circle (K=4)",
                  circle, -1.5, 1.5, -1.5, 1.5, resolution, n_seg);

    return 0;
}
