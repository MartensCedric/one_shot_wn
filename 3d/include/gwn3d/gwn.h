#pragma once

#include <Eigen/Core>
#include <array>
#include <chrono>
#include <string>
#include <vector>

#include "gwn3d/surface.h"

namespace gwn3d {

struct GwnOptions {
    /// Output grid resolution. Carried through into GwnResult for write_gwn_ppm;
    /// does not affect the computation when callers supply query_points directly.
    std::array<int, 2> resolution{500, 300};

    /// Run the pure fractional (solid-angle) method. Reserved; currently a no-op.
    bool fractional_only = false;

    /// Forwarded to region-splitting. Set false when the boundary's spherical
    /// projection has no numeric self-intersections (e.g. an analytic circle).
    bool perform_self_intersections = true;

    /// Remap the final GWN to handle multi-boundary surfaces with shared interior.
    bool symmetric_patch_adjustment = false;

    /// Flip the orientation of selected boundary patches before region-splitting.
    std::vector<bool> flip_patch_orientation;

    int compute_threads    = 19;
    int precompute_threads = 18;

    /// Upper bound on the ray parameter t for Surface::intersect_ray.
    double parametric_max_t = 5.0;
};

struct GwnTimingInfo {
    std::chrono::nanoseconds ray_shooting{0};
    std::chrono::nanoseconds boundary_processing{0};
    std::chrono::nanoseconds precompute_total{0};
    std::chrono::nanoseconds precompute_wall{0};

    int num_patches         = 0;
    int total_rays          = 0;
    int ray_length          = 0;
    int total_patch_samples = 0;
};

struct GwnResult {
    std::vector<double> gwn;
    GwnTimingInfo timing;
    std::array<int, 2> resolution{};
};

/// Three ways to query the GWN field. Pick the overload that matches your
/// layout; all three call the same underlying machinery.

/// 1) Single 3D query point. Returns the GWN scalar at that point.
double compute_gwn(
    Surface& surface,
    const Eigen::Vector3d& point,
    const GwnOptions& opts = {});

/// 2) Arbitrary query points grouped along rays for batched ray-shooting.
/// `rays` is the row decomposition: each inner vector is a list of indices
/// into `query_points`, in order along one ray. Every ray must have at
/// least two points so the ray direction can be derived. `surface` must
/// outlive this call.
GwnResult compute_gwn(
    Surface& surface,
    const Eigen::MatrixXd& query_points,
    const std::vector<std::vector<int>>& rays,
    const GwnOptions& opts = {});

/// 3) Axis-aligned 2D grid in 3D space, between two corner points. The
/// grid plane is perpendicular to whichever coordinate axis has the
/// smallest |p1 - p0| (pinned to the midpoint of that axis, or the common
/// value if p0 and p1 share it). `width` and `height` are resolutions
/// along the first and second remaining axes in (x, y, z) order. Rays
/// sweep the longer of `width` / `height` for batching. The returned
/// `GwnResult::resolution` is set to `{width, height}` so the result can
/// be handed directly to `write_gwn_ppm`.
GwnResult compute_gwn(
    Surface& surface,
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& p1,
    int width, int height,
    const GwnOptions& opts = {});

/// Write the GWN field as a binary P6 PPM. Values are clamped to [-1, 1] and
/// mapped through black: -1 -> red, 0 -> black, +1 -> green. The result must
/// have resolution[0] * resolution[1] entries in row-major order.
void write_gwn_ppm(const GwnResult& result, const std::string& filename);

} // namespace gwn3d
