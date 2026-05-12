#pragma once

#include <Eigen/Core>
#include <vector>

#include "patch.h"
#include "intersections.h"

namespace gwn3d {

/// Abstract surface used by compute_gwn. Derive from this to add a new
/// surface type: the only required methods are boundary_patches() and
/// intersect_ray().
///
/// Threading: compute_gwn calls prepare(n_threads) once, then intersect_ray
/// concurrently with thread_id in [0, n_threads). Allocate any per-thread
/// scratch (GSL solvers, RNGs, etc.) inside prepare.
class Surface {
public:
    virtual ~Surface() = default;

    /// Boundary curves used for region-splitting. The size of this vector
    /// defines the patch index space passed to intersect_ray.
    virtual std::vector<patch_t> boundary_patches() const = 0;

    /// Ray-surface intersections within parameter [0, t_max]. Setting
    /// valid_ray=false in the result asks compute_gwn to fall back to the
    /// per-point one-shot path for this (ray, patch).
    virtual all_intersections_with_normals_result intersect_ray(
        const Eigen::Vector3d& origin,
        const Eigen::Vector3d& direction,
        int patch_idx,
        double t_max,
        int thread_id) const = 0;

    virtual void prepare(int /*n_threads*/) {}

    /// True when intersect_ray returns one global hit set per ray regardless
    /// of patch_idx (the libigl mesh path). Skips per-patch region-splitting.
    virtual bool prefers_global_intersection() const { return false; }
};

} // namespace gwn3d
