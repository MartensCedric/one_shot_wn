// Point-query demo: the simplest GWN call shape. Pass a single 3D point and
// get the GWN scalar back. Probes a handful of interior / exterior locations
// against the bundled camel mesh.

#include <cstdio>
#include <string>

#include "gwn3d/gwn.h"
#include "gwn3d/mesh_surface.h"

int main(int argc, char** argv) {
    std::string mesh = argc > 1 ? argv[1] : "../../../inputs/camelhead.obj";
    auto surface = gwn3d::MeshSurface::from_file(mesh);

    const auto& V = surface.vertices();
    Eigen::Vector3d lo = V.colwise().minCoeff();
    Eigen::Vector3d hi = V.colwise().maxCoeff();
    Eigen::Vector3d center = 0.5 * (lo + hi);

    struct Probe { const char* label; Eigen::Vector3d p; };
    Probe probes[] = {
        { "center  (likely interior)", center },
        { "far +x  (likely exterior)", Eigen::Vector3d(hi.x() + (hi.x() - lo.x()), center.y(), center.z()) },
        { "far -y  (likely exterior)", Eigen::Vector3d(center.x(), lo.y() - (hi.y() - lo.y()), center.z()) },
        { "near bbox-min vertex      ", lo },
        { "near bbox-max vertex      ", hi },
    };

    for (const Probe& probe : probes) {
        double w = gwn3d::compute_gwn(surface, probe.p);
        std::printf("%s  p=(%+.3f, %+.3f, %+.3f)  GWN=%+.6f\n",
                    probe.label, probe.p.x(), probe.p.y(), probe.p.z(), w);
    }
    return 0;
}
