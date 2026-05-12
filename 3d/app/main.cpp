#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <cxxopts.hpp>

#include "gwn3d/gwn.h"
#include "gwn3d/mesh_surface.h"
#include "curve_net_parser.h"

int main(int argc, char** argv)
{
    cxxopts::Options options("one_shot_wn_3d",
        "Generalized winding numbers (one-shot method) for a triangle mesh");

    options.add_options()
        ("n,name",       "Experiment name (output file base)", cxxopts::value<std::string>())
        ("m,mesh",       "Mesh file (.obj or .off)",           cxxopts::value<std::string>())
        ("q,queries",    "Query points file",                  cxxopts::value<std::string>())
        ("r,resolution", "Grid resolution as WxH",             cxxopts::value<std::string>()->default_value("500x300"))
        ("h,help",       "Print help");

    auto result = options.parse(argc, argv);

    if (result.count("help") ||
        !result.count("mesh") ||
        !result.count("queries") ||
        !result.count("name")) {
        std::cout << options.help() << "\n";
        return 0;
    }

    const std::string mesh_file  = result["mesh"].as<std::string>();
    const std::string query_file = result["queries"].as<std::string>();
    const std::string name       = result["name"].as<std::string>();
    const std::string res_str    = result["resolution"].as<std::string>();

    int W = 500, H = 300;
    auto x_pos = res_str.find('x');
    if (x_pos != std::string::npos) {
        W = std::stoi(res_str.substr(0, x_pos));
        H = std::stoi(res_str.substr(x_pos + 1));
    }

    std::cout << "running: " << name << std::endl;

    gwn3d::MeshSurface surface = gwn3d::MeshSurface::from_file(mesh_file);

    Eigen::MatrixXd query_points = load_query_points(query_file);
    std::vector<int> dims = { W, H };
    std::vector<std::vector<int>> rays = dimension_to_rays(dims);

    gwn3d::GwnOptions opts;
    opts.resolution = { W, H };
    gwn3d::GwnResult res = gwn3d::compute_gwn(surface, query_points, rays, opts);

    gwn3d::write_gwn_ppm(res, name + ".ppm");
    std::cout << "wrote " << name << ".ppm" << std::endl;

    // Debug hook: GWN3D_DUMP_CHIS_M=<path> dumps the raw chi array in MATLAB
    // format for cross-version comparison.
    if (const char* dump = std::getenv("GWN3D_DUMP_CHIS_M")) {
        std::ofstream f(dump);
        f << "chis = [";
        for (size_t i = 0; i < res.gwn.size(); ++i)
            f << std::setprecision(10) << res.gwn[i] << ',';
        f << "]\n";
    }
    return 0;
}
