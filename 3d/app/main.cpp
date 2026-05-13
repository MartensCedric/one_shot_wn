// Renders the GWN field of a mesh on an axis-aligned 2D slice between two
// 3D corner points (defaults to a mid-z slice of the mesh's bounding box).

#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cxxopts.hpp>

#include "gwn3d/gwn.h"
#include "gwn3d/mesh_surface.h"

static Eigen::Vector3d parse_vec3(const std::string& s) {
    std::stringstream ss(s);
    Eigen::Vector3d v;
    char comma;
    if (!(ss >> v(0) >> comma >> v(1) >> comma >> v(2)))
        throw std::runtime_error("expected x,y,z but got: " + s);
    return v;
}

int main(int argc, char** argv)
{
    cxxopts::Options options("one_shot_wn_3d",
        "Generalized winding numbers (one-shot method) on a 2D slice of a triangle mesh");

    options.add_options()
        ("n,name",       "Experiment name (output file base)", cxxopts::value<std::string>())
        ("m,mesh",       "Mesh file (.obj or .off)",           cxxopts::value<std::string>())
        ("p0",           "Grid corner 0 as x,y,z (default: mesh bbox min, z = midpoint)",
                                                               cxxopts::value<std::string>())
        ("p1",           "Grid corner 1 as x,y,z (default: mesh bbox max, z = midpoint)",
                                                               cxxopts::value<std::string>())
        ("r,resolution", "Grid resolution as WxH",             cxxopts::value<std::string>()->default_value("500x300"))
        ("h,help",       "Print help");

    auto result = options.parse(argc, argv);

    if (result.count("help") || !result.count("mesh") || !result.count("name")) {
        std::cout << options.help() << "\n";
        return 0;
    }

    const std::string mesh_file = result["mesh"].as<std::string>();
    const std::string name      = result["name"].as<std::string>();
    const std::string res_str   = result["resolution"].as<std::string>();

    int W = 500, H = 300;
    auto x_pos = res_str.find('x');
    if (x_pos != std::string::npos) {
        W = std::stoi(res_str.substr(0, x_pos));
        H = std::stoi(res_str.substr(x_pos + 1));
    }

    gwn3d::MeshSurface surface = gwn3d::MeshSurface::from_file(mesh_file);

    Eigen::Vector3d p0, p1;
    if (result.count("p0") && result.count("p1")) {
        p0 = parse_vec3(result["p0"].as<std::string>());
        p1 = parse_vec3(result["p1"].as<std::string>());
    } else {
        // Default to an xy slice through the middle of the mesh's bbox.
        const auto& V = surface.vertices();
        Eigen::Vector3d lo = V.colwise().minCoeff();
        Eigen::Vector3d hi = V.colwise().maxCoeff();
        double mid_z = 0.5 * (lo.z() + hi.z());
        p0 = Eigen::Vector3d(lo.x(), lo.y(), mid_z);
        p1 = Eigen::Vector3d(hi.x(), hi.y(), mid_z);
    }

    std::cout << "running: " << name
              << " on slice " << p0.transpose() << " -> " << p1.transpose()
              << " at " << W << "x" << H << "\n";

    gwn3d::GwnResult res = gwn3d::compute_gwn(surface, p0, p1, W, H);
    gwn3d::write_gwn_ppm(res, name + ".ppm");
    std::cout << "wrote " << name << ".ppm" << std::endl;
    return 0;
}
