#pragma once

#include <Eigen/Dense>
#include <functional>
#include <vector>
#include <numeric>

#include "patch.h"
#include "math_util.h"
#include "spherical.h"
#include "region_splitting.h"

struct BezierCurvenet {
	std::vector<Eigen::MatrixXd> patches;
	Eigen::MatrixXd boundary;
};

std::function<Eigen::Vector3d(double)> create_cubic_bezier_func(const Eigen::MatrixXd& control_points);
std::function<Eigen::Vector3d(Eigen::Vector2d)> create_bezier_func(const Eigen::MatrixXd& control_points);
std::vector<patch_t> patch_from_bezier(int num_per_edge, const std::function<Eigen::Vector3d(Eigen::Vector2d)>& func);
Eigen::MatrixXd barycentric_mesh_grid(int resolution);
Eigen::MatrixXi triangle_indices(int resolution);
Eigen::MatrixXd quadratic_bezier(const Eigen::Matrix3d& control_points, int num_points);

std::function<Eigen::Vector3d(Eigen::Vector2d)> cubic_bezier_surface(const Eigen::MatrixXd& control_points);
std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> cubic_bezier_surface_jac(const Eigen::MatrixXd& control_points);

BezierCurvenet load_bezier_surface(const std::string& filename);
BezierCurvenet rescale_to_unit(const BezierCurvenet& bezier_cn);

void run_bezier_intersections();