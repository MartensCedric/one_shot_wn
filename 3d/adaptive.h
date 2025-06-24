#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <utility>
#include <vector>
#include <functional>
#include "boundary_processing.h"
#include "uv_util.h"
#include "math_util.h"
#include "bezier.h"
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include "curve_net_parser.h"


std::vector<double> winding_numbers_adaptive(const std::function<Eigen::Vector3d(Eigen::Vector2d)>& func, const std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)>& jac_func, const Eigen::MatrixXd& query_points, int max_depth, double tolerance);
double bem_double_integral(const Eigen::Vector2d& uv, const space_curve_t& space_curve, const Eigen::MatrixXd& df_dn, const BoundaryParametrization* boundary_param, const Eigen::Vector3d& q);
std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> create_bezier_jac_func(const Eigen::MatrixXd& control_points);
double adaptive_quadrature_winding_number(const Eigen::Vector2d& uv, const std::function<Eigen::Vector3d(Eigen::Vector2d)>& func, const std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)>& jac_func, const Eigen::Vector3d& q);
double bezier_triangle_winding_number(const Eigen::Vector2d& uv, const std::function<Eigen::Vector3d(Eigen::Vector2d)>& func, const std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)>& jac_func, const Eigen::Vector3d& q);

std::pair<std::vector<double>, std::chrono::nanoseconds> bezier_triangle_winding_number_timed(const Eigen::MatrixXd& control_points, const Eigen::MatrixXd& query_points, double tolerance);
std::vector<std::pair<double, std::chrono::nanoseconds>> bezier_triangle_wn_against_gt(const Eigen::MatrixXd& control_points, const Eigen::MatrixXd& query_points, const std::vector<double>& tolerances, const std::vector<double>& gt);
void run_bezier_triangle_aq_experiment(const Eigen::MatrixXd& query_points);