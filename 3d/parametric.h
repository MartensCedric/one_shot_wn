#pragma once

#include "intersections.h"
#include "math_util.h"
#include "patch.h"
#include <iostream>
#include <functional>
#include <Eigen/Dense>
#include <gsl/gsl_multiroots.h>
#include <random>

typedef std::function<Eigen::Vector3d(Eigen::Vector2d)> implicit_func_t;

struct parametric_solver_params
{
	implicit_func_t func;
	std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> jac;
	Eigen::Vector3d point;
	Eigen::Vector3d dir;

	std::function<bool(Eigen::Vector2d)> is_in_parametric_domain;
};

struct uv_solver_params
{
	implicit_func_t f;
	implicit_func_t g;
	Eigen::Vector2d safe_point;
	Eigen::Vector2d query_point;
};


int f_func_parametric(const gsl_vector* x, void* p, gsl_vector* f);
Eigen::Matrix<double, 3, 2> jacobian_fd_implicit(Eigen::Vector2d uv, implicit_func_t func);
std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> fd_jacobian(implicit_func_t func, double eps);
chi_result find_chi_implicit(gsl_multiroot_fsolver* s, const parametric_solver_params& params, double t_min, double t_max);
std::vector<int> find_all_chis_implicit(implicit_func_t func, const Eigen::MatrixXd& rays, Eigen::Vector3d point);
struct all_intersections_with_normals_result find_all_intersections_parametric(gsl_multiroot_fsolver* s, const struct parametric_solver_params& params, double t_min, double t_max);
struct all_intersections_with_normals_result find_all_intersections_parametric_gears(gsl_multiroot_fsolver* s, const struct parametric_solver_params& params, double t_min, double t_max);
Eigen::Vector3d implicit_test_func1(Eigen::Vector2d uv);
Eigen::Vector3d implicit_test_func2(Eigen::Vector2d uv);
bool implicit_test_func1_inside(Eigen::Vector2d uv);
bool implicit_test_func2_inside(Eigen::Vector2d uv);
Eigen::Vector3d parametric_torus_4_3(Eigen::Vector2d uv);
Eigen::Vector3d parametric_torus(Eigen::Vector2d uv);
Eigen::Vector3d spur_gear_parametric(Eigen::Vector2d uv);
Eigen::Vector3d spur_gear(Eigen::Vector2d uv);
Eigen::Vector3d spur_gear_parametric_right(Eigen::Vector2d uv);

Eigen::Vector3d infinite_singular_func(Eigen::Vector2d uv);
Eigen::Matrix<double, 3, 2> infinite_singular_func_jac(Eigen::Vector2d uv);

void compute_all_intersections(const patch_t& patch, const Eigen::MatrixXd& query_points, const std::function<Eigen::Vector3d(Eigen::Vector2d)>& param_func, const std::function<bool(Eigen::Vector2d)>& is_inside);
Eigen::MatrixXd evaluate_surface(const std::function<Eigen::Vector3d(Eigen::Vector2d)>& func, const Eigen::MatrixXd& uvs);
Eigen::MatrixXd discretize_curve(const std::function<Eigen::Vector3d(double)>& func, int num_discretizations);
Eigen::MatrixXd boundary_from_surface(const std::function<Eigen::Vector3d(Eigen::Vector2d)>& func, int num_elements);
Eigen::Vector3d parametric_vase_cubic(Eigen::Vector2d uv);
Eigen::Vector3d parametric_vase_cubic_2(Eigen::Vector2d uv);

Eigen::Matrix<double, 3, 2> parametric_vase_cubic_jac(Eigen::Vector2d uv);
Eigen::Matrix<double, 3, 2> parametric_vase_cubic_2_jac(Eigen::Vector2d uv);
Eigen::Vector3d evaluate_paper_func1(Eigen::Vector2d uv);
Eigen::Matrix<double, 3, 2> evaluate_paper_jacfunc1(Eigen::Vector2d uv);

void uniform_mesh_grid(int resolution, Eigen::MatrixXd& V, Eigen::MatrixXi& F);
Eigen::MatrixXd apply_function_to_uv(const Eigen::MatrixXd& uvs, const std::function<Eigen::Vector3d(Eigen::Vector2d)>& func);