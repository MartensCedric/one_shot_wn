#pragma once

#include <functional>

#include <Eigen/Dense>
#include <gsl/gsl_multiroots.h>

#include "intersections.h"
#include "math_util.h"
#include "patch.h"

typedef std::function<Eigen::Vector3d(Eigen::Vector2d)> implicit_func_t;

struct parametric_solver_params
{
	implicit_func_t func;
	std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> jac;
	Eigen::Vector3d point;
	Eigen::Vector3d dir;

	std::function<bool(Eigen::Vector2d)> is_in_parametric_domain;
};

int f_func_parametric(const gsl_vector* x, void* p, gsl_vector* f);
Eigen::Matrix<double, 3, 2> jacobian_fd_implicit(Eigen::Vector2d uv, implicit_func_t func);
std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> fd_jacobian(implicit_func_t func, double eps);
struct all_intersections_with_normals_result find_all_intersections_parametric(gsl_multiroot_fsolver* s, const struct parametric_solver_params& params, double t_min, double t_max);
struct all_intersections_with_normals_result find_all_intersections_parametric_gears(gsl_multiroot_fsolver* s, const struct parametric_solver_params& params, double t_min, double t_max);
