#pragma once

#include "math_util.h"
#include "uv_util.h"
#include "intersections.h"
#include <gsl/gsl_multiroots.h>


struct bem_solver_params
{
	const BoundaryParametrization* boundary_param;
	const space_curve_t& patch;
	const Eigen::MatrixXd& df_dn;
	Eigen::Vector3d point;
	Eigen::Vector3d dir;
};


int f_func(const gsl_vector* x, void* p, gsl_vector* f);
void print_solver_state(size_t iter, gsl_multiroot_fsolver* s);
chi_result find_chi(gsl_multiroot_fsolver* s, const bem_solver_params& params, double t_min, double t_max, double max_ray_length, bool abort_bad_ray);
struct all_intersections_with_normals_result all_ray_intersections(gsl_multiroot_fsolver* s, const bem_solver_params& params, double t_min, double t_max, double max_ray_length);