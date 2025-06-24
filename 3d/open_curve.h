#pragma once

#include "parametric.h"
#include "math_util.h"
#include "uv_util.h"
#include "bem_solver.h"
#include "boundary_processing.h"
#include <gsl/gsl_multiroots.h>
#include "curve_net.h"
#include "curve_net_wn.h"
#include "half_space.h"
#include "result.h"
#include "intersections.h"
#include <vector>

struct wn_result
{
	bool success;
	double wn;
};

struct ray_integral_t
{
	Eigen::MatrixXd rays;
	Eigen::VectorXd areas;
};
inline double signed_angle(Eigen::Vector2d p, Eigen::Vector2d q)
{
	p.normalize();
	q.normalize();

	double dot = p.dot(q);
	if (dot > 0.9999999)
		return 0;

	Eigen::Vector3d a(p(0), p(1), 0.0);
	Eigen::Vector3d b(q(0), q(1), 0.0);
	Eigen::Vector3d c = a.cross(b);
	double sign = 1.0;

	if (c(2) < 0.0)
		sign = -1.0;

	return sign * std::acos(dot);
}

typedef std::function<Eigen::Vector3d(Eigen::Vector2d)> parametric_func_t;
struct shared_open_patch_rootfinding_settings {
	gsl_multiroot_fsolver* surface_solver;
	gsl_multiroot_fsolver* uv_solver;
	parametric_func_t f1;
	parametric_func_t f2;
	Eigen::Vector3d point;
	Eigen::Vector3d dir;
	const open_boundary& bd1;
	const open_boundary& bd2;
};

int f_func_parametric_uv(const gsl_vector* x, void* p, gsl_vector* f);
double wn_2d_open_square(Eigen::Vector2d uv, double closeness);
wn_result find_chi_open(gsl_multiroot_fsolver* s, const bem_solver_params& params, double t_min, double t_max, double closeness);
std::vector<std::vector<double>> find_all_chis_open(const curve_net_sampler& sampler, const precomputed_curve_data& precompute, const ray_integral_t& rays_int);
bool is_inside_uv_polygon(gsl_multiroot_fsolver* parametric_solver, const Eigen::Vector2d& inside_point, const Eigen::Vector2d& query_point, implicit_func_t f, implicit_func_t g, const open_boundary& boundary_param, bool& has_hit_f_g_intersection);
winding_number_results winding_number_with_gaps(const std::vector<patch_t>& patches, const std::vector<std::pair<int, int>>& connected_patches, const Eigen::MatrixXd& query_points, const std::vector<open_boundary>& b_params);
all_intersections_with_normals_result all_intersections_shared_open_patch(const shared_open_patch_rootfinding_settings& config);
winding_number_results winding_number_with_gaps(const std::vector<patch_t>& patches, const std::vector<std::pair<int, int>>& connected_patches, const Eigen::MatrixXd& query_points, const std::vector<open_boundary>& b_params);
