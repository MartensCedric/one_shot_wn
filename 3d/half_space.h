#pragma once

#include <Eigen/Dense>
#include <vector>
#include <gsl/gsl_multiroots.h>
#include "parametric.h"
struct halfspace_intersections_t
{
	Eigen::Vector3d ray_dir;
	std::vector<double> roots_found;
	std::vector<bool> roots_makes_it_inside;
	std::vector<Eigen::Vector2d> root_uvs;
	bool is_valid;
};

struct halfspace_crossing_t
{
	double t;
	int dir;
	int patch_num;
	Eigen::Vector2d uv;
};

typedef std::function<bool(Eigen::Vector2d)> is_inside_f;

int chi_from_halfspace_intersections(const std::pair<halfspace_intersections_t, halfspace_intersections_t>& intersections, const std::pair<is_inside_f, is_inside_f>& is_inside_funcs, const std::pair<bool, bool>& inside_hs);
std::pair<halfspace_intersections_t, halfspace_intersections_t> all_halfspace_intersections_common_ray(gsl_multiroot_fsolver* s, const implicit_func_t& f1, const implicit_func_t& f2, const Eigen::Vector3d& point);
halfspace_intersections_t all_halfspace_intersections_for_ray(gsl_multiroot_fsolver* s, const implicit_func_t& f, const Eigen::Vector3d& point, const Eigen::Vector3d& dir);
halfspace_intersections_t all_halfspace_intersections(gsl_multiroot_fsolver* s, const implicit_func_t& f, const Eigen::Vector3d& point);
bool is_inside_half_space_implicit(gsl_multiroot_fsolver* s, const implicit_func_t& f, const Eigen::Vector3d& point);