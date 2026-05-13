#pragma once

#include <iostream>
#include <gsl/gsl_multiroots.h>
#include <random>
#include <chrono>

#include "patch.h"
#include "curve_net.h"
#include "math_util.h"
#include "boundary_processing.h"
#include "intersections.h"
#include "bem_solver.h"
#include "region_splitting.h"
#include "mesh.h"

struct precomputed_curve_data {
	std::vector<BoundaryParametrization*> int_params;
	std::vector<space_curve_t> full_patches;
	std::vector<Eigen::MatrixXd> df_dns;
	std::vector<box> bounding_boxes;
	std::chrono::nanoseconds precompute_result_time;
	std::chrono::nanoseconds precompute_total_time;
};

precomputed_curve_data precompute_patches(const std::vector<space_curve_t>& patches, const std::vector<double>& insidenesses, int n_threads);
void free_precompute(precomputed_curve_data& precomputed_data);
all_intersections_with_normals_result find_all_intersections_bem(const box& bounding_box, const struct bem_solver_params& bem_solver_params, gsl_multiroot_fsolver* solver, double max_ray_length);
std::vector<double> winding_numbers_along_ray(const std::vector<std::pair<double, int>>& intersections, const std::vector<region_weighted_rays_info>& region_infos, const std::vector<int>& ray, const Eigen::MatrixXd& query_points);
