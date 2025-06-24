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
#include "curve_net.h"
#include "bem_solver.h"
#include "parametric.h"
#include "region_splitting.h"
#include "curve_net_parser.h"
#include "mesh.h"

#include "result.h"
#include "octreeSDF.h"

struct precomputed_curve_data {
	std::vector<BoundaryParametrization*> int_params;
	std::vector<space_curve_t> full_patches;
	std::vector<Eigen::MatrixXd> df_dns;
	std::vector<box> bounding_boxes;
	std::chrono::nanoseconds precompute_result_time; // time to run all dense solves (multithreaded)
	std::chrono::nanoseconds precompute_total_time; // total time spent solving dense systems (sum of all single threaded tasks)
};


space_curve_t dirichlet_at_inf(const space_curve_t& patch, double scale);
precomputed_curve_data precompute_patches(const std::vector<space_curve_t>& patches, const std::vector<double>& insidenesses);
void free_precompute(precomputed_curve_data& precomputed_data);
std::vector<std::vector<int>> find_all_chis_from_regions_oneshot(const curve_net_sampler& sampler, const precomputed_curve_data& precompute);
std::vector<double> run_oneshot_rays(const curve_net_sampler& sampler, const precomputed_curve_data& precomputes_closed);
std::vector<double> run_one_ray_per_row(const curve_net_sampler& sampler, const precomputed_curve_data& precomputes_closed, const std::vector<std::vector<int>>& ray_shots);
//std::vector<std::vector<std::vector<std::pair<double, int>>>> find_all_intersections_from_ray_rows(const curve_net_sampler& sampler, const precomputed_curve_data& precompute, const std::vector<std::vector<int>>& ray_shots);
winding_number_results winding_number_mixed(const std::vector<patch_t>& patches, const Eigen::MatrixXd& query_points, std::vector<std::vector<int>> ray_shots, const surface_config& config);
winding_number_results winding_number_joint_patches(const std::vector<patch_t>& patches, const Eigen::MatrixXd& query_points, const std::vector<std::vector<int>>& ray_points, const surface_config& config);
all_intersections_with_normals_result find_all_intersections_bem(const box& bounding_box, const struct bem_solver_params& bem_solver_params, gsl_multiroot_fsolver* solver, double max_ray_length);
std::vector<double> winding_numbers_along_ray(const std::vector<std::pair<double, int>>& intersections, const std::vector<region_weighted_rays_info>& region_infos, const std::vector<int>& ray, const Eigen::MatrixXd& query_points);
void write_gwn_to_file(const winding_number_results& res, const curvenet_input& input);