#pragma once
#include <Eigen/Dense>
#include "math_util.h"
#include "spherical.h"
#include <vector>
typedef Eigen::Matrix<double, Eigen::Dynamic, 3> closed_spherical_curve_t;
typedef Eigen::Matrix<double, Eigen::Dynamic, 3> space_curve_t;

struct curve_net_sampler {
	std::vector<space_curve_t> patches;
	std::vector<Eigen::Vector3d> sample_points;
	bool is_closed;
	std::vector<std::vector<std::vector<Eigen::Vector3d>>> rays;
	std::vector<std::vector<std::vector<double>>> areas;
	std::vector<std::vector<std::vector<int>>> rel_wn;
	std::vector<std::vector<std::vector<closed_spherical_curve_t>>> spherical_regions;
	std::vector<double> insideness;
	
};

space_curve_t super_sample_patch(const space_curve_t& patch, int sampling_rate);

std::vector<space_curve_t> super_sample_patches(const std::vector<space_curve_t>& patches, int sampling_rate);
std::vector<double> gwn_from_chis_oneshot(const std::vector<std::vector<int>>& chis, const curve_net_sampler& sampler);
std::vector<double> gwn_from_chis_rayrows(const std::vector<std::vector<std::vector<std::pair<double, int>>>>& chis, const curve_net_sampler& sampler, const std::vector<std::vector<int>>& ray_shots);
std::vector<double> gwn_from_chis(const std::vector<std::vector<std::vector<int>>>& chis, const curve_net_sampler& sampler);
std::vector<std::vector<double>> gwn_from_chis_one_shot_per_patch(const std::vector<std::vector<int>>& res_chis, const curve_net_sampler& closed_curve_sampler);