#pragma once

#include <vector>
#include <Eigen/Dense>
#include <string>
#include "patch.h"
#include "parametric.h"
#include "boundary_processing.h"
#include "coons.h"

struct surface_config {
	SurfaceType surface_type = SurfaceType::MINIMAL;
	implicit_func_t parametric_func;
	std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> parametric_jac;
	bool use_fd = true;
	std::vector<coons_patch> coons_patches;
	std::string coons_folder_name;
	int num_coons_files = 0;

	double parametric_max_t = 5.0;
	std::string mesh_name;
	Eigen::MatrixXd mesh_vertices;
	Eigen::MatrixXi mesh_faces;
	std::vector<bool> coons_flip_normals;
	std::vector<bool> flip_patch_orientation;
	bool do_symmetric_patch_adjustment = false;
	bool do_jiggle_rays = false;
	bool alec_only = false;
	bool use_obj_folder = false;
	bool patch_from_uv_triangle = false;
	int num_patch_points_from_uv = 0;
	std::string override_output_name;
	bool is_override_output_name = false;
	std::vector<space_curve_t> boundaries;
	bool perform_self_intersections = true;
	std::function<bool(Eigen::Vector2d)> is_in_parametric_domain = [](Eigen::Vector2d uv) { return uv(0) >= 0 && uv(0) <= 1 && uv(1) >= 0 && uv(1) <= 1; };
};

struct curvenet_input {
	std::string input_name;
	std::string patches_name;
	std::vector<int> dimensions;
	std::vector<int> patches_to_remove;
	struct surface_config config;
	std::vector<std::vector<int>> predefined_mesh_boundary;
};


Eigen::MatrixXd remove_consecutive_duplicates(const Eigen::MatrixXd& curve);
std::vector<patch_t> load_all_patches(const std::string& patch_file);
std::vector<patch_t> remove_patches(const std::vector<patch_t>& patches, const std::vector<int>& ids_to_remove);
Eigen::MatrixXd load_query_points(const std::string& query_points);
Eigen::MatrixXd jiggle_rays(const Eigen::MatrixXd& query_points, const std::vector<std::vector<int>>& rays, double jiggle_amount);
std::vector<std::vector<int>> slice_to_rays(int num_rays, int ray_depth, int axis);
std::vector<patch_t> get_open_patches(const std::vector<patch_t>& patches);
std::vector<patch_t> get_closed_patches(const std::vector<patch_t>& patches);
std::vector<patch_t> subsample_patches(const std::vector<patch_t>& patches, int sampling_rate);
std::vector<space_curve_t> get_closed_patches_space_curves(const std::vector<patch_t>& patches);
std::vector<std::vector<int>> grid_to_rays(const std::vector<int>& dimensions, int axis);
std::vector<std::vector<int>> dimension_to_rays(std::vector<int> dimensions);
double curve_length(const Eigen::MatrixXd& curve);