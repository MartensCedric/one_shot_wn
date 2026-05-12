#pragma once

#include <vector>
#include <Eigen/Dense>
#include <string>
#include "patch.h"
#include "boundary_processing.h"

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
