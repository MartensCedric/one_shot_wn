#pragma once

#include <vector>
#include <Eigen/Dense>
#include <string>
#include "patch.h"
#include "boundary_processing.h"

Eigen::MatrixXd load_query_points(const std::string& query_points);
std::vector<std::vector<int>> slice_to_rays(int num_rays, int ray_depth, int axis);
std::vector<patch_t> get_closed_patches(const std::vector<patch_t>& patches);
std::vector<space_curve_t> get_closed_patches_space_curves(const std::vector<patch_t>& patches);
std::vector<std::vector<int>> grid_to_rays(const std::vector<int>& dimensions, int axis);
std::vector<std::vector<int>> dimension_to_rays(std::vector<int> dimensions);
