#pragma once
#include <Eigen/Dense>
#include <vector>
#include "math_util.h"
struct patch_t
{
	Eigen::MatrixXd curve;
	bool is_open;
	std::vector<int> gaps_ids;
	std::function<Eigen::Vector3d(Eigen::Vector2d)> func;
	std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> jac;
};

enum SurfaceType {
	MINIMAL,
	COONS,
	IMPLICIT,
	PARAMETRIC,
	REVOLVED,
	PARAMETRIC_GEARS, 
	MESH,
	HEIGHT_FIELD
};

patch_t flip_patch(const patch_t& patch);
box aabb_from_point_cloud(const Eigen::MatrixXd& points);