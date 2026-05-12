#include "patch.h"

patch_t flip_patch(const patch_t& patch)
{
	patch_t new_patch;
	new_patch.is_open = false;
	assert(!patch.is_open);
	Eigen::MatrixXd patch_values = patch.curve;
	new_patch.curve.resize(patch_values.rows(), 3);

	for (int k = 0; k < patch_values.rows(); k++)
	{
		new_patch.curve.row(k) = patch_values.row(patch_values.rows() - 1 - k);
	}

	return new_patch;
}

box aabb_from_point_cloud(const Eigen::MatrixXd& points)
{
	box b;
	b.x_min = points.col(0).minCoeff();
	b.x_max = points.col(0).maxCoeff();

	b.y_min = points.col(1).minCoeff();
	b.y_max = points.col(1).maxCoeff();

	b.z_min = points.col(2).minCoeff();
	b.z_max = points.col(2).maxCoeff();

	return b;
}