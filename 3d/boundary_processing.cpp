#include <Eigen/Dense>

#include "boundary_processing.h"
#include <vector>
#include <iostream>
#include "math_util.h"

space_curve_t create_space_curve(const boundary_curve_t& bd_curve, const std::vector<double>& data)
{
	space_curve_t space_curve;
	assert(data.size() == bd_curve.rows());
	space_curve.resize(bd_curve.rows(), 3);

	for (int i = 0; i < bd_curve.rows(); i++)
	{
		space_curve(i, 0) = bd_curve(i, 0);
		space_curve(i, 1) = bd_curve(i, 1);
		space_curve(i, 2) = data[i];
	}
	return space_curve;
}

space_curve_t create_space_curve(const boundary_curve_t& bd_curve, const Eigen::VectorXd& data)
{
	space_curve_t space_curve;
	assert(data.size() == bd_curve.rows());
	space_curve.resize(bd_curve.rows(), 3);

	for (int i = 0; i < bd_curve.rows(); i++)
	{
		space_curve(i, 0) = bd_curve(i, 0);
		space_curve(i, 1) = bd_curve(i, 1);
	}
	space_curve.col(2) = data;
	return space_curve;
}

curve_info_t compute_curve_info(const boundary_curve_t& curve)
{
	curve_info_t curve_info;
	boundary_curve_t next = circular_shift(curve, -1);
	curve_info.tangents = next - curve;

	curve_info.lengths.resize(curve.rows());
	for (int i = 0; i < curve.rows(); i++)
	{
		double v0 = curve_info.tangents(i, 0);
		double v1 = curve_info.tangents(i, 1);

		curve_info.lengths.coeffRef(i) = std::sqrt(v0 * v0 + v1 * v1);
		curve_info.tangents.coeffRef(i, 0) = curve_info.tangents(i, 0) / curve_info.lengths(i);
		curve_info.tangents.coeffRef(i, 1) = curve_info.tangents(i, 1) / curve_info.lengths(i);
	}


	curve_info.normals.resize(curve.rows(), 2);
	
	for (int i = 0; i < curve.rows(); i++)
	{
		curve_info.normals.coeffRef(i, 0) = curve_info.tangents(i, 1);
		curve_info.normals.coeffRef(i, 1) = -curve_info.tangents(i, 0);
	}
	
	return curve_info;
}

space_curve_t xyz_to_space_curve(const std::vector<double>& x_vec, const std::vector<double>& y_vec, const std::vector<double>& z_vec)
{
	assert(x_vec.size() == y_vec.size());
	assert(y_vec.size() == z_vec.size());

	space_curve_t space_curve;
	space_curve.resize(x_vec.size(), 3);
	for (int i = 0; i < space_curve.rows(); i++)
	{
		space_curve(i, 0) = x_vec[i];
		space_curve(i, 1) = y_vec[i];
		space_curve(i, 2) = z_vec[i];
	}

	return space_curve;
}

Eigen::Vector3d space_curve_means(const space_curve_t& space_curve)
{
	Eigen::Vector3d means = Eigen::Vector3d(0, 0, 0);
	for (int i = 0; i < space_curve.rows(); i++)
	{
		means += space_curve.row(i);
	}

	return means / space_curve.rows();
}

//Eigen::VectorXd fit_to_plane(const space_curve_t& open_square, const Eigen::MatrixXd& additional_uv_points)
//{
//	Eigen::VectorXd output;
//	Eigen::Vector3d centroid = open_square.colwise().mean();
//	Eigen::MatrixXd centered_points = open_square.colwise() - centroid;
//	Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered_points, Eigen::ComputeThinU | Eigen::ComputeThinV);
//	Eigen::Vector3d normal_vector = svd.matrixV().col(2);
//
//
//	for (int i = 0; i < additional_uv_points.rows(); i++)
//	{
//		Eigen::Vector2d uv = additional_uv_points.row(i);
//		double z = -(normal_vector(0) * uv(0) + normal_vector(1) * uv(1)) / normal_vector(2);
//		Eigen::Vector3d centered_point = Eigen::Vector3d(uv(0), uv(1), z);
//		Eigen::Vector3d new_point = centered_point + centroid;
//		output.row(i) = new_point;
//	}
//
//	return output;
//}