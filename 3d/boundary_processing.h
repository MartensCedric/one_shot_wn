#pragma once

#include <Eigen/Dense>
#include <vector>


typedef Eigen::Matrix<double, Eigen::Dynamic, 3> space_curve_t;
typedef Eigen::Matrix<double, Eigen::Dynamic, 2> boundary_curve_t;

struct curve_info_t {
	Eigen::VectorXd lengths;
	boundary_curve_t tangents;
	boundary_curve_t normals;
	bool is_open;
};

space_curve_t create_space_curve(const boundary_curve_t& bd_curve, const std::vector<double>& data);
space_curve_t create_space_curve(const boundary_curve_t& bd_curve, const Eigen::VectorXd& data);
space_curve_t xyz_to_space_curve(const std::vector<double>& x_vec, const std::vector<double>& y_vec, const std::vector<double>& z_vec);
curve_info_t compute_curve_info(const boundary_curve_t& curve);
//Eigen::VectorXd fit_to_plane(const space_curve_t& open_square, const Eigen::MatrixXd& additional_uv_points);
Eigen::Vector3d space_curve_means(const space_curve_t& space_curve);