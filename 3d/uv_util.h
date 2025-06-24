#pragma once

#include <Eigen/Dense>
#include <vector>

#include "boundary_processing.h"

struct open_boundary {
	std::vector<boundary_curve_t> curves;
};

double segment_segment_intersects(const Eigen::Vector2d& s1_p0, const Eigen::Vector2d& s1_p1, const Eigen::Vector2d& s2_p0, const Eigen::Vector2d& s2_p1, bool& intersects);

std::vector<double> boundary_segment_intersections(const std::vector<boundary_curve_t>& boundary_curves, const Eigen::Vector2d& p0, const Eigen::Vector2d& p1);
bool is_in_vector_double_epsilon(const std::vector<double>& v, double element, double eps_t);

// This code will add one level of indirection to the vtable. Should be ok but check performance
// if performance is an issue use function pointers on `get_inside` this is probably the most called function.
class BoundaryParametrization
{
protected:
	size_t n;
	std::vector<boundary_curve_t> boundary_curves;
	std::vector<curve_info_t> curve_infos;
	Eigen::MatrixXd normals;
	std::vector<int> boundary_signs;
	std::vector<bool> boundary_open;
	int total_segments;
	int total_points;

public:
	explicit BoundaryParametrization(size_t n) : n(n), total_segments(0), total_points(0) {};
	virtual void init_boundary() { 
		boundary_curves = this->create_boundaries();
		init_boundary(boundary_curves);
	}
	virtual void init_boundary(const std::vector<boundary_curve_t>& bd) {
		boundary_curves = bd;
		total_points = 0;
		n = bd.size();
		std::vector<Eigen::Vector2d> normals_vec;

		for (int i = 0; i < boundary_curves.size(); i++)
		{
			curve_info_t curve_info = compute_curve_info(boundary_curves[i]);
			curve_info.is_open = boundary_open[i];
			int rows = boundary_curves[i].rows();
			total_points += rows;
			if (curve_info.is_open)
			{
				int normal_rows = curve_info.normals.rows();
				Eigen::MatrixXd normals_open(normal_rows - 1, 2);
				Eigen::MatrixXd tangents_open(normal_rows - 1, 2);
				Eigen::VectorXd lengths_open(normal_rows - 1);

				for (int j = 0; j < rows - 1; j++)
				{
					normals_open.row(j) = curve_info.normals.row(j);
					tangents_open.row(j) = curve_info.tangents.row(j);
					lengths_open(j) = curve_info.lengths[j];
				}

				curve_info.normals = normals_open;
				curve_info.tangents = tangents_open;
				curve_info.lengths = lengths_open;
			}


			curve_infos.push_back(curve_info);

			for (int j = 0; j < curve_info.normals.rows(); j++)
			{
				normals_vec.push_back(curve_info.normals.row(j));
			}
		}

		normals.resize(normals_vec.size(), 2);

		for (int j = 0; j < normals_vec.size(); j++)
		{
			normals.coeffRef(j, 0) = normals_vec[j](0);
			normals.coeffRef(j, 1) = normals_vec[j](1);
		}

		total_segments = 0;
		for (int i = 0; i < boundary_curves.size(); i++)
			total_segments += get_segments(i);
	}
	virtual std::vector<boundary_curve_t> create_boundaries() = 0;
	virtual bool is_inside_boundary(Eigen::Vector2d uv) const = 0;
	const std::vector<boundary_curve_t>& get_boundary_curves() const { return boundary_curves; }
	const std::vector<curve_info_t>& get_curve_infos() const { return curve_infos; }
	const double get_boundary_sign(int i) const { return boundary_signs.at(i); }
	const bool is_open(int i) const { return boundary_open.at(i); }
	const int get_segments(int i) const { return is_open(i) ? (boundary_curves.at(i).rows() - 1) : boundary_curves.at(i).rows(); }
	const int get_total_segments() const { return total_segments; }
	const int get_total_points() const { return total_points; }
	const Eigen::MatrixXd& get_normals() const { return normals; }
	virtual std::vector<Eigen::Vector2d> sample_inside_points(int n) const = 0;
	virtual std::vector<Eigen::Vector2d> sample_outside_points(int n) const = 0;
};

class SquareParametrization : public BoundaryParametrization {
public:
	explicit SquareParametrization(size_t n) : BoundaryParametrization(n) { boundary_signs = { 1 }; boundary_open = { false }; };
	std::vector<boundary_curve_t> create_boundaries() override;
	bool is_inside_boundary(Eigen::Vector2d uv) const override;
	std::vector<Eigen::Vector2d> sample_inside_points(int n) const override;
	std::vector<Eigen::Vector2d> sample_outside_points(int n) const override;
};

class CircleParametrization : public BoundaryParametrization {
protected:
	double radius;
public:
	CircleParametrization(double radius, size_t n) : BoundaryParametrization(n), radius(radius) { boundary_signs = { 1 }; boundary_open = { false }; };
	std::vector<boundary_curve_t> create_boundaries() override;
	bool is_inside_boundary(Eigen::Vector2d uv) const override;
	std::vector<Eigen::Vector2d> sample_inside_points(int n) const override;
	std::vector<Eigen::Vector2d> sample_outside_points(int n) const override;
};

class AnnulusParametrization : public BoundaryParametrization {
protected:
	double small_radius;
	double big_radius;
public:
	AnnulusParametrization(double small_r, double big_r, size_t n) : BoundaryParametrization(n), small_radius(small_r), big_radius(big_r) {
		boundary_signs = { 1, -1 }; boundary_open = { false, false };
	};
	std::vector<boundary_curve_t> create_boundaries() override;
	bool is_inside_boundary(Eigen::Vector2d uv) const override;
	std::vector<Eigen::Vector2d> sample_inside_points(int n) const override;
	std::vector<Eigen::Vector2d> sample_outside_points(int n) const override;
};


class AnnulusOpenParametrization : public BoundaryParametrization {
protected:
	double big_radius;
	double closeness;
public:
	AnnulusOpenParametrization(double big_r, double closeness, size_t n) : BoundaryParametrization(n), big_radius(big_r), closeness(closeness) { boundary_signs = { 1, -1 }; boundary_open = { false, true}; };
	std::vector<boundary_curve_t> create_boundaries() override;
	bool is_inside_boundary(Eigen::Vector2d uv) const override;
	std::vector<Eigen::Vector2d> sample_inside_points(int n) const override;
	std::vector<Eigen::Vector2d> sample_outside_points(int n) const override;
};

//class SquareAnnulusParametrization : public BoundaryParametrization {
//protected:
//	double small_radius;
//	double big_radius;
//public:
//	SquareAnnulusParametrization(double small_r, double big_r, size_t n) : BoundaryParametrization(n), small_radius(small_r), big_radius(big_r) { boundary_signs = { 1, -1 }; };
//	boundary_curve_t create_boundary() override;
//	bool is_inside_boundary(Eigen::Vector2d uv) const override;
//	std::vector<Eigen::Vector2d> sample_inside_points(int n) const override;
//	std::vector<Eigen::Vector2d> sample_outside_points(int n) const override;
//};

std::vector<boundary_curve_t> create_circle_boundaries(size_t n, double radius);
std::vector<boundary_curve_t> create_circle_boundaries(size_t n, double radius, double initial_angle);
boundary_curve_t create_square_uv_parameterization_open(size_t n, double closedness);

space_curve_t create_space_curve_for_boundaries(const BoundaryParametrization* boundary_param, const Eigen::VectorXd& data);
std::vector<boundary_curve_t> create_square_boundaries(size_t n);
boundary_curve_t create_square_uv_parameterization_open(size_t n, double closedness);
Eigen::Vector3d fit_plane(const space_curve_t& patch);
Eigen::VectorXd fit_data_to_plane(const Eigen::Vector3d& plane_normal, const boundary_curve_t& data);
space_curve_t fit_data_to_patch_plane(const boundary_curve_t& planar_data, const boundary_curve_t& boundary_param, const space_curve_t& patch);


struct root_position {
	double u, v, t;
};

bool is_in_vector_epsilon(const std::vector<struct root_position>& v, const struct root_position& element, double eps_t, double eps_uv);