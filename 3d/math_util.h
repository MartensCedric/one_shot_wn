#pragma once
#define R_AT_INF 5.0
#define A_LOG_COEFF -1.0

#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <array>
#include <Eigen/Core>
#include "uv_util.h"



std::vector<double> linspace(double start, double end, int num_points, bool include_end);
Eigen::Vector3d interpolate_curve(double t, const std::vector<Eigen::Vector3d>& curve);
boundary_curve_t circular_shift(const boundary_curve_t& curve, int n);
Eigen::VectorXd circular_shift(const Eigen::VectorXd& vec, int n);

#define ASSERT_RELEASE(X, M) if(!(X)) { throw std::runtime_error(M);}

template <typename T>
inline bool is_within_epsilon(T v1, T v2, T epsilon)
{
	return std::abs(v1 - v2) < epsilon;
}

template <typename T>
bool is_in_vector_epsilon(const std::vector<T>& vec, T element, T epsilon)
{
	for (int i = 0; i < vec.size(); i++)
	{
		if (is_within_epsilon(vec[i], element, epsilon))
			return true;
	}
	return false;
}


struct ray_box_intersection_result
{
	bool intersects;
	double t_min;
	double t_max;
};

struct box {
	double x_min;
	double x_max;
	double y_min;
	double y_max;
	double z_min;
	double z_max;
};


struct boundary_at_inf {
	CircleParametrization exterior_boundary;
	curve_info_t exterior_curve_info;
	Eigen::MatrixXd df_dn_ext;
	Eigen::Vector3d I2;
	Eigen::Vector3d means; // 'b'
};


ray_box_intersection_result ray_box_intersection(Eigen::Vector3d origin, Eigen::Vector3d direction, const struct box& box);
box bounding_box(const space_curve_t& patch);

Eigen::VectorXd fundamental_solution_laplacian(Eigen::Vector2d p, const boundary_curve_t& boundary_curve, const curve_info_t& curve_info);
Eigen::Matrix<double, Eigen::Dynamic, 2> fundamental_solution_derivative_laplacian(Eigen::Vector2d p, const boundary_curve_t& boundary_curve, const curve_info_t& curve_info);
Eigen::Matrix<double, Eigen::Dynamic, 2> fundamental_solution_double_derivative_laplacian(Eigen::Vector2d p, const boundary_curve_t& boundary_curve, const curve_info_t& curve_info);

Eigen::Matrix<double, Eigen::Dynamic, 2> fundamental_solution_derivative_laplacian_for_boundaries(Eigen::Vector2d p, const BoundaryParametrization* boundary_param);
Eigen::VectorXd fundamental_solution_laplacian_for_boundaries(Eigen::Vector2d p, const BoundaryParametrization* boundary_param);

Eigen::MatrixXd compute_bem_H(const boundary_curve_t& boundary_curve, const curve_info_t& curve_info);
Eigen::MatrixXd compute_bem_G(const boundary_curve_t& boundary_curve, const curve_info_t& curve_info);

Eigen::MatrixXd compute_bem_G_from_boundaries(const BoundaryParametrization* boundary_param);
Eigen::MatrixXd compute_bem_H_from_boundaries(const BoundaryParametrization* boundary_param);

Eigen::VectorXd df_dn_from_boundary(const space_curve_t& space_curve, const curve_info_t& curve_info);
Eigen::VectorXd df_dn_from_G_and_H(const space_curve_t& space_curve, const Eigen::MatrixXd& G, const Eigen::MatrixXd& H);
Eigen::MatrixXd exterior_df_dn(const space_curve_t& space_curve, const Eigen::MatrixXd& G, const Eigen::MatrixXd& H, const curve_info_t& curve_info);
Eigen::Vector3d interpolate_curve(double t, const std::vector<Eigen::Vector3d>& curve);

Eigen::Vector3d representation_formula_interior(Eigen::Vector2d uv, const BoundaryParametrization* boundary_param, const space_curve_t& space_curve, const Eigen::MatrixXd& df_dn);
//Eigen::Vector3d representation_formula_exterior(Eigen::Vector2d uv, const BoundaryParametrization* boundary_param, const space_curve_t& space_curve, const Eigen::MatrixXd& df_dn);
Eigen::Vector3d evaluate_F(Eigen::Vector3d func_args, const BoundaryParametrization* boundary_param, const space_curve_t& space_curve, const Eigen::MatrixXd& df_dn, Eigen::Vector3d point, Eigen::Vector3d dir);
Eigen::Matrix<double, 3, 2> jacobian_F(Eigen::Vector2d point, const space_curve_t& space_curve, const Eigen::MatrixXd& df_dn, const BoundaryParametrization* boundary_param);


inline bool insideness_is_open(double insideness) {
	return std::abs(insideness - 1.0) > 0.000001;
}

inline int nmod(int k, int n) {
	return ((k %= n) < 0) ? k + n : k;
}

Eigen::Matrix<double, 3, 2> jacobian_fd(Eigen::Vector2d point, const std::function<Eigen::Vector3d(Eigen::Vector2d)>& func, double eps);

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T>& v) {

	// initialize original index locations
	std::vector<size_t> idx(v.size());
	std::iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	// using std::stable_sort instead of std::sort
	// to avoid unnecessary index re-orderings
	// when v contains elements of equal values 
	std::stable_sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

	return idx;
}

template <typename T>
std::vector<size_t> sort_indexes_rev(const std::vector<T>& v) {

	// initialize original index locations
	std::vector<size_t> idx(v.size());
	std::iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	// using std::stable_sort instead of std::sort
	// to avoid unnecessary index re-orderings
	// when v contains elements of equal values 
	std::stable_sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });

	return idx;
}

template<typename T>
std::vector<T> random_shuffle(const std::vector<T>& v, std::vector<int>& ordering)
{
	std::vector<T> output(v.size());
	std::random_shuffle(ordering.begin(), ordering.end());

	for (int i = 0; i < ordering.size(); i++)
	{
		output[i] = v[ordering[i]];
	}
	return output;
}

Eigen::MatrixXd random_shuffle(const Eigen::MatrixXd& v, std::vector<int>& ordering);

template<typename T>
std::vector<T> unshuffle_from_random(const std::vector<T>& v, const std::vector<int>& ordering)
{
	std::vector<T> output(v.size());

	for (int i = 0; i < ordering.size(); i++)
	{
		output[ordering[i]] = v[i];
	}

	return output;
}

template<typename T> 
std::vector<std::array<T, 3>> eigen_to_cpp_array(const Eigen::Matrix<T, Eigen::Dynamic, 3>& mat)
{
	const T* T_ptr = mat.data();
	std::vector<T> col0(T_ptr, T_ptr + mat.rows());
	std::vector<T> col1(T_ptr + mat.rows(), T_ptr + 2 * mat.rows());
	std::vector<T> col2(T_ptr + 2 * mat.rows(), T_ptr + 3 * mat.rows());
	std::vector<std::array<T, 3>> out;
	for (size_t i = 0; i < mat.rows(); ++i) {
		out.push_back({ col0.at(i), col1.at(i), col2.at(i) });
	}
	return out;
}

int find_by_row(const Eigen::MatrixXd& mat, const Eigen::RowVector3d& query, double tolerance, int start_idx);

Eigen::MatrixXd rescale_to_unit(const Eigen::MatrixXd& points);
std::vector<Eigen::MatrixXd> rescale_to_unit(const std::vector<Eigen::MatrixXd>& patches);
Eigen::MatrixXd slice_XY(int row_count);
Eigen::MatrixXd slice_XZ(int row_count);
std::vector<std::vector<int>> rays_from_slice(const Eigen::MatrixXd&);

void triangulate_unit_square(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int N);

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Arrangement_on_surface_2.h>
#include <CGAL/Arr_geodesic_arc_on_sphere_traits_2.h>
#include <CGAL/Arr_spherical_topology_traits_2.h>

template<typename T>
Eigen::Vector3d cgal_point_to_eigen(const T& cgal_p)
{
	double idx = CGAL::to_double(cgal_p.dx().approx());
	double idy = CGAL::to_double(cgal_p.dy().approx());
	double idz = CGAL::to_double(cgal_p.dz().approx());
	Eigen::Vector3d point = Eigen::Vector3d(idx, idy, idz);
	return point;
}