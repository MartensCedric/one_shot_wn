
#include "uv_util.h"
#include "math_util.h"
#include <random>


// https://youtu.be/bvlIYX9cgls
// RETURNS the coefficient of intersection from the FIRST segment. p0 + p0p1 * c
double segment_segment_intersects(const Eigen::Vector2d& s1_p0, const Eigen::Vector2d& s1_p1, const Eigen::Vector2d& s2_p0, const Eigen::Vector2d& s2_p1, bool& intersects)
{
	double epsilon = 0.00001;
	intersects = false;
	Eigen::Vector2d s1_dir = s1_p1 - s1_p0;
	Eigen::Vector2d s2_dir = s2_p1 - s2_p0;

	double x1 = s1_p0(0);
	double x2 = s1_p1(0);
	double x3 = s2_p0(0);
	double x4 = s2_p1(0);

	double y1 = s1_p0(1);
	double y2 = s1_p1(1);
	double y3 = s2_p0(1);
	double y4 = s2_p1(1);

	double b = (x4 - x3) * (y2 - y1) - (y4 - y3) * (x2 - x1);

	if (std::abs(b) < epsilon)
		return 0.0; // colinear

	double a = (x4 - x3) * (y3 - y1) - (y4 - y3) * (x3 - x1);
	double c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);

	double alpha = a / b;
	double beta = c / b;

	intersects = alpha > 0.0 && alpha < 1.0 && beta > 0.0 && beta < 1.0;
	return alpha;
}

std::vector<double> boundary_segment_intersections(const std::vector<boundary_curve_t>& boundary_curves, const Eigen::Vector2d& p0, const Eigen::Vector2d& p1)
{
	std::vector<double> output;

	for (int curve_index = 0; curve_index < boundary_curves.size(); curve_index++)
	{
		for (int i = 0; i < boundary_curves[curve_index].rows() - 1; i++)
		{
			Eigen::Vector2d s2_p0 = boundary_curves[curve_index].row(i);
			Eigen::Vector2d s2_p1 = boundary_curves[curve_index].row(i+1);

			bool intersects;
			double t = segment_segment_intersects(p0, p1, s2_p0, s2_p1, intersects);
			if(intersects)
				output.push_back(t);
		}
	}

	return output;
}

bool SquareParametrization::is_inside_boundary(Eigen::Vector2d uv) const
{
	return uv(0) >= 0.0 && uv(0) <= 1.0 && uv(1) >= 0.0 && uv(1) <= 1.0;
}

std::vector<boundary_curve_t> SquareParametrization::create_boundaries() 
{
	return create_square_boundaries(n);
}

std::vector<Eigen::Vector2d> SquareParametrization::sample_inside_points(int n) const
{
	std::random_device rd; 
	std::mt19937 gen(rd());
	gen.seed(0xced);
	std::uniform_real_distribution<double> dis(0, 1);

	std::vector<Eigen::Vector2d> samples_out;
	for (int i = 0; i < n; i++)
	{
		double u = dis(gen);
		double v = dis(gen);

		assert(is_inside_boundary(Eigen::Vector2d(u, v)));
		samples_out.emplace_back(u, v);
	}

	return samples_out;
}

std::vector<Eigen::Vector2d> SquareParametrization::sample_outside_points(int n) const
{
	std::random_device rd;
	std::mt19937 gen(rd());
	gen.seed(0xced);
	double padding = 3;
	std::uniform_real_distribution<double> dis(-padding, 1 + padding);

	std::vector<Eigen::Vector2d> samples_out;
	for (int i = 0; i < n; i++)
	{
		double u = 0;
		double v = 0;

		do {
			u = dis(gen);
			v = dis(gen);
		} while (is_inside_boundary(Eigen::Vector2d(u, v)));

		assert(!is_inside_boundary(Eigen::Vector2d(u, v)));
		samples_out.emplace_back(u, v);
	}

	return samples_out;
}

bool CircleParametrization::is_inside_boundary(Eigen::Vector2d uv) const
{
	return uv(0) * uv(0) + uv(1) * uv(1) < radius * radius;
}

std::vector<boundary_curve_t> create_square_boundaries(size_t n)
{
	boundary_curve_t boundary_curve;
	int x_count = n / 2;
	int y_count = n - x_count;

	int bottom_x_count = x_count / 2;
	int top_x_count = x_count - bottom_x_count;

	int left_y_count = y_count / 2;
	int right_y_count = y_count - left_y_count;

	assert(bottom_x_count + top_x_count + left_y_count + right_y_count == n);

	boundary_curve.resize(n, 2);

	std::vector<double> bottom_u = linspace(0, 1, bottom_x_count, false);
	std::vector<double> bottom_v = linspace(0, 0, bottom_x_count, false);

	std::vector<double> right_u = linspace(1, 1, right_y_count, false);
	std::vector<double> right_v = linspace(0, 1, right_y_count, false);

	std::vector<double> top_u = linspace(1, 0, top_x_count, false);
	std::vector<double> top_v = linspace(1, 1, top_x_count, false);

	std::vector<double> left_u = linspace(0, 0, left_y_count, false);
	std::vector<double> left_v = linspace(1, 0, left_y_count, false);

	std::vector<double> u_values;
	std::vector<double> v_values;

	u_values.resize(n);

	std::copy(bottom_u.begin(), bottom_u.end(), u_values.begin());
	std::copy(right_u.begin(), right_u.end(), u_values.begin() + bottom_u.size());
	std::copy(top_u.begin(), top_u.end(), u_values.begin() + bottom_u.size() + right_u.size());
	std::copy(left_u.begin(), left_u.end(), u_values.begin() + top_u.size() + bottom_u.size() + right_u.size());


	v_values.resize(n);

	std::copy(bottom_v.begin(), bottom_v.end(), v_values.begin());
	std::copy(right_v.begin(), right_v.end(), v_values.begin() + bottom_v.size());
	std::copy(top_v.begin(), top_v.end(), v_values.begin() + bottom_v.size() + right_v.size());
	std::copy(left_v.begin(), left_v.end(), v_values.begin() + bottom_v.size() + right_v.size() + top_v.size());


	for (size_t i = 0; i < n; i++)
	{
		boundary_curve.coeffRef(i, 0) = u_values[i];
		boundary_curve.coeffRef(i, 1) = v_values[i];
	}

	return std::vector<boundary_curve_t> { boundary_curve };
}

std::vector<boundary_curve_t> create_circle_boundaries(size_t n, double radius, double initial_angle)
{
	std::vector<double> thetas = linspace(initial_angle, initial_angle + 2.0 * EIGEN_PI, n, false);
	boundary_curve_t boundary_curve;
	boundary_curve.resize(n, 2);

	for (int i = 0; i < n; i++)
	{
		boundary_curve(i, 0) = radius * std::cos(thetas[i]);
		boundary_curve(i, 1) = radius * std::sin(thetas[i]);
	}
	return std::vector<boundary_curve_t> {boundary_curve};
}

std::vector<boundary_curve_t> create_circle_boundaries(size_t n, double radius)
{
	return create_circle_boundaries(n, radius, 0.0);
}

std::vector<boundary_curve_t> CircleParametrization::create_boundaries() 
{
	return create_circle_boundaries(n, radius);
}

std::vector<Eigen::Vector2d> CircleParametrization::sample_inside_points(int n) const
{
	std::random_device rd;
	std::mt19937 gen(rd());
	gen.seed(0xced);
	std::uniform_real_distribution<double> dis_r(0, radius);
	std::uniform_real_distribution<double> dis_theta(0, 2*EIGEN_PI);

	std::vector<Eigen::Vector2d> samples_out;
	for (int i = 0; i < n; i++)
	{
		double r = dis_r(gen);
		double theta = dis_theta(gen);
		double u = r * std::cos(theta);
		double v = r * std::sin(theta);
		assert(is_inside_boundary(Eigen::Vector2d(u, v)));
		samples_out.emplace_back(u, v);
	}

	return samples_out;
}

std::vector<Eigen::Vector2d> CircleParametrization::sample_outside_points(int n) const
{
	std::random_device rd;
	std::mt19937 gen(rd());
	gen.seed(0xced);
	std::uniform_real_distribution<double> dis_r(radius, radius*3.0);
	std::uniform_real_distribution<double> dis_theta(0, 2 * EIGEN_PI);

	std::vector<Eigen::Vector2d> samples_out;
	for (int i = 0; i < n; i++)
	{
		double r = dis_r(gen);
		double theta = dis_theta(gen);
		double u = r * std::cos(theta);
		double v = r * std::sin(theta);
		assert(!is_inside_boundary(Eigen::Vector2d(u, v)));
		samples_out.emplace_back(u, v);
	}

	return samples_out;
}

bool AnnulusParametrization::is_inside_boundary(Eigen::Vector2d uv) const
{
	bool is_inside_big_circle =  uv(0) * uv(0) + uv(1) * uv(1) < big_radius * big_radius;
	bool is_inside_small_circle = uv(0) * uv(0) + uv(1) * uv(1) < small_radius * small_radius;
	return is_inside_big_circle && !is_inside_small_circle;
}

std::vector<boundary_curve_t> AnnulusParametrization::create_boundaries()
{
	boundary_curve_t small_circle = create_circle_boundaries(n, small_radius)[0];
	boundary_curve_t big_circle = create_circle_boundaries(n, big_radius)[0];

	return std::vector<boundary_curve_t>{big_circle, small_circle};
}

std::vector<Eigen::Vector2d> AnnulusParametrization::sample_inside_points(int n) const
{
	std::random_device rd;
	std::mt19937 gen(rd());
	gen.seed(0xced);
	std::uniform_real_distribution<double> dis_r(small_radius, big_radius);
	std::uniform_real_distribution<double> dis_theta(0, 2 * EIGEN_PI);

	std::vector<Eigen::Vector2d> samples_out;
	for (int i = 0; i < n; i++)
	{
		double r = dis_r(gen);
		double theta = dis_theta(gen);
		double u = r * std::cos(theta);
		double v = r * std::sin(theta);
		assert(is_inside_boundary(Eigen::Vector2d(u, v)));
		samples_out.emplace_back(u, v);
	}

	return samples_out;
}

std::vector<Eigen::Vector2d> AnnulusParametrization::sample_outside_points(int n) const
{
	std::random_device rd;
	std::mt19937 gen(rd());
	gen.seed(0xced);
	std::uniform_real_distribution<double> dis_r(big_radius, big_radius*3.0);
	std::uniform_real_distribution<double> dis_theta(0, 2 * EIGEN_PI);

	std::vector<Eigen::Vector2d> samples_out;
	for (int i = 0; i < n; i++)
	{
		double r = dis_r(gen);
		double theta = dis_theta(gen);
		double u = r * std::cos(theta);
		double v = r * std::sin(theta);
		assert(!is_inside_boundary(Eigen::Vector2d(u, v)));
		samples_out.emplace_back(u, v);
	}

	return samples_out;
}


std::vector<boundary_curve_t> AnnulusOpenParametrization::create_boundaries() {
	std::vector<boundary_curve_t> boundaries;
	int num_elements_on_large_circle = std::round(static_cast<double>(n) / closeness);
	boundary_curve_t large_circle = create_circle_boundaries(num_elements_on_large_circle, R_AT_INF)[0];
	boundary_curve_t small_square_full = create_square_boundaries(num_elements_on_large_circle)[0];
	for (int i = 0; i < small_square_full.rows(); i++)
	{
		small_square_full.row(i) -= Eigen::Vector2d(0.5, 0.5);
	}

	boundary_curve_t small_square = small_square_full.block(0, 0, n, 2);

	return std::vector<boundary_curve_t>{large_circle, small_square};
}

boundary_curve_t create_square_uv_parameterization_open(size_t n, double closedness)
{
	boundary_curve_t parameterization = create_square_boundaries( static_cast<double>(n) / closedness)[0];
	boundary_curve_t boundary_curve;
	boundary_curve.resize(n, 2);
	for (int i = 0; i < n; i++)
		boundary_curve.row(i) = parameterization.row(i);
	return boundary_curve;
}

bool AnnulusOpenParametrization::is_inside_boundary(Eigen::Vector2d uv) const
{
	return uv(0) * uv(0) + uv(1) * uv(1) < big_radius;
}

std::vector<Eigen::Vector2d> AnnulusOpenParametrization::sample_inside_points(int n) const
{
	std::random_device rd;
	std::mt19937 gen(rd());
	gen.seed(0xced);
	std::uniform_real_distribution<double> dis_u(0.0, R_AT_INF);
	std::uniform_real_distribution<double> dis_v(0, 2.0 * EIGEN_PI);

	std::vector<Eigen::Vector2d> samples_out;
	for (int i = 0; i < n; i++)
	{
		double r, t;
		r = dis_u(gen);
		t = dis_v(gen);

		double u = r * std::cos(t);
		double v = r * std::sin(t);

		samples_out.emplace_back(u, v);
	}

	return samples_out;
}

std::vector<Eigen::Vector2d> AnnulusOpenParametrization::sample_outside_points(int n) const
{
	std::random_device rd;
	std::mt19937 gen(rd());
	gen.seed(0xced);
	std::uniform_real_distribution<double> dis_r(big_radius, big_radius * 3.0);
	std::uniform_real_distribution<double> dis_theta(0.0, 2.0 * EIGEN_PI);

	std::vector<Eigen::Vector2d> samples_out;
	for (int i = 0; i < n; i++)
	{
		double r = dis_r(gen);
		double theta = dis_theta(gen);
		double u = r * std::cos(theta);
		double v = r * std::sin(theta);
		assert(!is_inside_boundary(Eigen::Vector2d(u, v)));
		samples_out.emplace_back(u, v);
	}

	return samples_out;
}

space_curve_t create_space_curve_for_boundaries(const BoundaryParametrization* boundary_param, const Eigen::VectorXd& data)
{
	int rows = boundary_param->get_total_points();

	space_curve_t space_curve;
	space_curve.resize(rows, 3);
	int row_acc = 0;
	for(int i = 0; i < boundary_param->get_boundary_curves().size(); i++)
	{
		int current_rows = boundary_param->get_boundary_curves()[i].rows();
		space_curve.block(row_acc, 0, current_rows, 2) = boundary_param->get_boundary_curves()[i];
		row_acc += current_rows;
	}
	space_curve.block(0, 2, data.rows(), 1) = data;

	return space_curve;

}

Eigen::Vector3d fit_plane(const space_curve_t& patch)
{
	Eigen::Vector3d centroid = patch.colwise().mean();
	Eigen::MatrixXd centered_points = patch.rowwise() - centroid.transpose();

	Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered_points, Eigen::ComputeThinU | Eigen::ComputeThinV);

	Eigen::Vector3d normal_vector = svd.matrixV().col(2);
	return normal_vector;
}

Eigen::VectorXd fit_data_to_plane(const Eigen::Vector3d& plane_normal, const boundary_curve_t& data)
{
	Eigen::VectorXd output(data.rows());
	output.setZero();
	for (int i = 0; i < data.rows(); i++)
	{
		double x = data.coeff(i, 0);
		double y = data.coeff(i, 1);

		double z = -(plane_normal(0) * x + plane_normal(1) * y) / plane_normal(2);
		output(i) = z;
	}

	return output;
}

space_curve_t fit_data_to_patch_plane(const boundary_curve_t& planar_data, const boundary_curve_t& boundary_param, const space_curve_t& patch)
{
	Eigen::VectorXd x_values = patch.col(0);
	Eigen::VectorXd y_values = patch.col(1);
	Eigen::VectorXd z_values = patch.col(2);

	space_curve_t x_space_curve = create_space_curve(boundary_param, x_values);
	space_curve_t y_space_curve = create_space_curve(boundary_param, y_values);
	space_curve_t z_space_curve = create_space_curve(boundary_param, z_values);

	Eigen::Vector3d x_normal = fit_plane(x_space_curve);
	Eigen::Vector3d y_normal = fit_plane(y_space_curve);
	Eigen::Vector3d z_normal = fit_plane(z_space_curve);

	Eigen::Vector3d x_mean = x_space_curve.colwise().mean();
	Eigen::Vector3d y_mean = y_space_curve.colwise().mean();
	Eigen::Vector3d z_mean = z_space_curve.colwise().mean();

	Eigen::VectorXd far_x_values = fit_data_to_plane(x_normal, planar_data);
	Eigen::VectorXd far_y_values = fit_data_to_plane(y_normal, planar_data);
	Eigen::VectorXd far_z_values = fit_data_to_plane(z_normal, planar_data);

	space_curve_t output(planar_data.rows(), 3);
	output.col(0) = far_x_values.array() + x_mean(2);
	output.col(1) = far_y_values.array() + y_mean(2);
	output.col(2) = far_z_values.array() + z_mean(2);
	return output;
}

bool is_in_vector_epsilon(const std::vector<struct root_position>& v, const struct root_position& element, double eps_t, double eps_uv)
{
	for (int i = 0; i < v.size(); i++)
	{
		const struct root_position& root = v.at(i);
		if (is_within_epsilon(root.u, element.u, eps_uv) && is_within_epsilon(root.v, element.v, eps_uv) && is_within_epsilon(root.t, element.t, eps_t))
			return true;
	}
	return false;
}

bool is_in_vector_double_epsilon(const std::vector<double>& v, double element, double eps_t)
{
	for (int i = 0; i < v.size(); i++)
	{
		double current_val = v.at(i);
		if (is_within_epsilon(current_val, element, eps_t))
			return true;
	}
	return false;
}