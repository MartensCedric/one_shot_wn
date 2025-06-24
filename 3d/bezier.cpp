#include "bezier.h"
std::function<Eigen::Vector3d(double)> create_cubic_bezier_func(const Eigen::MatrixXd& control_points)
{
	std::function<Eigen::Vector3d(double)> bezier_func = [=](double t) {
			Eigen::Vector3d b0 = std::pow(1.0 - t, 3) * control_points.row(0);
			Eigen::Vector3d b1 = 3.0 * std::pow(1.0 - t, 2) * t * control_points.row(1);
			Eigen::Vector3d b2 = 3.0 * (1.0 - t) * std::pow(t,2) * control_points.row(2);
			Eigen::Vector3d b3 = std::pow(t, 3) * control_points.row(3);
			return b0 + b1 + b2 + b3;
		};
	return bezier_func;
}
std::function<Eigen::Vector3d(Eigen::Vector2d)> create_bezier_func(const Eigen::MatrixXd& control_points)
{
	std::function<Eigen::Vector3d(Eigen::Vector2d)> bezier_func = [=](Eigen::Vector2d uv) {
		// Cubic Bezier Triangle
		double u = uv(0);
		double v = uv(1);
		Eigen::RowVectorXd basis(10);
		basis(0) = -std::pow((u + v - 1), 3);
		basis(1) = 3. * v * std::pow((u + v - 1), 2);
		basis(2) = 3. * u * std::pow((u + v - 1), 2);
		basis(3) = -3. * std::pow(v, 2.) * (u + v - 1);
		basis(4) = -6. * u * v * (u + v - 1);
		basis(5) = -3. * std::pow(u, 2.) * (u + v - 1);
		basis(6) = std::pow(v, 3);
		basis(7) = 3. * u * std::pow(v, 2);
		basis(8) = 3. * std::pow(u, 2) *v;
		basis(9) = std::pow(u, 3);
		
		Eigen::Vector3d result = basis * control_points ;
		return result;
	};

	return bezier_func;
}

std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> create_bezier_jac_func(const Eigen::MatrixXd& control_points)
{
	std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> bezier_jac_func = [=](Eigen::Vector2d uv) {
		// Cubic Bezier Triangle
		double u = uv(0);
		double v = uv(1);
		const Eigen::MatrixXd& C = control_points;

		double A_1_1 = control_points(0, 0);
		double A_2_1 = control_points(1, 0);
		double A_3_1 = control_points(2, 0);
		double A_4_1 = control_points(3, 0);
		double A_5_1 = control_points(4, 0);
		double A_6_1 = control_points(5, 0);
		double A_7_1 = control_points(6, 0);
		double A_8_1 = control_points(7, 0);
		double A_9_1 = control_points(8, 0);
		double A_10_1 = control_points(9, 0);

		double A_1_2 = control_points(0, 1);
		double A_2_2 = control_points(1, 1);
		double A_3_2 = control_points(2, 1);
		double A_4_2 = control_points(3, 1);
		double A_5_2 = control_points(4, 1);
		double A_6_2 = control_points(5, 1);
		double A_7_2 = control_points(6, 1);
		double A_8_2 = control_points(7, 1);
		double A_9_2 = control_points(8, 1);
		double A_10_2 = control_points(9, 1);

		double A_1_3 = control_points(0, 2);
		double A_2_3 = control_points(1, 2);
		double A_3_3 = control_points(2, 2);
		double A_4_3 = control_points(3, 2);
		double A_5_3 = control_points(4, 2);
		double A_6_3 = control_points(5, 2);
		double A_7_3 = control_points(6, 2);
		double A_8_3 = control_points(7, 2);
		double A_9_3 = control_points(8, 2);
		double A_10_3 = control_points(9, 2);

		Eigen::Matrix<double, 3, 2> jac;
		
		jac(0, 0) = 3 * A_3_1 * (u + v - 1) * (u + v - 1) - 3 * A_1_1 * (u + v - 1) * (u + v - 1) - 3 * A_6_1 * u * u + 3 * A_10_1 * u * u - 3 * A_4_1 * v * v + 3 * A_8_1 * v * v + 3 * A_3_1 * u * (2 * u + 2 * v - 2) + 3 * A_2_1 * v * (2 * u + 2 * v - 2) - 6 * A_6_1 * u * (u + v - 1) - 6 * A_5_1 * v * (u + v - 1) - 6 * A_5_1 * u * v + 6 * A_9_1 * u * v;
		jac(0, 1) = 3 * A_2_1 * (u + v - 1) * (u + v - 1) - 3 * A_1_1 * (u + v - 1) * (u + v - 1) - 3 * A_6_1 * u * u + 3 * A_9_1 * u * u - 3 * A_4_1 * v * v + 3 * A_7_1 * v * v + 3 * A_3_1 * u * (2 * u + 2 * v - 2) + 3 * A_2_1 * v * (2 * u + 2 * v - 2) - 6 * A_5_1 * u * (u + v - 1) - 6 * A_4_1 * v * (u + v - 1) - 6 * A_5_1 * u * v + 6 * A_8_1 * u * v;

		jac(1, 0) = 3 * A_3_2 * (u + v - 1) * (u + v - 1) - 3 * A_1_2 * (u + v - 1) * (u + v - 1) - 3 * A_6_2 * u * u + 3 * A_10_2 * u * u - 3 * A_4_2 * v * v + 3 * A_8_2 * v * v + 3 * A_3_2 * u * (2 * u + 2 * v - 2) + 3 * A_2_2 * v * (2 * u + 2 * v - 2) - 6 * A_6_2 * u * (u + v - 1) - 6 * A_5_2 * v * (u + v - 1) - 6 * A_5_2 * u * v + 6 * A_9_2 * u * v;
		jac(1, 1) = 3 * A_2_2 * (u + v - 1) * (u + v - 1) - 3 * A_1_2 * (u + v - 1) * (u + v - 1) - 3 * A_6_2 * u * u + 3 * A_9_2 * u * u - 3 * A_4_2 * v * v + 3 * A_7_2 * v * v + 3 * A_3_2 * u * (2 * u + 2 * v - 2) + 3 * A_2_2 * v * (2 * u + 2 * v - 2) - 6 * A_5_2 * u * (u + v - 1) - 6 * A_4_2 * v * (u + v - 1) - 6 * A_5_2 * u * v + 6 * A_8_2 * u * v;

		jac(2, 0) = 3 * A_3_3 * (u + v - 1) * (u + v - 1) - 3 * A_1_3 * (u + v - 1) * (u + v - 1) - 3 * A_6_3 * u * u + 3 * A_10_3 * u * u - 3 * A_4_3 * v * v + 3 * A_8_3 * v * v + 3 * A_3_3 * u * (2 * u + 2 * v - 2) + 3 * A_2_3 * v * (2 * u + 2 * v - 2) - 6 * A_6_3 * u * (u + v - 1) - 6 * A_5_3 * v * (u + v - 1) - 6 * A_5_3 * u * v + 6 * A_9_3 * u * v;
		jac(2, 1) = 3 * A_2_3 * (u + v - 1) * (u + v - 1) - 3 * A_1_3 * (u + v - 1) * (u + v - 1) - 3 * A_6_3 * u * u + 3 * A_9_3 * u * u - 3 * A_4_3 * v * v + 3 * A_7_3 * v * v + 3 * A_3_3 * u * (2 * u + 2 * v - 2) + 3 * A_2_3 * v * (2 * u + 2 * v - 2) - 6 * A_5_3 * u * (u + v - 1) - 6 * A_4_3 * v * (u + v - 1) - 6 * A_5_3 * u * v + 6 * A_8_3 * u * v;
		return jac;
	};

	return bezier_jac_func;
}

Eigen::MatrixXd barycentric_mesh_grid(int resolution)
{
	int N = 2 * resolution;
	int len = (N + 1) * (N + 2) / 2;
	Eigen::MatrixXd pts(len, 3);

	for (int k = 0; k <= N; k++)
	{
		std::vector<int> a_range(N-k + 1, 0);
		std::iota(a_range.rbegin(), a_range.rend(), 0);
		int i = len - (N - k) * (N + 1 - k) / 2 - 1;

		for (int j = 0; j < a_range.size(); j++)
		{
			int index = i - N + k + j;
			double a_range_j = static_cast<double>(a_range[j]) / N;
			pts.row(index) = Eigen::RowVector3d(a_range_j, k * (1 / static_cast<double>(N)), 1 - k / static_cast<double>(N) - a_range_j);
		}
	}

	return pts;	
}

Eigen::MatrixXi triangle_indices(int resolution)
{
	int N = 2 * resolution;
	int len = (N) * (N + 1) / 2;
	int total = len + N + 1;
	Eigen::MatrixXi indices_1(len, 3);
	Eigen::MatrixXi indices_2(len - N, 3);

	for (int k = 0; k <= N; k++)
	{
		int i = total - (N - k + 1) * (N - k + 2) / 2 + 1;
		for (int j = 0; j < N - k; j++)
		{
			int index = j + i - k - 1;
			indices_1.row(index) = Eigen::RowVector3i(i+j - 1, i+1+j - 1, i + N -k + 1 + j - 1);
		}
	}

	for (int k = 1; k <= N; k++)
	{
		int i = total - (N - k + 1) * (N - k + 2) / 2 + 1;
		for (int j = 0; j <= i - 2 * k - 1 - (i - k - N); j++)
		{
			int index = i - k - N + j - 1;
			indices_2.row(index) = Eigen::RowVector3i(i + j - 1, i - N + k - 1 + j - 1, i + 1 + j - 1);
		}
	}

	Eigen::MatrixXi output(indices_1.rows() + indices_2.rows(), 3);
	output.block(0, 0, indices_1.rows(), 3) = indices_1;
	output.block(indices_1.rows(), 0, indices_2.rows(), 3) = indices_2;
	return output;
}

std::vector<patch_t> patch_from_bezier(int num_per_edge, const std::function<Eigen::Vector3d(Eigen::Vector2d)>& func)
{
	patch_t output;
	output.is_open = false;
	output.gaps_ids = {};
	std::vector<double> lin = linspace(0, 1, num_per_edge, false);

	output.curve.resize(num_per_edge * 3, 3);

	for (int i = 0; i < num_per_edge; i++)
	{
		Eigen::Vector2d uv(lin[i], 0);
		output.curve.row(i) = func(uv);
	}

	for (int i = 0; i < num_per_edge; i++)
	{
		Eigen::Vector2d uv(1 - lin[i], lin[i]);
		output.curve.row(num_per_edge + i) = func(uv);
	}


	for (int i = 0; i < num_per_edge; i++)
	{
		Eigen::Vector2d uv(0, 1 - lin[i]);
		output.curve.row(2 * num_per_edge + i) = func(uv);
	}

	return {output};
}

Eigen::MatrixXd quadratic_bezier(const Eigen::Matrix3d& control_points, int num_points)
{
	Eigen::MatrixXd output(num_points, 3);

	std::vector<double> ts = linspace(0, 1, num_points, true);

	Eigen::Vector3d p0 = control_points.row(0);
	Eigen::Vector3d p1 = control_points.row(1);
	Eigen::Vector3d p2 = control_points.row(2);

	for (int i = 0; i < num_points; i++)
	{
		double t = ts[i];
		output.row(i) = (1.0 - t) * ((1.0 - t) * p0 + t * p1) + t * ((1.0 - t) * p1 + t * p2);
	}

	return output;
}

void run_bezier_intersections()
{
	std::vector<std::pair< Eigen::Matrix3d, Eigen::Matrix3d>> positive_case_control_points;
	std::vector<std::pair< Eigen::Matrix3d, Eigen::Matrix3d>> negative_case_control_points;
	std::vector<std::vector<Eigen::Vector3d>> positive_case_intersections;

	for (int i = 0; i < 5000; i++)
	{
		Eigen::Matrix3d control_points_1 = Eigen::Matrix3d::Random();
		Eigen::Matrix3d control_points_2 = Eigen::Matrix3d::Random();
		Eigen::MatrixXd curve_1 = project_to_sphere(quadratic_bezier(control_points_1, 1000), Eigen::Vector3d::Zero());
		Eigen::MatrixXd curve_2 = project_to_sphere(quadratic_bezier(control_points_2, 1000), Eigen::Vector3d::Zero());

		int num_intersections = 0;

		std::vector<Eigen::Vector3d> intersections;

		for (int j = 0; j < curve_1.rows() - 1; j++)
		{
			int curve1_first_index = j;
			int curve1_second_index = j + 1;
			for (int k = 0; k < curve_2.rows() - 1; k++)
			{
				int curve2_first_index = k;
				int curve2_second_index = k + 1;

				intersection_t inter = spherical_segments_intersect(curve_1.row(curve1_first_index), curve_1.row(curve1_second_index),
					curve_2.row(curve2_first_index), curve_2.row(curve2_second_index));

				if (inter.intersects)
				{
					num_intersections++;
					intersections.push_back(inter.location);
				}
			}
		}

		if (num_intersections > 0 && positive_case_intersections.size() < 200)
		{
			positive_case_intersections.push_back(intersections);
			positive_case_control_points.push_back({ control_points_1, control_points_2 });
		}
		else if (num_intersections == 0 && negative_case_control_points.size() < 200)
		{
			negative_case_control_points.push_back({ control_points_1, control_points_2 });
		}

		if (negative_case_control_points.size() >= 200 && positive_case_intersections.size() >= 200)
			break;

	}

	Eigen::IOFormat matlab_fmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
	std::ofstream outfile("bezier_bezier_intersections.m");
	for (int i = 0; i < positive_case_intersections.size(); i++)
	{
		outfile << "pos_control_points_1{" + std::to_string(i + 1) << "} = " << positive_case_control_points[i].first.format(matlab_fmt) << ";" << std::endl;
		outfile << "pos_control_points_2{" + std::to_string(i + 1) << "} = " << positive_case_control_points[i].second.format(matlab_fmt) << ";" << std::endl;

		Eigen::MatrixXd intersections_mat(positive_case_intersections[i].size(), 3);

		for (int k = 0; k < positive_case_intersections[i].size(); k++)
		{
			intersections_mat.row(k) = positive_case_intersections[i][k];
		}
		outfile << "pos_intersections{" + std::to_string(i + 1) + "} = " << intersections_mat.format(matlab_fmt) << ";" << std::endl;
	}

	for (int i = 0; i < negative_case_control_points.size(); i++)
	{
		outfile << "neg_control_points_1{" + std::to_string(i + 1) << "} = " << negative_case_control_points[i].first.format(matlab_fmt) << ";" << std::endl;
		outfile << "neg_control_points_2{" + std::to_string(i + 1) << "} = " << negative_case_control_points[i].second.format(matlab_fmt) << ";" << std::endl;
	}

	outfile.close();
}

std::function<Eigen::Vector3d(Eigen::Vector2d)> cubic_bezier_surface(const Eigen::MatrixXd& control_points)
{
	assert(control_points.rows() == 16);
	assert(control_points.cols() == 3);
	std::function<Eigen::Vector3d(Eigen::Vector2d)> bezier_func = [=](Eigen::Vector2d uv) {
		double u = uv(0);
		double v = uv(1);
		Eigen::Vector3d B_0_0 = control_points.row(0);
		Eigen::Vector3d B_0_1 = control_points.row(1);
		Eigen::Vector3d B_0_2 = control_points.row(2);
		Eigen::Vector3d B_0_3 = control_points.row(3);
		Eigen::Vector3d B_1_0 = control_points.row(4);
		Eigen::Vector3d B_1_1 = control_points.row(5);
		Eigen::Vector3d B_1_2 = control_points.row(6);
		Eigen::Vector3d B_1_3 = control_points.row(7);
		Eigen::Vector3d B_2_0 = control_points.row(8);
		Eigen::Vector3d B_2_1 = control_points.row(9);
		Eigen::Vector3d B_2_2 = control_points.row(10);
		Eigen::Vector3d B_2_3 = control_points.row(11);
		Eigen::Vector3d B_3_0 = control_points.row(12);
		Eigen::Vector3d B_3_1 = control_points.row(13);
		Eigen::Vector3d B_3_2 = control_points.row(14);
		Eigen::Vector3d B_3_3 = control_points.row(15);

		return B_0_0 * std::pow(u - 1, 3) * std::pow(v - 1, 3) +
			B_3_3 * std::pow(u, 3) * std::pow(v, 3) - B_0_3 * std::pow(v, 3) * std::pow(u - 1, 3) - B_3_0 * std::pow(u, 3) * std::pow(v - 1, 3) + 3 * B_1_3 * u * std::pow(v, 3) * std::pow(u - 1, 2)
			- 3 * B_2_3 * std::pow(u, 2) * std::pow(v, 3) * (u - 1) + 3 * B_3_1 * std::pow(u, 3) * v * std::pow(v - 1, 2) - 3 * B_3_2 * std::pow(u, 3) * std::pow(v, 2) * (v - 1)
			- 3 * B_1_0 * u * std::pow(u - 1, 2) * std::pow(v - 1, 3) + 3 * B_2_0 * std::pow(u, 2) * (u - 1) * std::pow(v - 1, 3) - 3 * B_0_1 * v * std::pow(u - 1, 3) * std::pow(v - 1, 2)
			+ 3 * B_0_2 * std::pow(v, 2) * std::pow(u - 1, 3) * (v - 1) + 9 * B_1_1 * u * v * std::pow(u - 1, 2) * std::pow(v - 1, 2) - 9 * B_1_2 * u * std::pow(v, 2) * std::pow(u - 1, 2) * (v - 1)
			- 9 * B_2_1 * std::pow(u, 2) * v * (u - 1) * std::pow(v - 1, 2) + 9 * B_2_2 * std::pow(u, 2) * std::pow(v, 2) * (u - 1) * (v - 1);
		};
	
	return bezier_func;
}

std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> cubic_bezier_surface_jac(const Eigen::MatrixXd& control_points)
{
	assert(control_points.rows() == 16);
	assert(control_points.cols() == 3);
	std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> bezier_jac_func = [=](Eigen::Vector2d uv) {
		double u = uv(0);
		double v = uv(1);
		Eigen::Vector3d B_0_0 = control_points.row(0);
		Eigen::Vector3d B_0_1 = control_points.row(1);
		Eigen::Vector3d B_0_2 = control_points.row(2);
		Eigen::Vector3d B_0_3 = control_points.row(3);
		Eigen::Vector3d B_1_0 = control_points.row(4);
		Eigen::Vector3d B_1_1 = control_points.row(5);
		Eigen::Vector3d B_1_2 = control_points.row(6);
		Eigen::Vector3d B_1_3 = control_points.row(7);
		Eigen::Vector3d B_2_0 = control_points.row(8);
		Eigen::Vector3d B_2_1 = control_points.row(9);
		Eigen::Vector3d B_2_2 = control_points.row(10);
		Eigen::Vector3d B_2_3 = control_points.row(11);
		Eigen::Vector3d B_3_0 = control_points.row(12);
		Eigen::Vector3d B_3_1 = control_points.row(13);
		Eigen::Vector3d B_3_2 = control_points.row(14);
		Eigen::Vector3d B_3_3 = control_points.row(15);
		
		Eigen::Matrix<double, 3, 2> jac;
		jac.col(0) = 3 * B_0_0 * std::pow(u - 1, 2) * std::pow(v - 1, 3) - 3 * B_1_0 * std::pow(u - 1, 2) * std::pow(v - 1, 3) 
			- 3 * B_2_3 * std::pow(u, 2) * std::pow(v, 3) + 3 * B_3_3 * std::pow(u, 2) * std::pow(v, 3) - 3 * B_0_3 * std::pow(v, 3) * std::pow(u - 1, 2) 
			+ 3 * B_1_3 * std::pow(v, 3) * std::pow(u - 1, 2) + 3 * B_2_0 * std::pow(u, 2) * std::pow(v - 1, 3) - 3 * B_3_0 * std::pow(u, 2) * std::pow(v - 1, 3) 
			+ 3 * B_1_3 * u * std::pow(v, 3) * (2 * u - 2) + 6 * B_2_0 * u * (u - 1) * std::pow(v - 1, 3) - 9 * B_2_1 * std::pow(u, 2) * v * std::pow(v - 1, 2) 
			+ 9 * B_2_2 * std::pow(u, 2) * std::pow(v, 2) * (v - 1) + 9 * B_3_1 * std::pow(u, 2) * v * std::pow(v - 1, 2) - 9 * B_3_2 * std::pow(u, 2) * std::pow(v, 2) * (v - 1) 
			- 3 * B_1_0 * u * (2 * u - 2) * std::pow(v - 1, 3) - 9 * B_0_1 * v * std::pow(u - 1, 2) * std::pow(v - 1, 2) + 9 * B_0_2 * std::pow(v, 2) * std::pow(u - 1, 2) * (v - 1) 
			+ 9 * B_1_1 * v * std::pow(u - 1, 2) * std::pow(v - 1, 2) - 9 * B_1_2 * std::pow(v, 2) * std::pow(u - 1, 2) * (v - 1) - 6 * B_2_3 * u * std::pow(v, 3) * (u - 1) 
			- 18 * B_2_1 * u * v * (u - 1) * std::pow(v - 1, 2) + 18 * B_2_2 * u * std::pow(v, 2) * (u - 1) * (v - 1) + 9 * B_1_1 * u * v * (2 * u - 2) * std::pow(v - 1, 2)
			- 9 * B_1_2 * u * std::pow(v, 2) * (2 * u - 2) * (v - 1);

		jac.col(1) = 3 * B_0_0 * pow(u - 1, 3) * pow(v - 1, 2)
			- 3 * B_0_1 * pow(u - 1, 3) * pow(v - 1, 2)
			- 3 * B_3_2 * pow(u, 3) * pow(v, 2)
			+ 3 * B_3_3 * pow(u, 3) * pow(v, 2)
			+ 3 * B_0_2 * pow(v, 2) * pow(u - 1, 3)
			- 3 * B_0_3 * pow(v, 2) * pow(u - 1, 3)
			- 3 * B_3_0 * pow(u, 3) * pow(v - 1, 2)
			+ 3 * B_3_1 * pow(u, 3) * pow(v - 1, 2)
			- 9 * B_1_2 * u * pow(v, 2) * pow(u - 1, 2)
			+ 9 * B_1_3 * u * pow(v, 2) * pow(u - 1, 2)
			+ 9 * B_2_2 * pow(u, 2) * pow(v, 2) * (u - 1)
			- 9 * B_2_3 * pow(u, 2) * pow(v, 2) * (u - 1)
			+ 3 * B_3_1 * pow(u, 3) * v * (2 * v - 2)
			+ 6 * B_0_2 * v * pow(u - 1, 3) * (v - 1)
			- 9 * B_1_0 * u * pow(u - 1, 2) * pow(v - 1, 2)
			+ 9 * B_1_1 * u * pow(u - 1, 2) * pow(v - 1, 2)
			+ 9 * B_2_0 * pow(u, 2) * (u - 1) * pow(v - 1, 2)
			- 9 * B_2_1 * pow(u, 2) * (u - 1) * pow(v - 1, 2)
			- 3 * B_0_1 * v * (2 * v - 2) * pow(u - 1, 3)
			- 6 * B_3_2 * pow(u, 3) * v * (v - 1)
			- 18 * B_1_2 * u * v * pow(u - 1, 2) * (v - 1)
			+ 18 * B_2_2 * pow(u, 2) * v * (u - 1) * (v - 1)
			+ 9 * B_1_1 * u * v * (2 * v - 2) * pow(u - 1, 2)
			- 9 * B_2_1 * pow(u, 2) * v * (2 * v - 2) * (u - 1);
		return jac;			
		};
	return bezier_jac_func;
}

BezierCurvenet load_bezier_surface(const std::string& filename)
{
	BezierCurvenet bez_curvenet;

	std::ifstream input(filename);

	if (!input.is_open())
	{
		throw std::runtime_error("Cannot bezier surface file");
	}

	int num_surfaces;
	input >> num_surfaces;
	for (int i = 0; i < num_surfaces; i++)
	{
		Eigen::Matrix<double, 16, 3> control_points;
		for (int j = 0; j < 16; j++)
		{
			double x, y, z;
			input >> x >> y >> z;
			control_points.row(j) = Eigen::RowVector3d(x, y, z);
		}
		bez_curvenet.patches.push_back(control_points);
	}
	int boundary_input;
	input >> boundary_input;
	bez_curvenet.boundary.resize(boundary_input, 3);
	for (int i = 0; i < boundary_input; i++)
	{
		double x, y, z;
		input >> x >> y >> z;
		bez_curvenet.boundary.row(i) = Eigen::RowVector3d(x, y, z);
	}
	return bez_curvenet;
}

BezierCurvenet rescale_to_unit(const BezierCurvenet& bezier_cn)
{
	std::vector<Eigen::MatrixXd> clouds = bezier_cn.patches;

	clouds.push_back(bezier_cn.boundary);
	clouds = rescale_to_unit(clouds);
	int num_boundaries = 1;
	BezierCurvenet bcn;
	for (int i = 0; i < bezier_cn.patches.size(); i++)
		bcn.patches.push_back(clouds[i]);
	for (int i = 0; i < num_boundaries; i++)
		bcn.boundary = clouds[i + bezier_cn.patches.size()];
	return bcn;
}