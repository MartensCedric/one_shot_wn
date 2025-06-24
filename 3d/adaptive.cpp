#include "adaptive.h"



double bem_double_integral(const Eigen::Vector2d& uv, const space_curve_t& space_curve, const Eigen::MatrixXd& df_dn, const BoundaryParametrization* boundary_param, const Eigen::Vector3d& q)
{
	const Eigen::Vector3d& p = representation_formula_interior(uv, boundary_param, space_curve, df_dn);

	const Eigen::Vector3d r = p - q;

	const Eigen::MatrixXd& jac = jacobian_F(uv, space_curve, df_dn, boundary_param);
	Eigen::Vector3d dp_du = jac.col(0);
	Eigen::Vector3d dp_dv = jac.col(1);
	Eigen::Vector3d normal = dp_du.cross(dp_dv);
	normal.normalize();

	double r_norm = r.norm();
	double r_norm_3 = r_norm * r_norm * r_norm;
	return r.dot(normal) / r_norm_3;
}

double adaptive_quadrature_winding_number(const Eigen::Vector2d& uv, const std::function<Eigen::Vector3d(Eigen::Vector2d)>& func, const std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)>& jac_func, const Eigen::Vector3d& q)
{
	const Eigen::Vector3d p = func(uv);
	const Eigen::Vector3d r = p - q;

	Eigen::Matrix<double, 3, 2> jac = jac_func(uv);
	Eigen::Vector3d dp_du = jac.col(0);
	Eigen::Vector3d dp_dv = jac.col(1);
	Eigen::Vector3d normal = dp_du.cross(dp_dv);

	double normal_norm = normal.norm();
	if (normal_norm < 1e-10) {
		return 0.0; // Degenerate case, no contribution to the winding number
	}

	double r_norm = r.norm();
	if (r_norm < 1e-10) {
		return 0.0; // Avoid division by zero or near-zero
	}
	double r_norm_3 = r_norm * r_norm * r_norm;

	return r.dot(normal) / r_norm_3;
}

double bezier_triangle_winding_number(const Eigen::Vector2d& uv, const std::function<Eigen::Vector3d(Eigen::Vector2d)>& func, const std::function<Eigen::Matrix<double, 3,2>(Eigen::Vector2d)>& jac_func, const Eigen::Vector3d& q)
{
	const Eigen::Vector3d p = func(uv);

	const Eigen::Vector3d r = p - q;

	Eigen::Matrix<double, 3,2> jac = jac_func(uv);
	Eigen::Vector3d dp_du = jac.col(0);
	Eigen::Vector3d dp_dv = jac.col(1);
	Eigen::Vector3d normal = dp_du.cross(dp_dv);
	//normal.normalize();

	double r_norm = r.norm();
	double r_norm_3 = r_norm * r_norm * r_norm;
	return r.dot(normal) / r_norm_3;
}

std::pair<std::vector<double>, std::chrono::nanoseconds> bezier_triangle_winding_number_timed(const Eigen::MatrixXd& control_points, const Eigen::MatrixXd& query_points, double tolerance)
{
	auto func = create_bezier_func(control_points);
	auto jac_func = create_bezier_jac_func(control_points);
	boost::math::quadrature::gauss_kronrod<double, 15> integrator;
	std::chrono::high_resolution_clock::time_point aq_tic = std::chrono::high_resolution_clock::now();
	std::pair<std::vector<double>, std::chrono::nanoseconds> result;
	result.first = std::vector<double>(query_points.rows(), 0);
	result.second = std::chrono::nanoseconds(0);
	const int max_threads = 19;
#pragma omp parallel for num_threads(max_threads)
	for (int i = 0; i < query_points.rows(); i++)
	{
		auto inner_integrand = [&](double u) { // u + v  = 1
			double err;
			return integrator.integrate([&](double v) {
				return bezier_triangle_winding_number(Eigen::Vector2d(u, v), func, jac_func, query_points.row(i));
				}, 0.0, 1.0 - u, 20, tolerance, &err);
			};
		double error;
		double wn = integrator.integrate(inner_integrand, 0.0, 1.0, 20, tolerance, &error);
		result.first[i] = wn;
	}

	std::chrono::high_resolution_clock::time_point aq_toc = std::chrono::high_resolution_clock::now();
	result.second = (aq_toc - aq_tic) / query_points.rows();

	return result;
}

std::vector<std::pair<double, std::chrono::nanoseconds>> bezier_triangle_wn_against_gt(const Eigen::MatrixXd& control_points, const Eigen::MatrixXd& query_points, const std::vector<double>& tolerances, const std::vector<double>& gt)
{
	std::vector<std::pair<double, std::chrono::nanoseconds>> result;
	for (int i = 0; i < tolerances.size(); i++)
	{
		std::cout << "Tolerance: " << tolerances[i] << std::endl;
		std::pair<std::vector<double>, std::chrono::nanoseconds> wns = bezier_triangle_winding_number_timed(control_points, query_points, tolerances[i]);
		double error = 0.0;
		for (int j = 0; j < gt.size(); j++)
		{
			error += std::abs(gt[j] - wns.first[j]);
		}
		result.emplace_back(error / query_points.rows(), wns.second);
	}
	return result;
}

void run_bezier_triangle_aq_experiment(const Eigen::MatrixXd& query_points)
{
	std::vector<double> tolerances = { 5e-3, 5e-3,
								   5e-4, 1e-4,
								   5e-5, 1e-5,
								   5e-6, 1e-6,
								   5e-7, 1e-7,
								   5e-8, 1e-8,
								   5e-9, 1e-9,};

	Eigen::MatrixXd control_points(10, 3);

	control_points << 0.525074852359172, -0.278903363421520, -0.102120790292900,
		0.267975879627391, -0.272018745807755, 0.180600624945055,
		0.431344779247598, 0.071712035570697, -0.088579682401932,
		-0.153017612368211, -0.194730960250322, -0.034224316835228,
		-0.075251851075666, 0.010472623951161, -0.123097637107601,
		-0.003283903504825, 0.269654133300490, -0.087522050261528,
		-0.528835002641230, -0.352302828703711, -0.058733610651726,
		-0.360043137505598, 0.227047086288067, 0.123799708915016,
		-0.155802131082939, 0.268691296953398, -0.057550640014390,
		-0.011479503767503, 0.513635693551943, 0.079936366358547;

	std::cout << "Running ground truth" << std::endl;
	std::pair<std::vector<double>, std::chrono::nanoseconds> gt_res = bezier_triangle_winding_number_timed(control_points, query_points, 1e-11);
	std::cout << "Ground Truth in: " << gt_res.second.count() / 1000 << " us. " << std::endl;

	std::vector<std::pair<double, std::chrono::nanoseconds>> against_gt = bezier_triangle_wn_against_gt(control_points, query_points, tolerances, gt_res.first);

	std::cout << "tolerances: " << std::endl;
	for (int i = 0; i < tolerances.size(); i++)
	{
		std::cout << tolerances[i] << std::endl;
	}

	std::cout << std::endl;

	std::cout << "error: " << std::endl;
	for (int i = 0; i < against_gt.size(); i++)
	{
		std::cout << against_gt[i].first << std::endl;
	}

	std::cout << std::endl;

	std::cout << "time (us)" << std::endl;
	for (int i = 0; i < against_gt.size(); i++)
	{
		std::cout << against_gt[i].second.count() / 1000 << std::endl;
	}
}

std::vector<double> winding_numbers_adaptive(const std::function<Eigen::Vector3d(Eigen::Vector2d)>& func, const std::function<Eigen::Matrix<double, 3,2>(Eigen::Vector2d)>& jac_func, const Eigen::MatrixXd& query_points, int max_depth, double tolerance)
{
	std::vector<double> winding_numbers(query_points.rows(), 0);
	boost::math::quadrature::gauss_kronrod<double, 15> integrator;

	std::chrono::high_resolution_clock::time_point aq_tic = std::chrono::high_resolution_clock::now();
	
	const int max_threads = 19;
	std::vector<bool> finished_amount(query_points.rows(), false);
#pragma omp parallel for num_threads(max_threads)
	for (int i = 0; i < query_points.rows(); i++)
	{
		
		auto inner_integrand = [&](double u) { 
			double err;
			return integrator.integrate([&](double v) {
				return adaptive_quadrature_winding_number(Eigen::Vector2d(u, v), func, jac_func, query_points.row(i));
				}, 0.0, 1.0, max_depth, tolerance, &err);
			};
		double error;
		winding_numbers[i] = integrator.integrate(inner_integrand, 0.0, 1.0, max_depth, tolerance, &error) / (4.0 * EIGEN_PI);

		finished_amount[i] = true;
		if (i % 6400 == 0)
		{
			int done_count = 0;
			for (int k = 0; k < query_points.rows(); k++)
			{
				if (finished_amount[k])
					done_count++;
			}

			std::cout << (static_cast<double>(done_count) / static_cast<double>(query_points.rows())) * 100.0 << "%" << std::endl;
		}
	}

	std::chrono::high_resolution_clock::time_point aq_toc = std::chrono::high_resolution_clock::now();

	return winding_numbers;
}