#include "parametric.h"
#include "uv_util.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

int f_func_parametric(const gsl_vector* x, void* p, gsl_vector* f)
{
	struct parametric_solver_params* params = (struct parametric_solver_params*)p;

	Eigen::Vector3d point = params->point;
	Eigen::Vector3d dir = params->dir;

	implicit_func_t func = params->func;

	const double u = gsl_vector_get(x, 0);
	const double v = gsl_vector_get(x, 1);
	const double t = gsl_vector_get(x, 2);

	Eigen::Vector3d output =  func(Eigen::Vector2d(u, v))  - (point + t * dir);

	gsl_vector_set(f, 0, output(0));
	gsl_vector_set(f, 1, output(1));
	gsl_vector_set(f, 2, output(2));

	return GSL_SUCCESS;
}

Eigen::Matrix<double, 3, 2> jacobian_fd_implicit(Eigen::Vector2d uv, implicit_func_t func)
{
	double eps = 0.0000001;
	Eigen::Vector3d df_du = (func(uv + Eigen::Vector2d(eps, 0)) - func(uv + Eigen::Vector2d(-eps, 0))) / (2.0 * eps);
	Eigen::Vector3d df_dv = (func(uv + Eigen::Vector2d(0, eps)) - func(uv + Eigen::Vector2d(0, -eps))) / (2.0 * eps);

	Eigen::Matrix<double, 3, 2> output;
	output.col(0) = df_du;
	output.col(1) = df_dv;
	return output;
}

std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> fd_jacobian(implicit_func_t func, double eps)
{
	return [=](Eigen::Vector2d uv)
	{
		Eigen::Vector3d du = (func(uv + Eigen::Vector2d(eps, 0)) - func(uv + Eigen::Vector2d(-eps, 0))) / (2.0 * eps);
		Eigen::Vector3d dv = (func(uv + Eigen::Vector2d(0, eps)) - func(uv + Eigen::Vector2d(0, -eps))) / (2.0 * eps);

		Eigen::Matrix<double, 3, 2> jacobian;
		jacobian.col(0) = du;
		jacobian.col(1) = dv;
		return jacobian;
	};
}

struct all_intersections_with_normals_result find_all_intersections_parametric_gears(gsl_multiroot_fsolver* s, const struct parametric_solver_params& params, double t_min, double t_max)
{
	all_intersections_with_normals_result result;
	result.valid_ray = true;
	const double root_epsilon = 1e-5;
	const double fval_epsilon = 1e-9;
	const double t_epsilon = 1e-7;
	const size_t max_iterations = 1000;
	const double ray_dot_epsilon = 0.0;

	constexpr double uv_epsilon = 0.00;
	constexpr double u_min = uv_epsilon;
	constexpr double u_max = 1.0 - uv_epsilon;
	constexpr double v_min = uv_epsilon;
	constexpr double v_max = 1.0 - uv_epsilon;

	std::vector<double> roots_found;
	const int num_starting_ts = std::max<int>(2, (t_max - t_min) * 60);
	const int num_starting_uvs = 500;
	std::vector<double> t_linspace = linspace(t_min, t_max, num_starting_ts, true);
	std::vector<double> u_linspace = linspace(u_min, u_max, num_starting_uvs, true);
	std::vector<double> v_linspace = { 0.5 };

	for (double t0 : t_linspace)
	{
		for (double u0 : u_linspace)
		{
			for (double v0 : v_linspace)
			{
				double x_init[3] = { u0, v0, t0 };
				gsl_vector_set(s->x, 0, x_init[0]);
				gsl_vector_set(s->x, 1, x_init[1]);
				gsl_vector_set(s->x, 2, x_init[2]);
				s->function->params = (void*)&params;
				gsl_multiroot_fsolver_set(s, s->function, s->x);

				int status;
				size_t iter_num = 0;

				do
				{
					iter_num++;
					status = gsl_multiroot_fsolver_iterate(s);
					if (status)
						break;

					status = gsl_multiroot_test_residual(s->f, fval_epsilon);
				} while (status == GSL_CONTINUE && iter_num < max_iterations);

				if (status == GSL_SUCCESS)
				{
					double u = gsl_vector_get(s->x, 0);
					double v = gsl_vector_get(s->x, 1);
					double t = gsl_vector_get(s->x, 2);

					if (t < t_min - t_epsilon || t > t_max + t_epsilon)
						continue;
					if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0)
						continue;

					if (!is_in_vector_epsilon(roots_found, t, root_epsilon))
					{
						Eigen::Matrix<double, 3, 2> jacobian = jacobian_fd_implicit(Eigen::Vector2d(u, v), params.func);
						Eigen::Vector3d normal = jacobian.col(0).cross(jacobian.col(1));
						Eigen::Vector3d dir = params.dir;

						normal.normalize();
						dir.normalize();

						double ray_normal_dot = normal.dot(dir);

						if ((std::abs(ray_normal_dot) < ray_dot_epsilon || u < u_min || u > u_max || v < v_min || v > v_max))
						{
							result.valid_ray = false;
							result.all_intersections.clear();
							return result;
						}
						else
						{
							if (ray_normal_dot < 0)
								result.all_intersections.emplace_back(t, -1);
							else
								result.all_intersections.emplace_back(t, 1);

							roots_found.push_back(t);
						}
					}
				}
				else if (status != GSL_FAILURE && status != GSL_CONTINUE
					&& status != GSL_ENOPROG && status != GSL_ENOPROGJ)
				{
					std::cout << "status: " << status << std::endl;
					throw std::runtime_error("Different solver result");
				}
			}
		}
	}

	std::sort(result.all_intersections.begin(), result.all_intersections.end(), [=](const std::pair<double, int>& p1, const std::pair<double, int>& p2) { return p1.first < p2.first; });
	return result;
}


struct all_intersections_with_normals_result find_all_intersections_parametric(gsl_multiroot_fsolver* s, const struct parametric_solver_params& params, double t_min, double t_max)
{
	all_intersections_with_normals_result result;
	result.valid_ray = true;
	const double root_epsilon = 1e-5;
	const double fval_epsilon = 1e-9;
	const double t_epsilon = 1e-7;
	const size_t max_iterations = 1000;
	const double ray_dot_epsilon = 0.0;

	constexpr double uv_epsilon = 0.0;
	constexpr double u_min = uv_epsilon;
	constexpr double u_max = 1.0 - uv_epsilon;
	constexpr double v_min = uv_epsilon;
	constexpr double v_max = 1.0 - uv_epsilon;

	std::vector<struct root_position> roots_found;
	const int num_starting_ts = std::max<int>(2, (t_max - t_min) * 30);

	std::vector<double> t_linspace = linspace(t_min, t_max, num_starting_ts, true);
	std::vector<double> u_linspace = linspace(u_min, u_max, 20, true);
	std::vector<double> v_linspace = linspace(v_min, v_max, 10, true);

	for (double t0 : t_linspace)
	{
		for (double u0 : u_linspace)
		{
			for (double v0 : v_linspace)
			{
				if (!params.is_in_parametric_domain(Eigen::Vector2d(u0, v0)))
					continue;
				double x_init[3] = { u0, v0, t0 };
				gsl_vector_set(s->x, 0, x_init[0]);
				gsl_vector_set(s->x, 1, x_init[1]);
				gsl_vector_set(s->x, 2, x_init[2]);
				s->function->params = (void*)&params;
				gsl_multiroot_fsolver_set(s, s->function, s->x);

				int status;
				size_t iter_num = 0;

				do
				{
					iter_num++;
					status = gsl_multiroot_fsolver_iterate(s);
					if (status)
						break;

					status = gsl_multiroot_test_residual(s->f, fval_epsilon);
				} while (status == GSL_CONTINUE && iter_num < max_iterations);

				if (status == GSL_SUCCESS)
				{
					double u = gsl_vector_get(s->x, 0);
					double v = gsl_vector_get(s->x, 1);
					double t = gsl_vector_get(s->x, 2);

					if (t < t_min - t_epsilon || t > t_max + t_epsilon)
						continue;
					if (!params.is_in_parametric_domain(Eigen::Vector2d(u, v)))
						continue;

					struct root_position root_pos {u, v, t};

					if (!is_in_vector_epsilon(roots_found, root_pos, root_epsilon, 1e-3))
					{
						Eigen::Matrix<double, 3, 2> jacobian = params.jac(Eigen::Vector2d(u, v));
						Eigen::Vector3d normal = jacobian.col(0).cross(jacobian.col(1));
						Eigen::Vector3d dir = params.dir;

						normal.normalize();
						dir.normalize();

						double ray_normal_dot = normal.dot(dir);

						if ((std::abs(ray_normal_dot) < ray_dot_epsilon || u < u_min || u > u_max || v < v_min || v > v_max))
						{
							result.valid_ray = false;
							result.all_intersections.clear();
							return result;
						}
						else
						{
							if (ray_normal_dot < 0)
								result.all_intersections.emplace_back(t, -1);
							else
								result.all_intersections.emplace_back(t, 1);

							roots_found.push_back(root_pos);
						}
					}
				}
				else if (status != GSL_FAILURE && status != GSL_CONTINUE
					&& status != GSL_ENOPROG && status != GSL_ENOPROGJ)
				{
					std::cout << "status: " << status << std::endl;
					throw std::runtime_error("Different solver result");
				}
			}
		}
	}

	std::sort(result.all_intersections.begin(), result.all_intersections.end(), [=](const std::pair<double, int>& p1, const std::pair<double, int>& p2) { return p1.first < p2.first; });
	return result;
}
