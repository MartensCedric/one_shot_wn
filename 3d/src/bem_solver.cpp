#include "bem_solver.h"
#include <gsl/gsl_multiroots.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

int f_func(const gsl_vector* x, void* p, gsl_vector* f)
 {
	 struct bem_solver_params* params = (struct bem_solver_params*)p;
	 const space_curve_t& patch = params->patch;
	 
	 Eigen::Vector3d point = params->point;
	 Eigen::Vector3d dir = params->dir;

	 const double u = gsl_vector_get(x, 0);
	 const double v = gsl_vector_get(x, 1);
	 const double t = gsl_vector_get(x, 2);

	 const Eigen::MatrixXd& df_dn = params->df_dn;
	 Eigen::Vector3d uvt = Eigen::Vector3d(u, v, t);


	 Eigen::Vector3d output = evaluate_F(uvt, params->boundary_param, patch, df_dn, point, dir);

	 gsl_vector_set(f, 0, output(0));
	 gsl_vector_set(f, 1, output(1));
	 gsl_vector_set(f, 2, output(2));

	 return GSL_SUCCESS;
};

void print_solver_state(size_t iter, gsl_multiroot_fsolver* s)
{
	gsl_vector* dx_vec = gsl_multiroot_fsolver_dx(s);
	printf("Iteration = %3u u = % .3f, v = % .3f, t = % .3f "
		"f(x) = % .3e % .3e % .3e dx = % .3e % .3e % .3e\n",
		iter,
		gsl_vector_get(s->x, 0),
		gsl_vector_get(s->x, 1),
		gsl_vector_get(s->x, 2),
		gsl_vector_get(s->f, 0),
		gsl_vector_get(s->f, 1),
		gsl_vector_get(s->f, 2),
		gsl_vector_get(dx_vec, 0),
		gsl_vector_get(dx_vec, 1),
		gsl_vector_get(dx_vec, 2));
}

struct all_intersections_with_normals_result all_ray_intersections(gsl_multiroot_fsolver* s, const struct bem_solver_params& params, double t_min, double t_max, double max_ray_length)
{
	all_intersections_with_normals_result result;
	result.valid_ray = true;
	const double root_epsilon = 0.00005;
	const double fval_epsilon = 1e-6;
	const double t_epsilon = 1e-7;
	const size_t max_iterations = 1000;
	const double ray_dot_epsilon = 0.1;

	constexpr double uv_epsilon = 0.05;
	constexpr double u_min = uv_epsilon;
	constexpr double u_max = 1.0 - uv_epsilon;
	constexpr double v_min = uv_epsilon;
	constexpr double v_max = 1.0 - uv_epsilon;

	std::vector<double> roots_found;
	const int num_starting_ts = std::max<int>(2.0, ((t_max - t_min)/(max_ray_length)) * 15.0);
	const int num_starting_uvs = 3;
	std::vector<double> t_linspace = linspace(t_min, t_max, num_starting_ts, true);
	std::vector<double> u_linspace = linspace(u_min, u_max, num_starting_uvs, true);
	std::vector<double> v_linspace = linspace(v_min, v_max, num_starting_uvs, true);

	int solver_error_counter = 0;
	for (double t0 : t_linspace)
	{
		for (double u0 : u_linspace)
		{
			for (double v0 : v_linspace)
			{
				double x_init[3] = { u0, v0, t0 };
				const Eigen::Vector3d uvt(x_init);
				gsl_vector_set(s->x, 0, x_init[0]);
				gsl_vector_set(s->x, 1, x_init[1]);
				gsl_vector_set(s->x, 2, x_init[2]);
				s->function->params = (void*)&params;
				gsl_multiroot_fsolver_set(s, s->function, s->x); // this line is probably not needed here

				int status;
				size_t iter_num = 0;

				do
				{
					iter_num++;
					status = gsl_multiroot_fsolver_iterate(s);
					//print_solver_state(iter_num, s);
					if (status)
						break;

					status = gsl_multiroot_test_residual(s->f, fval_epsilon);
				} while (status == GSL_CONTINUE && iter_num < max_iterations);

				if (status == GSL_ENOPROG || status == GSL_ENOPROGJ)
				{
					solver_error_counter++;
				}
				else if (status == GSL_SUCCESS)
				{
					double u = gsl_vector_get(s->x, 0);
					double v = gsl_vector_get(s->x, 1);
					double t = gsl_vector_get(s->x, 2);


					if (t < t_min - t_epsilon || t > t_max + t_epsilon) // t out of range
						continue;


					if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) // outside uv space
						continue;


					if (!is_in_vector_epsilon(roots_found, t, root_epsilon))
					{
						Eigen::Matrix<double, 3, 2> jacobian = jacobian_F(Eigen::Vector2d(u, v), params.patch, params.df_dn, params.boundary_param);
						Eigen::Vector3d normal = jacobian.col(0).cross(jacobian.col(1));
						Eigen::Vector3d dir = params.dir;

						normal.normalize();
						dir.normalize();

						double ray_normal_dot = normal.dot(dir);

						if ((std::abs(ray_normal_dot) < ray_dot_epsilon || u < u_min || u > u_max || v < v_min || v > v_max))
						{
							//std::cout << "ray_normal_dot: " << ray_normal_dot << ", u:" << u << ", v:" << v << std::endl;
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
				else if (status != GSL_FAILURE && status != GSL_CONTINUE)
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

chi_result find_chi(gsl_multiroot_fsolver* s, const bem_solver_params& params, double t_min, double t_max, double max_ray_length, bool abort_bad_ray)
{
	chi_result result;
	result.chi = 0;
	result.success = false;

	// can be moved somewhere else
	const double root_epsilon = 0.00005;
	const double fval_epsilon = 1e-6;
	const double t_epsilon = 1e-7;
	const size_t max_iterations = 1000;
	const double ray_dot_epsilon = 0.075;

	constexpr double uv_epsilon = 0.01;
	constexpr double u_min = uv_epsilon;
	constexpr double u_max = 1.0 - uv_epsilon;
	constexpr double v_min = uv_epsilon;
	constexpr double v_max = 1.0 - uv_epsilon;

	std::vector<double> roots_found;
	const int num_starting_ts = std::max<int>(2.0, ((t_max - t_min)/ max_ray_length) * 15.0);
	const int num_starting_uvs = 3;
	std::vector<double> t_linspace = linspace(t_min, t_max, num_starting_ts, true);
	std::vector<double> u_linspace = linspace(u_min, u_max, num_starting_uvs, true);
	std::vector<double> v_linspace = linspace(v_min, v_max, num_starting_uvs, true);

	int solver_error_counter = 0;
	for (double t0 : t_linspace)
	{
		for (double u0 : u_linspace)
		{
			for (double v0 : v_linspace)
			{
				double x_init[3] = { u0, v0, t0};
				const Eigen::Vector3d uvt(x_init);
				gsl_vector_set(s->x, 0, x_init[0]);
				gsl_vector_set(s->x, 1, x_init[1]);
				gsl_vector_set(s->x, 2, x_init[2]);
				s->function->params = (void*)&params;
				gsl_multiroot_fsolver_set(s, s->function, s->x); // this line is probably not needed here

				int status;
				size_t iter_num = 0;

				do
				{
					iter_num++;
					status = gsl_multiroot_fsolver_iterate(s);
					//print_solver_state(iter_num, s);
					if (status)
						break;

					status = gsl_multiroot_test_residual(s->f, fval_epsilon);
				} while (status == GSL_CONTINUE && iter_num < max_iterations);

				if (status == GSL_ENOPROG || status == GSL_ENOPROGJ)
				{
					solver_error_counter++;
				}
				else if (status == GSL_SUCCESS)
				{
					double u = gsl_vector_get(s->x, 0);
					double v = gsl_vector_get(s->x, 1);
					double t = gsl_vector_get(s->x, 2);


					if (t < t_min - t_epsilon || t > t_max + t_epsilon) // t out of range
						continue;


					if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) // outside uv space
						continue;


					if (!is_in_vector_epsilon(roots_found, t, root_epsilon))
					{
						Eigen::Matrix<double, 3, 2> jacobian = jacobian_F(Eigen::Vector2d(u, v), params.patch, params.df_dn, params.boundary_param);
						Eigen::Vector3d normal = jacobian.col(0).cross(jacobian.col(1));
						Eigen::Vector3d dir = params.dir;

						normal.normalize();
						dir.normalize();

						double ray_normal_dot = normal.dot(dir);

						if (abort_bad_ray && (std::abs(ray_normal_dot) < ray_dot_epsilon || u < u_min || u > u_max || v < v_min || v > v_max))
						{
							//std::cout << "ray_normal_dot: " << ray_normal_dot << ", u:" << u << ", v:" << v << std::endl;
							return result; // discard ray
						}

						if (ray_normal_dot < 0)
							result.chi--;
						else
							result.chi++;


						roots_found.push_back(t);
					}
				}
				else if (status != GSL_FAILURE && status != GSL_CONTINUE)
				{
					std::cout << "status: " << status << std::endl;
					throw std::runtime_error("Different solver result");
				}
			}
		}
	}
	result.success = true;
	return result;
}