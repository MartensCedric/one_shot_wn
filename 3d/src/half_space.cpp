#include "half_space.h"
#include "math_util.h"
#include <Eigen/Dense>

halfspace_intersections_t all_halfspace_intersections_for_ray(gsl_multiroot_fsolver* s, const implicit_func_t& f, const Eigen::Vector3d& point, const Eigen::Vector3d& dir)
{
	halfspace_intersections_t result;
	result.is_valid = false;
	result.ray_dir = dir;
	std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> jac;
	parametric_solver_params solver_params = { f, jac, point, dir };

	double t_min = 0.0;
	double t_max = 10.0;
	double u_min = -3.0;
	double v_min = -3.0;
	double u_max = 3.0;
	double v_max = 3.0;

	int num_starting_ts = 15;
	int num_starting_uvs = 5;

	std::vector<double> t_linspace = linspace(0.0, t_max, num_starting_ts, true);
	std::vector<double> u_linspace = linspace(u_min, u_max, num_starting_uvs, true);
	std::vector<double> v_linspace = linspace(v_min, v_max, num_starting_uvs, true);

	const double root_epsilon = 0.00005;
	const double fval_epsilon = 1e-6;
	const double t_epsilon = 1e-7;
	const size_t max_iterations = 1000;
	const double ray_dot_epsilon = 0.001;


	int solver_error_counter = 0;
	// Use bounded optimization if this is slow.

	result.roots_found.clear();
	result.roots_makes_it_inside.clear();
	
	for (double u_start : u_linspace)
	{
		for (double v_start : v_linspace)
		{
			for (double t_start : t_linspace)
			{
				double x_init[3] = { u_start, v_start, t_start };

				const Eigen::Vector3d uvt(x_init);
				gsl_vector_set(s->x, 0, x_init[0]);
				gsl_vector_set(s->x, 1, x_init[1]);
				gsl_vector_set(s->x, 2, x_init[2]);
				s->function->params = (void*)&solver_params;

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


					if (t < t_min - t_epsilon || t > t_max + t_epsilon)  // t out of range
						continue;

					if (!is_in_vector_epsilon(result.roots_found, t, root_epsilon))
					{
						Eigen::Matrix<double, 3, 2> jacobian = jacobian_fd_implicit(Eigen::Vector2d(u, v), solver_params.func);
						Eigen::Vector3d normal = jacobian.col(0).cross(jacobian.col(1));
						Eigen::Vector3d dir = solver_params.dir;

						normal.normalize();
						dir.normalize();

						double ray_normal_dot = normal.dot(dir);

						if (std::abs(ray_normal_dot) < ray_dot_epsilon)
						{
							result.is_valid = false;
							result.roots_found.clear();
							result.roots_makes_it_inside.clear();
							result.root_uvs.clear();
							return result;
						}


						if (ray_normal_dot < 0)
							result.roots_makes_it_inside.push_back(true);
						else
							result.roots_makes_it_inside.push_back(false);


						result.roots_found.push_back(t);
						result.root_uvs.emplace_back(u, v);
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

	std::vector<double> roots_unsorted(result.roots_found.begin(), result.roots_found.end());
	std::vector<int> roots_sign_unsorted(result.roots_makes_it_inside.begin(), result.roots_makes_it_inside.end());
	std::vector<Eigen::Vector2d> roots_uv_unsorted(result.root_uvs.begin(), result.root_uvs.end());
	std::vector<size_t> sorted_indices = sort_indexes<double>(result.roots_found);

	for (int i = 0; i < sorted_indices.size(); i++)
	{
		result.roots_found[i] = roots_unsorted[sorted_indices[i]];
		result.roots_makes_it_inside[i] = roots_sign_unsorted[sorted_indices[i]];
		result.root_uvs[i] = roots_uv_unsorted[sorted_indices[i]];
	}
	result.is_valid = true;
	return result;
}

halfspace_intersections_t all_halfspace_intersections(gsl_multiroot_fsolver* s, const implicit_func_t& f, const Eigen::Vector3d& point)
{
	int max_attempts = 10;

	Eigen::Vector2d uv_on_surface = Eigen::Vector2d(0.5, 0.5);
	
	for (int i = 0; i < max_attempts; i++)
	{
		Eigen::Vector3d point_on_surface = f(uv_on_surface);
		Eigen::Vector3d ray_dir = (point_on_surface - point).normalized();

		halfspace_intersections_t result = all_halfspace_intersections_for_ray(s, f, point, ray_dir);
		if (result.is_valid)
			return result;

		uv_on_surface += Eigen::Vector2d(0.04, -0.02);
	}

	std::cout << "Max attempts reached for finding a non-tangent halfspace intersection" << std::endl;

	halfspace_intersections_t res;
	res.is_valid = false;
	return res;
}

std::pair<halfspace_intersections_t, halfspace_intersections_t> all_halfspace_intersections_common_ray(gsl_multiroot_fsolver* s, const implicit_func_t& f1, const implicit_func_t& f2, const Eigen::Vector3d& point)
{
	std::pair<halfspace_intersections_t, halfspace_intersections_t> result;
	result.first.is_valid = false;
	result.second.is_valid = false;
	
	int max_attempts = 10;

	Eigen::Vector2d uv_on_surface = Eigen::Vector2d(0.37314, 0.643243);

	for (int i = 0; i < max_attempts; i++)
	{
		Eigen::Vector3d point_on_surface = f1(uv_on_surface);
		Eigen::Vector3d ray_dir = (point_on_surface - point).normalized();

		halfspace_intersections_t result1 = all_halfspace_intersections_for_ray(s, f1, point, ray_dir);
		halfspace_intersections_t result2 = all_halfspace_intersections_for_ray(s, f2, point, ray_dir);
		if (result1.is_valid && result2.is_valid)
			return { result1, result2 };

		uv_on_surface += Eigen::Vector2d(0.04, -0.02);
	}

	std::cout << "Max attempts reached for finding a non-tangent halfspace intersection" << std::endl;
	return result;
}

bool is_inside_half_space_implicit(gsl_multiroot_fsolver* s, const implicit_func_t& f, const Eigen::Vector3d& point)
{
	halfspace_intersections_t result = all_halfspace_intersections(s, f, point);
	if (result.roots_makes_it_inside.empty())
	{
		std::cout << "WARNING: No roots found despite shooting a ray intersecting the surface! " << std::endl;
		return false;
	}
		
	//ASSERT_RELEASE(!result.roots_makes_it_inside.empty(), "No roots found despite shooting a ray intersecting the surface!");
	return !result.roots_makes_it_inside[0];
}

int chi_from_halfspace_intersections(const std::pair<halfspace_intersections_t, halfspace_intersections_t>& intersections, const std::pair<is_inside_f, is_inside_f>& is_inside_funcs, const std::pair<bool, bool>& inside_hs)
{
	int chi = 0;
	std::vector<halfspace_crossing_t> crossings;
	bool inside_p1 = inside_hs.first;
	bool inside_p2 = inside_hs.second;

	// can be done in O(N)
	for (int i = 0; i < intersections.first.roots_found.size(); i++)
	{
		halfspace_crossing_t crossing;
		crossing.t = intersections.first.roots_found[i];
		crossing.dir = intersections.first.roots_makes_it_inside[i] ? -1 : 1;
		crossing.uv = intersections.first.root_uvs[i];
		crossing.patch_num = 0;
		crossings.push_back(crossing);
	}

	for (int i = 0; i < intersections.second.roots_found.size(); i++)
	{
		halfspace_crossing_t crossing;
		crossing.t = intersections.second.roots_found[i];
		crossing.dir = intersections.second.roots_makes_it_inside[i] ? -1 : 1;
		crossing.uv = intersections.second.root_uvs[i];
		crossing.patch_num = 1;
		crossings.push_back(crossing);
	}
	std::sort(crossings.begin(), crossings.end(), [](const halfspace_crossing_t& a, const halfspace_crossing_t& b) { return a.t < b.t; });


	for (int i = 0; i < crossings.size(); i++)
	{
		if (inside_p1 && inside_p2)
		{
			//assert(crossings[i].dir == -1);
			if (crossings[i].patch_num == 0)
			{
				inside_p1 = false;

				if (is_inside_funcs.first(crossings[i].uv))
					chi++;
			}
			else if (crossings[i].patch_num == 1)
			{
				inside_p2 = false;
				if (is_inside_funcs.second(crossings[i].uv))
					chi++;
			}
		}
		else if (!inside_p1 && !inside_p2)
		{
			//assert(crossings[i].dir == 1);
			if (crossings[i].patch_num == 0)
			{
				inside_p1 = true;

				if (is_inside_funcs.first(crossings[i].uv))
					chi--;
			}
			else if (crossings[i].patch_num == 1)
			{
				inside_p2 = true;
				if (is_inside_funcs.second(crossings[i].uv))
					chi++;
			}
		}
		else if (!inside_p1 && inside_p2)
		{
			if (crossings[i].patch_num == 0)
			{
				inside_p1 = true;
				if (is_inside_funcs.first(crossings[i].uv))
					chi--;
			}
			else if (crossings[i].patch_num == 1)
			{
				inside_p2 = false;
				if (is_inside_funcs.second(crossings[i].uv))
					chi++;
			}
		}
		else if (inside_p1 && !inside_p2)
		{
			if (crossings[i].patch_num == 0)
			{
				inside_p1 = false;
				if (is_inside_funcs.first(crossings[i].uv))
					chi++;
			}
			else if (crossings[i].patch_num == 1)
			{
				inside_p2 = true;
				if (is_inside_funcs.second(crossings[i].uv))
					chi--;
			}
		}
	}


	return chi;
}