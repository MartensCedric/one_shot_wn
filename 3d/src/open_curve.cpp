#include "open_curve.h"
#include <iostream>
#include <random>

int f_func_parametric_uv(const gsl_vector* x, void* p, gsl_vector* f)
{
	uv_solver_params* params = (uv_solver_params*)p;
	Eigen::Vector2d safe_point = params->safe_point;
	Eigen::Vector2d query_point = params->query_point;
	implicit_func_t surface_f = params->f;
	implicit_func_t surface_g = params->g;

	const double u_g = gsl_vector_get(x, 0);
	const double v_g = gsl_vector_get(x, 1);
	const double s = gsl_vector_get(x, 2);
	Eigen::Vector2d point_in_f = safe_point + s * (query_point - safe_point);
	Eigen::Vector3d eval_f = surface_f(point_in_f);
	Eigen::Vector3d eval_g = surface_g(Eigen::Vector2d(u_g, v_g));
	Eigen::Vector3d output = eval_f - eval_g;
	
	gsl_vector_set(f, 0, output(0));
	gsl_vector_set(f, 1, output(1));
	gsl_vector_set(f, 2, output(2));
	return GSL_SUCCESS;
};


double wn_2d_open_square(Eigen::Vector2d uv, double closeness)
{
	double wn = 0;
	uv(0) += 0.5;
	uv(1) += 0.5;

	/// 0,0 to 1,0
	if (closeness >= 0.25)
	{
		Eigen::Vector2d p = -uv;
		Eigen::Vector2d q = Eigen::Vector2d(1, 0) - uv;
		wn += signed_angle(p, q) / (2.0 * M_PI);
	}
	else
	{
		Eigen::Vector2d p = -uv;
		Eigen::Vector2d q = Eigen::Vector2d(closeness, 0) - uv;
		wn += signed_angle(p, q) / (2.0 * M_PI);
		return wn;
	}

	closeness -= 0.25;
	if (closeness >= 0.25)
	{
		Eigen::Vector2d p = Eigen::Vector2d(1,0) -uv;
		Eigen::Vector2d q = Eigen::Vector2d(1, 1) - uv;
		wn += signed_angle(p, q) / (2.0 * M_PI);
	}
	else
	{
		Eigen::Vector2d p = Eigen::Vector2d(1, 0) - uv;
		Eigen::Vector2d q = Eigen::Vector2d(1, closeness) - uv;
		wn += signed_angle(p, q) / (2.0 * M_PI);
		return wn;
	}

	closeness -= 0.25;
	if (closeness >= 0.25)
	{
		Eigen::Vector2d p = Eigen::Vector2d(1, 1) - uv;
		Eigen::Vector2d q = Eigen::Vector2d(0, 1) - uv;
		wn += signed_angle(p, q) / (2.0 * M_PI);
	}
	else
	{
		Eigen::Vector2d p = Eigen::Vector2d(1, 1) - uv;
		Eigen::Vector2d q = Eigen::Vector2d(1- closeness, 1) - uv;
		wn += signed_angle(p, q) / (2.0 * M_PI);
		return wn;
	}


	closeness -= 0.25;
	if (closeness >= 0.25)
	{
		Eigen::Vector2d p = Eigen::Vector2d(0, 1) - uv;
		Eigen::Vector2d q = Eigen::Vector2d(0, 0) - uv;
		wn += signed_angle(p, q) / (2.0 * M_PI);
	}
	else
	{
		Eigen::Vector2d p = Eigen::Vector2d(0, 1) - uv;
		Eigen::Vector2d q = Eigen::Vector2d(0, 1 - closeness) - uv;
		wn += signed_angle(p, q) / (2.0 * M_PI);
		return wn;
	}
	return wn;
}


wn_result find_chi_open(gsl_multiroot_fsolver* s, const bem_solver_params& params, double t_min, double t_max, double closeness)
{
	wn_result result;
	result.wn = 0;
	result.success = false;

	// can be moved somewhere else
	const double root_epsilon = 0.00005;
	const double fval_epsilon = 1e-6;
	const double t_epsilon = 1e-7;
	const size_t max_iterations = 1000;
	const double ray_dot_epsilon = 0.15;

	constexpr double uv_epsilon = 0.02;
	constexpr double u_min = uv_epsilon;
	constexpr double u_max = 1.0 - uv_epsilon;
	constexpr double v_min = uv_epsilon;
	constexpr double v_max = 1.0 - uv_epsilon;

	std::vector<double> roots_found;
	const int num_starting_ts = std::max<int>(2, (t_max - t_min) * 30);
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


					if (!is_in_vector_epsilon(roots_found, t, root_epsilon))
					{
						Eigen::Matrix<double, 3, 2> jacobian = jacobian_F(Eigen::Vector2d(u, v), params.patch, params.df_dn, params.boundary_param);
						Eigen::Vector3d normal = jacobian.col(0).cross(jacobian.col(1));
						Eigen::Vector3d dir = params.dir;

						normal.normalize();
						dir.normalize();

						double ray_normal_dot = normal.dot(dir);

						double sign;
						if (ray_normal_dot < 0)
							sign = -1;
						else
							sign = 1;

						// 
						if (params.boundary_param->is_inside_boundary(Eigen::Vector2d(u, v)))
						{
							result.wn += sign * std::max<double>(0, wn_2d_open_square(Eigen::Vector2d(u, v), closeness));
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
	result.success = true;
	return result;
}

std::vector<std::vector<double>> find_all_chis_open(const curve_net_sampler& sampler, const precomputed_curve_data& precompute, const ray_integral_t& rays_int)
{
	const std::vector<double>& closenesses = sampler.insideness;
	std::cout << "Setting up solvers" << std::endl;
	std::default_random_engine rand_engine_template{ 0xced };
	const gsl_multiroot_fsolver_type* T;

	T = gsl_multiroot_fsolver_hybrids;
	std::cout << "Done setting up solvers" << std::endl;

	std::vector<std::vector<double>> patchwise_chis;
	for (int i = 0; i < sampler.patches.size(); i++)
	{
		patchwise_chis.push_back(std::vector<double>(sampler.sample_points.size(), 0));
	}
	constexpr int max_threads = 19;
	omp_set_num_threads(max_threads);

	gsl_multiroot_fsolver* s[max_threads];
	gsl_multiroot_function F[max_threads];
	for (int i = 0; i < max_threads; i++)
	{
		F[i].f = &f_func;
		F[i].n = 3;
		s[i] = gsl_multiroot_fsolver_alloc(T, F[i].n);
		s[i]->x = gsl_vector_alloc(F[i].n);
		s[i]->function = &F[i];
	}

	std::vector<bool> is_done(sampler.sample_points.size(), false);

#pragma omp parallel for num_threads(max_threads)
	for (int i = 0; i < sampler.sample_points.size(); i++)
	{
		if (i % 10 == 0)
			std::cout << i << std::endl;
		Eigen::Vector3d point = sampler.sample_points[i];


		for(int k = 0; k < rays_int.rays.rows(); k++)
		{
			Eigen::Vector3d dir = rays_int.rays.row(k);
			for (int j = 0; j < sampler.patches.size(); j++)
			{
				const space_curve_t& space_curve = precompute.full_patches[j];
				const Eigen::MatrixXd& df_dn = precompute.df_dns[j];
				box bounding_box = precompute.bounding_boxes[j];
				double closeness = closenesses[j];

				ray_box_intersection_result rb_result = ray_box_intersection(point, dir, bounding_box);

				if (!rb_result.intersects)
				{
					continue;
				}

				struct bem_solver_params params = { precompute.int_params[j], space_curve, df_dn, point, dir };
				int thread_id = omp_get_thread_num();
				F[thread_id].params = &params;

				wn_result result = find_chi_open(s[thread_id], params, std::max<double>(0.0, rb_result.t_min), rb_result.t_max * 2.0, closeness);
				//wn_result result = find_chi_open(s[thread_id], params, 0.0, 1.5, closeness);// std::max<double>(0.0, rb_result.t_min), rb_result.t_max, closeness);
	
				if (!result.success)
				{
					break;
				}
				patchwise_chis[j][i] += result.wn * rays_int.areas(k);
			}
		}

		is_done[i] = true;

		if (i % 10 == 0)
		{
			int num_done = 0;
			for (int j = 0; j < sampler.sample_points.size(); j++)
			{
				if (is_done[j])
					num_done++;
			}

			double done_percent = 100.0 * static_cast<double>(num_done) / static_cast<double>(sampler.sample_points.size());
			std::cout << done_percent << "%" << std::endl;
		}
	}

	//gsl_multiroot_fsolver_free(s);
	//gsl_vector_free(x); <- read access violation (?)

	std::cout << "done" << std::endl;
	return patchwise_chis;
}

winding_number_results winding_number_with_gaps(const std::vector<patch_t>& patches, const std::vector<std::pair<int, int>>& connected_patches, const Eigen::MatrixXd& query_points, const std::vector<open_boundary>& b_params)
{
	winding_number_results wn_res;
	
	wn_res.gwn.resize(query_points.rows(), 0.0);
	// todo: don't assume that there's a single gap per patch
	std::vector<double> closeness(patches.size(), 0);

	for (int i = 0; i < patches.size(); i++)
	{
		double gap_distance = (patches[i].curve.row(0) - patches[i].curve.row(patches[i].curve.rows() - 1)).norm();
		closeness[i] = gap_distance / (gap_distance + curve_length(patches[i].curve));
	}

	const gsl_multiroot_fsolver_type* T = gsl_multiroot_fsolver_hybrids;

	constexpr int max_threads = 18;
	omp_set_num_threads(max_threads);

	gsl_multiroot_fsolver* s[max_threads];
	gsl_multiroot_fsolver* param_s[max_threads];
	gsl_multiroot_function F[max_threads];
	gsl_multiroot_function param_F[max_threads];
	for (int i = 0; i < max_threads; i++)
	{
		F[i].f = &f_func_parametric;
		F[i].n = 3;
		s[i] = gsl_multiroot_fsolver_alloc(T, F[i].n);
		s[i]->x = gsl_vector_alloc(F[i].n);
		s[i]->function = &F[i];

		param_F[i].f = &f_func_parametric_uv;
		param_F[i].n = 3;
		param_s[i] = gsl_multiroot_fsolver_alloc(T, param_F[i].n);
		param_s[i]->x = gsl_vector_alloc(param_F[i].n);
		param_s[i]->function = &param_F[i];
	}

	// to be replaced with biharmonic BEM stuff
	implicit_func_t patch1_f = implicit_test_func1;
	implicit_func_t patch2_f = implicit_test_func2;

	std::vector<bool> query_points_done(query_points.rows(), false);

//#pragma omp parallel for num_threads(max_threads)
	for (int i = 0; i < query_points.rows(); i++)
	//for (int i = 80 * 33 + 48; i < query_points.rows(); i++)
	{
		Eigen::Vector3d query_point = query_points.row(i);
		for (int j = 0; j < connected_patches.size(); j++)
		{
			int patch1_id = connected_patches[j].first;
			int patch2_id = connected_patches[j].second;

			// doesnt need to be done at each iteration of this loop
			int patch1_rows = patches[patch1_id].curve.rows();
			int patch2_rows = patches[patch2_id].curve.rows();
			Eigen::MatrixXd combined_patches(patch1_rows + patch2_rows, 3);
			combined_patches.block(0, 0, patch1_rows, 3) = patches[patch1_id].curve;
			combined_patches.block(patch1_rows, 0, patch2_rows, 3) = patches[patch2_id].curve;

			combined_patches = keep_unique_points(combined_patches);

			int thread_id = omp_get_thread_num();

			Eigen::Vector3d ray = Eigen::Vector3d(0.0, 0.0, 1.0);

			std::default_random_engine gen(0xced);
			std::uniform_real_distribution<double> dis(-1.0, 1.0);

			shared_open_patch_rootfinding_settings config = {
				s[thread_id],
				param_s[thread_id],
				implicit_test_func1,
				implicit_test_func2,
				query_point, 
				ray,
				b_params[patch1_id],
				b_params[patch2_id]
			};
		
			bool ray_is_accepted = false;
			int bad_ray_count = 0;
			all_intersections_with_normals_result result;
			do
			{
				result = all_intersections_shared_open_patch(config);
				ray_is_accepted = result.valid_ray;

				if (!ray_is_accepted)
				{
					ray(0) = dis(gen);
					ray(1) = dis(gen);
					ray(2) = dis(gen);
					ray.normalize();
					config.dir = ray;
					bad_ray_count++;
				}
			} while (!ray_is_accepted && bad_ray_count < 20);

			int chi = 0; 
			for (int k = 0; k < result.all_intersections.size(); k++)
			{
				chi += result.all_intersections[k].second;
			}
			region_weighted_rays_info weighted_rays = get_weighted_rays(combined_patches, query_point);

	
			bool found_region = false;
			for (int k = 0; k < weighted_rays.polygonal_regions.size(); k++)
			{
				if (is_inside_polygon(weighted_rays.polygonal_regions[k], ray))
				{
					found_region = true;
					int base_wn = weighted_rays.relative_wn[k];
				

					int required_offset = chi - base_wn;

					for (int l = 0; l < weighted_rays.relative_wn.size(); l++)
					{
						double area = weighted_rays.areas[l];

						int ints = weighted_rays.relative_wn[l] + required_offset;
						double chi_final = static_cast<double>(ints);
						wn_res.gwn[i] += area * chi_final;
					}
				}
			}

			if (!ray_is_accepted)
				wn_res.gwn[i] = -3.0;

			ASSERT_RELEASE(found_region, "did not find region containing the ray");
		}

		query_points_done[i] = true;

	

		if (i % 100 == 0)
		{
			int total_done = 0;
			for (int j = 0; j < query_points_done.size(); j++)
			{
				if (query_points_done[j])
					total_done++;
			}

			std::cout << total_done << " out of " << query_points.rows() << std::endl;
		}
	}

	return wn_res;
	
}

bool is_inside_uv_polygon(gsl_multiroot_fsolver* parametric_solver, const Eigen::Vector2d& inside_point, const Eigen::Vector2d& query_point, implicit_func_t f, implicit_func_t g, const open_boundary& boundary_param, bool& has_hit_f_g_intersection)
{
	// we're shooting a ray from the SAFE_POINT to the QUERY_POINT, t=0 => SAFE_POINT, t=1 QUERY_POINT
	std::vector<double> all_intersections = boundary_segment_intersections(boundary_param.curves, inside_point, query_point);
	has_hit_f_g_intersection = false;
	const double root_epsilon = 0.00005;
	const double fval_epsilon = 1e-6;
	const double t_epsilon = 1e-7;

	const size_t max_iterations = 1000;

	double eps = 0.00001;
	double t_min = eps;
	double t_max = 1.0 - eps;
	int num_starting_ts = 30;
	Eigen::Vector2d dir = query_point - inside_point;
	const int num_starting_uvs = 4;
	constexpr double uv_epsilon = 0.0;
	constexpr double u_min = -3.0;
	constexpr double u_max = 3.0;
	constexpr double v_min = -3.0;
	constexpr double v_max = 3.0;
	int solver_error_counter = 0;
	std::vector<double> t_linspace = linspace(t_min, t_max, num_starting_ts, true);
	std::vector<double> u_linspace = linspace(u_min, u_max, num_starting_uvs, true);
	std::vector<double> v_linspace = linspace(v_min, v_max, num_starting_uvs, true);

	uv_solver_params params;
	params.f = f;
	params.g = g;
	params.safe_point = inside_point;
	params.query_point = query_point;

	for (double t0 : t_linspace)
	{
		for (double u0 : u_linspace)
		{
			for (double v0 : v_linspace)
			{
				double x_init[3] = { u0, v0, t0 };
				const Eigen::Vector3d uvt(x_init);
				gsl_vector_set(parametric_solver->x, 0, x_init[0]);
				gsl_vector_set(parametric_solver->x, 1, x_init[1]);
				gsl_vector_set(parametric_solver->x, 2, x_init[2]);
				parametric_solver->function->params = (void*)&params;
				gsl_multiroot_fsolver_set(parametric_solver, parametric_solver->function, parametric_solver->x); // this line is probably not needed here

				int status;
				size_t iter_num = 0;

				do
				{
					iter_num++;
					status = gsl_multiroot_fsolver_iterate(parametric_solver);
					//print_solver_state(iter_num, s);
					if (status)
						break;

					status = gsl_multiroot_test_residual(parametric_solver->f, fval_epsilon);
				} while (status == GSL_CONTINUE && iter_num < max_iterations);

				if (status == GSL_ENOPROG || status == GSL_ENOPROGJ)
				{
					solver_error_counter++;
				}
				else if (status == GSL_SUCCESS)
				{
					double g_u = gsl_vector_get(parametric_solver->x, 0);
					double g_v = gsl_vector_get(parametric_solver->x, 1);
					double t = gsl_vector_get(parametric_solver->x, 2);
					Eigen::Vector2d f_uv_point = inside_point + t * (query_point - inside_point);
					Eigen::Vector3d f_value = f(f_uv_point);
					Eigen::Vector3d g_value = g(Eigen::Vector2d(g_u, g_v));

					double norm = (f_value - g_value).norm();

					if (t < t_min - t_epsilon || t > t_max + t_epsilon) // t out of range
						continue;


					if (!is_in_vector_epsilon(all_intersections, t, root_epsilon))
					{
						all_intersections.push_back(t);
						has_hit_f_g_intersection = true;
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

	return all_intersections.size() % 2 == 0;
}

all_intersections_with_normals_result find_intersections_shared_boundary_parametric(gsl_multiroot_fsolver* surface_solver, gsl_multiroot_fsolver* parametric_solver, const parametric_solver_params& params, implicit_func_t g, const open_boundary& boundary_param)
{
	all_intersections_with_normals_result result;
	implicit_func_t f = params.func;

	const double root_epsilon = 0.00005;
	const double fval_epsilon = 1e-6;
	const double t_epsilon = 1e-7;
	const size_t max_iterations = 1000;
	const double ray_dot_epsilon = 0.15;

	constexpr double uv_epsilon = 0.02;
	constexpr double u_min = uv_epsilon;
	constexpr double u_max = 1.0 - uv_epsilon;
	constexpr double v_min = uv_epsilon;
	constexpr double v_max = 1.0 - uv_epsilon;
	
	constexpr double t_max = 5.0;
	constexpr double t_min = 0.0;

	std::vector<double> roots_found;
	const int num_starting_ts = std::max<int>(2, (t_max - t_min) * 30);
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
				gsl_vector_set(surface_solver->x, 0, x_init[0]);
				gsl_vector_set(surface_solver->x, 1, x_init[1]);
				gsl_vector_set(surface_solver->x, 2, x_init[2]);
				surface_solver->function->params = (void*)&params;
				gsl_multiroot_fsolver_set(surface_solver, surface_solver->function, surface_solver->x); // this line is probably not needed here

				int status;
				size_t iter_num = 0;

				do
				{
					iter_num++;
					status = gsl_multiroot_fsolver_iterate(surface_solver);
					//print_solver_state(iter_num, s);
					if (status)
						break;

					status = gsl_multiroot_test_residual(surface_solver->f, fval_epsilon);
				} while (status == GSL_CONTINUE && iter_num < max_iterations);

				if (status == GSL_ENOPROG || status == GSL_ENOPROGJ)
				{
					solver_error_counter++;
				}
				else if (status == GSL_SUCCESS)
				{
					double u = gsl_vector_get(surface_solver->x, 0);
					double v = gsl_vector_get(surface_solver->x, 1);
					double t = gsl_vector_get(surface_solver->x, 2);

					if (t < t_min - t_epsilon || t > t_max + t_epsilon) // t out of range
						continue;

					if (!is_in_vector_epsilon(roots_found, t, root_epsilon))
					{
						Eigen::Matrix<double, 3, 2> jacobian = jacobian_fd_implicit(Eigen::Vector2d(u, v), f);
						Eigen::Vector3d normal = jacobian.col(0).cross(jacobian.col(1));
						Eigen::Vector3d dir = params.dir;

						normal.normalize();
						dir.normalize();

						double ray_normal_dot = normal.dot(dir);

						int chi;
						if (ray_normal_dot < 0)
							chi = -1;
						else
							chi = 1;

						Eigen::Vector2d inside_point = Eigen::Vector2d(0.5, 0.1);
						Eigen::Vector2d query_point = Eigen::Vector2d(u, v);
						bool has_hit_f_g_intersection;
						bool is_inside_uv = is_inside_uv_polygon(parametric_solver, inside_point, query_point, f, g, boundary_param, has_hit_f_g_intersection);
						if (has_hit_f_g_intersection)
						{
							result.all_intersections.clear();
							result.valid_ray = false;
						}

						roots_found.push_back(t);
						if (is_inside_uv)
						{
							result.all_intersections.emplace_back(t, chi);
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
	return result;
}

all_intersections_with_normals_result all_intersections_shared_open_patch(const shared_open_patch_rootfinding_settings& config)
{
	gsl_multiroot_fsolver* surf_solver = config.surface_solver;
	gsl_multiroot_fsolver* uv_solver = config.uv_solver;
	const open_boundary& bd1 = config.bd1;
	const open_boundary& bd2 = config.bd2;

	parametric_solver_params params;
	params.dir = config.dir;
	params.func = config.f1;
	params.point = config.point;

 	all_intersections_with_normals_result results_f1 = find_intersections_shared_boundary_parametric(surf_solver, uv_solver, params, config.f2, bd1);
	if (!results_f1.valid_ray)
		return results_f1;

	params.func = config.f2;
	all_intersections_with_normals_result results_f2 = find_intersections_shared_boundary_parametric(surf_solver, uv_solver, params, config.f1, bd2);
	if (!results_f2.valid_ray)
		return results_f2;	
	
	all_intersections_with_normals_result result;
	result.valid_ray = true;

	result.all_intersections.insert(result.all_intersections.end(), results_f1.all_intersections.begin(), results_f1.all_intersections.end());
	result.all_intersections.insert(result.all_intersections.end(), results_f2.all_intersections.begin(), results_f2.all_intersections.end());
	std::sort(result.all_intersections.begin(), result.all_intersections.end(), [=](const std::pair<double, int>& p1, const std::pair<double, int>& p2) { return p1.first < p2.first; });

	return result;
}