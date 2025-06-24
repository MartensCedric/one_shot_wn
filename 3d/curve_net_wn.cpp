#include "curve_net_wn.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <atomic>
#include <Eigen/Dense>

#include <igl/barycenter.h>


#include <boost/math/quadrature/gauss_kronrod.hpp>
#include "adaptive.h"

#define WORKERS_TO_USE_SHUFFLED 18
#define WORKERS_TO_USE_UNSHUFFLED 40

space_curve_t dirichlet_at_inf(const space_curve_t& patch, double scale)
{
	space_curve_t values_at_inf = patch;
	Eigen::Vector3d means = space_curve_means(patch);
	for (int i = 0; i < values_at_inf.rows(); i++)
		values_at_inf.row(i) -= means;
	values_at_inf *= scale;
	for (int i = 0; i < values_at_inf.rows(); i++)
		values_at_inf.row(i) += means;
	return values_at_inf;
}

precomputed_curve_data precompute_patches(const std::vector<space_curve_t>& patches, const std::vector<double>& insidenesses)
{
	precomputed_curve_data precompute;
	std::cout << patches.size() << " patches" << std::endl;

	std::cout << "Preprocessing..." << std::endl;
	constexpr int max_threads = 18;

	precompute.bounding_boxes.resize(patches.size());
	precompute.df_dns.resize(patches.size());
	precompute.full_patches.resize(patches.size());
	precompute.int_params.resize(patches.size());

	precompute.precompute_total_time = std::chrono::nanoseconds::zero();
	std::chrono::high_resolution_clock::time_point precompute_tic = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(max_threads)
	for (int i = 0; i < patches.size(); i++)
	{
		std::chrono::high_resolution_clock::time_point precompute_start = std::chrono::high_resolution_clock::now();
		BoundaryParametrization* boundary_param = nullptr;
		bool is_open = insideness_is_open(insidenesses[i]);
		if (is_open)
			boundary_param = new AnnulusOpenParametrization(R_AT_INF, insidenesses[i], patches[i].rows());
		else
			boundary_param = new SquareParametrization(patches[i].rows());
			

		boundary_param->init_boundary();


		space_curve_t ext_full_values;

		if (is_open)
		{
			ext_full_values.resize(boundary_param->get_total_points(), 3);
			const boundary_curve_t& bd_large = boundary_param->get_boundary_curves()[0];
			const boundary_curve_t& bd_small = boundary_param->get_boundary_curves()[1];
			int n_large = bd_large.rows();
			int n_small = bd_small.rows();

			space_curve_t values_on_plane = fit_data_to_patch_plane(bd_large, bd_small, patches[i]);

			ext_full_values.block(0, 0, n_large, 3) = values_on_plane;
			ext_full_values.block(n_large, 0, n_small, 3) = patches[i];
		}
		else
		{
			ext_full_values = patches[i];		
		}
	

		const space_curve_t x_space_curve_ext = create_space_curve_for_boundaries(boundary_param, ext_full_values.col(0));
		const space_curve_t y_space_curve_ext = create_space_curve_for_boundaries(boundary_param, ext_full_values.col(1));
		const space_curve_t z_space_curve_ext = create_space_curve_for_boundaries(boundary_param, ext_full_values.col(2));

		const Eigen::MatrixXd G = compute_bem_G_from_boundaries(boundary_param);
		const Eigen::MatrixXd H = compute_bem_H_from_boundaries(boundary_param);

		//std::cout << G.col(0) << std::endl;

		const Eigen::VectorXd dx_dn = df_dn_from_G_and_H(x_space_curve_ext, G, H);
		const Eigen::VectorXd dy_dn = df_dn_from_G_and_H(y_space_curve_ext, G, H);
		const Eigen::VectorXd dz_dn = df_dn_from_G_and_H(z_space_curve_ext, G, H);

		Eigen::MatrixXd df_dn(dx_dn.rows(), 3);
		df_dn.col(0) = dx_dn;
		df_dn.col(1) = dy_dn;
		df_dn.col(2) = dz_dn;

		precompute.full_patches[i] = ext_full_values;
		precompute.df_dns[i] = df_dn; // There might be a way to emplace back this
		precompute.int_params[i] = boundary_param;
		precompute.bounding_boxes[i] = bounding_box(patches[i]);

		std::chrono::high_resolution_clock::time_point precompute_end = std::chrono::high_resolution_clock::now();
#pragma omp critical 
		{
			precompute.precompute_total_time += precompute_end - precompute_start;
		}
	
	}
	std::chrono::high_resolution_clock::time_point precompute_toc = std::chrono::high_resolution_clock::now();
	precompute.precompute_result_time = precompute_toc - precompute_tic;
	std::cout << "Preprocessing done!" << std::endl;
	return precompute;
}

all_intersections_with_normals_result find_all_intersections_bem(const box& bounding_box, const struct bem_solver_params& bem_solver_params, gsl_multiroot_fsolver* solver, double max_ray_length)
{
	all_intersections_with_normals_result result;
	result.valid_ray = true;
	ray_box_intersection_result rb_result = ray_box_intersection(bem_solver_params.point, bem_solver_params.dir, bounding_box);
	if (!rb_result.intersects)
		return result;

	return all_ray_intersections(solver, bem_solver_params, std::max(0.0, rb_result.t_min), rb_result.t_max, max_ray_length);
}

void free_precompute(precomputed_curve_data& precomputed_data)
{
	for (int i = 0; i < precomputed_data.int_params.size(); i++)
	{
		delete precomputed_data.int_params[i];
		precomputed_data.int_params[i] = nullptr;
	}

	//for (int i = 0; i < precomputed_data.ext_params.size(); i++)
	//{
	//	delete precomputed_data.ext_params[i];
	//	precomputed_data.ext_params[i] = nullptr;
	//}
}


std::vector<double> winding_numbers_along_ray(const std::vector<std::pair<double, int>>& intersections, const std::vector<region_weighted_rays_info>& region_infos, const std::vector<int>& ray, const Eigen::MatrixXd& query_points)
{
	bool found_region = false;
	std::vector<double> winding_numbers(ray.size(), 0);
	Eigen::Vector3d dir = (query_points.row(ray[1]) - query_points.row(ray[0])).normalized();

	//Eigen::IOFormat numpy_fmt(Eigen::FullPrecision, 0, ", ", ",\n", "[", "]", "[", "]");
	//std::cout << dir << std::endl;
	for (int i = 0; i < ray.size(); i++)
	//for (int i = 4; i < 5; i++)
	{
		int query_index = ray[i];
		const region_weighted_rays_info& region_info = region_infos[i];
		double distance = (query_points.row(ray[i]) - query_points.row(ray[0])).norm();

		int chi = 0;
		for (int l = intersections.size() - 1; l >= 0; l--)
		{
			if (intersections[l].first < distance)
				break;

			chi += intersections[l].second;
		}

		for (int region_index = 0; region_index < region_info.areas.size(); region_index++)
		{
			//std::cout << "region_index: " << region_index << std::endl;
			//std::cout << region_infos[i].polygonal_regions[region_index].format(numpy_fmt) << std::endl;
			if (is_inside_polygon(region_infos[i].polygonal_regions[region_index], dir))
			{

				found_region = true;
				int base_wn = region_infos[i].relative_wn[region_index];
			

				int required_offset = chi - base_wn;

				for (int l = 0; l < region_infos[i].relative_wn.size(); l++)
				{
					double area = region_infos[i].areas[l];

					int ints = region_infos[i].relative_wn[l] + required_offset;
					double chi_final = static_cast<double>(ints);
					winding_numbers[i] += area * chi_final;
				}

				break;
			}
		}
	}

	return winding_numbers;
}

winding_number_results winding_number_mixed(const std::vector<patch_t>& patches, const Eigen::MatrixXd& query_points, std::vector<std::vector<int>> ray_shots, const surface_config& config)
{
	winding_number_results results;
	curvenet_timing_info timing;
	bool one_shot_all_points = false; // instead of one-ray

	timing.num_patches = patches.size();
	timing.total_rays = ray_shots.size();
	timing.ray_length = ray_shots[0].size();
	timing.ray_shooting_result_time = std::chrono::nanoseconds::zero();
	timing.boundary_processing_result_time = std::chrono::nanoseconds::zero();
	
	std::cout << "Running on " << query_points.rows() << " query points on " << patches.size() << " patches" << std::endl;;
	std::cout << "Shooting " << ray_shots.size() << " rays of size " << ray_shots[0].size() << std::endl;
	
	timing.total_patch_samples = 0;
	for (int i = 0; i < patches.size(); i++)
		timing.total_patch_samples += patches[i].curve.rows();

	std::cout << "Patches have a total of " << timing.total_patch_samples << std::endl;

	std::vector<space_curve_t> closed_patches = get_closed_patches_space_curves(patches);

	std::vector<double> insideness(closed_patches.size(), 1.0);
	precomputed_curve_data precompute;
	
	if (config.surface_type == SurfaceType::MINIMAL)
	{
		precompute = precompute_patches(closed_patches, insideness);
	}
	std::cout << "precomputed done" << std::endl;
	//std::cout << "Precompute time: " << std::chrono::duration_cast<std::chrono::milliseconds>(precompute.precompute_result_time).count() << " ms" << std::endl;

	const gsl_multiroot_fsolver_type* T = gsl_multiroot_fsolver_hybrids;

	std::vector<double> winding_numbers(query_points.rows(), 0);
	std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> fd_jac;
	if (config.surface_type == SurfaceType::PARAMETRIC)
	{
		if (config.use_fd)
			fd_jac = fd_jacobian(config.parametric_func, 1e-8);
		else
			fd_jac = config.parametric_jac;

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

	std::vector<std::pair<int, int>> ignored_rays;

	std::vector<std::vector<all_intersections_with_normals_result>> intersections(ray_shots.size());

	std::vector<bool> rays_done(ray_shots.size(), false);
	std::vector<int> rays_ordering(ray_shots.size());
	std::iota(rays_ordering.begin(), rays_ordering.end(), 0);
	ray_shots = random_shuffle<std::vector<int>>(ray_shots, rays_ordering);

	std::chrono::high_resolution_clock::time_point ray_shot_tic = std::chrono::high_resolution_clock::now();
	int total_intersection_tests_required = 0;


	for (int ray_index = 0; ray_index < ray_shots.size(); ray_index++)
	{
		std::vector<all_intersections_with_normals_result> intersection_for_patches(closed_patches.size());
		intersections[ray_index] = intersection_for_patches;
		for (int patch_index = 0; patch_index < closed_patches.size(); patch_index++)
		{
			intersections[ray_index][patch_index].valid_ray = false;
		}
	}
	
#pragma omp parallel for num_threads(max_threads)
	for (int ray_index = 0; ray_index < ray_shots.size(); ray_index++)
	{
		if (one_shot_all_points)
			break;
		int first_query_point_idx = ray_shots[ray_index][0];
		int second_query_point_idx = ray_shots[ray_index][1];

		Eigen::Vector3d first_query_point_loc = query_points.row(first_query_point_idx);
		Eigen::Vector3d ray_dir = (query_points.row(second_query_point_idx) - query_points.row(first_query_point_idx)).normalized();

		if (config.surface_type == SurfaceType::MESH)
		{
			intersections[ray_index] = { find_all_intersections_mesh(first_query_point_loc, ray_dir, config.mesh_vertices, config.mesh_faces) };
			continue;
		}


		for (int patch_index = 0; patch_index < closed_patches.size(); patch_index++)
		{
			std::chrono::high_resolution_clock::time_point first_point_region_split_tic = std::chrono::high_resolution_clock::now();
			region_weighted_rays_info region_weighted_rays = get_weighted_rays(closed_patches[patch_index], first_query_point_loc);
			if (region_weighted_rays.polygonal_regions.size() == 0)
				break;
			std::chrono::high_resolution_clock::time_point first_point_region_split_toc = std::chrono::high_resolution_clock::now();
			std::chrono::nanoseconds first_point_region_split_time = first_point_region_split_toc - first_point_region_split_tic;

			std::chrono::high_resolution_clock::time_point first_point_ray_tic = std::chrono::high_resolution_clock::now();
			int thread_id = omp_get_thread_num();
			if (config.surface_type == SurfaceType::MINIMAL)
			{

				const box& bounding_box = precompute.bounding_boxes[patch_index];
				const Eigen::MatrixXd& df_dns = precompute.df_dns[patch_index];
				struct bem_solver_params solver_params = {
					precompute.int_params[patch_index],
					precompute.full_patches[patch_index],
					precompute.df_dns[patch_index],
					first_query_point_loc,
					ray_dir
				};

				F[thread_id].params = &solver_params;
				F[thread_id].f = &f_func;

				double max_diag = Eigen::Vector3d(bounding_box.x_max - bounding_box.x_min, bounding_box.y_max - bounding_box.y_min, bounding_box.z_max - bounding_box.z_min).norm();
				intersections[ray_index][patch_index] = find_all_intersections_bem(bounding_box, solver_params, s[thread_id], max_diag);


				if (intersections[ray_index][patch_index].valid_ray)
				{
					double threshold = 0.006;
					double min_dist_to_segment = find_minimum_distance_to_segment_all_polygons(region_weighted_rays.polygonal_regions, ray_dir);

					if (min_dist_to_segment < threshold)
						intersections[ray_index][patch_index].valid_ray = false;
				}
			}
			else if (config.surface_type == SurfaceType::PARAMETRIC)
			{
				struct parametric_solver_params solver_params = {
					config.parametric_func,
					fd_jac,
					first_query_point_loc,
					ray_dir,
					config.is_in_parametric_domain
				};
				F[thread_id].params = &solver_params;
				F[thread_id].f = &f_func_parametric;
	
				intersections[ray_index][patch_index] = find_all_intersections_parametric(s[thread_id], solver_params, 0.0, config.parametric_max_t);
				total_intersection_tests_required += intersections[ray_index][patch_index].all_intersections.size() + 1;

				//ray_all_intersections.valid_ray = false;
			}
			else if (config.surface_type == SurfaceType::PARAMETRIC_GEARS)
			{
				struct parametric_solver_params solver_params = {
					config.parametric_func,
					fd_jac,
					first_query_point_loc,
					ray_dir,
					config.is_in_parametric_domain
				};
				F[thread_id].params = &solver_params;
				F[thread_id].f = &f_func_parametric;

				intersections[ray_index][patch_index] = find_all_intersections_parametric_gears(s[thread_id], solver_params, 0.0, config.parametric_max_t);
				//ray_all_intersections.valid_ray = false;
			}
			else if (config.surface_type == SurfaceType::COONS)
			{
				struct parametric_solver_params solver_params = {
						config.coons_patches[patch_index].func,
						fd_jac,
						first_query_point_loc,
						ray_dir, 
						config.is_in_parametric_domain
				};
				F[thread_id].params = &solver_params;
				F[thread_id].f = &f_func_parametric;

				intersections[ray_index][patch_index] = find_all_intersections_parametric(s[thread_id], solver_params, 0.0, config.parametric_max_t);

				if (!config.coons_flip_normals.empty())
				{
					if (config.coons_flip_normals[patch_index])
					{
						for (int int_index = 0; int_index < intersections[ray_index][patch_index].all_intersections.size(); int_index++)
						{
							intersections[ray_index][patch_index].all_intersections[int_index].second = -intersections[ray_index][patch_index].all_intersections[int_index].second;
						}
					}
				}
			}
			else if (config.surface_type == SurfaceType::MESH)
			{
				ASSERT_RELEASE(false, "unsupported");
			}
			else if (config.surface_type == SurfaceType::HEIGHT_FIELD)
			{

			}
		}

		rays_done[ray_index] = true;
	}

	std::chrono::high_resolution_clock::time_point ray_shot_toc = std::chrono::high_resolution_clock::now();
	std::chrono::nanoseconds ray_shot_delta = ray_shot_toc - ray_shot_tic;
	timing.ray_shooting_result_time += ray_shot_delta;


	/*std::cout << "Num intersections tests required: " << total_intersection_tests_required << std::endl;
	std::cout << "Rays: " << ray_shots.size() << std::endl;
	std::cout << "Query Points: " << query_points.rows() << std::endl;*/

	ray_shots = unshuffle_from_random(ray_shots, rays_ordering);
	intersections = unshuffle_from_random(intersections, rays_ordering);
	std::cout << "Ray shooting done" << std::endl;

	std::chrono::high_resolution_clock::time_point boundary_processing_start = std::chrono::high_resolution_clock::now();

	if (config.surface_type != SurfaceType::MESH)
	{
#pragma omp parallel for num_threads(max_threads)
		for (int ray_index = 0; ray_index < ray_shots.size(); ray_index++)
		{
			for (int patch_index = 0; patch_index < closed_patches.size(); patch_index++)
			{
				if (intersections[ray_index][patch_index].valid_ray)
				{
					// march along the ray
					std::chrono::high_resolution_clock::time_point other_points_region_split_tic = std::chrono::high_resolution_clock::now();
					std::vector<region_weighted_rays_info> regions_infos(ray_shots[ray_index].size());
					for (int point_along_ray_index = 0; point_along_ray_index < ray_shots[ray_index].size(); point_along_ray_index++)
					{
						int query_index = ray_shots[ray_index][point_along_ray_index];
						regions_infos[point_along_ray_index] = get_weighted_rays_with_config(closed_patches[patch_index], query_points.row(query_index), config);
					}
					std::vector<double> winding_numbers_ray = winding_numbers_along_ray(intersections[ray_index][patch_index].all_intersections, regions_infos, ray_shots[ray_index], query_points);
						
					
#pragma omp critical 
					{
						for (int along_ray_index = 0; along_ray_index < ray_shots[ray_index].size(); along_ray_index++)
						{
							winding_numbers[ray_shots[ray_index][along_ray_index]] += winding_numbers_ray[along_ray_index];
						}
					}
				}
				else
				{
#pragma omp critical 
					{
						ignored_rays.emplace_back(ray_index, patch_index);
					}
				}

			}			
		}
	}
	else // is MESH
	{
#pragma omp parallel for num_threads(max_threads)
		for (int ray_index = 0; ray_index < ray_shots.size(); ray_index++)
		{
			for (int patch_index = 0; patch_index < patches.size(); patch_index++)
			{
				std::chrono::high_resolution_clock::time_point other_points_region_split_tic = std::chrono::high_resolution_clock::now();
				std::vector<region_weighted_rays_info> regions_infos(ray_shots[ray_index].size());

				for (int point_along_ray_index = 0; point_along_ray_index < ray_shots[ray_index].size(); point_along_ray_index++)
				{
					int query_index = ray_shots[ray_index][point_along_ray_index];
					regions_infos[point_along_ray_index] = get_weighted_rays(closed_patches[patch_index], query_points.row(query_index));
				}

				std::chrono::high_resolution_clock::time_point other_points_region_split_toc = std::chrono::high_resolution_clock::now();


				std::vector<double> winding_numbers_ray = winding_numbers_along_ray(intersections[ray_index][0].all_intersections, regions_infos, ray_shots[ray_index], query_points);


	#pragma omp critical 
				{
					for (int along_ray_index = 0; along_ray_index < ray_shots[ray_index].size(); along_ray_index++)
					{
						winding_numbers[ray_shots[ray_index][along_ray_index]] += winding_numbers_ray[along_ray_index];
					}
				}
			}
			// march along the ray				
		}
	}

	std::chrono::high_resolution_clock::time_point boundary_processing_stop = std::chrono::high_resolution_clock::now();
	std::chrono::nanoseconds boundary_processing_delta = boundary_processing_stop - boundary_processing_start;
	timing.boundary_processing_result_time += boundary_processing_delta;
	std::cout << "Boundary processing done" << std::endl;

	std::cout << "time: " << (std::chrono::high_resolution_clock::now() - ray_shot_tic).count() << std::endl;

	std::vector<std::pair<int, int>> query_point_and_patch_leftovers;
	for (int ignored_ray_index = 0; ignored_ray_index < ignored_rays.size(); ignored_ray_index++)
	{
		int ray_index = ignored_rays[ignored_ray_index].first;
		int patch_index = ignored_rays[ignored_ray_index].second;
		
		for (int along_ray_index = 0; along_ray_index < ray_shots[ray_index].size(); along_ray_index++)
		{
			query_point_and_patch_leftovers.emplace_back(ray_shots[ray_index][along_ray_index], patch_index);
		}
	}


	int print_step = std::max<int>(query_point_and_patch_leftovers.size() / 25.0, 20);
	std::vector<bool> is_done(query_point_and_patch_leftovers.size(), false);


	std::vector<Eigen::Vector3d> rays_leftover_indices(query_point_and_patch_leftovers.size());
	std::vector<std::vector<int>> relative_wns_leftover_indices(query_point_and_patch_leftovers.size());
	std::vector<std::vector<double>> areas_leftover_indices(query_point_and_patch_leftovers.size());

	std::chrono::high_resolution_clock::time_point one_shot_bp_tic = std::chrono::high_resolution_clock::now();
	std::cout << "Leftover: " << query_point_and_patch_leftovers.size() << std::endl;
#pragma omp parallel for num_threads(max_threads)
	for (int leftover_index = 0; leftover_index < query_point_and_patch_leftovers.size(); leftover_index++)
	{
		int chi = 0;
		int query_point_index = query_point_and_patch_leftovers[leftover_index].first;
		int patch_index = query_point_and_patch_leftovers[leftover_index].second;


		/*space_curve_t curve = closed_patches[patch_index];
		int K = std::ceil(curve.rows() * 0.1);
		for(int k = 0; k < K; k++)
		{
			int largest_index = largest_segment(project_to_sphere(curve, query_points.row(query_point_index)));
			double t_curve1 = parameter_t_at_index(curve, largest_index);
			double t_curve2 = parameter_t_at_index(curve, (largest_index + 1) % curve.rows());

			double new_t = (t_curve1 + t_curve2) / 2.0;
			Eigen::Vector3d new_point = new_point_from_t_triangle_uv(config.parametric_func, new_t);

		}*/

		region_weighted_rays_info weighted_rays = get_weighted_rays(closed_patches[patch_index], query_points.row(query_point_index));
		if (weighted_rays.polygonal_regions.size() == 0)
			continue;

		int thread_id = omp_get_thread_num();

		if (weighted_rays.rays.empty())
		{
			chi = 0;
			rays_leftover_indices[leftover_index] = Eigen::Vector3d::Zero();
			//std::cout << "Zero regions found for point " << query_point_index << " with patch " <<  patch_index << std::endl;
		}
		else
		{
			int ray_id = 0;
	
			Eigen::Vector3d dir = weighted_rays.rays[ray_id];
			rays_leftover_indices[leftover_index] = dir;
			relative_wns_leftover_indices[leftover_index] = weighted_rays.relative_wn;
			areas_leftover_indices[leftover_index] = weighted_rays.areas;
			ASSERT_RELEASE(weighted_rays.rays.size() > 0, "No ray to shoot");
		}

		is_done[leftover_index] = true;

		if (leftover_index % print_step == 0)
		{
			size_t num_done = 0;
			for (int i = 0; i < query_point_and_patch_leftovers.size(); i++)
			{
				if(is_done[i])
					num_done++;
			}

			double percent = 100.0 * (static_cast<double>(num_done) / static_cast<double>(query_point_and_patch_leftovers.size()));

			std::cout << "one-shot progress: " << percent << "% => " << num_done << " out of " << query_point_and_patch_leftovers.size() << std::endl;
		}
	}

	std::chrono::high_resolution_clock::time_point one_shot_bp_toc = std::chrono::high_resolution_clock::now();
	std::chrono::nanoseconds one_shot_bp_delta = one_shot_bp_toc - one_shot_bp_tic;
	timing.boundary_processing_result_time += one_shot_bp_delta;

	for (int d = 0; d < is_done.size(); d++)
		is_done[d] = false;

	std::chrono::high_resolution_clock::time_point one_shot_ray_shooting_tic = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(max_threads)
	for (int leftover_index = 0; leftover_index < query_point_and_patch_leftovers.size(); leftover_index++)
	{
		int thread_id = omp_get_thread_num();
		int chi = 0;
		int query_point_index = query_point_and_patch_leftovers[leftover_index].first;
		int patch_index = query_point_and_patch_leftovers[leftover_index].second;
		Eigen::Vector3d point = query_points.row(query_point_index);
		Eigen::Vector3d dir = rays_leftover_indices[leftover_index];
		const std::vector<int>& relative_wn = relative_wns_leftover_indices[leftover_index];
		const std::vector<double>& areas = areas_leftover_indices[leftover_index];
		bool has_no_rays = dir.isZero();

		if (config.surface_type == SurfaceType::MINIMAL)
		{
			struct bem_solver_params solver_params = {
						precompute.int_params[patch_index],
						precompute.full_patches[patch_index],
						precompute.df_dns[patch_index],
						point,
						dir,
			};

			const space_curve_t& space_curve = solver_params.patch;
			const Eigen::MatrixXd& df_dn = solver_params.df_dn;
			ray_box_intersection_result rb_result = ray_box_intersection(point, dir, precompute.bounding_boxes[patch_index]);
			if (!rb_result.intersects || areas.size() == 1 || has_no_rays)
			{
				chi = 0;
			}
			else
			{

				F[thread_id].params = &solver_params;
				F[thread_id].f = &f_func;
				const box& bb = precompute.bounding_boxes[patch_index];
				double max_ray_length = Eigen::Vector3d(bb.x_max - bb.x_min, bb.z_max - bb.z_min, bb.z_max - bb.z_min).norm();
				chi_result result = find_chi(s[thread_id], solver_params, std::max(0.0, rb_result.t_min), rb_result.t_max, max_ray_length, true);
				if (!result.success)
				{
					//std::cout << "Ray rejected" << std::endl;
					chi = 0;
				}
				else
				{
					chi = result.chi;
				}
			}
		}
		else if (config.surface_type == SurfaceType::PARAMETRIC)
		{
			struct parametric_solver_params solver_params = {
				config.parametric_func,
				fd_jac,
				point,
				dir,
				config.is_in_parametric_domain
			};
			F[thread_id].params = &solver_params;
			F[thread_id].f = &f_func_parametric;

			all_intersections_with_normals_result result = find_all_intersections_parametric(s[thread_id], solver_params, 0.0, config.parametric_max_t);

			if (!result.valid_ray)
			{
				//std::cout << "Ray rejected" << std::endl;
				chi = 0;
			}
			else
			{
				chi = 0;
				for (int int_index = 0; int_index < result.all_intersections.size(); int_index++)
				{
					chi += result.all_intersections[int_index].second;
				}
			}
		}
		else if (config.surface_type == SurfaceType::MESH)
		{
			//chi = 0;
			//ASSERT_RELEASE(false, "Rays should never be rejected so one-shot shouldnt happen");
		}

		for (int k = 0; k < areas.size(); k++)
		{
			double area = areas[k];
			int intersections = chi + relative_wn[k] - relative_wn[0];
			double final_chi = static_cast<double>(intersections);

#pragma omp critical 
			{
				winding_numbers[query_point_index] += area * final_chi;
	
			}
		}

		is_done[leftover_index] = true;

		if (leftover_index % print_step == 0)
		{
			size_t num_done = 0;
			for (int i = 0; i < query_point_and_patch_leftovers.size(); i++)
			{
				if (is_done[i])
					num_done++;
			}

			double percent = 100.0 * (static_cast<double>(num_done) / static_cast<double>(query_point_and_patch_leftovers.size()));

			std::cout << "1S ray progress: " << percent << "% => " << num_done << " out of " << query_point_and_patch_leftovers.size() << std::endl;
		}
	}

	std::chrono::high_resolution_clock::time_point one_shot_ray_shooting_toc = std::chrono::high_resolution_clock::now();
	std::chrono::nanoseconds one_shot_ray_shooting_delta = one_shot_ray_shooting_toc - one_shot_ray_shooting_tic;
	results.timing.ray_shooting_result_time += one_shot_ray_shooting_delta;
	results.timing = timing;
	results.gwn = winding_numbers;

	// multiple boundaries should work, but it's not implemented so do this hack for now
	if (config.do_symmetric_patch_adjustment)
	{
		for (int i = 0; i < results.gwn.size(); i++)
		{
			double wn_bad = results.gwn[i];
			if (wn_bad > 0)
				results.gwn[i] = 1.0 - 2.0 * (1.0 - wn_bad);
			else
				results.gwn[i] *= 2.0;
		}
	}

	return results;
}


void write_gwn_to_file(const winding_number_results& wn_res, const curvenet_input& input)
{
	const std::vector<double>& gwn = wn_res.gwn;

	std::string input_name = input.config.is_override_output_name ? input.config.override_output_name : input.input_name;
	std::string filename = "outputs/" + input_name + ".m";
	std::ofstream chi_output(filename);
	chi_output << "chis = [";
	for (int j = 0; j < gwn.size(); j++)
	{
		chi_output << std::setprecision(10) << gwn[j] << ",";
	}

	chi_output << "];\n";
	chi_output << "res = [";

	for (int j = input.dimensions.size() - 1; j >= 0; j--)
	{
		chi_output << std::to_string(input.dimensions[j]) << ",";
	}

	chi_output << "];\n";

	chi_output << "num_patches = " << std::to_string(wn_res.timing.num_patches) << ";" << std::endl;
	chi_output << "total_rays = " << std::to_string(wn_res.timing.total_rays) << ";" << std::endl;
	chi_output << "ray_length = " << std::to_string(wn_res.timing.ray_length) << ";" << std::endl;
	chi_output << "total_patch_samples = " << std::to_string(wn_res.timing.total_patch_samples) << ";" << std::endl;
	chi_output << "ray_shooting_result_time = " << std::to_string(wn_res.timing.ray_shooting_result_time.count()) << ";" << std::endl;
	chi_output << "boundary_processing_result_time = " << std::to_string(wn_res.timing.boundary_processing_result_time.count()) << ";" << std::endl;

	chi_output.close();
}

winding_number_results winding_number_joint_patches(const std::vector<patch_t>& patches, const Eigen::MatrixXd& query_points, const std::vector<std::vector<int>>& ray_points, const surface_config& config)
{
	winding_number_results results;
	curvenet_timing_info timing;
	bool one_shot_all_points = false; // instead of one-ray

	
	Eigen::MatrixXd ray_dirs(ray_points.size(), 3);
	for (int i = 0; i < ray_dirs.rows(); i++)
	{
		ray_dirs.row(i) = (query_points.row(ray_points[i][1]) - query_points.row(ray_points[i][0])).normalized();
	}

	//std::cout << "Running on " << query_points.rows() << " query points on " << patches.size() << " patches" << std::endl;;

	std::vector<double> insideness(patches.size(), 1.0);
	precomputed_curve_data precompute;

	//if (config.surface_type == SurfaceType::MINIMAL)
	//{
	//	precompute = precompute_patches(closed_patches, insideness);

	//}
	//std::cout << "precomputed done" << std::endl;
	//std::cout << "Precompute time: " << std::chrono::duration_cast<std::chrono::milliseconds>(precompute.precompute_result_time).count() << " ms" << std::endl;

	const gsl_multiroot_fsolver_type* T = gsl_multiroot_fsolver_hybrids;

	std::vector<double> winding_numbers(query_points.rows(), 0);

	constexpr int max_threads = 19;
	omp_set_num_threads(max_threads);

	gsl_multiroot_fsolver* s[max_threads];
	gsl_multiroot_function F[max_threads];
	for (int i = 0; i < max_threads; i++)
	{
		F[i].f = &f_func_parametric;
		F[i].n = 3;
		s[i] = gsl_multiroot_fsolver_alloc(T, F[i].n);
		s[i]->x = gsl_vector_alloc(F[i].n);
		s[i]->function = &F[i];
	}

	std::vector<std::vector<std::pair<double, int>>> signed_intersections(ray_points.size());

#pragma omp parallel for num_threads(max_threads)
	for (int ray_index = 0; ray_index < ray_points.size(); ray_index++)
	//for (int ray_index = 80; ray_index < ray_points.size(); ray_index++)
	{
		int thread_id = omp_get_thread_num();
		std::vector<std::pair<double, int>> intersections;
		Eigen::Vector3d ray_dir = ray_dirs.row(ray_index);
		Eigen::Vector3d point = query_points.row(ray_points[ray_index][0]);
		for (int patch_index = 0; patch_index < patches.size(); patch_index++)
		{
			if (config.surface_type == SurfaceType::PARAMETRIC)
			{
					struct parametric_solver_params solver_params = {
						patches[patch_index].func,
						patches[patch_index].jac,
						point,
						ray_dir,
						config.is_in_parametric_domain
					};
					F[thread_id].params = &solver_params;
					F[thread_id].f = &f_func_parametric;
					all_intersections_with_normals_result patch_intersections = find_all_intersections_parametric(s[thread_id], solver_params, 0.0, std::sqrt(3));
					
					for (int intersection_index = 0; intersection_index < patch_intersections.all_intersections.size(); intersection_index++)
						intersections.push_back(patch_intersections.all_intersections[intersection_index]);
			}
			else if(config.surface_type == SurfaceType::MINIMAL)
			{
				//intersections[ray_index][patch_index] = find_all_intersections_bem();
			}
		}

		std::sort(intersections.begin(), intersections.end(), [](const auto& p1, const auto& p2) {
			return p1.first < p2.first;
		});

		signed_intersections[ray_index] = intersections;

		//for (int int_id = 0; int_id < signed_intersections[ray_index].size(); int_id++)
		//{
		//	Eigen::Vector3d pos = point + ray_dir * signed_intersections[ray_index][int_id].first;
		//	std::cout << pos(0) << " " << pos(1) << " " << pos(2) << std::endl;

		//}
	}

//#pragma omp parallel for num_threads(max_threads)
	for (int ray_index = 0; ray_index < ray_points.size(); ray_index++)
	{
		int thread_id = omp_get_thread_num();
		for (int point_index = 0; point_index < ray_points[ray_index].size(); point_index++)
		{
			int query_index = ray_points[ray_index][point_index];
			Eigen::Vector3d point = query_points.row(query_index);
			const space_curve_t& boundary = config.boundaries[0];

			/*std::chrono::high_resolution_clock::time_point old_method_tic = std::chrono::high_resolution_clock::now();
			region_weighted_rays_info _region_weighted_rays = get_weighted_rays(boundary, point);
			std::chrono::high_resolution_clock::time_point old_method_toc = std::chrono::high_resolution_clock::now();
			std::chrono::nanoseconds old_method_time = old_method_toc - old_method_tic;*/
			
			//std::cout << "old: " << std::to_string(old_method_time.count()) << std::endl;
			
			std::chrono::high_resolution_clock::time_point cgal_tic = std::chrono::high_resolution_clock::now();
			region_weighted_rays_info region_weighted_rays = decompose_regions_fast({ boundary }, point);
			std::chrono::high_resolution_clock::time_point cgal_toc = std::chrono::high_resolution_clock::now();
			std::chrono::nanoseconds cgal_time = cgal_toc - cgal_tic;

			//std::cout << "cgal: " << std::to_string(cgal_time.count()) << std::endl;

			Eigen::Vector3d ray_dir = ray_dirs.row(ray_index);
			int region_id = find_region_index(region_weighted_rays.polygonal_regions, ray_dir);

			int chi = 0;

			const std::vector<std::pair<double, int>>& intersections_from_starting_point = signed_intersections[ray_index];
			double distance_from_starting_point = (query_points.row(query_index) - query_points.row(ray_points[ray_index][0])).norm();
			
			for (int intersection_index = 0; intersection_index < intersections_from_starting_point.size(); intersection_index++)
			{
				if (distance_from_starting_point < intersections_from_starting_point[intersection_index].first)
					chi += intersections_from_starting_point[intersection_index].second;
			}

			const std::vector<int>& relative_wn = region_weighted_rays.relative_wn;
			const std::vector<double>& areas = region_weighted_rays.areas;
			for (int k = 0; k < areas.size(); k++)
			{
				double area = areas[k];
				int intersections = chi + relative_wn[k] - relative_wn[region_id];
				double final_chi = static_cast<double>(intersections);

				winding_numbers[query_index] += area * final_chi;
			}
		}
	}
	results.gwn = winding_numbers;

	return results;
}