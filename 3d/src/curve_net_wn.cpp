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

