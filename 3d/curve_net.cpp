#include "curve_net.h"
#include <vector>
#include <iostream>

std::vector<double> gwn_from_chis_oneshot(const std::vector<std::vector<int>>& chis, const curve_net_sampler& sampler)
{
	std::vector<double> gwns(sampler.sample_points.size(), 0);
	for (int i = 0; i < sampler.sample_points.size(); i++) {
		for (int j = 0; j < sampler.patches.size(); j++) {
			double patch_gwn = 0.0;
			for (int k = 0; k < sampler.areas[j][i].size(); k++)
			{
				double area = sampler.areas[j][i][k];
				int intersections = chis[j][i] + sampler.rel_wn[j][i][k] - sampler.rel_wn[j][i][0];
				double chi = static_cast<double>(intersections);

				patch_gwn = area * chi;
				gwns[i] += patch_gwn;
			}
		}
	}
	return gwns;
}

std::vector<std::vector<double>> gwn_from_chis_one_shot_per_patch(const std::vector<std::vector<int>>& res_chis, const curve_net_sampler& closed_curve_sampler)
{
	std::vector<std::vector<double>> patchwise;

	for (int patch_id = 0; patch_id < closed_curve_sampler.patches.size(); patch_id++)
	{
		std::vector<double> gwn(res_chis[patch_id].size());

		for (int query_id = 0; query_id < res_chis[patch_id].size(); query_id++)
		{
			double patch_gwn_at_q = 0.0;
			for (int k = 0; k < closed_curve_sampler.areas[patch_id][query_id].size(); k++)
			{
				double area = closed_curve_sampler.areas[patch_id][query_id][k];
				int intersections = res_chis[patch_id][query_id] + closed_curve_sampler.rel_wn[patch_id][query_id][k] - closed_curve_sampler.rel_wn[patch_id][query_id][0];
				double chi = static_cast<double>(intersections);
				gwn[query_id] += area * chi;
			}

		}
		patchwise.push_back(gwn);
	}
	return patchwise;
}

std::vector<double> gwn_from_chis(const std::vector<std::vector<std::vector<int>>>& chis, const curve_net_sampler& sampler)
{
	std::vector<double> gwns(sampler.sample_points.size(), 0);
	for (int i = 0; i < sampler.sample_points.size(); i++) {
		for (int j = 0; j < sampler.patches.size(); j++) {
			double patch_gwn = 0.0;
			for (int k = 0; k < sampler.areas[j][i].size(); k++)
			{
				double area = sampler.areas[j][i][k];
				int intersections = chis[j][i][k];
				double chi = static_cast<double>(intersections);

				patch_gwn = area * chi;
				gwns[i] += patch_gwn;
			}
		}
	}
	return gwns;
}

space_curve_t super_sample_patch(const space_curve_t& patch, int sampling_rate)
{
	std::vector<Eigen::Vector3d> ss_patch;

	for (int i = 0; i < patch.rows(); i++)
	{
		Eigen::RowVector3d diff = patch.row(i) - patch.row((i + 1) % patch.rows());
		double distance = diff.norm();
		int v = std::ceil(std::max(2.0, static_cast<double>(sampling_rate)* distance));
		std::vector<double> x_lin = linspace(patch(i, 0), patch((i + 1) % patch.rows(), 0), v, false);
		std::vector<double> y_lin = linspace(patch(i, 1), patch((i + 1) % patch.rows(), 1), v, false);
		std::vector<double> z_lin = linspace(patch(i, 2), patch((i + 1) % patch.rows(), 2), v, false);

		for (int j = 0; j < x_lin.size(); j++)
		{
			ss_patch.emplace_back(x_lin[j], y_lin[j], z_lin[j]);
		}
	}
	space_curve_t result;
	result.resize(ss_patch.size(), 3);

	for (int i = 0; i < ss_patch.size(); i++)
	{
		result(i, 0) = ss_patch[i](0);
		result(i, 1) = ss_patch[i](1);
		result(i, 2) = ss_patch[i](2);
	}
	return result;
}

std::vector<space_curve_t> super_sample_patches(const std::vector<space_curve_t>& patches, int sampling_rate)
{
	std::vector<space_curve_t> space_curves(patches.size());

	for (int i = 0; i < patches.size(); i++)
		space_curves[i] = super_sample_patch(patches[i], sampling_rate);
	
	return space_curves;
}

std::vector<double> gwn_from_chis_rayrows(const std::vector<std::vector<std::vector<std::pair<double, int>>>>& intersections, const curve_net_sampler& sampler, const std::vector<std::vector<int>>& ray_shots)
{
	std::vector<double> output(sampler.sample_points.size(), 0);
	for (int i = 0; i < ray_shots.size(); i++)
	{
		Eigen::Vector3d dir = sampler.sample_points[ray_shots[i][1]] - sampler.sample_points[ray_shots[i][0]];
		dir.normalize();

		for (int j = 0; j < ray_shots[i].size(); j++)
		{
			int query_index = ray_shots[i][j];

			double distance = (sampler.sample_points[query_index] - sampler.sample_points[ray_shots[i][0]]).norm();
			for (int patch_index = 0; patch_index < sampler.patches.size(); patch_index++)
			{
				bool found_region = false;
				for (int k = 0; k < sampler.rel_wn[patch_index][query_index].size(); k++)
				{
					if (is_inside_polygon(sampler.spherical_regions[patch_index][query_index][k], dir))
					{
		     			found_region = true;
						int base_wn = sampler.rel_wn[patch_index][query_index][k];
						int chi = 0;
						for (int l = intersections[patch_index][i].size() - 1; l >= 0; l--)
						{
							if (intersections[patch_index][i][l].first < distance)
								break;

							chi += intersections[patch_index][i][l].second;
						}

						int required_offset = chi - base_wn;

						for (int l = 0; l < sampler.rel_wn[patch_index][query_index].size(); l++)
						{
							double area = sampler.areas[patch_index][query_index][l];

							int ints = sampler.rel_wn[patch_index][query_index][l] + required_offset;
							double chi_final = static_cast<double>(ints);
							output[query_index] += area * chi_final; 
						}
						
						break;
					}
				}

				/*if(!found_region)
					std::cout << "WARNING: no region found containing ray => " << query_index << " " << patch_index << std::endl;*/
				//ASSERT_RELEASE(found_region, "no region found containing ray...");
			}
		}
		//std::cout << i << " ==> " << output[ray_shots[i][0]] << std::endl;
	}
	return output;
}