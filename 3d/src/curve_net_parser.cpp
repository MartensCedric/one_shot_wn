#include "curve_net_parser.h"
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>
#include <iomanip>
#include <random>
#include <vector>
#include "curve_net.h"

Eigen::MatrixXd remove_consecutive_duplicates(const Eigen::MatrixXd& curve)
{
	double epsilon = 2.5e-9;
	// removes consecutive duplicates
	std::vector<Eigen::Vector3d> non_duplicates;
	non_duplicates.reserve(curve.rows());
	for (int i = 0; i < curve.rows(); i++)
	{
		int next_index = (i + 1) % curve.rows();

		Eigen::Vector3d current = curve.row(i);
		Eigen::Vector3d next = curve.row(next_index);
		Eigen::Vector3d diff = current - next;
		
		if (diff.squaredNorm() > epsilon)
			non_duplicates.push_back(current);
	}

	if (non_duplicates.size() == curve.size())
		return curve;

	Eigen::MatrixXd output(non_duplicates.size(), 3);

	for (int i = 0; i < non_duplicates.size(); i++)
		output.row(i) = non_duplicates[i];
	return output;
}

std::vector<patch_t> load_all_patches(const std::string& patch_file)
{
	int num_patches;
	std::ifstream file(patch_file);
	if (!file.is_open())
	{
		std::cerr << "Could not find: " << patch_file << std::endl;
		throw std::runtime_error("Could not find: " + patch_file);
	}
		
	file >> num_patches;
	std::vector<patch_t> patches(num_patches);

	for (int i = 0; i < num_patches; i++)
	{
		int patch_size, num_gaps;
		file >> patch_size >> num_gaps;

		patch_t patch;
		patch.is_open = num_gaps > 0;
		patch.gaps_ids.resize(num_gaps);
		patch.curve.resize(patch_size, 3);

		for (int j = 0; j < patch_size; j++)
		{
			double x, y, z;
			file >> x >> y >> z;
			patch.curve.row(j) = Eigen::Vector3d(x, y, z);
		}

		for (int j = 0; j < num_gaps; j++)
		{
			int gap_id;
			file >> gap_id;
			patch.gaps_ids[j] = gap_id;
		}

		patches[i] = patch;
	}

	return patches;
}

Eigen::MatrixXd load_query_points(const std::string& query_points_file)
{
	int num_query_points;
	std::ifstream file(query_points_file);
	if (!file.is_open())
		throw std::runtime_error("Could not find: " + query_points_file);
	file >> num_query_points;
	Eigen::MatrixXd output;
	output.resize(num_query_points, 3);

	for (int i = 0; i < num_query_points; i++)
	{
		double x, y, z;
		file >> x >> y >> z;
		output.row(i) = Eigen::Vector3d(x, y, z);
	}
	return output;
}

std::vector<patch_t> remove_patches(const std::vector<patch_t>& patches, const std::vector<int>& ids_to_remove)
{
	std::vector<patch_t> output(patches.begin(), patches.end());
	std::vector<int> ids_to_remove_ordered(ids_to_remove.begin(), ids_to_remove.end());
	std::sort(ids_to_remove_ordered.rbegin(), ids_to_remove_ordered.rend());
	for (int i = 0; i < ids_to_remove_ordered.size(); i++)
		output.erase(output.begin() + ids_to_remove_ordered[i]);
	return output;
}

std::vector<std::vector<int>> grid_to_rays(const std::vector<int>& dimensions, int axis)
{
	ASSERT_RELEASE(dimensions.size() == 3, "wrong dimension");
	ASSERT_RELEASE(axis >= 0 && axis <= 2, "wrong axis");
	std::vector<std::vector<std::vector<int>>> dense_grid;
	int acc = 0;
	for (int x = 0; x < dimensions[0]; x++)
	{
		std::vector<std::vector<int>> x_vec;
		for (int y = 0; y < dimensions[1]; y++)
		{
			std::vector<int> y_vec;
			for (int z = 0; z < dimensions[2]; z++)
			{
				y_vec.push_back(acc++);
			}
			x_vec.push_back(y_vec);
		}
		dense_grid.push_back(x_vec);
	}

	std::vector<std::vector<int>> output;

	int ray_length = dimensions[axis];
	if (axis == 0)
	{
		int dim1 = dimensions[1];
		int dim2 = dimensions[2];
		for (int i = 0; i < dim1; i++)
		{
			for (int j = 0; j < dim2; j++)
			{
				std::vector<int> ray;
				for (int depth = 0; depth < ray_length; depth++)
					ray.push_back(dense_grid[depth][i][j]);
				output.push_back(ray);
			}
		}
	}
	else if (axis == 1)
	{
		int dim1 = dimensions[0];
		int dim2 = dimensions[2];
		for (int i = 0; i < dim1; i++)
		{
			for (int j = 0; j < dim2; j++)
			{
				std::vector<int> ray;
				for (int depth = 0; depth < ray_length; depth++)
					ray.push_back(dense_grid[i][depth][j]);
				output.push_back(ray);
			}
		}
	}
	else if(axis == 2)
	{ 
		int dim1 = dimensions[0];
		int dim2 = dimensions[1];
		for (int i = 0; i < dim1; i++)
		{
			for (int j = 0; j < dim2; j++)
			{
				std::vector<int> ray;
				for (int depth = 0; depth < ray_length; depth++)
					ray.push_back(dense_grid[i][j][depth]);
				output.push_back(ray);
			}
		}
	}
	return output;
}

std::vector<std::vector<int>> dimension_to_rays(std::vector<int> dimensions)
{
	ASSERT_RELEASE(dimensions.size() == 2 || dimensions.size() == 3, "Dimension not supported");
	std::vector<int>::const_iterator max_el = std::max_element(dimensions.begin(), dimensions.end());
	int axis = max_el - dimensions.begin();
	int max_dimension = *max_el;

	if (dimensions.size() == 2)
	{
		dimensions.erase(max_el);
		return slice_to_rays(dimensions.front(), max_dimension, axis);
	}
	else if (dimensions.size() == 3)
	{
		return grid_to_rays(dimensions, axis);
	}

	ASSERT_RELEASE(false, "invalid state");
	return {};
}

std::vector<std::vector<int>> slice_to_rays(int num_rays, int ray_depth, int axis)
{
	ASSERT_RELEASE(axis == 0 || axis == 1, "invalid axis");
	std::vector<std::vector<int>> output;
	for (int i = 0; i < num_rays; i++)
	{
		std::vector<int> ray(ray_depth, 0);
		if(axis == 1)
			std::iota(ray.begin(), ray.end(), i * ray_depth);
		else if(axis == 0)
		{
			for (int j = 0; j < ray_depth; j++)
			{
				ray[j] = i + num_rays * j;
			}
		}
		output.push_back(ray);
	}
	return output;
}

std::vector<patch_t> get_open_patches(const std::vector<patch_t>& patches)
{
	std::vector<patch_t> only_open;
	std::copy_if(patches.begin(), patches.end(), std::back_inserter(only_open), [](const patch_t& p) { return p.is_open; });
	return only_open;
}

std::vector<patch_t> get_closed_patches(const std::vector<patch_t>& patches)
{	
	std::vector<patch_t> only_closed;
	std::copy_if(patches.begin(), patches.end(), std::back_inserter(only_closed), [](const patch_t& p) { return !p.is_open; });
	return only_closed;
}

std::vector<space_curve_t> get_closed_patches_space_curves(const std::vector<patch_t>& patches)
{
	std::vector<space_curve_t> only_closed;
	for (int i = 0; i < patches.size(); i++)
		only_closed.push_back(patches[i].curve);
	return only_closed;
}

std::vector<patch_t> subsample_patches(const std::vector<patch_t>& patches, int sampling_rate)
{
	std::vector<patch_t> space_curves(patches.size());

	for (int i = 0; i < patches.size(); i++)
	{
		space_curves[i].curve = super_sample_patch(patches[i].curve, sampling_rate);
		space_curves[i].is_open = patches[i].is_open;
	}
		

	return space_curves;
}


double curve_length(const Eigen::MatrixXd& curve)
{
	double total_length = 0;

	for (int i = 0; i < curve.rows(); i++)
	{
		int first_index = i;
		int second_index = (i + 1) % curve.rows();
		double distance = (curve.row(second_index) - curve.row(first_index)).norm();
		total_length += distance;
	}

	return total_length;
}

Eigen::MatrixXd jiggle_rays(const Eigen::MatrixXd& query_points, const std::vector<std::vector<int>>& rays, double jiggle_amount)
{
	Eigen::MatrixXd query_points_jiggled(query_points.rows(), query_points.cols());

	std::default_random_engine gen(0xced);
	std::uniform_real_distribution<double> dis(-jiggle_amount, jiggle_amount);

	for (int i = 0; i < rays.size(); i++)
	{
		double x_diff = dis(gen);
		double y_diff = dis(gen);
		double z_diff = dis(gen);

		Eigen::RowVector3d diff = Eigen::RowVector3d(x_diff, y_diff, z_diff);

		for (int j = 0; j < rays[i].size(); j++)
		{
			query_points_jiggled.row(rays[i][j]) = query_points.row(rays[i][j]) + diff;
		}
	}

	return query_points_jiggled;
}