#include "curve_net_parser.h"
#include <string>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>
#include <vector>
#include "curve_net.h"

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
