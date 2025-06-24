#pragma once

#include <vector>
#include <utility>
struct all_intersections_with_normals_result
{
	std::vector<std::pair<double, int>> all_intersections;
	bool valid_ray;
};

struct all_intersections_result
{
	std::vector<double> all_intersections;
	bool valid_ray;
};


struct chi_result
{
	bool success;
	int chi;
};