#pragma once
#include "spherical.h"
#include "patch.h"
#include "curve_net.h"
#include "curve_net_parser.h"
#include <map>
#include <set>

struct intersection_t {
	bool intersects;
	int end_point_intersection;
	Eigen::Vector3d location;
};

struct spherical_intersection_entrance
{
	int intersection_point_id;
	int entrance_id;

	bool operator<(const spherical_intersection_entrance& other) const
	{
		if (intersection_point_id < other.intersection_point_id)
			return true;
		else if (intersection_point_id > other.intersection_point_id)
			return false;
		return entrance_id < other.entrance_id;
	}
};

struct spherical_intersection_direction
{
	int exit_id;
	int direction; // -1 or 1
};

struct projection_intersection_info
{
	closed_spherical_curve_t extended_curve;
	int num_intersections;
	std::vector<int> curve_ids;
	std::map<spherical_intersection_entrance, spherical_intersection_direction> turn_left_index;
	bool is_valid = false;
};

struct region_splitting_info
{
	std::vector<closed_spherical_curve_t> regions;
	std::map<int, std::set<std::pair<int, int>>> neighborhood_map;
};

struct region_weighted_rays_info
{
	std::vector<double> areas;
	std::vector<Eigen::Vector3d> rays;
	std::vector<int> relative_wn;
	std::vector<closed_spherical_curve_t> polygonal_regions;
};

struct weighted_rays_info
{
	std::vector<double> areas;
	std::vector<Eigen::Vector3d> rays;
	std::vector<int> relative_wn;
};

typedef std::pair<Eigen::Vector3d, Eigen::Vector3d> segment_t;
region_weighted_rays_info get_weighted_rays(const closed_curve_t& curve, const Eigen::Vector3d& point);
region_weighted_rays_info get_weighted_rays_with_config(const closed_curve_t& curve, const Eigen::Vector3d& point, const surface_config& config);
region_weighted_rays_info decompose_regions_fast(const std::vector <space_curve_t>& boundaries, const Eigen::Vector3d& point);

region_splitting_info compute_regions_for_curve(const closed_spherical_curve_t& spherical_curve);
closed_spherical_curve_t project_to_sphere(const closed_curve_t& curve, const Eigen::Vector3d& point);
closed_spherical_curve_t curve_data_from_ids(const std::vector<int>& ids, const closed_spherical_curve_t& curve_data);
std::map<int, std::set<std::pair<int, int>>>  get_neighborhood_map(int num_regions, const std::vector<int>& regions_from_left, const std::vector<int>& regions_from_right);
region_splitting_info create_regions_with_relative_wn(const std::vector<int>& curve_ids, std::map<spherical_intersection_entrance, spherical_intersection_direction>& turn_left, const closed_curve_t& curve_data, int num_intersections);
std::vector<int> relative_wn_from_adjacency(const std::map<int, std::set<std::pair<int, int>>>& neighborhood_map);
Eigen::Vector3d find_ray_in_region(const closed_spherical_curve_t& curve, int start_index);
Eigen::Vector3d find_best_ray_in_region(const closed_spherical_curve_t& curve);

projection_intersection_info find_all_spherical_intersections(const closed_spherical_curve_t& spherical_curve);

curve_net_sampler run_region_splitting(const std::vector<patch_t>& closed_patches, const Eigen::MatrixXd& query_points);
closed_curve_t keep_unique_points(const closed_curve_t& curve);
double minimum_distance_to_segment(const closed_spherical_curve_t& curve, const Eigen::Vector3d& point);
double find_minimum_distance_to_segment_all_polygons(const std::vector<closed_spherical_curve_t>& regions, const Eigen::Vector3d& point);

closed_spherical_curve_t remove_duplicates(const closed_spherical_curve_t& curve);

intersection_t spherical_segments_intersect(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const Eigen::Vector3d& p3, const Eigen::Vector3d& p4);
intersection_t spherical_segments_intersect_faster(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const Eigen::Vector3d& p3, const Eigen::Vector3d& p4);
void compute_all_sphere_intersections(const patch_t& patch, const Eigen::MatrixXd& query_points);

int largest_segment(const closed_spherical_curve_t& curve);
double parameter_t_at_index(const closed_curve_t& curve, int largest_index);
int find_region_index(const std::vector<space_curve_t>& regions, const Eigen::Vector3d& point);


