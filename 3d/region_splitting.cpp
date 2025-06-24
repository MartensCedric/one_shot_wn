#include "region_splitting.h"
#include "spherical.h"
#include <vector>
#include <numeric>
#include <map>
#include <Eigen/Dense>
#include <set>
#include <map>
#include <queue>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include "math_util.h"
#include <CGAL/Surface_sweep_2_algorithms.h>
#include <CGAL/number_utils.h>
#include "curve_net_parser.h"

intersection_t spherical_segments_intersect(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const Eigen::Vector3d& p3, const Eigen::Vector3d& p4)
{
    double epsilon = 0.000000005;

    struct intersection_t result;
    result.intersects = false;
    result.end_point_intersection = -1;

    Eigen::Vector3d great_circle_1_normal = p1.cross(p2).normalized();
    Eigen::Vector3d great_circle_2_normal = p3.cross(p4).normalized();

    Eigen::Vector3d p3_proj_dir = -great_circle_1_normal.dot(p3) * great_circle_1_normal;
    Eigen::Vector3d p4_proj_dir = -great_circle_1_normal.dot(p4) * great_circle_1_normal;

    if (p3_proj_dir.dot(p4_proj_dir) > 0)
        return result;

    Eigen::Vector3d p1_proj_dir = -great_circle_2_normal.dot(p1) * great_circle_2_normal;
    Eigen::Vector3d p2_proj_dir = -great_circle_2_normal.dot(p2) * great_circle_2_normal;

    if (p1_proj_dir.dot(p2_proj_dir) > 0)
        return result;

    double g1_norm = great_circle_1_normal.norm();
    double g2_norm = great_circle_2_normal.norm();

    great_circle_1_normal /= g1_norm;
    great_circle_2_normal /= g2_norm;

    if (1.0 - great_circle_1_normal.dot(great_circle_2_normal) < epsilon) // warning: epsilon might not be a valid thresh value ebcause we stopped normalizing
        return result;


    p3_proj_dir /= g1_norm;
    p4_proj_dir /= g1_norm;

    p1_proj_dir /= g2_norm;
    p2_proj_dir /= g2_norm;

    if (p1_proj_dir.norm() < epsilon)
    {
        result.end_point_intersection = 0;
        return result; // at endpoint 0
    }

    if (p2_proj_dir.norm() < epsilon)
    {
        result.end_point_intersection = 1;
        return result; // at endpoint 1
    }

    if (p3_proj_dir.norm() < epsilon)
    {
        result.end_point_intersection = 2;
        return result; // at endpoint 2
    }

    if (p4_proj_dir.norm() < epsilon)
    {
        result.end_point_intersection = 3;
        return result; // at endpoint 3
    }




    double t = -p1.dot(great_circle_2_normal) / great_circle_2_normal.dot((p2 - p1));
    Eigen::Vector3d intersection = p1 + (p2 - p1) * t;
    intersection.normalize();
    Eigen::Vector3d antipodal_intersection = -intersection;

    double theta_1_1 = std::acos(p1.dot(intersection));
    double theta_2_1 = std::acos(p2.dot(intersection));
    double theta_3_1 = std::acos(p1.dot(antipodal_intersection));
    double theta_4_1 = std::acos(p2.dot(antipodal_intersection));

    Eigen::Vector3d plane_1_intersection;
    if (theta_1_1 + theta_2_1 < theta_3_1 + theta_4_1)
        plane_1_intersection = intersection;
    else
        plane_1_intersection = antipodal_intersection;

    double theta_1_2 = std::acos(p3.dot(intersection));
    double theta_2_2 = std::acos(p4.dot(intersection));
    double theta_3_2 = std::acos(p3.dot(antipodal_intersection));
    double theta_4_2 = std::acos(p4.dot(antipodal_intersection));

    Eigen::Vector3d plane_2_intersection;
    if (theta_1_2 + theta_2_2 < theta_3_2 + theta_4_2)
        plane_2_intersection = intersection;
    else
        plane_2_intersection = antipodal_intersection;


    if (plane_1_intersection.dot(plane_2_intersection) < 0)
        return result; // don't intersect

    result.location = plane_1_intersection;
    result.intersects = true;
    return result;
}

intersection_t spherical_segments_intersect_faster(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const Eigen::Vector3d& p3, const Eigen::Vector3d& p4)
{
    double epsilon = 0.000000005;
    
    struct intersection_t result;
    result.intersects = false;
    result.end_point_intersection = -1;

    Eigen::Vector3d great_circle_1_normal = p1.cross(p2); // .normalized();
    Eigen::Vector3d great_circle_2_normal = p3.cross(p4); // .normalized();

    Eigen::Vector3d p3_proj_dir = -great_circle_1_normal.dot(p3) * great_circle_1_normal;
    Eigen::Vector3d p4_proj_dir = -great_circle_1_normal.dot(p4) * great_circle_1_normal;

    if (p3_proj_dir.dot(p4_proj_dir) > 0)
        return result;

    Eigen::Vector3d p1_proj_dir = -great_circle_2_normal.dot(p1) * great_circle_2_normal;
    Eigen::Vector3d p2_proj_dir = -great_circle_2_normal.dot(p2) * great_circle_2_normal;

    if (p1_proj_dir.dot(p2_proj_dir) > 0)
        return result;

    double g1_norm = great_circle_1_normal.norm();
    double g2_norm = great_circle_2_normal.norm();

    great_circle_1_normal /= g1_norm;
    great_circle_2_normal /= g2_norm;

    if (1.0 - great_circle_1_normal.dot(great_circle_2_normal) < epsilon) // warning: epsilon might not be a valid thresh value ebcause we stopped normalizing
        return result;


    p3_proj_dir /= g1_norm;
    p4_proj_dir /= g1_norm;

    p1_proj_dir /= g2_norm;
    p2_proj_dir /= g2_norm;

    if (p1_proj_dir.norm() < epsilon)
    {
        result.end_point_intersection = 0;
        return result; // at endpoint 0
    }

    if (p2_proj_dir.norm() < epsilon)
    {
        result.end_point_intersection = 1;
        return result; // at endpoint 1
    }

    if (p3_proj_dir.norm() < epsilon)
    {
        result.end_point_intersection = 2;
        return result; // at endpoint 2
    }

    if (p4_proj_dir.norm() < epsilon)
    {
        result.end_point_intersection = 3;
        return result; // at endpoint 3
    }




    double t = -p1.dot(great_circle_2_normal) / great_circle_2_normal.dot((p2 - p1));
    Eigen::Vector3d intersection = p1 + (p2 - p1) * t;
    intersection.normalize();
    Eigen::Vector3d antipodal_intersection = -intersection;

    double theta_1_1 = std::acos(p1.dot(intersection));
    double theta_2_1 = std::acos(p2.dot(intersection));
    double theta_3_1 = std::acos(p1.dot(antipodal_intersection));
    double theta_4_1 = std::acos(p2.dot(antipodal_intersection));

    Eigen::Vector3d plane_1_intersection;
    if (theta_1_1 + theta_2_1 < theta_3_1 + theta_4_1)
        plane_1_intersection = intersection;
    else
        plane_1_intersection = antipodal_intersection;

    double theta_1_2 = std::acos(p3.dot(intersection));
    double theta_2_2 = std::acos(p4.dot(intersection));
    double theta_3_2 = std::acos(p3.dot(antipodal_intersection));
    double theta_4_2 = std::acos(p4.dot(antipodal_intersection));

    Eigen::Vector3d plane_2_intersection;
    if (theta_1_2 + theta_2_2 < theta_3_2 + theta_4_2)
        plane_2_intersection = intersection;
    else
        plane_2_intersection = antipodal_intersection;


    if (plane_1_intersection.dot(plane_2_intersection) < 0)
        return result; // don't intersect

    result.location = plane_1_intersection;
    result.intersects = true;
    return result;
}

region_splitting_info compute_regions_for_curve(const closed_spherical_curve_t& spherical_curve)
{
    projection_intersection_info proj_info = find_all_spherical_intersections(spherical_curve);
    region_splitting_info region_splitting = create_regions_with_relative_wn(proj_info.curve_ids, proj_info.turn_left_index, proj_info.extended_curve, proj_info.num_intersections);
    return region_splitting;
}

std::vector<int> relative_wn_from_adjacency(const std::map<int, std::set<std::pair<int, int>>>& neighborhood_map)
{
    int num_regions = neighborhood_map.size();
    std::vector<int> relative_wn(num_regions, 0);
    if (num_regions == 0)
        return relative_wn;
    std::vector<bool> visited(num_regions, false);
    std::queue<std::pair<int, int>> bfs_queue;
    bfs_queue.push({ 0,0, });
    while (!bfs_queue.empty())
    {
        std::pair<int, int> region_wn = bfs_queue.front();
        bfs_queue.pop();
        int current_region = region_wn.first;
        int current_wn = region_wn.second;
     
        visited[current_region] = true;

        relative_wn[current_region] = current_wn;
        const std::set<std::pair<int, int>>& neighbors = neighborhood_map.at(current_region);
        for (std::set<std::pair<int, int>>::const_iterator it = neighbors.begin(); it != neighbors.end(); it++)
        {
            int neighboring_region_id = it->first;
            int neighboring_direction = it->second;
            if (!visited[neighboring_region_id])
                bfs_queue.push({ neighboring_region_id, current_wn + neighboring_direction });
        }
    }
    return relative_wn;
}

curve_net_sampler run_region_splitting(const std::vector<patch_t>& closed_patches, const Eigen::MatrixXd& query_points)
{
    std::cout << "Region splitting on " << closed_patches.size() << " patches and " << query_points.rows() << " query points" << std::endl;
    curve_net_sampler sampler;
    sampler.patches.resize(closed_patches.size());

    for (int i = 0; i < closed_patches.size(); i++)
        sampler.patches[i] = closed_patches[i].curve;

    sampler.sample_points.resize(query_points.rows());
    sampler.areas.resize(closed_patches.size());
    sampler.rays.resize(closed_patches.size());
    sampler.rel_wn.resize(closed_patches.size());
    sampler.spherical_regions.resize(closed_patches.size());

    for (int i = 0; i < closed_patches.size(); i++)
    {
        sampler.areas[i].resize(query_points.rows());
        sampler.rays[i].resize(query_points.rows());
        sampler.rel_wn[i].resize(query_points.rows());
        sampler.spherical_regions[i].resize(query_points.rows());
    }

    constexpr int max_threads = 18;
    omp_set_num_threads(max_threads);

#pragma omp parallel for num_threads(max_threads)
    for (int i = 0; i < query_points.rows(); i++)
    {
        for (int j = 0; j < closed_patches.size(); j++)
        {
            Eigen::Vector3d q = query_points.row(i);
            const patch_t& patch = closed_patches[j];
            assert(!patch.is_open);
            region_weighted_rays_info weighted_rays = get_weighted_rays(patch.curve, q);

            sampler.areas[j][i] = weighted_rays.areas;
            sampler.rays[j][i] = weighted_rays.rays;
            sampler.rel_wn[j][i] = weighted_rays.relative_wn;
            sampler.spherical_regions[j][i] = weighted_rays.polygonal_regions;
        }

        sampler.sample_points[i] = query_points.row(i);
    }

    sampler.insideness.resize(closed_patches.size());
    std::fill(sampler.insideness.begin(), sampler.insideness.end(), 1.0);
    std::cout << "Region splitting done" << std::endl;
    return sampler;
}
region_weighted_rays_info get_weighted_rays(const closed_curve_t& curve, const Eigen::Vector3d& point)
{
    surface_config config;
    return get_weighted_rays_with_config(curve, point, config);
}
region_weighted_rays_info get_weighted_rays_with_config(const closed_curve_t& curve, const Eigen::Vector3d& point, const surface_config& config)
{
    closed_spherical_curve_t spherical_curve = project_to_sphere(curve, point);

    region_splitting_info region_info;

    if (config.perform_self_intersections)
    {
        region_info = compute_regions_for_curve(spherical_curve);
    }
    else
    {
        // sometimes we can know this analytically.
        // for now this is when the boundary is a circle
        projection_intersection_info proj_info;
        proj_info.extended_curve = spherical_curve;
        closed_spherical_curve_t& extended_curve = proj_info.extended_curve; // curve with intersections at the end
        int current_curve_length = spherical_curve.rows();
        int next_index = current_curve_length;

        std::vector<int>& curve_ids = proj_info.curve_ids; // using a vector will be HIGHLY inefficient. Look into std::forward_list
        curve_ids.resize(spherical_curve.rows());
        std::iota(curve_ids.begin(), curve_ids.end(), 0);

        proj_info.num_intersections = 0;

        region_info = create_regions_with_relative_wn(proj_info.curve_ids, proj_info.turn_left_index, proj_info.extended_curve, proj_info.num_intersections);
    }
    


    const std::vector<closed_spherical_curve_t>& regions = region_info.regions;
    std::vector<double> areas;
    std::map<int, std::set<std::pair<int, int>>> adjusted_neighborhood_map;
    std::vector<int> sorted_relative_wn;
    std::vector<Eigen::Vector3d> rays;
    constexpr double area_min_size = 0.0008 / (4.0 * EIGEN_PI);
    int first_valid_region = std::numeric_limits<int>::max();

    std::vector<bool> region_was_deleted(regions.size(), false);
    std::vector<int> region_post_deletion_id(regions.size(), 0);
    int current_post_deletion_id = 0;
    std::vector<closed_spherical_curve_t> valid_regions;
    for (int i = 0; i < regions.size(); i++)
    {
        double area = spherical_polygon_area(regions[i]);

        if (area > area_min_size)
        {
            areas.push_back(area);
            valid_regions.push_back(regions[i]);
            Eigen::Vector3d ray = find_ray_in_region(regions[i], 0);
            //ASSERT_RELEASE(is_inside_polygon(regions[i], ray), "ray not inside polygon");
            rays.push_back(ray);
            region_was_deleted[i] = false;
            region_post_deletion_id[i] = current_post_deletion_id;
            current_post_deletion_id++;
        }
        else
        {
            region_was_deleted[i] = true;
        }
    }
 
    int adjusted_neighbors_index = 0;
    for (int i = 0; i < regions.size(); i++)
    {
        if (region_was_deleted[i])
            continue;
        std::set<std::pair<int, int>> adjusted_neighbors;
        for (const auto& pair : region_info.neighborhood_map[i])
        {
            std::pair<int, int> p = pair;
            p.first = region_post_deletion_id[p.first];
            if (!region_was_deleted[pair.first])
            {
                adjusted_neighbors.insert(p);
            }
        }
        adjusted_neighborhood_map[adjusted_neighbors_index] = adjusted_neighbors;
        adjusted_neighbors_index++;
    }

    // sort from bigger area to smaller area
    std::vector<size_t> sorted_indices = sort_indexes_rev(areas);
    std::map<int, std::set<std::pair<int, int>>> sorted_neighborhood_map;
    std::vector<double> sorted_areas(areas.size(), 0);
    std::vector<Eigen::Vector3d> sorted_rays(rays.size());
    std::vector<closed_spherical_curve_t> sorted_regions(areas.size());
    std::map<int, int> new_sorted_id;
    for (int i = 0; i < sorted_indices.size(); i++)
    {
        new_sorted_id[sorted_indices[i]] = i;
    }


    for (int i = 0; i < sorted_indices.size(); i++)
    {
        std::set<std::pair<int, int>> neighbors = adjusted_neighborhood_map[sorted_indices[i]];

        std::set<std::pair<int, int>> new_neighbors;
        for (const auto& pair : neighbors)
        {
            std::pair<int, int> p;
            p.first = new_sorted_id[pair.first];
            p.second = pair.second;
            new_neighbors.insert(p);
        }
        sorted_neighborhood_map[i] = new_neighbors;
        sorted_areas[i] = areas[sorted_indices[i]];
        sorted_rays[i] = rays[sorted_indices[i]];
        sorted_regions[i] = valid_regions[sorted_indices[i]];
    }
    sorted_relative_wn = relative_wn_from_adjacency(sorted_neighborhood_map);
    region_weighted_rays_info w_region_info;
    w_region_info.areas = sorted_areas;
    w_region_info.rays = sorted_rays;
    w_region_info.relative_wn = sorted_relative_wn;
    w_region_info.polygonal_regions = sorted_regions;

    // find 10 rays in the region, pick the BEST one.
    //w_region_info.rays[0] = find_best_ray_in_region(regions[0]);

    return w_region_info;
}

closed_spherical_curve_t curve_data_from_ids(const std::vector<int>& ids, const closed_spherical_curve_t& curve_data)
{
    closed_spherical_curve_t output(ids.size(), 3);

    for (int i = 0; i < ids.size(); i++)
        output.row(i) = curve_data.row(ids[i]);
    return output;
}

std::map<int, std::set<std::pair<int, int>>> get_neighborhood_map(int num_regions, const std::vector<int>& regions_from_left, const std::vector<int>& regions_from_right)
{
    std::map<int, std::set<std::pair<int, int>>> n_map;

    for (int i = 0; i < num_regions; i++)
    {
        n_map[i] = std::set<std::pair<int, int>>();
    }

    // todo FIX THIS PROPERLY
    // they are neighbors if they have a shared EDGE.
    // shared edge with intersections is MISSING

    for (int i = 0; i < regions_from_left.size(); i++)
    {
        int index = i;
        int next_index = (i + 1) % regions_from_left.size();

        int l_1 = regions_from_left[index];
        int l_2 = regions_from_left[next_index];
        int r_1 = regions_from_right[index];
        int r_2 = regions_from_right[next_index];

        if (l_1 >= 0 && l_2 >= 0 && r_1 >= 0 && r_2 >= 0 && l_1 == l_2 && r_1 == r_2)
        {
            n_map[l_1].insert({ r_1, -1 });
            n_map[r_1].insert({ l_1, 1 });
        }
    }

    //for (int i = 0; i < regions_from_left.size(); i++)
    //{
    //    int l_i = regions_from_left[i];
    //    int r_i = regions_from_right[i];

    //    if (l_i >= 0 && r_i >= 0)
    //    {
    //        n_map[l_i].insert({ r_i, -1 });
    //        n_map[r_i].insert({ l_i, 1 });
    //    }
    //}
    return n_map;
}

projection_intersection_info find_all_spherical_intersections(const closed_spherical_curve_t& spherical_curve)
{
    projection_intersection_info projection_intersection;
    projection_intersection.extended_curve = spherical_curve;
    closed_spherical_curve_t& extended_curve = projection_intersection.extended_curve; // curve with intersections at the end
    int current_curve_length = spherical_curve.rows();
    int next_index = current_curve_length;

    std::vector<int>& curve_ids = projection_intersection.curve_ids; // using a vector will be HIGHLY inefficient. Look into std::forward_list
    curve_ids.resize(spherical_curve.rows());
    std::iota(curve_ids.begin(), curve_ids.end(), 0);

    if (spherical_curve.rows() == 3)
    {
        projection_intersection.num_intersections = 0;
        return projection_intersection;
    }

    std::vector<bool> is_less_quarter_segments(spherical_curve.rows());
    std::vector<int> segment_next_idx(spherical_curve.rows());
    std::vector<box> bounding_box(spherical_curve.rows());
    for (int i = 0; i < spherical_curve.rows(); i++)
    {
        int idx1 = i;
        int idx2 = (i + 1) % spherical_curve.rows();

        segment_next_idx[idx1] = idx2;
        /*       bool need_bb = spherical_curve.row(idx1).dot(spherical_curve.row(idx2)) >= 0.0;
               is_less_quarter_segments[idx1] = need_bb;

               if (need_bb)
               {
                   box bb;
                   bb.x_min = std::min(spherical_curve.coeff(idx1, 0), spherical_curve.coeff(idx2, 0));
                   bb.x_max = std::max(spherical_curve.coeff(idx1, 0), spherical_curve.coeff(idx2, 0));
                   bb.y_min = std::min(spherical_curve.coeff(idx1, 1), spherical_curve.coeff(idx2, 1));
                   bb.y_max = std::max(spherical_curve.coeff(idx1, 1), spherical_curve.coeff(idx2, 1));
                   bb.z_min = std::min(spherical_curve.coeff(idx1, 2), spherical_curve.coeff(idx2, 2));
                   bb.z_max = std::max(spherical_curve.coeff(idx1, 2), spherical_curve.coeff(idx2, 2));
                   bounding_box[idx1] = bb;
               }*/
    }

    // 0, 1, 2, 6,  3, 4,6, 5.
    // 0, 1, 2, 3, 4, 5 <=
    std::map<spherical_intersection_entrance, spherical_intersection_direction>& turn_left = projection_intersection.turn_left_index;

    for (int i = 0; i < curve_ids.size(); i++)
    {
        for (int j = i + 2; j < curve_ids.size(); j++)
        {
            int idx1 = i;
            int idx2 = (i + 1) % curve_ids.size();// segment_next_idx[idx1];//; 
            int idx3 = j;
            int idx4 = (j + 1) % curve_ids.size(); // segment_next_idx[idx3];

            // skip neighboring segments
            if ((idx3 - idx1) <= 1)
                continue;
            if (idx4 == 0 && idx1 == 0)
                continue;


            assert(idx1 != idx2 && idx1 != idx3 && idx1 != idx4);
            assert(idx2 != idx3 && idx2 != idx4);
            assert(idx3 != idx4);


            if (!(curve_ids[idx1] != curve_ids[idx2] && curve_ids[idx1] != curve_ids[idx3] && curve_ids[idx1] != curve_ids[idx4]))
                continue;
            if (!(curve_ids[idx2] != curve_ids[idx3] && curve_ids[idx2] != curve_ids[idx4]))
                continue;
            if (!(curve_ids[idx3] != curve_ids[idx4]))
                continue;

            /*   if (is_less_quarter_segments[curve_ids[idx1]] && is_less_quarter_segments[curve_ids[idx2]])
               {
                   const box& bb1 = bounding_box[curve_ids[idx1]];
                   const box& bb2 = bounding_box[curve_ids[idx2]];
                   if (bb1.x_max < bb2.x_min || bb2.x_max < bb1.x_min
                    || bb1.y_max < bb2.y_min || bb2.y_max < bb1.y_min
                    || bb1.z_max < bb2.z_min || bb2.z_max < bb1.z_min)
                   {
                       continue;
                   }
               }
   */

            intersection_t intersection = spherical_segments_intersect(extended_curve.row(curve_ids[idx1]), extended_curve.row(curve_ids[idx2]), 
                extended_curve.row(curve_ids[idx3]), extended_curve.row(curve_ids[idx4]));
            if (intersection.intersects)
            {
                if (intersection.end_point_intersection != -1)
                {
                    // it's an endpoint intersection
                    int id_of_endpoint;
                    switch (intersection.end_point_intersection)
                    {
                    case 0:
                        id_of_endpoint = idx1;
                        break;
                    case 1:
                        id_of_endpoint = idx2;
                        break;
                    case 2:
                        id_of_endpoint = idx3;
                        break;
                    case 3:
                        id_of_endpoint = idx4;
                        break;
                    default:
                        ASSERT_RELEASE(false, "impossible state");
                        break;
                    }
                    curve_ids.erase(curve_ids.begin() + id_of_endpoint);
                    ASSERT_RELEASE(false, "unimplemented");

                    next_index++;
                }
                else
                {
                    // 0, 1,7, 2, 3, 4, 7, 5, 6, 
                    // bb0, bb1, bb2, bb3, bb4, bb5, bb6
                    //bb0, bb1, bb7, bb2, bb3, bb4, bb7, bb5, bb6
                    curve_ids.insert(curve_ids.begin() + idx3 + 1, next_index);
                    curve_ids.insert(curve_ids.begin() + idx1 + 1, next_index);
                    /*
                         segment_next_idx.resize(curve_ids.size());
                         for (int k = 0; k < curve_ids.size(); k++)
                         {
                             int idx1 = k;
                             int idx2 = (k + 1) % curve_ids.size();

                             segment_next_idx[idx1] = curve_ids[idx2];
                         }*/


                         /*          const box& b1 = bounding_box[curve_ids[idx1]];
                                   bounding_box.push_back(b1);

                                   bool b1_less_than_quarter = is_less_quarter_segments[curve_ids[idx1]];
                                   is_less_quarter_segments.push_back(b1_less_than_quarter);*/

                    extended_curve.conservativeResize(extended_curve.rows() + 1, extended_curve.cols());
                    extended_curve.row(extended_curve.rows() - 1) = intersection.location;
                    //std::cout << intersection.location(0) << " " << intersection.location(1) << " " << intersection.location(2) << std::endl;
                    current_curve_length++;
                    next_index++;
                }
            }
        }
    }

    // For every intersection we just added
    // find the 4 neighbors (2 per segment)
    int num_new_segments = current_curve_length - spherical_curve.rows();
    std::vector<std::vector<int>> four_indices(num_new_segments);
    for (int i = 0; i < num_new_segments; i++)
    {
        int idx = spherical_curve.rows() + i;
        std::vector<int> indices;
        for (int j = 0; j < curve_ids.size(); j++)
        {
            if (indices.size() == 4)
            {
                break;
            }
            else if (idx == curve_ids[j])
            {
                indices.push_back(curve_ids[(j - 1) % curve_ids.size()]);
                indices.push_back(curve_ids[(j + 1) % curve_ids.size()]);
            }
        }

        ASSERT_RELEASE(indices.size() == 4, "Cannot complete left turn");
        four_indices[i] = indices;
    }

    // Creates a map where you can fetch how to "turn left" 
    for (int i = 0; i < num_new_segments; i++)
    {
        int idx = spherical_curve.rows() + i;
        std::vector<int> indices = four_indices[i];
        int idx1 = indices[0];
        int idx2 = indices[1];
        int idx3 = indices[2];
        int idx4 = indices[3];

        Eigen::Vector3d s1_dir = extended_curve.row(idx2) - extended_curve.row(idx1);
        Eigen::Vector3d s2_dir = extended_curve.row(idx4) - extended_curve.row(idx3);

        if (s1_dir.cross(s2_dir).dot(extended_curve.row(idx1)) > 0.0)
        {
            turn_left[{idx, idx1}] = { idx4, 1 };
            turn_left[{idx, idx2}] = { idx3, -1 };
            turn_left[{idx, idx3}] = { idx1, -1 };
            turn_left[{idx, idx4}] = { idx2, 1 };
        }
        else
        {
            turn_left[{idx, idx1}] = { idx3, -1 };
            turn_left[{idx, idx2}] = { idx4, 1 };
            turn_left[{idx, idx3}] = { idx2, 1 };
            turn_left[{idx, idx4}] = { idx1, -1 };
        }
    }

    ASSERT_RELEASE(turn_left.size() % 4 == 0, "ill formed intersections");

    projection_intersection.num_intersections = next_index - spherical_curve.rows();
    return projection_intersection;
}


// This code is quite hard to understand due to the multiple levels of indirections used in the data structures
region_splitting_info create_regions_with_relative_wn(const std::vector<int>& curve_ids, std::map<spherical_intersection_entrance, spherical_intersection_direction>& turn_left, const closed_curve_t& curve_data, int num_intersections)
{
    std::vector<closed_spherical_curve_t> regions;
    int curve_ids_len = curve_ids.size();
    int num_non_shared_pts = curve_ids.size() - 2 * num_intersections;
    int num_pts = num_non_shared_pts + num_intersections;

    // >= 0  region id occupying the boundary point
    // -1 is an eligible start
    // -2 is not an eligible start

    std::vector<int> eligible_starts_left(curve_ids.size(), -1);
    std::vector<int> eligible_starts_right(curve_ids.size(), -1);

    for (int i = 0; i < curve_ids.size(); i++)
    {
        eligible_starts_left[i] = curve_ids[i] < num_non_shared_pts ? -1 : -2;
    }
    std::copy(eligible_starts_left.begin(), eligible_starts_left.end(), eligible_starts_right.begin());

    int current_region_id = 0;
    assert(num_intersections >= 0);
    if (num_intersections == 0)
    {
        std::vector<int> curve_ids_reversed(curve_ids.rbegin(), curve_ids.rend());
        regions.push_back(curve_data_from_ids(curve_ids, curve_data));
        regions.push_back(curve_data_from_ids(curve_ids_reversed, curve_data));
    }
    else
    {
        auto first_eligible_index_left = std::find(eligible_starts_left.begin(), eligible_starts_left.end(), -1);
        auto first_eligible_index_right = std::find(eligible_starts_right.begin(), eligible_starts_right.end(), -1);
        bool left_eligible = first_eligible_index_left != eligible_starts_left.end();
        bool right_eligible = first_eligible_index_right != eligible_starts_right.end();
        while (left_eligible || right_eligible)
        {
            std::vector<int> current_region;
            int start_index = -1;
            int current_direction = 0; // right
            if (left_eligible)
            {
                start_index = std::distance(eligible_starts_left.begin(), first_eligible_index_left);
                eligible_starts_left[start_index] = current_region_id;
                current_direction = 1;
            }
            else
            {
                start_index = std::distance(eligible_starts_right.begin(), first_eligible_index_right);
                eligible_starts_right[start_index] = current_region_id;
                current_direction = -1;
            }

            int current_index = start_index;
            bool double_intersection = false;
            while (true)
            {
                if (!double_intersection)
                    current_index = nmod((current_index + current_direction), curve_ids_len);

                double_intersection = false;

                bool is_at_intersection = curve_ids[current_index] >= num_non_shared_pts;
                if (is_at_intersection)
                {
                    current_region.push_back(current_index);
                    int previous_index = -1;
                    if (current_region.size() >= 2)
                        previous_index = current_region[current_region.size() - 2];
                    else
                        previous_index = nmod(current_index - current_direction, curve_ids_len);

                    spherical_intersection_entrance entrance = { curve_ids[current_index], curve_ids[previous_index] };
                    spherical_intersection_direction direction = turn_left[entrance];
                    current_direction = direction.direction;

                    if (direction.exit_id >= num_non_shared_pts)
                        double_intersection = true;

                    auto exit_p = std::find(curve_ids.begin(), curve_ids.end(), direction.exit_id);
                    assert(exit_p != curve_ids.end());
                    current_index = nmod(std::distance(curve_ids.begin(), exit_p), curve_ids_len);
                }

                if (current_direction == 1)
                {
                    eligible_starts_left[current_index] = current_region_id;
                }
                else
                {
                    eligible_starts_right[current_index] = current_region_id;
                }

                if (!double_intersection)
                    current_region.push_back(current_index);

                if (current_index == start_index)
                    break;
            }

            for (int k = 0; k < current_region.size(); k++)
                current_region[k] = curve_ids[current_region[k]];


           /* if (current_region.size() > curve_ids.size())
                std::cout << "WARN: current_region.size() > curve_ids.size() => " << current_region.size() << " > " << curve_ids.size() << std::endl;*/
            regions.push_back(curve_data_from_ids(current_region, curve_data));
            current_region_id++;
            first_eligible_index_left = std::find(eligible_starts_left.begin(), eligible_starts_left.end(), -1);
            left_eligible = first_eligible_index_left != eligible_starts_left.end();
            right_eligible = false;
            if (!left_eligible)
            {
                first_eligible_index_right = std::find(eligible_starts_right.begin(), eligible_starts_right.end(), -1);
                right_eligible = first_eligible_index_right != eligible_starts_right.end();
            }
        }
    }

    region_splitting_info region_splitting;
    region_splitting.regions = regions;
    if (num_intersections == 0)
    {
        std::set<std::pair<int, int>> s1;
        std::set<std::pair<int, int>> s2;
        s1.emplace(1, -1);
        s2.emplace(0, 1);
        region_splitting.neighborhood_map.insert({ 0, s1 });
        region_splitting.neighborhood_map.insert({ 1, s2 });
    }
    else
    {
        region_splitting.neighborhood_map = get_neighborhood_map(regions.size(), eligible_starts_left, eligible_starts_right);
    }

    return region_splitting;
}

Eigen::Vector3d find_best_ray_in_region(const closed_spherical_curve_t& curve)
{
    int num_to_search = std::min<int>(100, curve.rows());

    Eigen::Vector3d best_ray;
    double best_ray_distance = 0;
    for (int i = 0; i < num_to_search; i++)
    {
        Eigen::Vector3d ray = find_ray_in_region(curve, i);
        double distance = find_minimum_distance_to_segment_all_polygons({ curve }, ray);

        if (distance > best_ray_distance)
        {
            best_ray_distance = distance;
            best_ray = ray;
        }
    }

    return best_ray;
}

Eigen::Vector3d find_ray_in_region(const closed_spherical_curve_t& curve, int start_index)
{
    int first_index = start_index;
    int second_index = (start_index + 1) % curve.rows();
    Eigen::Vector3d centroid = curve.colwise().mean();
    centroid.normalize();
    if (curve.rows() == 3)
        return centroid;

    double initial_point_weight = 0.2;
    Eigen::Vector3d initial_point = initial_point_weight * centroid + 0.5 * (1.0 - initial_point_weight) * curve.row(first_index).transpose() + 0.5 * (1.0 - initial_point_weight) * curve.row(second_index).transpose();
    initial_point.normalize();
    double eps = 0.000000005;
    Eigen::Vector3d great_circle_normal = initial_point.cross(centroid);
    great_circle_normal.normalize();
    std::vector<Eigen::Vector3d> intersections;
    std::vector<Eigen::Vector3d> intersection_directions;

    for (int i = 0; i < curve.rows(); i++)
    {
        int first = i;
        int second = (i + 1) % curve.rows();
        Eigen::Vector3d p1 = curve.row(first);
        Eigen::Vector3d p2 = curve.row(second);

        Eigen::Vector3d p1_proj_dir = -great_circle_normal.dot(p1) * great_circle_normal;
        Eigen::Vector3d p2_proj_dir = -great_circle_normal.dot(p2) * great_circle_normal;

        if (p1_proj_dir.dot(p2_proj_dir) < 0.0)
        {
            if (p1_proj_dir.norm() < eps)
            {
                intersections.push_back(p1);
            }
            else if (p2_proj_dir.norm() < eps)
            {
                intersections.push_back(p2);
            }
            else
            {
                double t = -(p1.dot(great_circle_normal)) / (p2 - p1).dot(great_circle_normal);
                Eigen::Vector3d intersection = p1 + (p2 - p1) * t;
                intersections.push_back(intersection);
            }

            Eigen::Vector3d intersection_direction = curve.row(second) - curve.row(first);
            intersection_direction.normalize();
            intersection_directions.push_back(intersection_direction);
        }
    }

    assert(!intersections.empty());
    if (intersections.empty())
        throw std::runtime_error("no intersections found for centroid marching algorithm");

    Eigen::Vector3d first_basis = initial_point;
    Eigen::Vector3d second_basis = great_circle_normal.cross(initial_point);
    second_basis.normalize();
    Eigen::Matrix3d unit_circle_to_3d;
    unit_circle_to_3d.setZero();
    unit_circle_to_3d.col(0) = first_basis;
    unit_circle_to_3d.col(1) = second_basis;

    std::vector<double> thetas(intersections.size(), 0);
    std::vector<bool> directions(intersections.size(), 0);
    for (int i = 0; i < intersections.size(); i++)
    {
        Eigen::Vector3d intersection_point = intersections[i];
        Eigen::Vector3d intersection_direction = intersection_directions[i];

        double angle = std::acos(intersection_point.dot(initial_point));
        if (great_circle_normal.dot(initial_point.cross(intersection_point)) < 0)
            angle = 2.0 * EIGEN_PI - angle;

        Eigen::Vector3d tangent = unit_circle_to_3d * Eigen::Vector3d(std::sin(angle), -std::cos(angle), 0);
        directions[i] = intersection_point.dot(intersection_direction.cross(tangent)) < 0;
        thetas[i] = angle;
    }

    std::vector<size_t> indices_sorted = sort_indexes(thetas);
    std::vector<bool> directions_sorted(directions.size(), false);
    std::vector<Eigen::Vector3d> intersections_sorted(intersections.size());

    for (int i = 0; i < directions.size(); i++)
        directions_sorted[i] = directions[indices_sorted[i]];

    for (int i = 0; i < intersections.size(); i++)
        intersections_sorted[i] = intersections[indices_sorted[i]];

    ASSERT_RELEASE(directions_sorted.size() > 0, "insufficient directions");
    ASSERT_RELEASE(directions_sorted.size() % 2 == 0, "centroid algorithm finds odd number of intersections");

    std::vector<std::pair<int, int>> interior_regions;
    for (int i = 0; i < intersections.size(); i++)
    {
        int first = i;
        int second = (i + 1) % intersections.size();
        // direction_sorted[i] is true if we're entering a region
        if (directions_sorted[first] && !directions_sorted[second])
        {
            int index_first = first;
            int index_second = second;
            interior_regions.emplace_back(first, second);
        }
    }

    ASSERT_RELEASE(interior_regions.size() > 0, "no interior regions");
    std::vector<double> interior_regions_length(interior_regions.size());
    for (int i = 0; i < interior_regions.size(); i++)
    {
        int first = interior_regions[i].first;
        int second = interior_regions[i].second;
        double angle = std::acos(intersections_sorted[first].dot(intersections_sorted[second]));
        if (great_circle_normal.dot(intersections_sorted[first].cross(intersections_sorted[second])) < 0)
        {
            angle = 2.0 * EIGEN_PI - angle;
        }
        interior_regions_length[i] = angle;
    }

    std::vector<size_t> interior_regions_indices = sort_indexes(interior_regions_length);
    std::vector<double> sorted_interior_regions_lengths(interior_regions_length.size());
    std::vector<std::pair<int, int>> sorted_interior_regions(interior_regions_length.size());
    for (int i = 0; i < interior_regions_length.size(); i++)
        sorted_interior_regions_lengths[i] = interior_regions_length[interior_regions_indices[i]];
    for (int i = 0; i < interior_regions_length.size(); i++)
        sorted_interior_regions[i] = interior_regions[interior_regions_indices[i]];


    int index_first = sorted_interior_regions.back().first;
    int index_second = sorted_interior_regions.back().second;

    Eigen::Vector3d interior_point = (intersections_sorted[index_first] + intersections_sorted[index_second]) / 2.0;
    interior_point.normalize();

    if (sorted_interior_regions_lengths.back() > EIGEN_PI)
        interior_point = -interior_point;

    return interior_point;
}



closed_curve_t keep_unique_points(const closed_curve_t& curve)
{
    double epsilon = 0.00005;
    std::vector<Eigen::Vector3d> points;
    std::vector<bool> is_unique(curve.rows(), true);
    closed_spherical_curve_t new_curve;

    int removed_points = 0;
    for (int i = 0; i < curve.rows(); i++)
    {
        points.push_back(curve.row(i));
        if (!is_unique[i]) continue;
        bool found_duplicate = false;
        for (int j = i + 1; j < curve.rows(); j++)
        {
            if (!is_unique[j]) continue;
            double dis = (curve.row(i) - curve.row(j)).norm();
            if (dis < epsilon)
            {
                found_duplicate = true;
                is_unique[i] = false;
                is_unique[j] = false;
                removed_points += 2;
                break;
            }
        }
    }

    new_curve.resize(curve.rows() - removed_points, 3);
    int acc = 0;
    for (int i = 0; i < is_unique.size(); i++)
        if (is_unique[i])
            new_curve.row(acc++) = points[i];

    return new_curve;
}

closed_spherical_curve_t remove_duplicates(const closed_spherical_curve_t& curve)
{
    double epsilon = 0.00005;
    std::vector<Eigen::Vector3d> points;
    closed_spherical_curve_t new_curve;

    for (int i = 0; i < curve.rows(); i++)
    {
        bool found_duplicate = false;
        for (int j = i + 1; j < curve.rows(); j++)
        {
            double dis = (curve.row(i) - curve.row(j)).norm();
            if (dis < epsilon)
            {
                found_duplicate = true;
                break;
            }
        }

        if (!found_duplicate)
            points.push_back(curve.row(i));
    }

    new_curve.resize(points.size(), 3);
    for (int i = 0; i < points.size(); i++)
        new_curve.row(i) = points[i];

    return new_curve;
}

double find_minimum_distance_to_segment_all_polygons(const std::vector<closed_spherical_curve_t>& regions, const Eigen::Vector3d& point)
{
    double minimum_distance = 1.0;

    for (int i = 0; i < regions.size(); i++)
    {
        double distance = minimum_distance_to_segment(regions[i], point);
        if (distance < minimum_distance)
            minimum_distance = distance;
    }
    return minimum_distance;
}


// this function isnt exact and won't work at all if all the segments are 2pi away
// however if the goal is just to find if a segment is within epsilon of a point, it will work just fine
double minimum_distance_to_segment(const closed_spherical_curve_t& curve, const Eigen::Vector3d& point)
{
    double minimum_distance = 1.0;

    for (int i = 0; i < curve.rows(); i++)
    {
        int first_index = i;
        int second_index = (i + 1) % curve.rows();

        Eigen::Vector3d p1 = curve.row(first_index);
        Eigen::Vector3d p2 = curve.row(second_index);
        Eigen::Vector3d mid_point = (p1 + p2).normalized();
        // ignore segments  2pi away
        if (mid_point.dot(point) > 0)
        {
            Eigen::Vector3d great_circle = p1.cross(p2).normalized();
            Eigen::Vector3d proj_dir = -great_circle.dot(point) * great_circle;
            double distance = proj_dir.norm();
            if (distance < minimum_distance)
                minimum_distance = distance;
        }

    }

    return minimum_distance;
}

void compute_all_sphere_intersections(const patch_t& patch, const Eigen::MatrixXd& query_points)
{
    std::vector<Eigen::Vector3d> qs;
    std::vector<std::vector<Eigen::Vector3d>> int_points;

    for (int i = 0; i < query_points.rows(); i++)
    {
        closed_spherical_curve_t proj_curve = project_to_sphere(patch.curve, query_points.row(i * 7));
        projection_intersection_info proj_info = find_all_spherical_intersections(proj_curve);

        if (proj_info.num_intersections == 0)
        {

            std::vector<Eigen::Vector3d> intersections;
            for (int k = 0; k < proj_info.num_intersections; k++)
            {
                intersections.push_back(proj_info.extended_curve.row(proj_curve.rows() + k));
            }

            qs.push_back(query_points.row(i));
            int_points.push_back(intersections);
            if (qs.size() >= 200)
                break;
        }
        std::cout << i * 7 << " " << query_points.rows() << std::endl;
    }	
    Eigen::IOFormat matlab_fmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
    std::ofstream outfile("proj_int_points.m");
    Eigen::MatrixXd query_mat(qs.size(), 3);

    for (int i = 0; i < qs.size(); i++)
    {
        query_mat.row(i) = qs[i];
    }

    outfile << "query_points = " << query_mat.format(matlab_fmt) << ";" << std::endl;

    for (int i = 0; i < int_points.size(); i++)
    {
        Eigen::MatrixXd intersections_pts(int_points[i].size(), 3);
        for (int k = 0; k < int_points[i].size(); k++)
            intersections_pts.row(k) = int_points[i][k];
        outfile << "intersections{"  << std::to_string(i+1) << "} = " << intersections_pts.format(matlab_fmt) << ";" << std::endl;
    }

    outfile.close();
}

int largest_segment(const closed_spherical_curve_t& curve)
{
    double largest_segment = 0;
    int largest_segment_id = 0;

    for (int i = 0; i < curve.rows(); i++)
    {
        int first_index = i;
        int second_index = (i + 1) % curve.rows();


        Eigen::Vector3d p1 = curve.row(first_index);
        Eigen::Vector3d p2 = curve.row(second_index);

        double distance = (p1 - p2).norm();

        if (largest_segment < distance)
        {
            largest_segment_id = first_index;
            largest_segment = distance;
        }
    }

    return largest_segment;
}
double parameter_t_at_index(const closed_curve_t& curve, int largest_index)
{
    double total_curve_length = 0;

    double distance_at_t = 0.0;
    for (int i = 0; i < curve.rows(); i++)
    {
        int first_index = i;
        int second_index = (i + 1) % curve.rows();


        Eigen::Vector3d p1 = curve.row(first_index);
        Eigen::Vector3d p2 = curve.row(second_index);

        double distance = (p1 - p2).norm();


        if (largest_index == first_index)
        {
            distance_at_t = distance;
            break;
        }
            

        total_curve_length += distance;
    }

    return distance_at_t;

}

int find_region_index(const std::vector<space_curve_t>& regions, const Eigen::Vector3d& point)
{
    for (int i = 0; i < regions.size(); i++)
    {
        if (is_inside_polygon(regions[i], point))
            return i;
    }
    
    throw std::runtime_error("Point not found in any region!");
}

region_weighted_rays_info decompose_regions_fast(const std::vector<space_curve_t>& boundaries, const Eigen::Vector3d& point)
{
    typedef CGAL::Exact_predicates_exact_constructions_kernel         Kernel;
    typedef CGAL::Arr_geodesic_arc_on_sphere_traits_2<Kernel>         Geom_traits;
    typedef Geom_traits::Point_2                                      Point;
    typedef Geom_traits::Curve_2                                      Curve;
    typedef CGAL::Arr_spherical_topology_traits_2<Geom_traits>        Topol_traits;
    typedef CGAL::Arrangement_on_surface_2<Geom_traits, Topol_traits> Arrangement;
    typedef CGAL::Arr_naive_point_location<Arrangement>         Naive_pl;

    region_weighted_rays_info info;
    Geom_traits traits;
    Arrangement arr(&traits);
    auto ctr_p = traits.construct_point_2_object();
    auto ctr_cv = traits.construct_curve_2_object();

    std::list<Curve> arcs;
    for (int boundary_index = 0; boundary_index < boundaries.size(); boundary_index++)
    {
        closed_spherical_curve_t spherical_curve = project_to_sphere(boundaries[boundary_index], point);
        for (int curve_index = 0; curve_index < spherical_curve.rows(); curve_index++)
        {
            int first_index = curve_index;
            int second_index = (curve_index + 1) % spherical_curve.rows();

            Eigen::RowVector3d f_p = spherical_curve.row(first_index);
            Eigen::RowVector3d s_p = spherical_curve.row(second_index);
            arcs.push_back(ctr_cv(ctr_p(f_p(0), f_p(1), f_p(2)), ctr_p(s_p(0), s_p(1), s_p(2))));
        }
    }
    std::list<Point> pts;

    // Perform the sweep and obtain the subcurves.
   // CGAL::compute_intersection_points(arcs.begin(), arcs.end(), std::back_inserter(pts));
    CGAL::insert(arr, arcs.begin(), arcs.end());

    std::vector<bool> is_inner_ccb;

    typename Arrangement::Face_const_iterator fit;
    std::map<Arrangement::Ccb_halfedge_const_circulator, int> halfedge_map;
    int face_acc = 0;
    for(fit = arr.faces_begin(); fit != arr.faces_end(); ++fit)
    {
        if (fit->is_unbounded())
        {
            throw std::runtime_error("Unbounded face!");
        }

        std::vector<Eigen::Vector3d> points;
        is_inner_ccb.push_back(!fit->has_outer_ccb());
       
    	for (auto ici = fit->inner_ccbs_begin(); ici != fit->inner_ccbs_end(); ici++)
    	{
    		Arrangement::Ccb_halfedge_const_circulator h = *ici;
    		Arrangement::Ccb_halfedge_const_circulator start_h = *ici;

    		do
    		{
                halfedge_map[h] = face_acc;
    			Arrangement::Vertex_const_handle ve_handle = h->target();
    			Arrangement::Vertex v = *ve_handle;
    			Point p = v.point();

                double x = CGAL::to_double(p.dx());
                double y = CGAL::to_double(p.dy());
                double z = CGAL::to_double(p.dz());


                Eigen::Vector3d vec3(x,y,z);

                points.push_back(vec3);

    			h = h->next();
    		} while (start_h != h);
    	}

        ASSERT_RELEASE(!points.empty() || fit->outer_ccbs_begin() != fit->outer_ccbs_end(), "Duplicate or no CCB for this face");

    	for (auto ici = fit->outer_ccbs_begin(); ici != fit->outer_ccbs_end(); ici++)
    	{
    		Arrangement::Ccb_halfedge_const_circulator h = *ici;
    		Arrangement::Ccb_halfedge_const_circulator start_h = *ici;

    		do
    		{
                halfedge_map[h] = face_acc;
    			Arrangement::Vertex_const_handle ve_handle = h->target();
    			Arrangement::Vertex v = *ve_handle;
    			Point p = v.point();

                double x = CGAL::to_double(p.dx());
                double y = CGAL::to_double(p.dy());
                double z = CGAL::to_double(p.dz());
                
                Eigen::Vector3d vec3(x,y,z);
                points.push_back(vec3);

    			h = h->next();
    		} while (start_h != h);
    	}

        Eigen::MatrixXd region(points.size(), 3);

        for (int p_idx = 0; p_idx < points.size(); p_idx++)
            region.row(p_idx) = points[p_idx];
        info.polygonal_regions.push_back(region);
        info.areas.push_back(spherical_polygon_area(region));
        face_acc++;
    }

    std::map<int, std::vector<Arrangement::Ccb_halfedge_const_circulator>> face_to_h_map;

    for (auto const& p : halfedge_map)
    {
        const Arrangement::Ccb_halfedge_const_circulator& h = p.first;
        int face_id = p.second;
        if (face_to_h_map.find(face_id) == face_to_h_map.end())
            face_to_h_map[face_id] = { h };
        else
            face_to_h_map[face_id].push_back(h);
    }

    int num_faces = face_acc;

    std::map<int, std::set<std::pair<int, int>>> adjacency;
    for (int i = 0; i < num_faces; i++)
    {
        const std::vector<Arrangement::Ccb_halfedge_const_circulator>& hs = face_to_h_map.at(i);
        std::map<int, bool> encountered;

        for (int j = 0; j < num_faces; j++)
            encountered[j] = false;

        std::set<std::pair<int, int>> neighbors;
        for (int j = 0; j < hs.size(); j++)
        {
            const Arrangement::Ccb_halfedge_const_circulator& h = hs[j];
            int neighbor_face = halfedge_map[h->twin()];
            if (!encountered[neighbor_face])
            {
                encountered[neighbor_face] = true;
                bool dir_l_r = h->direction() == CGAL::Arr_halfedge_direction::ARR_LEFT_TO_RIGHT;
          
         /*       if (h->is_on_outer_ccb())
                    dir_l_r = !dir_l_r;*/
                neighbors.insert({ neighbor_face ,  dir_l_r ? -1 : 1 });
            }            
        }
        adjacency[i] = neighbors;
    }

    info.relative_wn = relative_wn_from_adjacency(adjacency);


    std::vector<int> to_remove;
    for (int i = 0; i < info.areas.size(); i++)
    {
        if (info.areas[i] <= 1e-4)
        {
            to_remove.push_back(i);
        }
    }

    std::sort(to_remove.begin(), to_remove.end(), std::greater<int>());

    for (int i = 0; i < to_remove.size(); i++)
    {
        info.areas.erase(info.areas.begin() + to_remove[i]);
        info.polygonal_regions.erase(info.polygonal_regions.begin() + to_remove[i]);
        info.relative_wn.erase(info.relative_wn.begin() + to_remove[i]);
    }


    return info;
}

