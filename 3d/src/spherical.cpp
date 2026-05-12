#include "spherical.h"
#include "region_splitting.h"
#include <iostream>

closed_spherical_curve_t project_to_sphere(const closed_curve_t& curve, const Eigen::Vector3d& point)
{
	closed_spherical_curve_t spherical_curve(curve.rows(), 3);
	
	for (int i = 0; i < curve.rows(); i++)
	{
		Eigen::RowVector3d diff = curve.row(i) - point.transpose();
		spherical_curve.row(i) = diff.normalized();;
	}

	return spherical_curve;
}

lat_lon_curve_t cartesian_to_lat_lon(const closed_spherical_curve_t& cartesian)
{
	lat_lon_curve_t lat_lon(cartesian.rows(), 2);

	for (int i = 0; i < cartesian.rows(); i++)
	{
		double x = cartesian(i, 0);
		double y = cartesian(i, 1);
		double z = cartesian(i, 2);
		double theta = std::atan2(y, x);
		double phi = EIGEN_PI / 2.0 - std::acos(z);
		lat_lon(i, 0) = 180.0 * phi / EIGEN_PI;
		lat_lon(i, 1) = 360.0 * theta / (2.0 * EIGEN_PI);
	}
	return lat_lon;
}


// half-versed-sine
inline double hav(double x)
{
	return (1.0 - std::cos(x)) / 2.0;
}

// Ported directly from:
// https://github.com/lcx366/SphericalPolygon/blob/master/sphericalpolygon/excess_area.py
double spherical_polygon_excess(const lat_lon_curve_t& points)
{
	int n = points.rows();
	double sum_excess = 0;

	for (int i = 0; i < n; i++)
	{
		int current_index = i;
		int next_index = (i + 1) % n;
		Eigen::Vector2d p1 = points.row(current_index) * EIGEN_PI / 180.0;
		Eigen::Vector2d p2 = points.row(next_index) * EIGEN_PI / 180.0;
		double pdlat = p2(0) - p1(0);
		double pdlon = p2(1) - p1(1);
		double dlon = std::abs(pdlon);
		if (dlon < 1e-6 && std::abs(pdlat) < 1e-6)
			continue;

		if (dlon > EIGEN_PI)
			dlon = 2.0 * EIGEN_PI - dlon;
		if (pdlon < -EIGEN_PI)
			p2(1) = p2(1) + 2.0 * EIGEN_PI;
		if (pdlon > EIGEN_PI)
			p2(1) = p2(1) - 2.0 * EIGEN_PI;


		double havb = hav(pdlat) + std::cos(p1(0)) * std::cos(p2(0)) * hav(dlon);
		double b = 2.0 * std::asin(std::sqrt(havb));
		double a = EIGEN_PI / 2.0 - p1(0);
		double c = EIGEN_PI / 2.0 - p2(0);
		double s = 0.5 * (a + b + c);
		double t = std::tan(s/2.0) * std::tan((s-a)/2.0)*std::tan((s-b)/2.0)*std::tan((s-c)/2.0);
		double excess = 4.0 * std::atan(std::sqrt(std::abs(t)));
		if ((p2(1) - p1(1)) < 0)
			excess = -excess;

		sum_excess += excess;
	}

	return sum_excess;

}

double spherical_polygon_area(const closed_spherical_curve_t& points)
{
	lat_lon_curve_t lat_lon = cartesian_to_lat_lon(points);
	double area = spherical_polygon_excess(lat_lon);
	double eps = 2e-4;
	if (area < -eps)
		area += 4.0 * EIGEN_PI;
	else if (area < 0.0)
		return 0.0;
	return area / (4.0 * EIGEN_PI);
}

bool is_inside_polygon(const closed_spherical_curve_t& curve, const Eigen::Vector3d& point)
{
	Eigen::Vector3d antipodal_point = -point;
	Eigen::Vector3d plane_normal = point;
	
	double epsilon = 0.0005;
	Eigen::Vector3d point_on_polygon;
	
	// find point on polygon such that the great circle is not colinear with the edge it started from.
	bool found_start_point = false;
	double maximum_value = 0.0;
	Eigen::Vector3d point_on_polygon_candidate;

	for (int i = 0; i < curve.rows(); i++)
	{
		int first_index = i;
		int second_index = (i + 1) % curve.rows();
		point_on_polygon_candidate = ((curve.row(first_index) + curve.row(second_index))).normalized();

		Eigen::Vector3d great_circle_normal = point.cross(point_on_polygon_candidate).normalized();

		double first_point_dot = std::abs(curve.row(first_index).normalized().dot(great_circle_normal));
		double second_point_dot = std::abs(curve.row(second_index).normalized().dot(great_circle_normal));
		if (first_point_dot > epsilon && second_point_dot > epsilon)
		{
			point_on_polygon = point_on_polygon_candidate;
			found_start_point = true;
			break;
		}

		// do this just to actually get a point even in the most garbage cases
		double min = std::min(first_point_dot, second_point_dot);
		if (maximum_value < min)
		{
			maximum_value = min;
			point_on_polygon = point_on_polygon_candidate;
		}
			
	}

	//if (!found_start_point)
	//	std::cout << "Could not find a starting polygon point, took the most likely to be ok." << std::endl;

	Eigen::Vector3d point_at_pi_2 = point.cross(point_on_polygon).cross(point).normalized();

	closed_spherical_curve_t great_circle(4, 3);
	great_circle.row(0) = point;
	great_circle.row(1) = point_at_pi_2;
	great_circle.row(2) = antipodal_point;
	great_circle.row(3) = -point_at_pi_2;

	//Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ",\n", "[", "]", "[", "]");
	//std::cout << "great_circle" << std::endl;
	//std::cout << great_circle.format(HeavyFmt) << std::endl;

	//std::cout << "curve" << std::endl;
	//std::cout << curve.format(HeavyFmt) << std::endl;

	//std::cout << "the point" << std::endl;
	//std::cout << point << std::endl;

	//std::cout << "on polygon point" << std::endl;
	//std::cout << point_on_polygon << std::endl;


	// check if first intersection is from inside.
	for (int j = 0; j < great_circle.rows(); j++)
	{
		bool has_intersection = false;
		Eigen::Vector3d closest_intersection;
		double closest_intersection_distance = 2.0 * EIGEN_PI;
		Eigen::Vector3d closest_intersections_direction;
		Eigen::Vector3d closest_intersections_curve_dir;

		for (int i = 0; i < curve.rows(); i++)
		{
			int idx1 = i;
			int idx2 = (i + 1) % curve.rows();
			int idx3 = j;
			int idx4 = (j + 1) % great_circle.rows();


			intersection_t result = spherical_segments_intersect(curve.row(idx1), curve.row(idx2), great_circle.row(idx3), great_circle.row(idx4));
			if (result.intersects)
			{
				ASSERT_RELEASE(result.end_point_intersection == -1, "unsupported");
				Eigen::Vector3d great_circle_segment_start = great_circle.row(idx3);
				Eigen::Vector3d intersection_dir = great_circle_segment_start - result.location;
				Eigen::Vector3d curve_dir = curve.row(idx2) - curve.row(idx1);
				double intersection_distance = intersection_dir.norm();
				if (intersection_distance < closest_intersection_distance)
				{
					has_intersection = true;
					closest_intersection_distance = intersection_distance;
					closest_intersections_curve_dir = curve_dir;
					closest_intersection = result.location;
					closest_intersections_direction = intersection_dir.normalized();
				}
			}
		}

		if (has_intersection)
		{
			Eigen::Vector3d intersection_dir = closest_intersections_direction;
			Eigen::Vector3d curve_dir = closest_intersections_curve_dir;
			return closest_intersection.dot(intersection_dir.cross(curve_dir)) < 0;
		}
	}

	ASSERT_RELEASE(false, "This situation should never happen because we took a great circle that would for sure intersect the polygon.")
	return false;
}