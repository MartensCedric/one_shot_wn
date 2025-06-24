#pragma once
#include <vector>
#include <Eigen/Dense>
#include "math_util.h"

typedef Eigen::MatrixXd closed_curve_t;
typedef Eigen::Matrix<double, Eigen::Dynamic, 3> closed_spherical_curve_t;
typedef Eigen::Matrix<double, Eigen::Dynamic, 2> lat_lon_curve_t;
struct spherical_polygon
{
	closed_spherical_curve_t points;
	double area;
};

typedef std::pair<Eigen::Vector3d, Eigen::Vector3d> segment_t;

closed_spherical_curve_t project_to_sphere(const closed_curve_t& curve, const Eigen::Vector3d& point);
bool is_inside_polygon(const closed_spherical_curve_t& curve, const Eigen::Vector3d& point);
lat_lon_curve_t cartesian_to_lat_lon(const closed_spherical_curve_t& cartesian);
double spherical_polygon_excess(const lat_lon_curve_t& points);
double spherical_polygon_area(const closed_spherical_curve_t& points);

inline double hav(double x);