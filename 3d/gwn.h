#pragma once

#include <Eigen/Dense>
#include <igl/triangulated_grid.h>
#include <igl/cotmatrix.h>
#include <igl/boundary_loop.h>
#include <igl/grid.h>
#include <igl/delaunay_triangulation.h>
#include <igl/copyleft/cgal/delaunay_triangulation.h>
#include <igl/writeOBJ.h>
#include <igl/winding_number.h>
#include <chrono>
#include <string>
#include <vector>

#include "curve_net_parser.h"
#include "math_util.h"
#include "boundary_processing.h"
#include "uv_util.h"


typedef Eigen::Matrix<double, Eigen::Dynamic, 3> space_curve_t;
typedef Eigen::Matrix<double, Eigen::Dynamic, 2> boundary_curve_t;

struct gwn_timing_info {
	int num_patches = 0;
	int total_faces = 0;
	std::chrono::nanoseconds total_triangulation_time = std::chrono::nanoseconds::zero();
	std::chrono::nanoseconds total_meshing_time = std::chrono::nanoseconds::zero();
	std::chrono::nanoseconds winding_number_result_time = std::chrono::nanoseconds::zero();
};

void mesh_implicit_from_boundary(Eigen::MatrixXd& V, Eigen::MatrixXi& F, const Eigen::MatrixXd& boundary, gwn_timing_info& timing_info);
std::vector<double> gwn_from_meshes(const std::vector<Eigen::MatrixXd>& Vs, const std::vector<Eigen::MatrixXi>& Fs, const Eigen::MatrixXd& query_points, gwn_timing_info& timing_info);
void write_fem_gwn_to_file(const std::vector<double>& gwn, const gwn_timing_info& timing_info, const curvenet_input& input);
space_curve_t test_boundary1();