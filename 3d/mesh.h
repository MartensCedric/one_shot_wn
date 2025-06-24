#pragma once

#include <vector>
#include <string>
#include <Eigen/Dense>
#include <array>
#include "patch.h"
#include "intersections.h"

#include "octreeSDF.h"


void read_off(const std::string& filename, Eigen::MatrixXd& vertices, Eigen::MatrixXi& faces);
void read_obj(const std::string& filename, Eigen::MatrixXd& vertices, Eigen::MatrixXi& faces);
void write_mesh_to_obj(const std::string& filename, const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces);
std::vector<patch_t> extract_mesh_boundary(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces);
struct all_intersections_with_normals_result find_all_intersections_mesh(const Eigen::Vector3d& point, const Eigen::Vector3d& dir, const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces);
struct all_intersections_with_normals_result find_all_mesh_tree_intersections(const SDF* mesh_tree, std::array<double, 3>& query, std::array<double, 3>& dir);
bool ray_triangle_intersect(const Eigen::Vector3d& point, const Eigen::Vector3d& dir, const Eigen::Vector3d& v0, const Eigen::Vector3d& v1, const Eigen::Vector3d& v2,
	double eps,
	double& t,
	double& u,
	double& v,
	bool& parallel);