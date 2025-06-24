#include "mesh.h"
#include "igl/boundary_loop.h"
#include "igl/ray_triangle_intersect.h"
#include "igl/ray_mesh_intersect.h"
#include <Eigen/Dense>
#include <iostream>
#include <fstream> 
#include "octreeSDF.h"

std::vector<patch_t> extract_mesh_boundary(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces)
{
	std::vector<std::vector<Eigen::Index>> loop_vertices;
	igl::boundary_loop(faces, loop_vertices);

	std::vector<patch_t> patches;
	for (int i = 0; i < loop_vertices.size(); i++)
	{
		if (loop_vertices[i].size() <= 3) continue;
		patch_t loop;
		loop.is_open = false;
		loop.curve.resize(loop_vertices[i].size(), 3);
		for (int j = 0; j < loop_vertices[i].size(); j++)
			loop.curve.row(j) = vertices.row(loop_vertices[i][j]);

		patches.push_back(loop);
	}

	std::sort(patches.rbegin(), patches.rend(), [](const auto& a, const auto& b) { return a.curve.rows() < b.curve.rows(); });
	return patches;
}

void read_off(const std::string& filename, Eigen::MatrixXd& vertices, Eigen::MatrixXi& faces) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Error opening OFF file: " << filename << std::endl;
		return;
	}

	std::string header;
	int num_vertices, num_faces, num_edges;

	// Read and check the OFF header
	std::getline(file, header);
	if (header != "OFF") {
		std::cerr << "Invalid OFF file format." << std::endl;
		return;
	}

	// Read the number of vertices, faces, and edges
	file >> num_vertices >> num_faces >> num_edges;

	// Initialize Eigen matrices
	vertices.resize(num_vertices, 3); // Rows: vertices, Columns: x, y, z
	faces.resize(num_faces, 3);

	// Read vertices
	for (int i = 0; i < num_vertices; ++i) {
		double x, y, z;
		file >> x >> y >> z;
		vertices.row(i) << x, y, z;
	}

	// Read faces
	for (int i = 0; i < num_faces; ++i) {
		int n_verts_on_face, v0, v1, v2;
		file >> n_verts_on_face >> v0 >> v1 >> v2;
		faces.row(i) << v0, v1, v2;
	}

	file.close();
}

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Dense>

void read_obj(const std::string& filename, Eigen::MatrixXd& vertices, Eigen::MatrixXi& faces) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Error: Unable to open OBJ file: " << filename << std::endl;
		return;
	}

	std::vector<Eigen::Vector3d> temp_vertices;
	std::vector<Eigen::Vector3i> temp_faces;

	std::string line;
	while (std::getline(file, line)) {
		std::istringstream iss(line);
		std::string type;
		iss >> type;

		if (type == "v") {
			double x, y, z;
			iss >> x >> y >> z;
			temp_vertices.push_back(Eigen::Vector3d(x, y, z));
		}
		else if (type == "f") {
			int v1, v2, v3;
			// Note: OBJ indices are 1-based
			iss >> v1 >> v2 >> v3;
			temp_faces.push_back(Eigen::Vector3i(v1 - 1, v2 - 1, v3 - 1));
		}
		// Ignore other lines (comments, textures, etc.)
	}

	// Resize Eigen matrices and copy data
	vertices.resize(temp_vertices.size(), 3);
	faces.resize(temp_faces.size(), 3);
	for (int i = 0; i < temp_vertices.size(); ++i) {
		vertices.row(i) = temp_vertices[i];
	}
	for (int i = 0; i < temp_faces.size(); ++i) {
		faces.row(i) = temp_faces[i];
	}
}

void write_mesh_to_obj(const std::string& filename, const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces)
{
	std::ofstream outfile(filename);

	if (!outfile.is_open()) {
		std::cerr << "Error opening file: " << filename << std::endl;
		return; // Or throw an exception
	}

	// Write vertices
	for (int i = 0; i < vertices.rows(); ++i) {
		outfile << "v " << vertices(i, 0) << " " << vertices(i, 1) << " " << vertices(i, 2) << std::endl;
	}

	// Write faces (OBJ indices start at 1, not 0)
	for (int i = 0; i < faces.rows(); ++i) {
		outfile << "f " << faces(i, 0) + 1 << " " << faces(i, 1) + 1 << " " << faces(i, 2) + 1 << std::endl;
	}

	outfile.close();
	std::cout << "Mesh written to " << filename << std::endl;
}


bool ray_triangle_intersect(const Eigen::Vector3d& point,
	const Eigen::Vector3d& dir,
	const Eigen::Vector3d& v0,
	const Eigen::Vector3d& v1,
	const Eigen::Vector3d& v2,
	double eps,
	double& t,
	double& u,
	double& v,
	bool& parallel) {
	// Edge vectors
	Eigen::Vector3d edge1 = v1 - v0;
	Eigen::Vector3d edge2 = v2 - v0;

	// Vector perpendicular to the triangle 
	Eigen::Vector3d pvec = dir.cross(edge2);

	// Determinant 
	double det = edge1.dot(pvec);

	// Check for near-parallelism 
	if (std::abs(det) < eps) {
		// Calculate normal of the triangle
		Eigen::Vector3d normal = (v1 - v0).cross(v2 - v0).normalized();

		// Check if the ray's origin lies within the triangle's plane
		double distance_to_plane = std::abs((point - v0).dot(normal));
		if (distance_to_plane <= eps) {
			parallel = true; // Ray is on the plane
			return false;
		}
		else {
			parallel = false; // Ray is not on the plane
		}
	}

	// Distance from vertex to ray origin
	Eigen::Vector3d tvec = point - v0;

	// Barycentric coordinate u
	u = tvec.dot(pvec) * (1.0 / det);
	if (u < 0.0 || u > 1.0) {
		return false; // Outside triangle
	}

	// Vector from vertex perpendicular to edge1
	Eigen::Vector3d qvec = tvec.cross(edge1);

	// Barycentric coordinate v
	v = dir.dot(qvec) * (1.0 / det);
	if (v < 0.0 || u + v > 1.0) {
		return false; // Outside triangle
	}

	// Distance along ray to intersection
	t = edge2.dot(qvec) * (1.0 / det);

	return true;
}

struct all_intersections_with_normals_result find_all_intersections_mesh(const Eigen::Vector3d& point, const Eigen::Vector3d& dir, const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces)
{
	all_intersections_with_normals_result result;
	result.valid_ray = true;

	for (int i = 0; i < faces.rows(); i++)
	{
		Eigen::RowVector3i face = faces.row(i);
		double u, v, t;
		double eps = 1e-12;
		bool parallel;
		Eigen::Vector3d v0 = vertices.row(face(0));
		Eigen::Vector3d v1 = vertices.row(face(1));
		Eigen::Vector3d v2 = vertices.row(face(2));
		if (igl::ray_triangle_intersect(point, dir, v0, v1, v2, eps, t, u, v, parallel))
		{
			Eigen::Vector3d l1 = vertices.row(face(1)) - vertices.row(face(0));
			Eigen::Vector3d l2 = vertices.row(face(2)) - vertices.row(face(0));
			Eigen::Vector3d normal = l1.cross(l2);
			//normal.normalize();
			//double normal_dot = std::abs(normal.dot(dir));

			//if (normal_dot < 0.01)
			//	continue;

			if (normal.dot(dir) > 0.0)
			{
				result.all_intersections.emplace_back(t, 1);
			}
			else
			{
				result.all_intersections.emplace_back(t, -1);
			}
		}

	/*	if (parallel)
		{
			result.valid_ray = false;
			return result;
		}*/
	}

	std::vector<igl::Hit> hits;

	//if (igl::ray_mesh_intersect(point, dir, vertices, faces, hits)) {

	//	// Iterate over the hits
	//	for (const auto& hit : hits) {/*
	//		std::cout << "Intersection at face: " << hit.id << std::endl;
	//		std::cout << "Barycentric coordinates: " << hit.u << ", " << hit.v << std::endl;
	//		std::cout << "Distance along the ray: " << hit.t << std::endl;*/

	//		Eigen::Vector3i face = faces.row(hit.id);
	//		Eigen::Vector3d l1 = vertices.row(face(1)) - vertices.row(face(0));
	//		Eigen::Vector3d l2 = vertices.row(face(2)) - vertices.row(face(0));
	//		Eigen::Vector3d normal = l1.cross(l2);
	//		normal.normalize();
	//		double normal_dot = std::abs(normal.dot(dir));

	//		if (normal_dot < 0.01)
	//			continue;

	//		if (normal.dot(dir) > 0.0)
	//		{
	//			result.all_intersections.emplace_back(hit.t, 1);
	//		}
	//		else
	//		{
	//			result.all_intersections.emplace_back(hit.t, -1);
	//		}
	//	}
	//}
	//
	std::sort(result.all_intersections.begin(), result.all_intersections.end(), [=](const std::pair<double, int>& p1, const std::pair<double, int>& p2) { return p1.first < p2.first; });
	return result;
}

struct all_intersections_with_normals_result find_all_mesh_tree_intersections(const SDF* mesh_tree, std::array<double, 3>& query, std::array<double, 3>& dir)
{
	all_intersections_with_normals_result result;
	result.valid_ray = true;

	std::vector<std::pair<double, int>> intersections = mesh_tree->query_triangle_index(query, dir);

	for (int i = 0; i < intersections.size(); i++)
	{
		double t = intersections[i].first;
		int face_index = intersections[i].second;
		std::array<double, 3> v0 = mesh_tree->V[mesh_tree->F[face_index][0]];
		std::array<double, 3> v1 = mesh_tree->V[mesh_tree->F[face_index][1]];
		std::array<double, 3> v2 = mesh_tree->V[mesh_tree->F[face_index][2]];

		Eigen::Vector3d ev0 = Eigen::Vector3d(v0[0], v0[1], v0[2]);
		Eigen::Vector3d ev1 = Eigen::Vector3d(v1[0], v1[1], v1[2]);
		Eigen::Vector3d ev2 = Eigen::Vector3d(v2[0], v2[1], v2[2]);

		Eigen::Vector3d l1 = ev1 - ev0;
		Eigen::Vector3d l2 = ev2 - ev0;
		Eigen::Vector3d normal = l1.cross(l2);

		Eigen::Vector3d dir = Eigen::Vector3d(dir[0], dir[1], dir[2]);

		if (normal.dot(dir) > 0.0)
		{
			result.all_intersections.emplace_back(t, 1);
		}
		else
		{
			result.all_intersections.emplace_back(t, -1);
		}
		
	}

	std::sort(result.all_intersections.begin(), result.all_intersections.end(), [=](const std::pair<double, int>& p1, const std::pair<double, int>& p2) { return p1.first < p2.first; });
	return result;
}
