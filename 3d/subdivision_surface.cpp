#include "subdivision_surface.h"
#include <igl/loop.h>
#include <igl/adjacency_list.h>

void intersect_subd_loop(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces, const Eigen::MatrixXd& query_points)
{

	//Eigen::MatrixXd current_vertices = vertices;
	//Eigen::MatrixXi current_faces = faces;
	//
	//
	//std::vector<std::vector<int>> current_adjacency_list;
	//

	////std::vector<int> intersecting_faces = find_intersecting_faces(current_vertices, current_faces, query_points.row(0));
	//igl::adjacency_list(faces, current_adjacency_list);
	//
	/*Eigen::MatrixXd new_vertices;
	Eigen::MatrixXi new_faces;
	igl::loop(current_vertices, current_faces, new_vertices, new_faces, 1);*/


}