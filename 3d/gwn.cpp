
#include <Eigen/Dense>
#include "gwn.h"
#include <limits>
#include <igl/fast_winding_number.h>


space_curve_t test_boundary1()
{
	int n = 100;
	space_curve_t output(n, 3);
	std::vector<double> thetas = linspace(0, 2.0 * EIGEN_PI, 100, true);
	
	for (int i = 0; i < thetas.size(); i++)
	{
		output.row(i) = Eigen::Vector3d(std::cos(thetas[i]), std::sin(thetas[i]), std::sin(8.0 * thetas[i]));
	}
	return output;
}

void mesh_implicit_from_boundary(Eigen::MatrixXd& V, Eigen::MatrixXi& F, const Eigen::MatrixXd& boundary, gwn_timing_info& timing_info)
{
	int rows = 150;
	int cols = 150;

	Eigen::MatrixXd uv_vertex_positions;
	Eigen::MatrixXi triangulated_faces;

	// don't time this part
	igl::triangulated_grid(rows, cols, uv_vertex_positions, triangulated_faces);
	std::vector<int> loop_indices;
	
	igl::boundary_loop(triangulated_faces, loop_indices);

	std::vector<Eigen::Vector2d> interior_uv_vertices;
	for (int i = 0; i < uv_vertex_positions.rows(); i++)
	{
		if (std::find(loop_indices.begin(), loop_indices.end(), i) == loop_indices.end())
		{
			interior_uv_vertices.push_back(uv_vertex_positions.row(i));
		}
	}

	boundary_curve_t bd = create_square_boundaries(boundary.rows())[0];

	Eigen::MatrixXd all_vertices_uv(interior_uv_vertices.size() + bd.rows(), 2); // interior first
	for (int i = 0; i < interior_uv_vertices.size(); i++)
		all_vertices_uv.row(i) = interior_uv_vertices[i];
	for (int i = 0; i < bd.rows(); i++)
		all_vertices_uv.row(i + interior_uv_vertices.size()) = bd.row(i);
	
	// time starting here:
	std::chrono::high_resolution_clock::time_point triangulation_tic = std::chrono::high_resolution_clock::now();
	igl::copyleft::cgal::delaunay_triangulation(all_vertices_uv, F);
	std::chrono::high_resolution_clock::time_point triangulation_toc = std::chrono::high_resolution_clock::now();
	std::chrono::nanoseconds triangulation_time = triangulation_toc - triangulation_tic;

	std::chrono::high_resolution_clock::time_point fem_tic = std::chrono::high_resolution_clock::now();
	Eigen::VectorXd boundary_x = boundary.col(0);
	Eigen::VectorXd boundary_y = boundary.col(1);
	Eigen::VectorXd boundary_z = boundary.col(2);

	int num_interior = interior_uv_vertices.size();
	int num_b = bd.rows();

	Eigen::SparseMatrix<double> L;
	igl::cotmatrix(all_vertices_uv, F, L);
	Eigen::SparseMatrix<double> L_ii = L.block(0, 0, num_interior, num_interior);
	Eigen::SparseMatrix<double> L_ib = L.block(0, num_interior, num_interior, num_b);
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
	solver.compute(L_ii);

	if (solver.info() != Eigen::Success)
	{
		throw std::runtime_error("Matrix is not PSD!");
	}

	Eigen::VectorXd sol_x = solver.solve(-L_ib * boundary_x);
	Eigen::VectorXd sol_y = solver.solve(-L_ib * boundary_y);
	Eigen::VectorXd sol_z = solver.solve(-L_ib * boundary_z);


	V.resize(sol_x.rows() + boundary_x.rows(), 3);

	V.block(0, 0, sol_x.rows(), 1) = sol_x;
	V.block(sol_x.rows(), 0, boundary_x.rows(), 1) = boundary_x;
	V.block(0, 1, sol_y.rows(), 1) = sol_y;
	V.block(sol_y.rows(), 1, boundary_y.rows(), 1) = boundary_y;
	V.block(0, 2, sol_z.rows(), 1) = sol_z;
	V.block(sol_z.rows(), 2, boundary_z.rows(), 1) = boundary_z;
	std::chrono::high_resolution_clock::time_point fem_toc = std::chrono::high_resolution_clock::now();
	std::chrono::nanoseconds fem_time = fem_toc - fem_tic;
	//igl::writeOBJ("test.obj", V, F);

	timing_info.total_triangulation_time += triangulation_time;
	timing_info.total_meshing_time += fem_time;
}

std::vector<double> gwn_from_meshes(const std::vector<Eigen::MatrixXd>& Vs, const std::vector<Eigen::MatrixXi>& Fs, const Eigen::MatrixXd& query_points, gwn_timing_info& timing_info)
{
	std::vector<double> gwn(query_points.rows(), 0.0);
	for (int i = 0; i < Vs.size(); i++)
	{
		Eigen::VectorXd gwn_for_patch;
		//igl::writeOBJ("test.obj", Vs[i], Fs[i]);
		std::chrono::high_resolution_clock::time_point wn_tic = std::chrono::high_resolution_clock::now();
		igl::winding_number(Vs[i], Fs[i], query_points, gwn_for_patch);
		std::chrono::high_resolution_clock::time_point wn_toc = std::chrono::high_resolution_clock::now();
		std::chrono::nanoseconds wn_time = wn_toc - wn_tic;
		timing_info.winding_number_result_time += wn_time;

		for (int j = 0; j < gwn.size(); j++)
		{
			gwn[j] += gwn_for_patch(j);
		}
	}

	return gwn;
}

void write_fem_gwn_to_file(const std::vector<double>& gwn, const gwn_timing_info& timing_info, const curvenet_input& input)
{
	std::string outname = input.config.is_override_output_name ? input.config.override_output_name : input.input_name;
	std::string filename = "outputs/" + outname + "_alec.m";
	std::ofstream chi_output(filename);
	chi_output << "chis = [";
	for (int j = 0; j < gwn.size(); j++)
	{
		chi_output << std::setprecision(16) << gwn[j] << ",";
	}

	chi_output << "];\n";
	chi_output << "res = [";

	for (int j = input.dimensions.size() - 1; j >= 0; j--)
	{
		chi_output << std::to_string(input.dimensions[j]) << ",";
	}

	chi_output << "];\n";

	chi_output << "num_patches = " << std::to_string(timing_info.num_patches) << ";" << std::endl;
	chi_output << "total_faces = " << std::to_string(timing_info.total_faces) << ";" << std::endl;
	chi_output << "total_meshing_time = " << std::to_string(timing_info.total_meshing_time.count()) << ";" << std::endl;
	chi_output << "total_triangulation_time = " << std::to_string(timing_info.total_triangulation_time.count()) << ";" << std::endl;
	chi_output << "winding_number_result_time = " << std::to_string(timing_info.winding_number_result_time.count()) << ";" << std::endl;
	

	chi_output.close();
}