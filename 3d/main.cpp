#include <iostream>
#include <fstream>
#include <random>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <list>

#include <gsl/gsl_multiroots.h>
#include <cxxopts.hpp>

#include "math_util.h"
#include "uv_util.h"
#include "bem_solver.h"
#include "boundary_processing.h"
#include "curve_net.h"
#include "curve_net_wn.h"
#include "coons.h"
#include "open_curve.h"
#include "surface.h"
#include "uv_util.h"
#include "curve_net_parser.h"
#include "spherical.h"
#include "spherical.h"
#include "region_splitting.h"
#include "mesh.h"
#include "gwn.h"
#include "adaptive.h"
#include "bezier.h"


std::vector<std::string> get_patches(const std::string& name, int n)
{
	std::vector<std::string> files;

	for (int i = 0; i < n; i++)
		files.push_back(name + "_" + std::to_string(n) + ".coon_in");
	return files;
}

int main(int argc, char** argv)
{
	cxxopts::Options options("one_shot_wn_3d", "Winding number evaluation");

    options.add_options()
		("n,name", "Experiment Name", cxxopts::value<std::string>())
        ("m,mesh", "Mesh file", cxxopts::value<std::string>())
        ("q,queries", "Query points file", cxxopts::value<std::string>())
        ("h,help", "Print help");

    auto result = options.parse(argc, argv);

    if (result.count("help") ||
        !result.count("mesh") ||
        !result.count("queries") ||
		!result.count("name"))
    {
        std::cout << options.help() << "\n";
        return 0;
    }

    std::string mesh_file   = result["mesh"].as<std::string>();
	std::string query_file  = result["queries"].as<std::string>();
	std::string experiment_name  = result["name"].as<std::string>();
	std::vector<curvenet_input> inputs_to_run;

	curvenet_input input_from_args;
	input_from_args.input_name = experiment_name;
	input_from_args.dimensions = {500, 300};
	input_from_args.query_point_file = query_file;
	input_from_args.config.surface_type = SurfaceType::MESH;
	input_from_args.config.mesh_name = mesh_file;

	inputs_to_run.push_back(input_from_args);
	for (int i = 0; i < inputs_to_run.size(); i++)
	{
		curvenet_input input = inputs_to_run[i];

		if (input.config.alec_only)
			continue;

		std::cout << "running: " << input.input_name << std::endl;
		const std::string patches_filename = input.patches_name;
		const std::string query_points_filename = input.query_point_file;
		std::vector<int> dimensions = input.dimensions;

		std::vector<patch_t> patches;
		if (input.config.surface_type == SurfaceType::MESH)
		{

			size_t pos = input.config.mesh_name.rfind(".off");
			if (pos != std::string::npos && pos == (input.config.mesh_name.size() - std::string(".off").size())) {
				read_off(input.config.mesh_name, input.config.mesh_vertices, input.config.mesh_faces);
			}
			pos = input.config.mesh_name.rfind(".obj");
			if (pos != std::string::npos && pos == (input.config.mesh_name.size() - std::string(".obj").size())) {
				read_obj(input.config.mesh_name, input.config.mesh_vertices, input.config.mesh_faces);
			}

			if (input.predefined_mesh_boundary.empty())
			{
				patches = extract_mesh_boundary(input.config.mesh_vertices, input.config.mesh_faces);
			}
			else
			{
				for (int j = 0; j < input.predefined_mesh_boundary.size(); j++)
				{
					patch_t patch;
					patch.is_open = false;
					patch.curve.resize(input.predefined_mesh_boundary[j].size(), 3);
					for (int k = 0; k < input.predefined_mesh_boundary[j].size(); k++)
					{
						//patch.curve.row(k) = input.config.mesh_vertices.row(k);
						patch.curve.row(k) = input.config.mesh_vertices.row(input.predefined_mesh_boundary[j][k]);
					}

					patch.curve = remove_consecutive_duplicates(patch.curve);

					patches.push_back(patch);

					//eigen::ioformat numpy_fmt(eigen::fullprecision, 0, ", ", ",\n", "[", "]", "[", "]");
					//std::cout << patches.back().curve.format(numpy_fmt) << std::endl;
				}
			}

		}
		else if (input.config.surface_type == SurfaceType::COONS)
		{
			load_coons_patches_from_objs("inputs/coons", input.config.coons_folder_name, input.config.num_coons_files, patches, input.config.coons_patches);
			patches = remove_patches(patches, input.patches_to_remove);
			input.config.coons_patches = remove_coons_patches(input.config.coons_patches, input.patches_to_remove);
			input.config.flip_patch_orientation = remove_bool_vec(input.config.flip_patch_orientation, input.patches_to_remove);
			input.config.coons_flip_normals = remove_bool_vec(input.config.coons_flip_normals, input.patches_to_remove);
			std::vector<int> out = remove_int_vec(input.config.num_coons_files, input.patches_to_remove);

			std::cout << "patches left: ";
			for (int k = 0; k < out.size(); k++)
				std::cout << out[k] << ", ";
			std::cout << std::endl;

		}
		else if (input.config.patch_from_uv_triangle)
		{
			ASSERT_RELEASE(input.config.num_patch_points_from_uv % 3 == 0, "invalid bezier boundary count");
			int num_per_edge = input.config.num_patch_points_from_uv / 3;

			patches = patch_from_bezier(num_per_edge, input.config.parametric_func);
			patches = remove_patches(patches, input.patches_to_remove);



		}
		else
		{
			patches = load_all_patches(patches_filename);
			patches = remove_patches(patches, input.patches_to_remove);
		}

		Eigen::MatrixXd query_points = load_query_points(query_points_filename);


		std::vector<std::vector<int>> row_rays = dimension_to_rays(dimensions);

		if (input.config.do_jiggle_rays)
		{
			// we dont want to do this for the results
			query_points = jiggle_rays(query_points, row_rays, 5e-4);
		}

		if (!input.config.flip_patch_orientation.empty())
		{
			for (int patch_index = 0; patch_index < patches.size(); patch_index++)
			{
				if (input.config.flip_patch_orientation[patch_index])
					patches[patch_index] = flip_patch(patches[patch_index]);
			}

		}

		auto before_1s = std::chrono::high_resolution_clock::now();
		winding_number_results wn_res_1s;
		wn_res_1s = winding_number_mixed(patches, query_points, row_rays, input.config);
		auto after_1s = std::chrono::high_resolution_clock::now();
		write_gwn_to_file(wn_res_1s, input);
	}

	return 0;
}
