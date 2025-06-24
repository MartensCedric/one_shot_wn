#include <iostream>
#include "math_util.h"
#include "uv_util.h"
#include "bem_solver.h"
#include "boundary_processing.h"
#include "curve_net.h"
#include "curve_net_wn.h"
#include "coons.h"
#include "open_curve.h"
#include "surface.h"
#include <gsl/gsl_multiroots.h>
#include <random>
#include <iostream>
#include <fstream>
#include <numeric>
#include "uv_util.h"
#include "curve_net_parser.h"
#include "spherical.h"
#include <iomanip>
#include <chrono>
#include "spherical.h"
#include "region_splitting.h"
#include "mesh.h"
#include "gwn.h"

#include "adaptive.h"

#include <list>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Arrangement_on_surface_2.h>
#include <CGAL/Arr_geodesic_arc_on_sphere_traits_2.h>
#include <CGAL/Arr_spherical_topology_traits_2.h>
#include "arr_print.h"
#include <chrono>
#include "bezier.h"


std::vector<std::string> get_patches(const std::string& name, int n)
{
	std::vector<std::string> files;

	for (int i = 0; i < n; i++)
		files.push_back(name + "_" + std::to_string(n) + ".coon_in");
	return files;
}

void unit_tests()
{
	std::vector<Eigen::Vector3d> t1 = {
		Eigen::Vector3d(0., 0, 0),
		Eigen::Vector3d(1., 0., 0),
	};

	double err = 1e-5;
	assert((interpolate_curve(0.5, t1) - Eigen::Vector3d(0.5, 0.0, 0.0)).norm() < err);
	assert((interpolate_curve(0.8, t1) - Eigen::Vector3d(0.8, 0.0, 0.0)).norm() < err);

	std::vector<Eigen::Vector3d> t2 = {
		Eigen::Vector3d(0., 0, 0),
		Eigen::Vector3d(2., 0.0, 0),
		Eigen::Vector3d(4., 0.0, 0),
		Eigen::Vector3d(6., 0, 0)
	};
	assert((interpolate_curve(0.0, t2) - Eigen::Vector3d(0.0, 0.0, 0.0)).norm() < err);
	assert((interpolate_curve(1.0, t2) - Eigen::Vector3d(6.0, 0.0, 0.0)).norm() < err);
	assert((interpolate_curve(0.5, t2) - Eigen::Vector3d(3.0, 0.0, 0.0)).norm() < err);
	assert((interpolate_curve(4.0/6.0, t2) - Eigen::Vector3d(4.0, 0.0, 0.0)).norm() < err);
	assert((interpolate_curve(2.0/6.0, t2) - Eigen::Vector3d(2.0, 0.0, 0.0)).norm() < err);
	
	std::vector<Eigen::Vector3d> t3 = {
		Eigen::Vector3d(0., 0, 0),
		Eigen::Vector3d(1., 0.0, 0),
		Eigen::Vector3d(4., 0.0, 0),
		Eigen::Vector3d(10., 0, 0)
	};

	assert((interpolate_curve(0.0, t3) - Eigen::Vector3d(0.0, 0.0, 0.0)).norm() < err);
	assert((interpolate_curve(0.5, t3) - Eigen::Vector3d(5.0, 0.0, 0.0)).norm() < err);
	assert((interpolate_curve(0.9, t3) - Eigen::Vector3d(9.0, 0.0, 0.0)).norm() < err);

}


void region_tests()
{
	int n = 100;
	double closeness = 0.7;
	Eigen::Vector2d p_i(0.2, 0.7);
	std::vector<double> u_vals = linspace(0, 2 * M_PI, n, false);
	boundary_curve_t boundary_curve = create_square_uv_parameterization_open(n, closeness);
	space_curve_t space_curve;
	space_curve.resize(n, 3);
	for (int i = 0; i < boundary_curve.rows(); i++)
	{
		space_curve.coeffRef(i, 0) = boundary_curve.coeff(i, 0);
		space_curve.coeffRef(i, 1) = boundary_curve.coeff(i, 1);
		space_curve.coeffRef(i, 2) = std::sin(u_vals[i]);
	}

	curve_info_t curve_info = compute_curve_info(boundary_curve);
	curve_info.is_open = true;

	
	//std::cout << fundamental_solution_laplacian(p_i, boundary_curve, curve_info, true) << std::endl;
	//std::cout << fundamental_solution_derivative_laplacian(p_i, boundary_curve, curve_info, true) << std::endl;
}

void wn_tests()
{
	assert(std::abs(signed_angle(Eigen::Vector2d(1.0, 1.0), Eigen::Vector2d(-1.0, -1.0)) - M_PI) < 1e-5);
	assert(std::abs(signed_angle(Eigen::Vector2d(1.0, 1.0), Eigen::Vector2d(1.0, 1.0)) - 0.0) < 1e-5);
	assert(std::abs(signed_angle(Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(0.0, 1.0)) - (M_PI / 2.0)) < 1e-5);
	assert(std::abs(signed_angle(Eigen::Vector2d(0.0, 1.0), Eigen::Vector2d(1.0, 0.0)) - -(M_PI / 2.0)) < 1e-5);
}


//void spherical_segments_test()
//{
//	intersection_t res = spherical_segments_intersect({Eigen::Vector3d(1, 0.5, 0.0).normalized(), Eigen::Vector3d(1, -0.5, 0.0).normalized() }, 
//								 {Eigen::Vector3d(1, 0.0, 0.5).normalized(), Eigen::Vector3d(1, 0.0, -0.5).normalized() });
//	assert(res.intersects);
//	assert((res.location - Eigen::Vector3d(1, 0, 0)).norm() < 0.001);
//
//	res = spherical_segments_intersect({ Eigen::Vector3d(-1, 0.5, 0.0).normalized(), Eigen::Vector3d(-1, -0.5, 0.0).normalized() },
//		{ Eigen::Vector3d(1, 0.0, 0.5).normalized(), Eigen::Vector3d(1, 0.0, -0.5).normalized() });
//	assert(!res.intersects);
//}

Eigen::Vector3d test_implicit(Eigen::Vector2d uv)
{
	double x = uv(0);
	double y = uv(1);
	double z = std::cos(uv(0));
	return Eigen::Vector3d(x, y, z);
}

void test_inside_polygon()
{
	std::string patches_filename = "inputs/patches/square.patches"; // simple square, single patch
	std::vector<patch_t> patches = load_all_patches(patches_filename);
	Eigen::MatrixXd query_points(2, 3);
	query_points.row(0) = Eigen::Vector3d(1, 0, 0);
	query_points.row(1) = Eigen::Vector3d(-1, 0, 0);
	std::vector<patch_t> closed_patches = get_closed_patches(patches);

	curve_net_sampler sampler = run_region_splitting(closed_patches, query_points);

	Eigen::Vector3d ray = Eigen::Vector3d(0, 1, 0);
	
	// This ray always intersect nothing and should be in the largest region (first one)

	assert(is_inside_polygon(sampler.spherical_regions[0][0][0], ray));
	assert(is_inside_polygon(sampler.spherical_regions[0][1][0], ray));

	patches_filename = "inputs/patches/bunny.patches"; // simple square, single patch
	patches = load_all_patches(patches_filename);
	patches.erase(patches.begin() + 1, patches.end());
	query_points = load_query_points("inputs/query_points/bunny_40_40.points");
	closed_patches = get_closed_patches(patches);

	sampler = run_region_splitting(closed_patches, query_points);

	ray = Eigen::Vector3d(0, 1, 0);

	for (int i = 0; i < 40; i++)
	{
		assert(is_inside_polygon(sampler.spherical_regions[0][i][0], ray));
	}


}
curvenet_input results_bug_1000_285()
{
	curvenet_input input;
	input.input_name = "bug_1000_285";
	input.patches_name = "inputs/patches/bug.patches";
	input.dimensions = { 1000, 285 };
	input.patches_to_remove = { 13, 49, 18, 32, 42,51 };
	return input;
}



curvenet_input results_bug_top_windows_600_514()
{
	curvenet_input input;
	input.input_name = "bug_top_windows_600_514";
	input.patches_name = "inputs/patches/bug.patches";
	input.dimensions = { 600, 514};
	input.patches_to_remove = { 13, 49, 18, 32, 42,51 };
	return input;
}

curvenet_input results_bug_top_windows_500_428()
{
	curvenet_input input;
	input.input_name = "bug_top_windows_500_428";
	input.patches_name = "inputs/patches/bug.patches";
	input.dimensions = { 500, 428 };
	input.patches_to_remove = { 13, 49, 18, 32, 42,51 };
	return input;
}

curvenet_input results_bug_top_windows_350_190()
{
	curvenet_input input;
	input.input_name = "bug_top_windows_350_190";
	input.patches_name = "inputs/patches/bug.patches";
	input.dimensions = { 350, 190 };
	input.patches_to_remove = { 13, 49, 18, 32, 42,51 };
	return input;
}
curvenet_input results_bug_side_centered_600_327()
{
	curvenet_input input;
	input.input_name = "bug_side_centered_600_327";
	input.patches_name = "inputs/patches/bug.patches";
	input.dimensions = { 600, 327 };
	input.patches_to_remove = { 13, 49, 18, 32, 42, 51 };
	return input;
}

curvenet_input results_bug_side_600_327()
{
	curvenet_input input;
	input.input_name = "bug_side_600_327";
	input.patches_name = "inputs/patches/bug.patches";
	input.dimensions = { 600, 327 };
	input.patches_to_remove = { 13, 49, 18, 32, 42, 51 };
	return input;
}

curvenet_input results_bunny_face_600_600()
{
	curvenet_input input;
	input.input_name = "bunny_face_600_600";
	input.patches_name = "inputs/patches/bunny.patches";
	input.dimensions = { 600, 600 };
	input.patches_to_remove = {};
	
	return input;
}

curvenet_input results_bunny_diagonal_600_630()
{
	curvenet_input input;
	input.input_name = "bunny_diagonal_600_630";
	input.patches_name = "inputs/patches/bunny.patches";
	input.dimensions = { 600, 630 };
	input.patches_to_remove = {};

	return input;
}
curvenet_input results_hand_four_fingers_500_272()
{
	curvenet_input input;
	input.input_name = "hand_four_fingers_500_272";
	input.patches_name = "inputs/patches/hand2.patches";
	input.dimensions = { 500, 272 };
	input.patches_to_remove = { 28 };

	return input;
}

curvenet_input results_hand_one_finger_500_272()
{
	curvenet_input input;
	input.input_name = "hand_one_finger_500_272";
	input.patches_name = "inputs/patches/hand2.patches";
	input.dimensions = { 500, 272 };
	input.patches_to_remove = { 28 };

	return input;
}

curvenet_input results_SGA_Fighter_500_208()
{
	curvenet_input input;
	input.input_name = "SGA_Fighter_500_208";
	input.patches_name = "inputs/patches/SGA_Fighter.patches";
	input.dimensions = { 500, 208 };
	input.patches_to_remove = { 10, 13, 24, 27 };

	return input;
}

curvenet_input results_SGA_Fighter_vox_200_81_204()
{
	curvenet_input input;
	input.input_name = "SGA_Fighter_vox_200_81_204";
	input.patches_name = "inputs/patches/SGA_Fighter.patches";
	input.dimensions = { 200, 81, 204 };
	input.patches_to_remove = {};

	return input;
}

curvenet_input results_siggraphSpacecraft13_500_156()
{
	curvenet_input input;
	input.input_name = "siggraphSpacecraft13_500_156";
	input.patches_name = "inputs/patches/siggraphSpacecraft13.patches";
	input.dimensions = { 500, 156 };
	input.patches_to_remove = { };
	return input;
}

curvenet_input results_siggraphSpacecraft19_500_104()
{
	curvenet_input input;
	input.input_name = "siggraphSpacecraft19_500_104";
	input.patches_name = "inputs/patches/siggraphSpacecraft19.patches";
	input.dimensions = { 500, 104 };
	input.patches_to_remove = { };
	return input;
}

curvenet_input results_siggraphSpacecraft20_500_500()
{
	curvenet_input input;
	input.input_name = "siggraphSpacecraft20_500_500";
	input.patches_name = "inputs/patches/siggraphSpacecraft20.patches";
	input.dimensions = { 500, 500 };
	input.patches_to_remove = { };
	return input;
}

curvenet_input results_siggraphSpacecraft23_500_500()
{
	curvenet_input input;
	input.input_name = "siggraphSpacecraft23_500_500";
	input.patches_name = "inputs/patches/siggraphSpacecraft23.patches";
	input.dimensions = { 500, 500 };
	input.patches_to_remove = { };
	return input;
}

curvenet_input results_siggraphSpacecraft26_500_227()
{
	curvenet_input input;
	input.input_name = "siggraphSpacecraft26_500_227";
	input.patches_name = "inputs/patches/siggraphSpacecraft26.patches";
	input.dimensions = { 500, 227 };
	input.patches_to_remove = { 6, 8 };
	return input;
}

curvenet_input results_siggraphSpacecraft26_side_500_227()
{
	curvenet_input input;
	input.input_name = "siggraphSpacecraft26_side_500_227";
	input.patches_name = "inputs/patches/siggraphSpacecraft26.patches";
	input.dimensions = { 500, 227 };
	input.patches_to_remove = { 6, 8 };
	return input;
}

//curvenet_input results_siggraphSpacecraft30_500_227()
//{
//	curvenet_input input;
//	input.input_name = "siggraphSpacecraft30_500_227";
//	input.patches_name = "inputs/patches/siggraphSpacecraft30.patches";
//	input.dimensions = { 500, 227 };
//	input.patches_to_remove = { };
//	return input;
//}


// has bad patch but looks super cool
//curvenet_input results_siggraphSpacecraft30_side_500_227()
//{
//	curvenet_input input;
//	input.input_name = "siggraphSpacecraft30_side_500_227";
//	input.patches_name = "inputs/patches/siggraphSpacecraft30.patches";
//	input.dimensions = { 500, 227 };
//	input.patches_to_remove = { 0, 2, };
//	return input;
//}

curvenet_input results_siggraphSpacecraft37_500_227()
{
	curvenet_input input;
	input.input_name = "siggraphSpacecraft37_500_227";
	input.patches_name = "inputs/patches/siggraphSpacecraft37.patches";
	input.dimensions = { 500, 227 };
	input.patches_to_remove = { 6, 9, 22, 23 };
	return input;
}


curvenet_input results_siggraphSpacecraft38_500_113()
{
	curvenet_input input;
	input.input_name = "siggraphSpacecraft38_500_113";
	input.patches_name = "inputs/patches/siggraphSpacecraft38.patches";
	input.dimensions = { 500, 113 };
	input.patches_to_remove = {7, 15, 20, 23};
	return input;
}

// has trash patches
//curvenet_input results_siggraphSpacecraft44_500_500()
//{
//	curvenet_input input;
//	input.input_name = "siggraphSpacecraft44_500_500";
//	input.patches_name = "inputs/patches/siggraphSpacecraft44.patches";
//	input.dimensions = { 500, 500 };
//	input.patches_to_remove = { };
//	return input;
//}


curvenet_input results_siggraphSpacecraft56_500_113()
{
	curvenet_input input;
	input.input_name = "siggraphSpacecraft56_500_113";
	input.patches_name = "inputs/patches/siggraphSpacecraft56.patches";
	input.dimensions = { 500, 113 };
	input.patches_to_remove = { 5, 18 };
	return input;
}


//// pretty trash
//curvenet_input results_siggraphSpacecraft57_500_500()
//{
//	curvenet_input input;
//	input.input_name = "siggraphSpacecraft57_500_500";
//	input.patches_name = "inputs/patches/siggraphSpacecraft57.patches";
//	input.dimensions = { 500, 500 };
//	input.patches_to_remove = { };
//	return input;
//}
//
//// garbage patch
//curvenet_input results_siggraphSpacecraft75_500_500()
//{
//	curvenet_input input;
//	input.input_name = "siggraphSpacecraft75_500_500";
//	input.patches_name = "inputs/patches/siggraphSpacecraft75.patches";
//	input.dimensions = { 500, 500 };
//	input.patches_to_remove = { };
//	return input;
//}
//
//// pretty trash
//curvenet_input results_siggraphSpacecraft79_500_500()
//{
//	curvenet_input input;
//	input.input_name = "siggraphSpacecraft79_500_500";
//	input.patches_name = "inputs/patches/siggraphSpacecraft79.patches";
//	input.dimensions = { 500, 500 };
//	input.patches_to_remove = { };
//	return input;
//}


curvenet_input results_siggraphSpacecraft87_500_375()
{
	curvenet_input input;
	input.input_name = "siggraphSpacecraft87_500_375";
	input.patches_name = "inputs/patches/siggraphSpacecraft87.patches";
	input.dimensions = { 500, 375 };
	input.patches_to_remove = { };
	return input;
}

curvenet_input results_enterprise_200_1250()
{
	curvenet_input input;
	input.input_name = "enterprise_200_1250";
	input.patches_name = "inputs/patches/enterprise.patches";
	input.dimensions = { 200, 1250 };
	input.patches_to_remove = {11, 117};
	return input;
}

curvenet_input results_phoenix_500_333()
{
	curvenet_input input;
	input.input_name = "pheonix_500_333";
	input.patches_name = "inputs/patches/pheonix.patches";
	input.dimensions = { 500, 333 };
	input.patches_to_remove = {};
	return input;
}

curvenet_input results_SGA_Hawk_vox_210_112_80()
{
	curvenet_input input;
	input.input_name = "SGA_Hawk_vox_210_112_80";
	input.patches_name = "inputs/patches/SGA_Hawk.patches";
	input.dimensions = { 210, 112, 80};
	input.patches_to_remove = {};
	return input;
}

curvenet_input results_bunny_vox_220_215_166()
{
	curvenet_input input;
	input.input_name = "bunny_vox_220_215_166";
	input.patches_name = "inputs/patches/bunny.patches";
	input.dimensions = { 220, 215, 166 };
	input.patches_to_remove = {};
	return input;
}

curvenet_input results_SGA_Doghead_vox_170_106_210()
{
	curvenet_input input;
	input.input_name = "SGA_Doghead_vox_170_106_210";
	input.patches_name = "inputs/patches/SGA_Doghead.patches";
	input.dimensions = { 170, 106, 210 };
	input.patches_to_remove = {};
	return input;
}

Eigen::Vector3d roll_parametric(Eigen::Vector2d uv)
{
	double u = uv(0);
	double v = uv(1);
	return 0.1 * Eigen::Vector3d(-5. + 10. * u, (5.0 - 4.0 * v) * std::cos(v * 7 * EIGEN_PI), (5.0 - 4.0 * v) * std::sin(v * 7 * EIGEN_PI));
}

Eigen::Vector3d half_roll_parametric(Eigen::Vector2d uv)
{
	double u = uv(0);
	double v = uv(1);
	return 0.1 * Eigen::Vector3d(-5. + 10. * u, (5.0 - 4.0 * v) * std::cos(v * 1.5 * EIGEN_PI), (5.0 - 4.0 * v) * std::sin(v * 1.5 * EIGEN_PI));
}

Eigen::Vector3d almost_roll_parametric(Eigen::Vector2d uv)
{
	double u = uv(0);
	double v = uv(1);
	return 0.1 * Eigen::Vector3d(-5. + 10. * u, (5.0 - 4.0 * v) * std::cos(v * 3.0 * EIGEN_PI), (5.0 - 4.0 * v) * std::sin(v * 3.0 * EIGEN_PI));
}

Eigen::Vector3d quasi_roll_parametric(Eigen::Vector2d uv)
{
	double u = uv(0);
	double v = uv(1);
	return 0.1 * Eigen::Vector3d(-5. + 10. * u, (5.0 - 4.0 * v) * std::cos(v * 5.5 * EIGEN_PI), (5.0 - 4.0 * v) * std::sin(v * 5.5 * EIGEN_PI));
}

Eigen::Vector3d barely_roll_parametric(Eigen::Vector2d uv)
{
	double u = uv(0);
	double v = uv(1);
	return 0.1 * Eigen::Vector3d(-5. + 10. * u, (5.0 - 4.0 * v) * std::cos(v * 2.75 * EIGEN_PI), (5.0 - 4.0 * v) * std::sin(v * 2.75 * EIGEN_PI));
}


Eigen::Vector3d singularity_parametric(Eigen::Vector2d uv)
{
	uv(0) -= 0.5;
	uv(1) -= 0.5;

	return Eigen::Vector3d(uv(0),
						   uv(1),
						   1.0 / std::sqrt(uv(0) * uv(0) + uv(1) * uv(1)));
}

Eigen::Vector3d circular_tube_parametric(Eigen::Vector2d uv)
{
	double u = uv(0) * 2.0 * EIGEN_PI;
	double v = (uv(1) - 0.15) * 8.29;

	return Eigen::Vector3d(
						(2 + 0.25 * std::cos(u)) * std::cos(v * 2),
						(2 + 0.25 * std::cos(u)) * std::sin(v * 2),
						0.25 * std::sin(u) + v * 0.28 + 0.15 * std::sin(7.0 * v));
}

Eigen::Vector3d paraboloid_parametric(Eigen::Vector2d uv)
{
	return Eigen::Vector3d(uv(0), uv(1), uv(0) * uv(0) + uv(1) * uv(1));
}

curvenet_input results_roll_100_100()
{
	curvenet_input input;
	input.input_name = "roll_parametric_100_100";
	input.patches_name = "inputs/patches/roll_parametric.patches";
	input.dimensions = { 100, 100 };
	input.patches_to_remove = {};
	input.config.surface_type = SurfaceType::PARAMETRIC;
	input.config.parametric_func = roll_parametric;
	return input;
}


curvenet_input results_gear_100_100()
{
	curvenet_input input;
	input.input_name = "spur_gear_parametric_100_100";
	input.patches_name = "inputs/patches/spur_gear_parametric.patches";
	input.dimensions = { 100, 100 };
	input.patches_to_remove = {};
	input.config.surface_type = SurfaceType::PARAMETRIC_GEARS;
	input.config.parametric_func = spur_gear_parametric;
	input.config.parametric_max_t = 4.0;
	input.config.do_symmetric_patch_adjustment = false;// true;

	return input;
}

curvenet_input results_gear_500_499()
{
	curvenet_input input;
	input.input_name = "spur_gear_parametric_500_499";
	input.patches_name = "inputs/patches/spur_gear_parametric.patches";
	input.dimensions = { 500, 499 };
	input.patches_to_remove = {};
	input.config.surface_type = SurfaceType::PARAMETRIC_GEARS;
	input.config.parametric_func = spur_gear_parametric;
	input.config.parametric_max_t = 4.0;
	input.config.do_symmetric_patch_adjustment = true; // true; 
	input.config.do_jiggle_rays = false;

	return input;
}

curvenet_input results_gear_1000_499()
{
	curvenet_input input;
	input.input_name = "spur_gear_parametric_1000_499";
	input.patches_name = "inputs/patches/spur_gear_parametric_left.patches";
	input.dimensions = { 1000, 499 };
	input.patches_to_remove = {};
	input.config.surface_type = SurfaceType::PARAMETRIC_GEARS;
	input.config.parametric_func = spur_gear_parametric;
	input.config.parametric_max_t = 6.0;
	input.config.do_symmetric_patch_adjustment = true; // true; 
	input.config.do_jiggle_rays = false;

	return input;
}

curvenet_input results_gear_zoom_500_500()
{
	curvenet_input input;
	input.input_name = "spur_gear_parametric_zoom_500_500";
	input.patches_name = "inputs/patches/spur_gear_parametric_left.patches";
	input.dimensions = { 500, 500 };
	input.patches_to_remove = {};
	input.config.surface_type = SurfaceType::PARAMETRIC_GEARS;
	input.config.parametric_func = spur_gear_parametric;
	input.config.parametric_max_t = 1.0;
	input.config.do_symmetric_patch_adjustment = true; // true; 
	input.config.do_jiggle_rays = false;

	return input;
}


curvenet_input results_gear_zoom_right_500_500()
{
	curvenet_input input;
	input.input_name = "spur_gear_parametric_zoom_right_500_500";
	input.patches_name = "inputs/patches/spur_gear_parametric_right.patches";
	input.dimensions = { 500, 500 };
	input.patches_to_remove = {};
	input.config.surface_type = SurfaceType::PARAMETRIC_GEARS;
	input.config.parametric_func = spur_gear_parametric_right;
	input.config.parametric_max_t = 1.0;
	input.config.do_symmetric_patch_adjustment = true; // true; 
	input.config.do_jiggle_rays = false;

	return input;
}


curvenet_input results_gear_ALEC_ONLY_1000_499()
{
	curvenet_input input;
	input.input_name = "spur_gear_parametric_1000_499";
	input.dimensions = { 1000, 499 };
	input.patches_to_remove = {};
	input.config.surface_type = SurfaceType::MESH;
	input.config.mesh_name = "inputs/mesh/gear1_bad_res.obj";
	input.config.alec_only = true;
	return input;
}


curvenet_input results_gear_right_ALEC_ONLY_1000_499()
{
	curvenet_input input;
	input.input_name = "spur_gear_parametric_right_1000_499";
	input.dimensions = { 1000, 499 };
	input.patches_to_remove = {};
	input.config.surface_type = SurfaceType::MESH;
	input.config.mesh_name = "inputs/mesh/gear2_bad_res.obj";
	input.config.alec_only = true;
	return input;
}


curvenet_input results_gear_right_1000_499()
{
	curvenet_input input;
	input.input_name = "spur_gear_parametric_right_1000_499";
	input.patches_name = "inputs/patches/spur_gear_parametric_right.patches";
	input.dimensions = { 1000, 499 };
	input.patches_to_remove = {};
	input.config.surface_type = SurfaceType::PARAMETRIC_GEARS;
	input.config.parametric_func = spur_gear_parametric_right;
	input.config.parametric_max_t = 6.0;
	input.config.do_symmetric_patch_adjustment = true; // true; 
	input.config.do_jiggle_rays = false;

	return input;
}

curvenet_input results_gear_zoom_ALEC_ONLY_500_500()
{
	curvenet_input input;
	input.input_name = "spur_gear_parametric_zoom_500_500";
	input.patches_name = "inputs/patches/spur_gear_parametric_left.patches";
	input.config.surface_type = SurfaceType::MESH;
	input.config.mesh_name = "inputs/mesh/gear1_bad_res.obj";
	input.config.alec_only = true;
	return input;
}


curvenet_input results_gear_zoom_right_ALEC_ONLY_500_500()
{
	curvenet_input input;
	input.input_name = "spur_gear_parametric_zoom_right_500_500";
	input.patches_name = "inputs/patches/spur_gear_parametric_right.patches";
	input.config.surface_type = SurfaceType::MESH;
	input.config.mesh_name = "inputs/mesh/gear2_bad_res.obj";
	input.config.alec_only = true;
	return input;
}


curvenet_input results_roll_500_500()
{
	curvenet_input input;
	input.input_name = "roll_parametric_500_500";
	input.patches_name = "inputs/patches/roll_parametric.patches";
	input.dimensions = { 500, 500 };
	input.patches_to_remove = {};
	input.config.surface_type = SurfaceType::PARAMETRIC;
	input.config.parametric_func = roll_parametric;
	return input;
}
curvenet_input results_roll_80_80()
{
	curvenet_input input;
	input.input_name = "roll_parametric_80_80";
	input.patches_name = "inputs/patches/roll_parametric.patches";
	input.dimensions = { 80, 80 };
	input.patches_to_remove = {};
	input.config.surface_type = SurfaceType::PARAMETRIC;
	input.config.parametric_func = roll_parametric;
	return input;
}

curvenet_input results_paraboloid_100_100()
{
	curvenet_input input;
	input.input_name = "paraboloid_parametric_100_100";
	input.patches_name = "inputs/patches/paraboloid_parametric.patches";
	input.dimensions = { 100, 100 };
	input.patches_to_remove = {};
	input.config.surface_type = SurfaceType::PARAMETRIC;
	input.config.parametric_func = paraboloid_parametric;
	return input;
}

curvenet_input results_singularity_500_500()
{
	curvenet_input input;
	input.input_name = "singularity_parametric_500_500";
	input.patches_name = "inputs/patches/singularity_parametric.patches";
	input.dimensions = { 500, 500 };
	input.patches_to_remove = {};
	input.config.surface_type = SurfaceType::PARAMETRIC;
	input.config.parametric_func = singularity_parametric;
	return input;
}

curvenet_input results_circular_tube_500_500()
{
	curvenet_input input;
	input.input_name = "circular_tube_parametric_500_500";
	input.patches_name = "inputs/patches/circular_tube_parametric.patches";
	input.dimensions = { 500, 500 };
	input.patches_to_remove = {};
	input.config.surface_type = SurfaceType::PARAMETRIC;
	input.config.parametric_func = circular_tube_parametric;
	input.config.parametric_max_t = 7.0;
	return input;
}

curvenet_input results_eiffel_200_420()
{
	curvenet_input input;
	input.input_name = "eiffel_200_420";
	input.patches_name = "inputs/patches/eiffel.patches";
	input.dimensions = { 200, 420 };
	input.patches_to_remove = {};
	return input;
}

curvenet_input results_camel_head_500_300()
{
	curvenet_input input;
	input.input_name = "camelhead_500_300";
	input.dimensions = { 500, 300 };
	input.config.surface_type = SurfaceType::MESH;
	input.config.mesh_name = "inputs/mesh/camelhead.off";
	input.patches_to_remove = {};
	
	return input;
}

curvenet_input results_camel_head_800_600()
{
	curvenet_input input;
	input.input_name = "camelhead_800_600";
	input.dimensions = { 800, 600 };
	input.config.surface_type = SurfaceType::MESH;
	input.config.mesh_name = "inputs/mesh/camelhead.off";
	input.patches_to_remove = {};

	return input;
}

curvenet_input results_bunny_mesh_600_600()
{
	curvenet_input input;
	input.input_name = "bunny_mesh_600_600";
	input.dimensions = { 600, 600 };
	input.config.surface_type = SurfaceType::MESH;
	input.config.mesh_name = "inputs/mesh/bunny.off";
	input.patches_to_remove = {};

	return input;
}

curvenet_input results_bunny_mesh_600_720()
{
	curvenet_input input;
	input.input_name = "bunny_mesh_600_720";
	input.dimensions = { 600, 720 };
	input.config.surface_type = SurfaceType::MESH;
	input.config.mesh_name = "inputs/mesh/bunny.off";
	input.patches_to_remove = {};

	return input;	
}


curvenet_input results_bunny_one_hole_600_783()
{
	curvenet_input input;
	input.input_name = "bunny_one_hole_600_783";
	input.dimensions = { 600, 783 };
	input.config.surface_type = SurfaceType::MESH;
	input.config.mesh_name = "inputs/mesh/bunny_1_hole.off";
	input.patches_to_remove = {};

	return input;
}

curvenet_input results_piggy_600_1000()
{
	curvenet_input input;
	input.input_name = "piggy_600_1000";
	input.dimensions = { 600, 1000};
	input.config.surface_type = SurfaceType::MESH;
	input.config.mesh_name = "inputs/mesh/piggy.obj";
	input.patches_to_remove = {};
	input.predefined_mesh_boundary = { {5543, 5101, 5100, 5155, 3864, 3865 } };
	return input;
}

curvenet_input results_piggy_zoom_600_660()
{
	curvenet_input input;
	input.input_name = "piggy_zoom_600_660";
	input.dimensions = { 600, 660 };
	input.config.surface_type = SurfaceType::MESH;
	input.config.mesh_name = "inputs/mesh/piggy.obj";
	input.patches_to_remove = {};
	input.predefined_mesh_boundary = { {5543, 5101, 5100, 5155, 3864, 3865 } };
	return input;
}

curvenet_input results_teapot_filled_600_262()
{
	curvenet_input input;
	input.input_name = "teapot_filled_600_262";
	input.dimensions = { 600, 262 };
	input.config.surface_type = SurfaceType::MESH;
	input.config.mesh_name = "inputs/mesh/teapot_filled.obj";
	input.patches_to_remove = {};

	input.predefined_mesh_boundary = { { 3450, 3474, 3496, 3517, 3536, 3548, 3559, 3563, 3560, 3549, 3537, 3518, 3497, 3475, 3451, 3431, 3401, 3396, 3400, 3430, } };

	return input;
}


curvenet_input results_julia_vase_11_600_375()
{
	curvenet_input input;
	input.input_name = "julia_vase_11_600_375";
	input.dimensions = { 600, 375 };
	input.config.surface_type = SurfaceType::MESH;
	input.config.mesh_name = "inputs/mesh/julia_vase_11.obj";
	input.patches_to_remove = {};

	return input;
}

curvenet_input results_bunny_one_hole_600_600()
{
	curvenet_input input;
	input.input_name = "bunny_one_hole_600_600";
	input.dimensions = { 600, 600 };
	input.config.surface_type = SurfaceType::MESH;
	input.config.mesh_name = "inputs/mesh/bunny_1_hole.off";
	input.patches_to_remove = {};

	return input;
}

curvenet_input results_SGA_Fighter_100_40_102()
{
	curvenet_input input;
	input.input_name = "SGA_Fighter_vox_100_40_102";
	input.dimensions = { 100, 40, 102 };
	input.config.surface_type = SurfaceType::MINIMAL;
	input.patches_name = "inputs/patches/SGA_Fighter.patches";
	input.patches_to_remove = {};

	return input;
}
curvenet_input results_car_coons_1000_375()
{
	curvenet_input input;
	input.input_name = "Car_simpified_1000_375";
	input.dimensions = { 1000, 375};
	input.config.coons_folder_name = "Car_simpified";
	input.config.num_coons_files = 25;
	input.config.coons_flip_normals.resize(input.config.num_coons_files, false);
	input.config.coons_flip_normals[1] = true;
	input.config.coons_flip_normals[6] = true;
	input.config.coons_flip_normals[23] = true;
	input.config.surface_type = SurfaceType::COONS;
	input.config.parametric_max_t = 3.0;
	input.patches_to_remove = {}; // { 21, 23 };
	//input.patches_to_remove = {0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24};

	return input;
}

curvenet_input results_car_coons_691_259()
{
	curvenet_input input;
	input.input_name = "Car_simpified_691_259";
	input.dimensions = { 691, 259 };
	input.config.coons_folder_name = "Car_simpified";
	input.config.num_coons_files = 25;
	input.config.coons_flip_normals.resize(input.config.num_coons_files, false);
	input.config.coons_flip_normals[1] = true;
	input.config.coons_flip_normals[6] = true;
	input.config.coons_flip_normals[23] = true;
	input.config.surface_type = SurfaceType::COONS;
	input.config.parametric_max_t = 3.0;
	input.patches_to_remove = {}; // { 21, 23 };
	//input.patches_to_remove = {0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24};

	return input;
}

curvenet_input results_car_coons_713_267()
{
	curvenet_input input;
	input.input_name = "Car_simpified_713_267";
	input.dimensions = { 713, 267 };
	input.config.coons_folder_name = "Car_simpified";
	input.config.num_coons_files = 25;
	input.config.coons_flip_normals.resize(input.config.num_coons_files, false);
	input.config.coons_flip_normals[1] = true;
	input.config.coons_flip_normals[6] = true;
	input.config.coons_flip_normals[23] = true;
	input.config.surface_type = SurfaceType::COONS;
	input.config.parametric_max_t = 3.0;
	input.patches_to_remove = {}; // { 21, 23 };
	//input.patches_to_remove = {0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24};

	return input;
}

curvenet_input results_car_coons_1000_285()
{
	curvenet_input input;
	input.input_name = "Car_coons_1000_285";
	input.dimensions = { 1000, 285 };
	input.config.coons_folder_name = "Car_simpified";
	input.config.num_coons_files = 25;
	input.config.coons_flip_normals.resize(input.config.num_coons_files, false);
	input.config.coons_flip_normals[1] = true;
	input.config.coons_flip_normals[6] = true;
	input.config.coons_flip_normals[23] = true;
	input.config.surface_type = SurfaceType::COONS;
	input.config.parametric_max_t = 3.0;
	input.patches_to_remove = {}; // { 21, 23 };
	//input.patches_to_remove = {0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24};

	return input;
}


curvenet_input results_car_coons_694_260()
{
	curvenet_input input;
	input.input_name = "Car_simpified_694_260";
	input.dimensions = { 694, 260 };
	input.config.coons_folder_name = "Car_simpified";
	input.config.num_coons_files = 25;
	input.config.coons_flip_normals.resize(input.config.num_coons_files, false);
	input.config.coons_flip_normals[1] = true;
	input.config.coons_flip_normals[6] = true;
	input.config.coons_flip_normals[23] = true;
	input.config.surface_type = SurfaceType::COONS;
	input.config.parametric_max_t = 3.0;
	input.patches_to_remove = {}; // { 21, 23 };
	//input.patches_to_remove = {0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24};

	return input;
}



curvenet_input results_car_coons_700_262()
{
	curvenet_input input;
	input.input_name = "Car_simpified_700_262";
	input.dimensions = { 700, 262 };
	input.config.coons_folder_name = "Car_simpified";
	input.config.num_coons_files = 25;
	input.config.coons_flip_normals.resize(input.config.num_coons_files, false);
	input.config.coons_flip_normals[1] = true;
	input.config.coons_flip_normals[6] = true;
	input.config.coons_flip_normals[23] = true;
	input.config.surface_type = SurfaceType::COONS;
	input.config.parametric_max_t = 3.0;
	input.patches_to_remove = {}; // { 21, 23 };
	//input.patches_to_remove = {0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24};

	return input;
}

curvenet_input results_car_coons_600_225()
{
	curvenet_input input;
	input.input_name = "Car_simpified_600_225";
	input.dimensions = { 600, 225 };
	input.config.coons_folder_name = "Car_simpified";
	input.config.num_coons_files = 25;
	input.config.coons_flip_normals.resize(input.config.num_coons_files, false);
	input.config.coons_flip_normals[1] = true;
	input.config.coons_flip_normals[6] = true;
	input.config.coons_flip_normals[23] = true;
	input.config.surface_type = SurfaceType::COONS;
	input.config.parametric_max_t = 3.0;
	input.patches_to_remove = {}; // { 21, 23 };
	//input.patches_to_remove = {0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24};

	return input;
}

curvenet_input results_boat_coons_600_437_0()
{
	curvenet_input input;
	input.input_name = "Boat_simplified_600_437_0";
	input.dimensions = { 600, 437 };
	input.config.coons_folder_name = "Boat_simplified";
	input.config.num_coons_files = 30;
	input.config.coons_flip_normals.resize(input.config.num_coons_files, false);
	//input.config.coons_flip_normals[24] = true;
	input.config.flip_patch_orientation.resize(input.config.num_coons_files, false);

	input.config.surface_type = SurfaceType::COONS;
	input.config.parametric_max_t = 3.0;
	input.patches_to_remove = { 0,1,2,3,4,5,6, 7,8,9,11, 12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 }; // keep 10

	return input;
}

curvenet_input results_boat_coons_600_437_1()
{
	curvenet_input input;
	input.input_name = "Boat_simplified_600_437_1";
	input.dimensions = { 600, 437 };
	input.config.coons_folder_name = "Boat_simplified";
	input.config.num_coons_files = 30;
	input.config.coons_flip_normals.resize(input.config.num_coons_files, false);
	//input.config.coons_flip_normals[24] = true;
	input.config.flip_patch_orientation.resize(input.config.num_coons_files, false);

	input.config.surface_type = SurfaceType::COONS;
	input.config.parametric_max_t = 3.0;
	input.patches_to_remove = {0,1,2,3,4,5,6, 7,8,9,10, 12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29}; // keep 11

	return input;
}

curvenet_input results_boat_coons_600_437_2()
{
	curvenet_input input;
	input.input_name = "Boat_simplified_600_437_2";
	input.dimensions = { 600, 437 };
	input.config.coons_folder_name = "Boat_simplified";
	input.config.num_coons_files = 30;
	input.config.coons_flip_normals.resize(input.config.num_coons_files, false);
	//input.config.coons_flip_normals[24] = true;
	input.config.flip_patch_orientation.resize(input.config.num_coons_files, false);

	input.config.surface_type = SurfaceType::COONS;
	input.config.parametric_max_t = 3.0;
	input.patches_to_remove = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29 }; // keep 24

	return input;
}

curvenet_input results_boat_coons_600_437_3()
{
	curvenet_input input;
	input.input_name = "Boat_simplified_600_437_3";
	input.dimensions = { 600, 437 };
	input.config.coons_folder_name = "Boat_simplified";
	input.config.num_coons_files = 30;
	input.config.coons_flip_normals.resize(input.config.num_coons_files, false);
	//input.config.coons_flip_normals[24] = true;
	input.config.flip_patch_orientation.resize(input.config.num_coons_files, false);
	//input.config.flip_patch_orientation[15] = true;
	input.config.flip_patch_orientation[15] = true;

	input.config.surface_type = SurfaceType::COONS;
	input.config.parametric_max_t = 3.0;
	input.patches_to_remove = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14, 24 };

	return input;
}


curvenet_input results_boat_coons_600_437_4()
{
	curvenet_input input;
	input.input_name = "Boat_simplified_600_437_4";
	input.dimensions = { 600, 437 };
	input.config.coons_folder_name = "Boat_simplified";
	input.config.num_coons_files = 30;
	input.config.coons_flip_normals.resize(input.config.num_coons_files, false);
	//input.config.coons_flip_normals[24] = true;
	input.config.flip_patch_orientation.resize(input.config.num_coons_files, false);
	//input.config.flip_patch_orientation[15] = true;
	input.config.flip_patch_orientation[15] = true;

	input.config.surface_type = SurfaceType::COONS;
	input.config.parametric_max_t = 3.0;
	//input.patches_to_remove = { 10,11, 24 }; // remove from experiment 1 & 2
	input.patches_to_remove = { 1,2,3,4, 5, 8, 9, 10, 11, 12,13,14,15,16,17,18,19,20,21,22,23, 24, 25,26,27,28,29 }; // 0, 6, 7
	return input;
}


curvenet_input results_boat_coons_600_437_5()
{
	curvenet_input input;
	input.input_name = "Boat_simplified_600_437_5";
	input.dimensions = { 600, 437 };
	input.config.coons_folder_name = "Boat_simplified";
	input.config.num_coons_files = 30;
	input.config.coons_flip_normals.resize(input.config.num_coons_files, false);
	//input.config.coons_flip_normals[24] = true;
	input.config.flip_patch_orientation.resize(input.config.num_coons_files, false);
	//input.config.flip_patch_orientation[15] = true;
	input.config.flip_patch_orientation[15] = true;

	input.config.surface_type = SurfaceType::COONS;
	input.config.parametric_max_t = 3.0;
	//input.patches_to_remove = { 10,11, 24 }; // remove from experiment 1 & 2
	input.patches_to_remove = {0, 1,2,3,4,5,6,7,8,9,10, 11, 15,16,17,18,19,20,21,22,23, 24, 25,26,27,28,29 }; // 12, 13, 14
	return input;
}

curvenet_input results_boat_coons_600_437_6()
{
	curvenet_input input;
	input.input_name = "Boat_simplified_600_437_6";
	input.dimensions = { 600, 437 };
	input.config.coons_folder_name = "Boat_simplified";
	input.config.num_coons_files = 30;
	input.config.coons_flip_normals.resize(input.config.num_coons_files, false);
	//input.config.coons_flip_normals[24] = true;
	input.config.flip_patch_orientation.resize(input.config.num_coons_files, false);
	//input.config.flip_patch_orientation[15] = true;
	input.config.flip_patch_orientation[15] = true;

	input.config.surface_type = SurfaceType::COONS;
	input.config.parametric_max_t = 3.0;
	//input.patches_to_remove = { 10,11, 24 }; // remove from experiment 1 & 2
	input.patches_to_remove = {0, 6,7,8,9, 10, 11, 12,13,14,15,16,17,18,19,20,21,22,23, 24, 25,26,27,28,29 };

	return input;
}

curvenet_input results_boat_coons_600_437_7()
{
	curvenet_input input;
	input.input_name = "Boat_simplified_600_437_7";
	input.dimensions = { 600, 437 };
	input.config.coons_folder_name = "Boat_simplified";
	input.config.num_coons_files = 30;
	input.config.coons_flip_normals.resize(input.config.num_coons_files, false);
	//input.config.coons_flip_normals[24] = true;
	input.config.flip_patch_orientation.resize(input.config.num_coons_files, false);
	//input.config.flip_patch_orientation[15] = true;
	input.config.flip_patch_orientation[15] = true;

	input.config.surface_type = SurfaceType::COONS;
	input.config.parametric_max_t = 3.0;
	input.patches_to_remove = {0, 1,2,3,4, 5, 6, 7, 9, 10, 11, 12,13,14,15,16,17,18,19,20,21,22,23, 24, 25,26,27,28,29 }; // 8
	return input;
}

curvenet_input results_boat_coons_600_437_8()
{
	curvenet_input input;
	input.input_name = "Boat_simplified_600_437_8";
	input.dimensions = { 600, 437 };
	input.config.coons_folder_name = "Boat_simplified";
	input.config.num_coons_files = 30;
	input.config.coons_flip_normals.resize(input.config.num_coons_files, false);
	//input.config.coons_flip_normals[24] = true;
	input.config.flip_patch_orientation.resize(input.config.num_coons_files, false);
	//input.config.flip_patch_orientation[15] = true;
	input.config.flip_patch_orientation[15] = true;

	input.config.surface_type = SurfaceType::COONS;
	input.config.parametric_max_t = 3.0;
	input.patches_to_remove = { 0, 1,2,3,4, 5, 6, 7, 8, 10, 11, 12,13,14,15,16,17,18,19,20,21,22,23, 24, 25,26,27,28,29 }; // 9
	return input;
}

curvenet_input results_boat_coons_600_437_all()
{
	curvenet_input input;
	input.input_name = "Boat_simplified_600_437_all";
	input.dimensions = { 600, 437 };
	input.config.coons_folder_name = "Boat_simplified";
	input.config.num_coons_files = 30;
	input.config.coons_flip_normals.resize(input.config.num_coons_files, false);
	//input.config.coons_flip_normals[24] = true;
	input.config.flip_patch_orientation.resize(input.config.num_coons_files, false);
	//input.config.flip_patch_orientation[15] = true;
	input.config.flip_patch_orientation[15] = true;

	input.config.surface_type = SurfaceType::COONS;
	input.config.parametric_max_t = 3.0;
	input.patches_to_remove = {};// { 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 }; // 9
	return input;
}


curvenet_input results_bezier_100_100_template()
{
	curvenet_input input;
	input.dimensions = { 100, 100 };
	input.patches_to_remove = {};
	input.input_name = "bezier_100_100";
	input.config.surface_type = SurfaceType::PARAMETRIC;

	Eigen::MatrixXd control_points(10, 3);

	control_points << 0.525074852359172, -0.278903363421520, -0.102120790292900,
		0.267975879627391, -0.272018745807755, 0.180600624945055,
		0.431344779247598, 0.071712035570697, -0.088579682401932,
		-0.153017612368211, -0.194730960250322, -0.034224316835228,
		-0.075251851075666, 0.010472623951161, -0.123097637107601,
		-0.003283903504825, 0.269654133300490, -0.087522050261528,
		-0.528835002641230, -0.352302828703711, -0.058733610651726,
		-0.360043137505598, 0.227047086288067, 0.123799708915016,
		-0.155802131082939, 0.268691296953398, -0.057550640014390,
		-0.011479503767503, 0.513635693551943, 0.079936366358547;

	std::function<Eigen::Vector3d(Eigen::Vector2d)> cubic_bez_func = create_bezier_func(control_points);

	input.config.parametric_func = cubic_bez_func;
	input.config.parametric_max_t = 1.5;
	input.config.is_in_parametric_domain = [](Eigen::Vector2d uv) { return uv(0) + uv(1) <= 1 && uv(0) >= 0 && uv(1) >= 0;  };
	input.config.patch_from_uv_triangle = true;
	input.config.is_override_output_name = true;
	return input;

}

curvenet_input results_bezier_100_100_gt()
{
	curvenet_input input;
	input.dimensions = { 100, 100 };
	input.patches_to_remove = {};
	input.input_name = "bezier_100_100";
	input.config.alec_only = true;
	input.config.mesh_name = "inputs/mesh/bezier_cubic_triangle_gt.obj";
	input.config.surface_type = SurfaceType::MESH;
	return input;
}


std::vector<std::function<curvenet_input(void)>> create_bezier_tests()
{
	std::vector<int> samples;
	samples.resize(20);
	for (int i = 0; i < samples.size(); i++)
	{
		samples[i] = 99 + i * 33;
	}

	std::vector<std::function<curvenet_input(void)>> bezier_test_out;

	//for (int i = 8; i < 9; i++)
	for (int i = 0; i < samples.size(); i++)
	{
		curvenet_input input = results_bezier_100_100_template();
		input.config.num_patch_points_from_uv = samples[i];
		input.config.override_output_name = "bezier_100_100_" + std::to_string(input.config.num_patch_points_from_uv);

		bezier_test_out.push_back([=](void) { return input; });
	}

	return bezier_test_out;
}

curvenet_input results_debug_side_600_327()
{
	curvenet_input input;
	input.input_name = "bug_side_600_327";
	input.patches_name = "inputs/patches/bug.patches";
	input.dimensions = { 600, 327 };
	input.patches_to_remove = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, };
	return input;
}

curvenet_input results_water_bezier_100_100()
{
	curvenet_input input;
	input.input_name = "water_100_100";
	input.dimensions = { 100, 100 };


	return input;
}

//int main()
//{
//	Eigen::IOFormat OctaveFmt(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]");
//	std::function<Eigen::Vector3d(Eigen::Vector2d)> func = infinite_singular_func;
//	std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> jac_func = infinite_singular_func_jac;
//
//	Eigen::MatrixXd Q = slice_XZ(5);
//	std::vector<std::vector<int>> slice_rays = rays_from_slice(Q);
//	return 0;
//}


curvenet_input results_infinite_singular_800()
{
	curvenet_input input;
	input.input_name = "infinite_singularity_800_800";
	input.patches_name = "inputs/patches/large_circle.patches";
	input.dimensions = { 800, 800 };
	input.config.parametric_max_t = 25.0;
	input.config.parametric_func = infinite_singular_func;
	input.config.surface_type = SurfaceType::PARAMETRIC;
	input.config.use_fd = false;
	input.config.parametric_jac = infinite_singular_func_jac;
	input.config.is_in_parametric_domain = [=](Eigen::Vector2d) { return true; };
	return input;
}

curvenet_input results_infinite_singular_100()
{
	curvenet_input input;
	input.input_name = "infinite_singularity_100_100";
	input.patches_name = "inputs/patches/large_circle.patches";
	input.dimensions = { 100, 100 };
	input.config.parametric_max_t = 25.0;
	input.config.parametric_func = infinite_singular_func;
	input.config.surface_type = SurfaceType::PARAMETRIC;
	input.config.use_fd = false;
	input.config.parametric_jac = infinite_singular_func_jac;
	input.config.is_in_parametric_domain = [=](Eigen::Vector2d) { return true; };
	return input;
}


curvenet_input results_infinite_singular_10_10()
{
	curvenet_input input;
	input.input_name = "infinite_singularity_10_10";
	input.patches_name = "inputs/patches/large_circle.patches";
	input.dimensions = { 10, 10 };
	input.config.parametric_max_t = 10.0;
	input.config.parametric_func = infinite_singular_func;
	input.config.surface_type = SurfaceType::PARAMETRIC;
	input.config.use_fd = false;
	input.config.parametric_jac = infinite_singular_func_jac;
	input.config.is_in_parametric_domain = [=](Eigen::Vector2d){ return true; };
	return input;
}

curvenet_input results_revolved_100_100()
{
	curvenet_input input;
	input.input_name = "revolved_100";
	input.patches_name = "inputs/patches/revolved_circle_zhalf.patches";
	input.dimensions = { 100, 100 };
	input.config.parametric_max_t = 3.0;
	input.config.parametric_func = parametric_vase_cubic;
	input.config.parametric_jac = parametric_vase_cubic_jac;
	input.config.perform_self_intersections = false;
	input.config.surface_type = SurfaceType::PARAMETRIC;
	input.config.use_fd = true;
	return input;
}

curvenet_input results_revolved_800_800()
{
	curvenet_input input;
	input.input_name = "revolved_800";
	input.patches_name = "inputs/patches/revolved_circle_zhalf.patches";
	input.dimensions = { 800, 800 };
	input.config.parametric_max_t = 3.0;
	input.config.parametric_func = parametric_vase_cubic;
	input.config.parametric_jac = parametric_vase_cubic_jac;
	input.config.perform_self_intersections = false;
	input.config.surface_type = SurfaceType::PARAMETRIC;
	input.config.use_fd = false;
	return input;
}

curvenet_input results_revolved_2_800_800()
{
	curvenet_input input;
	input.input_name = "revolved_800_2";
	input.patches_name = "inputs/patches/revolved_circle_zhalf.patches";
	input.dimensions = { 800, 800 };
	input.config.parametric_max_t = 3.0;
	input.config.parametric_func = parametric_vase_cubic_2;
	input.config.parametric_jac = parametric_vase_cubic_2_jac;
	input.config.perform_self_intersections = false;
	input.config.surface_type = SurfaceType::PARAMETRIC;
	input.config.use_fd = false;
	return input;
}

curvenet_input results_revolved_500_500()
{
	curvenet_input input;
	input.input_name = "revolved_500";
	input.patches_name = "inputs/patches/revolved_circle_zhalf.patches";
	input.dimensions = { 500, 500 };
	input.config.parametric_max_t = 3.0;
	input.config.parametric_func = parametric_vase_cubic;
	input.config.parametric_jac = parametric_vase_cubic_jac;
	input.config.perform_self_intersections = false;
	input.config.surface_type = SurfaceType::PARAMETRIC;
	input.config.use_fd = false;
	return input;
}

curvenet_input results_double_torus_open()
{
	curvenet_input input;
	input.input_name = "torus_4_3_800";
	input.patches_name = "inputs/patches/torus_4_3.patches";
	input.dimensions = { 400, 400 };
	input.config.parametric_max_t = 9.0;
	input.config.is_in_parametric_domain = [](Eigen::Vector2d uv) {return !(uv(0) >= 0.3 && uv(0) <= 0.7 && uv(1) >= 0.5 && uv(1) <= 1.0) && uv(0) >= 0.0 && uv(0) <= 1.0 && uv(1) >= 0 && uv(1) <= 1.0; };
	input.config.parametric_func = parametric_torus_4_3;
	input.config.perform_self_intersections = true;
	input.config.surface_type = SurfaceType::PARAMETRIC;
	input.config.use_fd = true;
	return input;
}

curvenet_input results_torus_open()
{
	curvenet_input input;
	input.input_name = "torus_reg_800";
	input.patches_name = "inputs/patches/torus_reg.patches";
	input.dimensions = { 800, 800 };
	input.config.parametric_max_t = 3.0;
	input.config.is_in_parametric_domain = [](Eigen::Vector2d uv) {return !(uv(0) >= 0.0 && uv(0) <= 0.5 && uv(1) >= 0.3 && uv(1) <= 0.7) && uv(0) >= 0.0 && uv(0) <= 1.0 && uv(1) >= 0 && uv(1) <= 1.0; };
	input.config.parametric_func = parametric_torus;
	input.config.perform_self_intersections = true;
	input.config.surface_type = SurfaceType::PARAMETRIC;
	input.config.use_fd = true;
	return input;
}

int main()
{
	//Eigen::IOFormat OctaveFmt(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]");
	//Eigen::MatrixXd mesh_grid;
	//Eigen::MatrixXi faces;
	//uniform_mesh_grid(100, mesh_grid, faces);
	//Eigen::MatrixXd V = apply_function_to_uv(mesh_grid, parametric_torus);
	//write_mesh_to_obj("torus.obj", V, faces);
	//return 0;
	/*std::vector<double> diff;
	auto fd_jac = fd_jacobian(parametric_vase_cubic_2, 1e-12);
	std::vector<Eigen::MatrixXd> jacs;
	std::vector<Eigen::MatrixXd> jacs_fd;
	for (int i = 0; i < 200; i++)
	{
		for (int j = 0; j < 200; j++)
		{
			double u = static_cast<double>(i) / 200.0;
			double v = static_cast<double>(j) / 200.0;

			Eigen::MatrixXd jac_fd = fd_jac(Eigen::Vector2d(u, v));
			Eigen::MatrixXd jac = parametric_vase_cubic_2_jac(Eigen::Vector2d(u, v));
			Eigen::MatrixXd diff_sq = (jac - jac_fd).array().square();
			diff.push_back(diff_sq.sum());
			jacs.push_back(jac);
			jacs_fd.push_back(jac_fd);
		}
	}

	std::cout << std::accumulate(diff.begin(), diff.end(), 0.0) << std::endl;
	std::vector<double>::iterator worse_index = std::max_element(diff.begin(), diff.end());
	int worse_index_int = std::distance(diff.begin(), worse_index);

	std::cout << *worse_index << std::endl;
	std::cout << "jac:" << std::endl;
	std::cout << jacs[worse_index_int] << std::endl;
	std::cout << "jac_fd:" << std::endl;
	std::cout << jacs_fd[worse_index_int] << std::endl;
	
	
	return 0;*/

	std::vector<std::function<curvenet_input(void)>> inputs_to_run;
	/*std::vector<std::function<curvenet_input(void)>> bezier_inputs = create_bezier_tests(); */

	inputs_to_run.push_back(results_double_torus_open);
	//inputs_to_run.push_back(results_infinite_singular_800);
	//inputs_to_run.push_back(bezier_inputs[10]);
	//inputs_to_run.push_back(results_bezier_100_100_gt);
	//for (auto b : bezier_inputs)
	//	inputs_to_run.push_back(b);
	//inputs_to_run.push_back(results_julia_vase_11_600_375);
	//inputs_to_run.push_back(results_bunny_one_hole_600_600);
	//inputs_to_run.push_back(results_camel_head_800_600);
	//inputs_to_run.push_back(results_piggy_600_1000);
	//inputs_to_run.push_back(results_piggy_zoom_600_660);

	//inputs_to_run.push_back(results_revolved_800_800);

	//inputs_to_run.push_back(results_gear_1000_499);
	//inputs_to_run.push_back(results_gear_right_1000_499);
	// 
	//inputs_to_run.push_back(results_gear_zoom_500_500);
	//inputs_to_run.push_back(results_gear_zoom_right_500_500);

	//inputs_to_run.push_back(results_gear_alec_only_1000_499);
	//inputs_to_run.push_back(results_gear_right_alec_only_1000_499);

	//inputs_to_run.push_back(results_gear_zoom_alec_only_500_500);
	//inputs_to_run.push_back(results_gear_zoom_right_alec_only_500_500);

	//inputs_to_run.push_back(results_car_coons_1000_285);
	//inputs_to_run.push_back(results_bug_1000_285);
	//inputs_to_run.push_back(results_car_coons_691_259);
	//inputs_to_run.push_back(results_car_coons_694_260);
	//inputs_to_run.push_back(results_car_coons_713_267);

	/*inputs_to_run.push_back(results_boat_coons_600_437_0);
	inputs_to_run.push_back(results_boat_coons_600_437_1);
	inputs_to_run.push_back(results_boat_coons_600_437_2);
	inputs_to_run.push_back(results_boat_coons_600_437_3);
	inputs_to_run.push_back(results_boat_coons_600_437_4);
	inputs_to_run.push_back(results_boat_coons_600_437_5);
	inputs_to_run.push_back(results_boat_coons_600_437_6);*/
	//inputs_to_run.push_back(results_boat_coons_600_437_7);
	//inputs_to_run.push_back(results_boat_coons_600_437_8);
	//inputs_to_run.push_back(results_boat_coons_600_437_all);

	//inputs_to_run.push_back(results_sga_fighter_100_40_102);



	//inputs_to_run.push_back(results_roll_500_500);



	//inputs_to_run.push_back(results_bug_side_600_327);
	//inputs_to_run.push_back(results_bug_top_windows_600_514);
	//inputs_to_run.push_back(results_hand_four_fingers_500_272);
	//inputs_to_run.push_back(results_hand_one_finger_500_272);
	//inputs_to_run.push_back(results_enterprise_200_1250);
	//inputs_to_run.push_back(results_siggraphspacecraft37_500_227);

	//inputs_to_run.push_back(results_sga_fighter_500_208);
	//inputs_to_run.push_back(results_siggraphspacecraft19_500_104);

	//inputs_to_run.push_back(results_siggraphspacecraft23_500_500);

	//inputs_to_run.push_back(results_siggraphspacecraft26_500_227);
	//inputs_to_run.push_back(results_siggraphspacecraft26_side_500_227);

	//inputs_to_run.push_back(results_siggraphspacecraft38_500_113);
	//inputs_to_run.push_back(results_siggraphspacecraft56_500_113);
	//inputs_to_run.push_back(results_siggraphspacecraft87_500_375);

	/*inputs_to_run.push_back(results_singularity_500_500);
	inputs_to_run.push_back(results_roll_500_500);*/
	//inputs_to_run.push_back(results_circular_tube_500_500);

	//inputs_to_run.push_back(results_phoenix_500_333);
	// 


	//inputs_to_run.push_back(results_sga_fighter_vox_200_81_204);
	//inputs_to_run.push_back(results_sga_hawk_vox_210_112_80);

	//inputs_to_run.push_back(results_sga_doghead_vox_170_106_210);

	//inputs_to_run.push_back(results_bunny_vox_220_215_166);

	for (int i = 0; i < inputs_to_run.size(); i++)
	{
		curvenet_input input = inputs_to_run[i]();

		if (input.config.alec_only)
			continue;

		std::cout << "running: " << input.input_name << std::endl;
		const std::string patches_filename = input.patches_name;
		const std::string query_points_name = input.input_name;
		const std::string query_points_filename = "inputs/query_points/" + query_points_name + ".points";
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





		//eigen::ioformat numpy_fmt(eigen::fullprecision, 0, ", ", ",\n", "[", "]", "[", "]");
		//std::cout << project_to_sphere(patches[0].curve, query_points.row(5388 - 1)).format(numpy_fmt) << std::endl << std::endl;

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

	//	std::cout << "Done 1s" << std::endl;
	//	auto before_aq = std::chrono::high_resolution_clock::now();
	//	winding_number_results wn_res_aq;
	//	wn_res_aq.gwn = winding_numbers_adaptive(parametric_vase_cubic, parametric_vase_cubic_jac, query_points, 15, 1e-7);
	//	auto after_aq = std::chrono::high_resolution_clock::now();
	//	std::cout << "Done AQ" << std::endl;
	//
	//	input.config.is_override_output_name = true;
	//	input.config.override_output_name = input.input_name + "_aq";
	//	//write_gwn_to_file(wn_res_aq, input);

	///*	Eigen::MatrixXd mesh_grid;
	//	Eigen::MatrixXi faces;
	//	uniform_mesh_grid(250, mesh_grid, faces);
	//	Eigen::MatrixXd V = apply_function_to_uv(mesh_grid, parametric_vase_cubic_2);
	//	write_mesh_to_obj("parametric_vase_2.obj", V, faces);*/


	//	auto before_gwn = std::chrono::high_resolution_clock::now();
	//	/*winding_number_results wn_res_gwn;
	//	gwn_timing_info gwn_timing;
	//	wn_res_gwn.gwn = gwn_from_meshes({ V }, { faces}, query_points, gwn_timing); */
	//	auto after_gwn = std::chrono::high_resolution_clock::now();
	//	input.config.override_output_name = input.input_name + "_alec";
	//	//write_gwn_to_file(wn_res_gwn, input);

	//	

	//	auto diff_1s = std::chrono::duration_cast<std::chrono::milliseconds>(after_1s - before_1s);
	//	auto diff_aq = std::chrono::duration_cast<std::chrono::milliseconds>(after_aq - before_aq);
	//	auto diff_gwn = std::chrono::duration_cast<std::chrono::milliseconds>(after_gwn - before_gwn);

	//	std::cout << "Elapsed time 1S: " << diff_1s.count() << " ms" << std::endl;
	//	std::cout << "Elapsed time AQ: " << diff_aq.count() << " ms" << std::endl;
	//	std::cout << "Elpased time GWN: " << diff_gwn.count() << " ms" << std::endl;
	}
	

	//for (int i = 0; i < inputs_to_run.size(); i++)
	//{
	//	curvenet_input input = inputs_to_run[i]();
	//	std::cout << "running (alec's gwn) : " << input.input_name << std::endl;
	//	const std::string patches_filename = input.patches_name;
	//	const std::string query_points_name = input.input_name;
	//	const std::string query_points_filename = "inputs/query_points/" + query_points_name + ".points";
	//	std::vector<int> dimensions = input.dimensions;

	//	std::vector<eigen::matrixxd> vs;
	//	std::vector<eigen::matrixxi> fs;

	//	std::vector<patch_t> patches;

	//	gwn_timing_info gwn_timing;
	//	if (input.config.surface_type == surfacetype::mesh)
	//	{
	//		size_t pos = input.config.mesh_name.rfind(".off");
	//		if (pos != std::string::npos && pos == (input.config.mesh_name.size() - std::string(".off").size())) {
	//			read_off(input.config.mesh_name, input.config.mesh_vertices, input.config.mesh_faces);
	//		}
	//		pos = input.config.mesh_name.rfind(".obj");
	//		if (pos != std::string::npos && pos == (input.config.mesh_name.size() - std::string(".obj").size())) {
	//			read_obj(input.config.mesh_name, input.config.mesh_vertices, input.config.mesh_faces);
	//		}

	//		patches = extract_mesh_boundary(input.config.mesh_vertices, input.config.mesh_faces);

	//		vs.push_back(input.config.mesh_vertices);
	//		fs.push_back(input.config.mesh_faces);

	//		gwn_timing.total_faces += input.config.mesh_faces.rows();
	//	}
	//	else
	//	{
	//		patches = load_all_patches(patches_filename);
	//		patches = remove_patches(patches, input.patches_to_remove);

	//		for (int patch_index = 0; patch_index < patches.size(); patch_index++)
	//		{
	//			eigen::matrixxd v;
	//			eigen::matrixxi f;
	//			mesh_implicit_from_boundary(v, f, patches[patch_index].curve, gwn_timing);
	//			vs.push_back(v);
	//			fs.push_back(f);
	//			gwn_timing.total_faces += f.rows();
	//		}
	//	}


	//
	//	eigen::matrixxd query_points = load_query_points(query_points_filename);


	//	gwn_timing.num_patches = patches.size();

	//	std::vector<double> gwn = gwn_from_meshes(vs, fs, query_points, gwn_timing);
	//	double hgwn_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gwn_timing.winding_number_result_time).count();
	//	std::cout << "alec hgwn time: " << hgwn_time_ms << " ms" << std::endl;

	//	write_fem_gwn_to_file(gwn, gwn_timing, input);
	//}

	return 0;
}
