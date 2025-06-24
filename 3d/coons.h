#pragma once

#include "intersections.h"
#include "patch.h"
#include <iostream>
#include <gsl/gsl_multiroots.h>
#include <random>
#include <vector>
typedef std::function<Eigen::Vector3d(Eigen::Vector2d)> curve_3d_func_t;
typedef std::function<Eigen::Vector3d(double)> curve_3d_param_func_t;
struct coons_patch {
	curve_3d_param_func_t c0;
	curve_3d_param_func_t c1;
	curve_3d_param_func_t d0;
	curve_3d_param_func_t d1;

	curve_3d_param_func_t dc0;
	curve_3d_param_func_t dc1;
	curve_3d_param_func_t dd0;
	curve_3d_param_func_t dd1;


	std::vector<Eigen::Vector3d> c0_v;
	std::vector<Eigen::Vector3d> c1_v;
	std::vector<Eigen::Vector3d> d0_v;
	std::vector<Eigen::Vector3d> d1_v;
	curve_3d_func_t func;
	bool is_closed;
};

Eigen::Matrix<double, 3, 2> coons_jacobian(double s, double t, const coons_patch& patch);

struct coon_patch_sampler {
	std::vector<coons_patch> c_patch;
	std::vector<Eigen::Vector3d> sample_points;
	bool is_closed;
	std::vector<std::vector<int>> num_regions;
	std::vector<std::vector<std::vector<Eigen::Vector3d>>> rays;
	std::vector<std::vector<std::vector<double>>> areas;
	std::vector<std::vector<std::vector<int>>> rel_wn;
};

struct coon_solver_params {
	coons_patch patch;
	Eigen::Vector3d point;
	Eigen::Vector3d dir;
};

coons_patch build_coon_patch(curve_3d_param_func_t c0, curve_3d_param_func_t c1, curve_3d_param_func_t d0, curve_3d_param_func_t d1);
coons_patch build_discrete_coon_patch(const std::vector<Eigen::Vector3d>& c0, const std::vector<Eigen::Vector3d>& c1, const std::vector<Eigen::Vector3d>& d0, const std::vector<Eigen::Vector3d>& d1);
chi_result find_chi_coon(gsl_multiroot_fsolver* s, const coon_solver_params& params, double t_min, double t_max, bool abort_bad_ray);
coons_patch bohemian_dome();
int f_coon(const gsl_vector* x, void* p, gsl_vector* f);
Eigen::Matrix<double, 3, 2> bohemian_dome_jac(Eigen::Vector2d uv);

std::vector<std::vector<std::vector<int>>> find_all_chis_parametric_oneshot(const coon_patch_sampler& sampler);
std::vector<int> find_all_chis_parametric_random_rays(const coon_patch_sampler& sampler);
std::vector<std::vector<std::vector<int>>> find_all_chis_parametric_from_regions(const coon_patch_sampler& sampler);
std::vector<double> gwn_from_chis_parametric(const std::vector<std::vector<std::vector<int>>>& res_chis, const coon_patch_sampler& coon_sampler);
coon_patch_sampler coon_sampler_from_file(const std::string& filename);


void load_coons_patches_from_objs(const std::string& folder_name, const std::string& base_filename, int num_patches, std::vector<patch_t>& surface_boundaries, std::vector<coons_patch>& coon_patches);
std::vector<coons_patch> remove_coons_patches(const std::vector<coons_patch>& patches, const std::vector<int>& ids_to_remove);
std::vector<bool> remove_bool_vec(const std::vector<bool>& patches, const std::vector<int>& ids_to_remove);
std::vector<int> remove_int_vec(int num_patches, const std::vector<int>& ids_to_remove);