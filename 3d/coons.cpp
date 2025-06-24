#include "coons.h"
#include "math_util.h"
#include "mesh.h"
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>
#include <igl/boundary_loop.h>

coons_patch build_coons_patch(curve_3d_param_func_t c0, curve_3d_param_func_t c1, curve_3d_param_func_t d0, curve_3d_param_func_t d1)
{
	coons_patch patch;
	patch.c0 = c0;
	patch.c1 = c1;
	patch.d0 = d0;
	patch.d1 = d1;

	curve_3d_func_t Lc = [=](Eigen::Vector2d in) {
		double s = in(0);
		double t = in(1);
		Eigen::Vector3d c0s = c0(s);
		Eigen::Vector3d c1s = c1(s);
		Eigen::Vector3d res = (1. - t) * c0s + t * c1s;
		return res;
	};

	curve_3d_func_t Ld = [=](Eigen::Vector2d in) {
		double s = in(0);
		double t = in(1);
		Eigen::Vector3d res = (1. - s) * d0(t) + s * d1(t);
		return res;
	};

	curve_3d_func_t bil = [=](Eigen::Vector2d in) {
		double s = in(0);
		double t = in(1);
		Eigen::Vector3d res = c0(0) * (1. - s) * (1. - t) + c0(1.) * s * (1. - t) + c1(0.0) * (1. - s) * t + c1(1.0) * s * t;
		return res;
	};

	patch.func = [=](Eigen::Vector2d in) { 
		//if(in(0) < 0.0 || in(0) > 1.0 || in(1) < 0.0 || in(1) > 1.0 || std::isnan(in(0)) || std::isnan(in(1)))
		//{ 
		//	double inf = std::numeric_limits<double>::infinity();
		//	return Eigen::Vector3d(inf, inf, inf);
		//}
		Eigen::Vector3d res = Lc(in) + Ld(in) - bil(in);
		return res;
	};

	return patch;
}
//
Eigen::Matrix<double, 3, 2> coons_jacobian(double s, double t, const coons_patch& patch)
{
	Eigen::Matrix<double, 3, 2> jac;
	Eigen::Vector3d dLcds = (1 - t) * patch.dc0(s) + t * patch.dc1(s);
	Eigen::Vector3d dLcdt = -patch.c0(s) + patch.c1(s);
	Eigen::Vector3d dLdds = -patch.d0(t) + patch.d1(t);
	Eigen::Vector3d dLddt = (1 - s) * patch.dd0(t) + s * patch.dd1(t);
	Eigen::Vector3d dBds = -patch.c0(0) * (1 - t) + patch.c0(1) * (1 - t) - patch.c1(0) * t + patch.c1(1) * t;
	Eigen::Vector3d dBdt = -patch.c0(0) * (1 - s) - patch.c0(1) * s * t + patch.c1(0) * (1-s) + patch.c1(1) * s;
	Eigen::Vector3d dCds = dLcds + dLdds - dBds;
	Eigen::Vector3d dCdt = dLcdt + dLddt - dBdt;
	jac(0, 0) = dCds(0);
	jac(1, 0) = dCds(1);
	jac(2, 0) = dCds(2);

	jac(0, 1) = dCdt(0);
	jac(1, 1) = dCdt(1);
	jac(2, 1) = dCdt(2);

	double eps = 1e-9;
	Eigen::Vector3d df_ds = (patch.func(Eigen::Vector2d(s + eps, t))- patch.func(Eigen::Vector2d(s - eps, t))) / (2.*eps);
	Eigen::Vector3d df_dt = (patch.func(Eigen::Vector2d(s, t + eps)) - patch.func(Eigen::Vector2d(s, t - eps))) / (2. * eps);

	std::cout << (df_ds - jac.col(0)).norm() << std::endl;
	std::cout << (df_dt - jac.col(1)).norm() << std::endl;

	jac.col(0) = df_ds;
	jac.col(1) = df_dt;
	return jac;
}
//int f_coon(const gsl_vector* x, void* p, gsl_vector* f)
//{
//	coon_solver_params* params = (coon_solver_params*)p;
//	Eigen::Vector3d point = params->point;
//	Eigen::Vector3d dir = params->dir;
//	coons_patch patch = params->patch;
//	const double u = gsl_vector_get(x, 0);
//	const double v = gsl_vector_get(x, 1);
//	const double t = gsl_vector_get(x, 2);
//
//	//if (std::isnan(u) || std::isnan(v) || std::isnan(t))
//	//{
//	//	double inf = std::numeric_limits<double>::infinity();
//	//	gsl_vector_set(f, 0, inf);
//	//	gsl_vector_set(f, 1, inf);
//	//	gsl_vector_set(f, 2, inf);
//	//	return GSL_SUCCESS;
//	//}
//
//	Eigen::Vector3d uvt = Eigen::Vector3d(u, v, t);
//	Eigen::Vector3d output = patch.func(Eigen::Vector2d(u, v)) - (point + dir * t);
//
//	//std::cout << "pts = [";
//	//for (int i = 0; i < 600; i++)
//	//{
//	//	Eigen::Vector3d p = patch.func(Eigen::Vector2d(static_cast<double>(rand()) / RAND_MAX, static_cast<double>(rand()) / RAND_MAX));
//	//	std::cout << p.x() << ", " << p.y() << ", " << p.z() << "; ";
//	//}
//
//	//std::cout << "];" << std::endl;
//	//std::cout << interpolate_curve(0.8, patch.c0_v) << std::endl << std::endl;
//	//std::cout << patch.func(Eigen::Vector2d(0.2, 0.0)) << std::endl << std::endl;
//	//std::cout << patch.func(Eigen::Vector2d(0.8, 0.0)) << std::endl << std::endl;;
//	//std::cout << patch.func(Eigen::Vector2d(0.0, 0.8)) << std::endl << std::endl;;
//	//std::cout << patch.func(Eigen::Vector2d(0.8, 1)) << std::endl << std::endl;;
//
//
//	//Eigen::Vector3d p0 = patch.func(Eigen::Vector2d(u, v));
//	Eigen::Matrix<double, 3, 2> fd_jac;
//
//	double eps = 1e-8;
//
//	//Eigen::Vector3d fu1 = patch.func(Eigen::Vector2d(u + eps, v));
//	//Eigen::Vector3d fu2 = patch.func(Eigen::Vector2d(u - eps, v));
//	//Eigen::Vector3d fu = fu2 - fu1;
//	//Eigen::Vector3d df_du = (fu) / (2 * eps);
//
//
//	/*Eigen::Vector3d f_u = (patch.func(Eigen::Vector2d(u + eps, v)) - patch.func(Eigen::Vector2d(u - eps, v))) / (2. * eps);
//	Eigen::Vector3d f_v = (patch.func(Eigen::Vector2d(u, v + eps)) - patch.func(Eigen::Vector2d(u, v - eps))) / (2. * eps);
//	fd_jac.col(0) = f_u;
//	fd_jac.col(1) = f_v;
//	std::cout << fd_jac << std::endl << std::endl;*/
//
//	//Eigen::Matrix<double, 3, 2> cedric_jac = coon_jacobian(u, v, patch);
//
//	//std::cout << cedric_jac - fd_jac << std::endl << std::endl;
//
//	/*std::cout << u << " " << v << " " << t << " => " << output(0) << " " << output(1) << " " << output(2) << std::endl << std::endl;*/
//
//	gsl_vector_set(f, 0, output(0));
//	gsl_vector_set(f, 1, output(1));
//	gsl_vector_set(f, 2, output(2));
//	return GSL_SUCCESS;
//}


typedef std::function<Eigen::Vector3d(double)> curve_3d_param_func_t;
coons_patch build_discrete_coons_patch(const std::vector<Eigen::Vector3d>& c0, const std::vector<Eigen::Vector3d>& c1, const std::vector<Eigen::Vector3d>& d0, const std::vector<Eigen::Vector3d>& d1)
{
	curve_3d_param_func_t c0_f = [=](double t) { return interpolate_curve(t, c0); };
	curve_3d_param_func_t c1_f = [=](double t) { return interpolate_curve(t, c1); };
	curve_3d_param_func_t d0_f = [=](double t) { return interpolate_curve(t, d0); };
	curve_3d_param_func_t d1_f = [=](double t) { return interpolate_curve(t, d1); };

	std::vector<Eigen::Vector3d> dc0, dc1, dd0, dd1;

	for (int i = 0; i < c0.size(); i++)
	{
		int next = (i + 1) % c0.size();
		dc0.push_back(c0[next] - c0[i]);
	}

	for (int i = 0; i < c1.size(); i++)
	{
		int next = (i + 1) % c1.size();
		dc1.push_back(c1[next] - c1[i]);
	}

	for (int i = 0; i < d0.size(); i++)
	{
		int next = (i + 1) % d0.size();
		dd0.push_back(d0[next] - d0[i]);
	}

	for (int i = 0; i < d1.size(); i++)
	{
		int next = (i + 1) % d1.size();
		dd1.push_back(d1[next] - d1[i]);
	}
	
	curve_3d_param_func_t dc0_f = [=](double t) { return interpolate_curve(t, dc0); };
	curve_3d_param_func_t dc1_f = [=](double t) { return interpolate_curve(t, dc1); };
	curve_3d_param_func_t dd0_f = [=](double t) { return interpolate_curve(t, dd0); };
	curve_3d_param_func_t dd1_f = [=](double t) { return interpolate_curve(t, dd1); };
	coons_patch c_patch = build_coons_patch(c0_f, c1_f, d0_f, d1_f);

	c_patch.c0_v = c0;
	c_patch.c1_v = c1;
	c_patch.d0_v = d0;
	c_patch.d1_v = d1;
	c_patch.dc0 = dc0_f;
	c_patch.dc1 = dc1_f;
	c_patch.dd0 = dd0_f;
	c_patch.dd1 = dd1_f;

	return c_patch;
}
#include <igl/triangulated_grid.h>
#include <igl/writeOBJ.h>

void load_coons_patches_from_objs(const std::string& folder_name, const std::string& base_filename, int num_patches, std::vector<patch_t>& surface_boundaries, std::vector<coons_patch>& coons_patches)
{
	for (int i = 0; i < num_patches; i++)
	{
		std::string filename = folder_name + "/" + base_filename + "/" + base_filename + "_o" + std::to_string(i) + ".obj";

		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		read_obj(filename, V, F);

		std::vector<int> c0;
		std::vector<int> c1;
		std::vector<int> d0;
		std::vector<int> d1;

		std::ifstream coons_data(folder_name + "/" + base_filename + "/" + base_filename + "_o" + std::to_string(i) + ".c_patch");
		assert(coons_data.is_open());

		std::vector<std::vector<int>> coons_funcs;
		for (int coons_func_index = 0; coons_func_index < 4; coons_func_index++)
		{
			std::vector<int> c_func;
			int num_values;
			coons_data >> num_values;
			for (int k = 0; k < num_values; k++)
			{
				int v;
				coons_data >> v;
				c_func.push_back(v);
			}

			coons_funcs.push_back(c_func);
		}

		c0 = coons_funcs[0];
		d1 = coons_funcs[1];
		d0 = coons_funcs[2];
		c1 = coons_funcs[3];

		std::vector<std::vector<Eigen::Index>> boundary_loops;
		igl::boundary_loop(F, boundary_loops);
		assert(boundary_loops.size() > 0);
		std::vector<Eigen::Index> boundary = boundary_loops[0];

		std::vector<Eigen::Vector3d> c0_v(c0.size());
		std::vector<Eigen::Vector3d> c1_v(c1.size());
		std::vector<Eigen::Vector3d> d0_v(d0.size());
		std::vector<Eigen::Vector3d> d1_v(d1.size());
		for (int k = 0; k < c0.size(); k++)
			c0_v[k] = V.row(c0[k]);
		for (int k = 0; k < c1.size(); k++)
			c1_v[k] = V.row(c1[k]);
		for (int k = 0; k < d0.size(); k++)
			d0_v[k] = V.row(d0[k]);
		for (int k = 0; k < d1.size(); k++)
			d1_v[k] = V.row(d1[k]);
			
		coons_patch c_patch = build_discrete_coons_patch(c0_v, c1_v, d0_v, d1_v);
		coons_patches.push_back(c_patch);

		patch_t patch;
		patch.is_open = false;
		patch.curve.resize(boundary.size(), 3);
		for (int k = 0; k < boundary.size(); k++)
			patch.curve.row(k) = V.row(boundary[k]);

		surface_boundaries.push_back(patch);

		//int rows = 40;
		//int cols = 40;

		//Eigen::MatrixXd uv_vertex_positions;
		//Eigen::MatrixXi triangulated_faces;

		//igl::triangulated_grid(rows, cols, uv_vertex_positions, triangulated_faces);
		//Eigen::MatrixXd output(uv_vertex_positions.rows(), 3);
		//for (int k = 0; k < uv_vertex_positions.rows(); k++)
		//	output.row(k) = c_patch.func(uv_vertex_positions.row(k));

		//igl::writeOBJ(std::string("boat_patch_") + std::to_string(i) + ".obj", output, triangulated_faces);

		//std::cout << "c0: ";
		//for (int k = 0; k < c0.size(); k++)
		//	std::cout << c0[k] << ",";
		//std::cout << std::endl << "c1: ";
		//for (int k = 0; k < c1.size(); k++)
		//	std::cout << c1[k] << ",";
		//std::cout << std::endl << "d0: ";
		//for (int k = 0; k < d0.size(); k++)
		//	std::cout << d0[k] << ",";
		//std::cout << std::endl << "d1: ";
		//for (int k = 0; k < d1.size(); k++)
		//	std::cout << d1[k] << ",";
		//std::cout << std::endl << std::endl;
	}

	std::cout << "Done coons loading" << std::endl;
}

std::vector<coons_patch> remove_coons_patches(const std::vector<coons_patch>& patches, const std::vector<int>& ids_to_remove)
{
	std::vector<coons_patch> output(patches.begin(), patches.end());
	std::vector<int> ids_to_remove_ordered(ids_to_remove.begin(), ids_to_remove.end());
	std::sort(ids_to_remove_ordered.rbegin(), ids_to_remove_ordered.rend());
	for (int i = 0; i < ids_to_remove_ordered.size(); i++)
		output.erase(output.begin() + ids_to_remove_ordered[i]);
	return output;
}


std::vector<bool> remove_bool_vec(const std::vector<bool>& patches, const std::vector<int>& ids_to_remove)
{
	std::vector<bool> output(patches.begin(), patches.end());
	std::vector<int> ids_to_remove_ordered(ids_to_remove.begin(), ids_to_remove.end());
	std::sort(ids_to_remove_ordered.rbegin(), ids_to_remove_ordered.rend());
	for (int i = 0; i < ids_to_remove_ordered.size(); i++)
		output.erase(output.begin() + ids_to_remove_ordered[i]);
	return output;
}

std::vector<int> remove_int_vec(int num_patches, const std::vector<int>& ids_to_remove)
{
	std::vector<int> output(num_patches);
	std::iota(output.begin(), output.end(), 0);
	std::vector<int> ids_to_remove_ordered(ids_to_remove.begin(), ids_to_remove.end());
	std::sort(ids_to_remove_ordered.rbegin(), ids_to_remove_ordered.rend());
	for (int i = 0; i < ids_to_remove_ordered.size(); i++)
		output.erase(output.begin() + ids_to_remove_ordered[i]);
	return output;
}