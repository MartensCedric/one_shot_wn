#include "parametric.h"
#include "uv_util.h"
#include <vector>
#include <random>
#include "adaptive.h"

int f_func_parametric(const gsl_vector* x, void* p, gsl_vector* f)
{
	struct parametric_solver_params* params = (struct parametric_solver_params*)p;

	Eigen::Vector3d point = params->point;
	Eigen::Vector3d dir = params->dir;

	implicit_func_t func = params->func;

	const double u = gsl_vector_get(x, 0);
	const double v = gsl_vector_get(x, 1);
	const double t = gsl_vector_get(x, 2);

	Eigen::Vector3d output =  func(Eigen::Vector2d(u, v))  - (point + t * dir);

	gsl_vector_set(f, 0, output(0));
	gsl_vector_set(f, 1, output(1));
	gsl_vector_set(f, 2, output(2));

	return GSL_SUCCESS;
};

Eigen::Matrix<double, 3, 2> jacobian_fd_implicit(Eigen::Vector2d uv, implicit_func_t func)
{
	double eps = 0.0000001;
	Eigen::Vector3d df_du = (func(uv + Eigen::Vector2d(eps, 0)) - func(uv + Eigen::Vector2d(-eps, 0))) / (2.0 * eps);
	Eigen::Vector3d df_dv = (func(uv + Eigen::Vector2d(0, eps)) - func(uv + Eigen::Vector2d(0, -eps))) / (2.0 * eps);

	Eigen::Matrix<double, 3, 2> output;
	output.col(0) = df_du;
	output.col(1) = df_dv;
	return output;
}



Eigen::Vector3d implicit_test_func1(Eigen::Vector2d uv)
{
	//return Eigen::Vector3d(uv(0), uv(1), uv(0) * uv(0));
	return Eigen::Vector3d(uv(0), uv(1), 1.0 - (uv(0) * uv(0) + uv(1) * uv(1)));
}
Eigen::Vector3d implicit_test_func2(Eigen::Vector2d uv)
{
	//return Eigen::Vector3d(1.0 + uv(0), uv(1), 1.0);
	return Eigen::Vector3d(uv(0) + 1, uv(1), -uv(0) - uv(1));
}

bool implicit_test_func1_inside(Eigen::Vector2d uv)
{
	/*if (uv(1) >= 0.25 && uv(1) <= 0.75 && uv(0) > 0)
		return true;*/
	if (uv(1) >= 0 && uv(1) <= 1.0 && uv(0) > 0)
		return true;
	return uv(0) >= 0 && uv(0) <= 1.0 && uv(1) >= 0 && uv(1) <= 1.0;
}

bool implicit_test_func2_inside(Eigen::Vector2d uv)
{
	//if (uv(1) >= 0.25 && uv(1) <= 0.75 && uv(0) < 0)
	//	return true;
	if (uv(1) >= 0 && uv(1) <= 1.0 && uv(0) < 0)
		return true;
	return uv(0) >= 0 && uv(0) <= 1.0 && uv(1) >= 0 && uv(1) <= 1.0;
}

void print_solver_state_2(size_t iter, gsl_multiroot_fsolver* s)
{
	gsl_vector* dx_vec = gsl_multiroot_fsolver_dx(s);
	printf("Iteration = %3u u = % .3f, v = % .3f, t = % .3f "
		"f(x) = % .3e % .3e % .3e dx = % .3e % .3e % .3e\n",
		iter,
		gsl_vector_get(s->x, 0),
		gsl_vector_get(s->x, 1),
		gsl_vector_get(s->x, 2),
		gsl_vector_get(s->f, 0),
		gsl_vector_get(s->f, 1),
		gsl_vector_get(s->f, 2),
		gsl_vector_get(dx_vec, 0),
		gsl_vector_get(dx_vec, 1),
		gsl_vector_get(dx_vec, 2));
}

struct all_intersections_with_normals_result find_all_intersections_parametric_gears(gsl_multiroot_fsolver* s, const struct parametric_solver_params& params, double t_min, double t_max)
{
	all_intersections_with_normals_result result;
	result.valid_ray = true;
	const double root_epsilon = 1e-5;
	const double fval_epsilon = 1e-9;
	const double t_epsilon = 1e-7;
	const size_t max_iterations = 1000;
	const double ray_dot_epsilon = 0.0;

	constexpr double uv_epsilon = 0.00;
	constexpr double u_min = uv_epsilon;
	constexpr double u_max = 1.0 - uv_epsilon;
	constexpr double v_min = uv_epsilon;
	constexpr double v_max = 1.0 - uv_epsilon;

	std::vector<double> roots_found;
	const int num_starting_ts = std::max<int>(2, (t_max - t_min) * 60);
	const int num_starting_uvs = 500;
	std::vector<double> t_linspace = linspace(t_min, t_max, num_starting_ts, true);
	std::vector<double> u_linspace = linspace(u_min, u_max, num_starting_uvs, true);
	std::vector<double> v_linspace = { 0.5 };  //linspace(v_min, v_max, num_starting_uvs, true);

	int solver_error_counter = 0;
	for (double t0 : t_linspace)
	{
		for (double u0 : u_linspace)
		{
			for (double v0 : v_linspace)
			{
				double x_init[3] = { u0, v0, t0 };
				const Eigen::Vector3d uvt(x_init);
				gsl_vector_set(s->x, 0, x_init[0]);
				gsl_vector_set(s->x, 1, x_init[1]);
				gsl_vector_set(s->x, 2, x_init[2]);
				s->function->params = (void*)&params;
				gsl_multiroot_fsolver_set(s, s->function, s->x); // this line is probably not needed here

				int status;
				size_t iter_num = 0;

				do
				{
					iter_num++;
					status = gsl_multiroot_fsolver_iterate(s);
					//print_solver_state_2(iter_num, s);
					if (status)
						break;

					status = gsl_multiroot_test_residual(s->f, fval_epsilon);
				} while (status == GSL_CONTINUE && iter_num < max_iterations);

				if (status == GSL_ENOPROG || status == GSL_ENOPROGJ)
				{
					solver_error_counter++;
				}
				else if (status == GSL_SUCCESS)
				{
					double u = gsl_vector_get(s->x, 0);
					double v = gsl_vector_get(s->x, 1);
					double t = gsl_vector_get(s->x, 2);


					if (t < t_min - t_epsilon || t > t_max + t_epsilon) // t out of range
						continue;


					if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) // outside uv space
						continue;


					if (!is_in_vector_epsilon(roots_found, t, root_epsilon))
					{
						Eigen::Matrix<double, 3, 2> jacobian = jacobian_fd_implicit(Eigen::Vector2d(u, v), params.func);
						Eigen::Vector3d normal = jacobian.col(0).cross(jacobian.col(1));
						Eigen::Vector3d dir = params.dir;

						normal.normalize();
						dir.normalize();

						double ray_normal_dot = normal.dot(dir);

						if ((std::abs(ray_normal_dot) < ray_dot_epsilon || u < u_min || u > u_max || v < v_min || v > v_max))
						{
							//std::cout << "ray_normal_dot: " << ray_normal_dot << ", u:" << u << ", v:" << v << std::endl;
							result.valid_ray = false;
							result.all_intersections.clear();
							return result;
						}
						else
						{
							if (ray_normal_dot < 0)
								result.all_intersections.emplace_back(t, -1);
							else
								result.all_intersections.emplace_back(t, 1);

							roots_found.push_back(t);
						}

					}
				}
				else if (status != GSL_FAILURE && status != GSL_CONTINUE)
				{
					std::cout << "status: " << status << std::endl;
					throw std::runtime_error("Different solver result");
				}
			}
		}
	}

	std::sort(result.all_intersections.begin(), result.all_intersections.end(), [=](const std::pair<double, int>& p1, const std::pair<double, int>& p2) { return p1.first < p2.first; });
	return result;
}


struct all_intersections_with_normals_result find_all_intersections_parametric(gsl_multiroot_fsolver* s, const struct parametric_solver_params& params, double t_min, double t_max)
{
	all_intersections_with_normals_result result;
	result.valid_ray = true;
	const double root_epsilon = 1e-5;
	const double fval_epsilon = 1e-9;
	const double t_epsilon = 1e-7;
	const size_t max_iterations = 1000;
	const double ray_dot_epsilon = 0.0;

	constexpr double uv_epsilon = 0.0;
	constexpr double u_min = uv_epsilon;
	constexpr double u_max = 1.0 - uv_epsilon;
	constexpr double v_min = uv_epsilon;
	constexpr double v_max = 1.0 - uv_epsilon;

	std::vector<struct root_position> roots_found;
	const int num_starting_ts = std::max<int>(2, (t_max - t_min) * 30);

	//const int num_starting_uvs = 35;
	std::vector<double> t_linspace = linspace(t_min, t_max, num_starting_ts, true);
	std::vector<double> u_linspace = linspace(u_min, u_max, 20, true);
	std::vector<double> v_linspace = linspace(v_min, v_max, 10, true);

	int solver_error_counter = 0;
	for (double t0 : t_linspace)
	{
		for (double u0 : u_linspace)
		{
			for (double v0 : v_linspace)
			{
				if (!params.is_in_parametric_domain(Eigen::Vector2d(u0, v0))) // outside uv space
					continue;
				double x_init[3] = { u0, v0, t0 };
				const Eigen::Vector3d uvt(x_init);
				gsl_vector_set(s->x, 0, x_init[0]);
				gsl_vector_set(s->x, 1, x_init[1]);
				gsl_vector_set(s->x, 2, x_init[2]);
				s->function->params = (void*)&params;
				gsl_multiroot_fsolver_set(s, s->function, s->x); // this line is probably not needed here

				int status;
				size_t iter_num = 0;

				do
				{
					iter_num++;
					status = gsl_multiroot_fsolver_iterate(s);
					//print_solver_state_2(iter_num, s);
					if (status)
						break;

					status = gsl_multiroot_test_residual(s->f, fval_epsilon);
				} while (status == GSL_CONTINUE && iter_num < max_iterations);

				if (status == GSL_ENOPROG || status == GSL_ENOPROGJ)
				{
					solver_error_counter++;
				}
				else if (status == GSL_SUCCESS)
				{
					double u = gsl_vector_get(s->x, 0);
					double v = gsl_vector_get(s->x, 1);
					double t = gsl_vector_get(s->x, 2);


					if (t < t_min - t_epsilon || t > t_max + t_epsilon) // t out of range
						continue;


					if (!params.is_in_parametric_domain(Eigen::Vector2d(u, v))) // outside uv space
						continue;

					struct root_position root_pos {u,v, t};

					if (!is_in_vector_epsilon(roots_found, root_pos, root_epsilon, 1e-3))
					{
 						Eigen::Matrix<double, 3, 2> jacobian = params.jac(Eigen::Vector2d(u, v));
						auto jac_func_fd = fd_jacobian(params.func, 1e-8);
						Eigen::Matrix<double, 3, 2> jac_fd = jac_func_fd(Eigen::Vector2d(u, v));
						
						Eigen::Vector3d normal = jacobian.col(0).cross(jacobian.col(1));
						Eigen::Vector3d dir = params.dir;

						normal.normalize();
						dir.normalize();

						double ray_normal_dot = normal.dot(dir);

						if ((std::abs(ray_normal_dot) < ray_dot_epsilon || u < u_min || u > u_max || v < v_min || v > v_max))
						{
							//std::cout << "ray_normal_dot: " << ray_normal_dot << ", u:" << u << ", v:" << v << std::endl;
							result.valid_ray = false;
							result.all_intersections.clear();
							return result;
						}
						else
						{
							if (ray_normal_dot < 0)
								result.all_intersections.emplace_back(t, -1);
							else
								result.all_intersections.emplace_back(t, 1);

							roots_found.push_back(root_pos);
						}

					}
				}
				else if (status != GSL_FAILURE && status != GSL_CONTINUE)
				{
					std::cout << "status: " << status << std::endl;
					throw std::runtime_error("Different solver result");
				}
			}
		}
	}

	std::sort(result.all_intersections.begin(), result.all_intersections.end(), [=](const std::pair<double, int>& p1, const std::pair<double, int>& p2) { return p1.first < p2.first; });
	return result;
}
Eigen::Vector2d custom_involute(double t)
{
	double v = 0.2 * t;
	v = std::max(0.0, std::min(v, 1.0));
	double normalization = Eigen::Vector2d(1.0, std::sqrt(0.2)).norm();
	return Eigen::Vector2d(t, std::sqrt(v)) / normalization;
}

Eigen::Vector3d spur_gear(Eigen::Vector2d uv)
{
	double teeth_separation_angle = 24.0;
	double involute_angle = 6.0;
	double teeth_front_angle = 8.0;
	double theta = uv(0) * 2.0 * EIGEN_PI;

	double theta_deg = uv(0) * 360.0;

	Eigen::Vector2d axle_hole_circle = 0.2 * Eigen::Vector2d(std::cos(theta), std::sin(theta));
	Eigen::Vector2d base_circle = 0.8 * Eigen::Vector2d(std::cos(theta), std::sin(theta));
	double angle_leftover = teeth_separation_angle * ((theta_deg / teeth_separation_angle) - std::floor(theta_deg / teeth_separation_angle));
	Eigen::Vector2d teeth_increment = Eigen::Vector2d().Zero();
	double teeth_distance = 0.2;

	if (angle_leftover < involute_angle)
	{
		teeth_increment = teeth_distance * custom_involute(angle_leftover / (involute_angle));
	}
	else if (angle_leftover < involute_angle + teeth_front_angle)
	{
		angle_leftover -= (involute_angle + teeth_front_angle) / 2.0;
		teeth_increment = teeth_distance * Eigen::Vector2d(1.0, 0.0);

	}
	else if (angle_leftover < involute_angle * 2 + teeth_front_angle)
	{
		angle_leftover -= involute_angle + teeth_front_angle;
		teeth_increment = teeth_distance * custom_involute(1.0 - (angle_leftover / (involute_angle)));
	}
	
		
	double teeth_increment_dist = teeth_increment.norm();
	Eigen::Vector2d interped_val = (1.0 - uv(1)) * axle_hole_circle + uv(1) * (base_circle + teeth_increment_dist * Eigen::Vector2d(std::cos(theta), std::sin(theta)));
	return Eigen::Vector3d(interped_val(0), interped_val(1), 0);
}

Eigen::Vector3d spur_gear_parametric_right(Eigen::Vector2d uv)
{
	double angle = 4.0 * M_PI / 180.0;

	Eigen::AngleAxisd rotation_vector(angle, Eigen::Vector3d::UnitZ());

	Eigen::Matrix3d rotation_matrix = rotation_vector.toRotationMatrix();

	return rotation_matrix * spur_gear_parametric(uv) + Eigen::Vector3d(1.88, 0, 0);
}

Eigen::Vector3d spur_gear_parametric(Eigen::Vector2d uv)
{
	double gear_height = 0.2;
	double u = std::max(0.0, std::min(1.0, uv(0)));
	double v = std::max(0.0, std::min(1.0, uv(1)));
	return spur_gear(Eigen::Vector2d(u, 1.0)) + Eigen::Vector3d(0, 0, gear_height * v);

	//Eigen::Vector3d output;
	//if (v < 1.0 / 3.0)
	//{
	//	output = spur_gear(Eigen::Vector2d(u, v * 3.0));
	//	return output;
	//}
	//else if (v < 2.0 / 3.0)
	//{
	//	double current_gear_height = gear_height * (v - 1.0 / 3.0) * 3.0;
	//	output = spur_gear(Eigen::Vector2d(u, 1.0)) + Eigen::Vector3d(0, 0, current_gear_height);
	//	return output;
	//}

	//output = spur_gear(Eigen::Vector2d(u, 1.0 - (v - 2.0 / 3.0) * 3.0)) + Eigen::Vector3d(0,0, gear_height);
	//return output;
}

void compute_all_intersections(const patch_t& patch, const Eigen::MatrixXd& query_points, const std::function<Eigen::Vector3d(Eigen::Vector2d)>& param_func, const std::function<bool(Eigen::Vector2d)>& is_inside)
{

	std::vector<Eigen::Vector3d> rays;
	std::vector<Eigen::Vector3d> qs;
	std::vector<double> ts;
	std::mt19937 gen;
	std::uniform_real_distribution<double> dis(-1, 1);

	const gsl_multiroot_fsolver_type* T;
	T = gsl_multiroot_fsolver_hybrids;
	gsl_multiroot_fsolver* s;
	gsl_multiroot_function F;


	F.f = &f_func_parametric;
	F.n = 3;
	s = gsl_multiroot_fsolver_alloc(T, F.n);
	s->x = gsl_vector_alloc(F.n);
	s->function = &F;

	std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> jac_func = fd_jacobian(param_func, 1e-8);

	for (int i = 0; i < query_points.rows(); i++)
	{
		if (dis(gen) > 0.02)
			continue;
		Eigen::Vector3d ray;
		ray(0) = dis(gen);
		ray(1) = dis(gen);
		ray(2) = dis(gen);

		ray.normalize();
		parametric_solver_params solver_params = {
			param_func,
			jac_func,
			query_points.row(i),
			ray,
			is_inside
		};

		all_intersections_with_normals_result intersections = find_all_intersections_parametric(s, solver_params, 0, 1.4);

		if (intersections.valid_ray && intersections.all_intersections.size() == 0)
		{
			rays.push_back(ray);
			qs.push_back(query_points.row(i));
			//ts.push_back(intersections.all_intersections.back().first);

			if (rays.size() >= 200)
				break;
		}
		
	}

	Eigen::IOFormat matlab_fmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
	Eigen::MatrixXd rays_mat(rays.size(), 3);
	Eigen::MatrixXd query_mat(qs.size(), 3);
	Eigen::VectorXd ts_mat(ts.size());

	for (int i = 0; i < rays_mat.rows(); i++)
	{
		rays_mat.row(i) = rays[i];
		query_mat.row(i) = qs[i];
		//ts_mat(i) = ts[i];
	}

	std::cout << rays.size() << " rays" << std::endl;
	std::ofstream outfile("ray_surface.m");
	outfile << "rays = " << rays_mat.format(matlab_fmt) << ";" << std::endl;
	outfile << "query_points = " << query_mat.format(matlab_fmt)<< ";" << std::endl;
	//outfile << "ts = " << ts_mat.format(matlab_fmt) << ";" << std::endl;
	outfile.close();

}

void uniform_mesh_grid(int resolution, Eigen::MatrixXd& V, Eigen::MatrixXi& F) {


	// Calculate the number of vertices and faces.
	int num_vertices = (resolution) * (resolution);
	resolution--;
	int num_faces = 2 * resolution * resolution;

	// Calculate the step size for the grid.
	double h = 1.0 / resolution;

	// Resize the matrices.
	V.resize(num_vertices, 2);
	F.resize(num_faces, 3);

	// Generate the vertices.
	for (int i = 0; i <= resolution; ++i) {
		for (int j = 0; j <= resolution; ++j) {
			V(i * (resolution + 1) + j, 0) = j * h;
			V(i * (resolution + 1) + j, 1) = i * h;
		}
	}

	// Generate the faces.
	int face_index = 0;
	for (int i = 0; i < resolution; ++i) {
		for (int j = 0; j < resolution; ++j) {
			int v1 = i * (resolution + 1) + j;
			int v2 = i * (resolution + 1) + j + 1;
			int v3 = (i + 1) * (resolution + 1) + j;
			int v4 = (i + 1) * (resolution + 1) + j + 1;

			F(face_index, 0) = v1;
			F(face_index, 1) = v2;
			F(face_index++, 2) = v3;

			F(face_index, 0) = v2;
			F(face_index, 1) = v4;
			F(face_index++, 2) = v3;
		}
	}
}

Eigen::MatrixXd evaluate_surface(const std::function<Eigen::Vector3d(Eigen::Vector2d)>& func, const Eigen::MatrixXd& uvs)
{
	Eigen::MatrixXd result(uvs.rows(), 3);
	for (int i = 0; i < uvs.rows(); i++)
		result.row(i) = func(uvs.row(i));
	return result;
}

std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> fd_jacobian(implicit_func_t func, double eps)
{
	std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> jac_func = [=](Eigen::Vector2d uv)
		{
		
			Eigen::Vector3d du = (func(uv + Eigen::Vector2d(eps, 0)) - func(uv + Eigen::Vector2d(-eps, 0))) / (2.0 * eps);
			Eigen::Vector3d dv = (func(uv + Eigen::Vector2d(0, eps)) - func(uv + Eigen::Vector2d(0, -eps))) / (2.0 * eps);

			Eigen::Matrix<double, 3, 2> jacobian;
			jacobian.col(0) = du;
			jacobian.col(1) = dv;
			return jacobian;
		};
	return jac_func;
}

Eigen::MatrixXd discretize_curve(const std::function<Eigen::Vector3d(double)>& func, int num_discretizations)
{
	Eigen::MatrixXd result;
	std::vector<double> ts = linspace(0, 1, num_discretizations, true);
	result.resize(ts.size(), 3);

	for (int i = 0; i < ts.size(); i++)
	{
		double t = ts[i];
		result.row(i) = func(t);
	}
	return result;
}

Eigen::MatrixXd boundary_from_surface(const std::function<Eigen::Vector3d(Eigen::Vector2d)>& func, int num_elements)
{
	ASSERT_RELEASE(num_elements % 4 == 0, "not divisible by 4");
	Eigen::MatrixXd b(num_elements, 3);
	int num_e = num_elements / 4;
	std::vector<double> ts = linspace(0, 1, num_e, false);
	
	int idx = 0;

	for (int i = 0; i < num_e; i++)
		b.row(idx++) = func(Eigen::Vector2d(ts[i], 0));
	for (int i = 0; i < num_e; i++)
		b.row(idx++) = func(Eigen::Vector2d(1, ts[i]));
	for (int i = 0; i < num_e; i++)
		b.row(idx++) = func(Eigen::Vector2d(1 - ts[i], 1));
	for (int i = 0; i < num_e; i++)
		b.row(idx++) = func(Eigen::Vector2d(0, 1 - ts[i]));

	return b;
}

Eigen::Vector3d evaluate_paper_func1(Eigen::Vector2d uv) {

	double x = uv(0);
	double y = uv(1);
	double z = std::exp(x - 1.0) * std::sin(y) - std::exp(x) * std::cos(y);
	return Eigen::Vector3d(x, y, z);
}

Eigen::Matrix<double, 3, 2> evaluate_paper_jacfunc1(Eigen::Vector2d uv) {
	
	double x = uv(0);
	double y = uv(1);
	Eigen::Vector3d df_du = Eigen::Vector3d(1.0, 0.0, std::exp(x - 1.0) * std::sin(y) - std::exp(x) * std::cos(y));
	Eigen::Vector3d df_dv = Eigen::Vector3d(0.0, 1.0, std::exp(x - 1.0) * std::cos(y) + std::exp(x) * std::sin(y));
	
	Eigen::Matrix<double, 3, 2> jac;
	jac.col(0) = df_du;
	jac.col(1) = df_dv;
	return jac;
}

std::vector<double> run_aq_parametric_paper_func(std::function<Eigen::Vector3d(Eigen::Vector2d)> func, 
					   std::function<Eigen::Matrix<double, 3, 2>(Eigen::Vector2d)> jac_func,
					   const Eigen::MatrixXd& query_points)
{
	std::vector<double> wns = winding_numbers_adaptive(func, jac_func, query_points, 12, 1e-7);
	return wns;
}

Eigen::Vector3d infinite_singular_func(Eigen::Vector2d uv)
{
	return Eigen::Vector3d(uv(0), uv(1), 1.0 / std::sqrt(uv(0) * uv(0) + uv(1) + uv(1)));
}

Eigen::Matrix<double, 3, 2> infinite_singular_func_jac(Eigen::Vector2d uv)
{
	Eigen::Matrix<double, 3, 2> jac;
	double u = uv(0);
	double v = uv(1);
	jac.col(0) = Eigen::Vector3d(1.0, 0.0, -u / std::pow(u*u + v*v, 1.5));
	jac.col(1) = Eigen::Vector3d(0.0, 1.0, -v / std::pow(u*u + v*v, 1.5));
	return jac;
}

Eigen::Vector3d parametric_vase_cubic(Eigen::Vector2d uv)
{
	double x = uv(0);
	double xx = x * x;
	double xxx = xx * x;
	double theta = uv(1) * 2.0 * EIGEN_PI;

	double a = 5.5;
	double b = -10.0;
	double c = 5.0;
	double radius = 0.5 * (a * xxx + b * xx + c * x);

	return Eigen::Vector3d(radius * std::cos(theta), radius * std::sin(theta), x);
}

Eigen::Matrix<double, 3, 2> parametric_vase_cubic_jac(Eigen::Vector2d uv)
{
	Eigen::Matrix<double, 3, 2> jac;
	double a = 5.5;
	double b = -10.0;
	double c = 5.0;
	double u = uv(0);
	double uu = u * u;
	double uuu = u * uu;
	double v = uv(1);

	double poly = 0.5 * (a * uuu + b * uu + c * u);
	double poly_d = 0.5 * (3.0 * a * uu + 2.0 * b * u + c);
	double cv = std::cos(2.0 * EIGEN_PI * v);
	double sv = std::sin(2.0 * EIGEN_PI * v);
	jac.col(0) = Eigen::Vector3d(cv * poly_d, sv * poly_d, 1.0);
	jac.col(1) = Eigen::Vector3d(sv * poly * (-2.0 * EIGEN_PI), cv * poly * 2.0 * EIGEN_PI, 0.0);
	
	return jac;
}


Eigen::Vector3d parametric_torus_4_3(Eigen::Vector2d uv)
{
	constexpr int p = 4;  // Number of times the curve winds around the minor circle
	constexpr int q = 3;  // Number of times the curve winds around the major circle
	constexpr double tube_radius = 0.2;  // Radius of the tube

	// Rescale u and v from [0, 1] to [0, 2*pi*q] and [0, 2*pi] respectively
	double u = uv(0) * 2 * M_PI;
	double v = uv(1) * 2 * M_PI;

	// Torus knot path equations (center of the tube)
	double x_center = (2 + cos(p * u)) * cos(q * u);
	double y_center = (2 + cos(p * u)) * sin(q * u);
	double z_center = sin(p * u);

	// Calculate tangent vector
	double dx_du = -p * sin(p * u) * cos(q * u) - q * (2 + cos(p * u)) * sin(q * u);
	double dy_du = -p * sin(p * u) * sin(q * u) + q * (2 + cos(p * u)) * cos(q * u);
	double dz_du = p * cos(p * u);

	Eigen::Vector3d tangent(dx_du, dy_du, dz_du);
	tangent.normalize();

	// Calculate second derivatives for curvature vector
	double ddx_du2 = -p * p * cos(p * u) * cos(q * u) + 2 * p * q * sin(p * u) * sin(q * u) - q * q * (2 + cos(p * u)) * cos(q * u);
	double ddy_du2 = -p * p * cos(p * u) * sin(q * u) - 2 * p * q * sin(p * u) * cos(q * u) - q * q * (2 + cos(p * u)) * sin(q * u);
	double ddz_du2 = -p * p * sin(p * u);

	Eigen::Vector3d curvature_vector(ddx_du2, ddy_du2, ddz_du2);
	curvature_vector -= curvature_vector.dot(tangent) * tangent;

	// Calculate normal and binormal vectors
	Eigen::Vector3d normal = curvature_vector.normalized();
	Eigen::Vector3d binormal = tangent.cross(normal);

	// Calculate the point on the tube surface
	double tube_x = x_center + tube_radius * (normal(0) * cos(v) + binormal(0) * sin(v));
	double tube_y = y_center + tube_radius * (normal(1) * cos(v) + binormal(1) * sin(v));
	double tube_z = z_center + tube_radius * (normal(2) * cos(v) + binormal(2) * sin(v));

	return Eigen::Vector3d(tube_x, tube_y, tube_z);
}

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

Eigen::Vector3d parametric_torus(Eigen::Vector2d uv) {
	// Major and minor radii
	double R = 0.4;  // Major radius (must satisfy R + r <= 0.5)
	double r = 0.1;  // Minor radius

	// Ensure the torus fits within the unit cube
	if (R + r > 0.5) {
		throw std::invalid_argument("R + r must be <= 0.5 to fit within the unit cube.");
	}

	// Rescale u and v from [0, 1] to [0, 2*pi]
	double u = uv(0) * 2 * M_PI;
	double v = uv(1) * 2 * M_PI;

	// Torus equations
	double x = (R + r * std::cos(v)) * std::cos(u);
	double y = (R + r * std::cos(v)) * std::sin(u);
	double z = r * std::sin(v);

	// Shift the torus to the center of the unit cube
	x += 0.5;
	y += 0.5;
	//z += 0.5;

	return Eigen::Vector3d(x, y, z);
}

#include <boost/multiprecision/cpp_dec_float.hpp>

Eigen::Vector3d parametric_vase_cubic_2(Eigen::Vector2d uv)
{
	double x = uv(0);
	double theta = uv(1) * 2.0 * EIGEN_PI;
	Eigen::Vector3d base = parametric_vase_cubic(uv);

	double k = -2.2;
	double g =  (1.0 - std::pow(-2.0 * x + 1.0, 2.0));
	double radius_sine = 0.1 * g * std::sin(k * 2.0 * EIGEN_PI * x);

	Eigen::Vector3d func_l = Eigen::Vector3d(radius_sine * std::cos(theta), radius_sine * std::sin(theta), 0.0);
	using BigFloat = boost::multiprecision::cpp_dec_float_100;
	BigFloat n = 2000.0;
	BigFloat v = 0.9;
	BigFloat w = 0.05;
	BigFloat xb = uv(0);
	double radius_m = static_cast<double>(1.0 / (1.0 + boost::multiprecision::exp(-n * (xb - v))) - 1.0 / (1.0 + boost::multiprecision::exp(-n * (xb - v - w))));
	Eigen::Vector3d func_m = Eigen::Vector3d(radius_m * std::cos(theta), radius_m * std::sin(theta), 0.0);
	Eigen::Vector3d func_res = base + func_l + 0.1 * func_m;

	return func_res;
}

Eigen::Matrix<double, 3, 2> parametric_vase_cubic_2_jac(Eigen::Vector2d uv)
{
	Eigen::Matrix<double, 3, 2> jac = parametric_vase_cubic_jac(uv);
	Eigen::Matrix<double, 3, 2> jac_m;


	using BigFloat = boost::multiprecision::cpp_dec_float_100;

	BigFloat xb = uv(0);
	double x = uv(0);
	double y = uv(1);
	BigFloat n = 2000.0;
	BigFloat v = 0.9;
	BigFloat w = 0.05;

	
	BigFloat exp_d = ((n * boost::multiprecision::exp(n * (v + w - xb))) / boost::multiprecision::pow(boost::multiprecision::exp(n * (v + w - xb)) + 1.0, 2.0) - (n * boost::multiprecision::exp(n * (v - xb))) / boost::multiprecision::pow(boost::multiprecision::exp(n * (v - xb)) + 1.0, 2.0));
	jac_m.col(0) = Eigen::Vector3d(
		- std::cos(2.0 * EIGEN_PI * y) * static_cast<double>(exp_d),
		- std::sin(2.0 * EIGEN_PI * y) * static_cast<double>(exp_d),
		 0.0);
	BigFloat exp_f = (1.0 / (exp(n * (v + w - x)) + 1.0) - 1.0 / (exp(n * (v - x)) + 1.0));
	jac_m.col(1) = Eigen::Vector3d(
		2.0 * EIGEN_PI * std::sin(2.0 * EIGEN_PI * y) * static_cast<double>(exp_f),
		-2.0 * EIGEN_PI * std::cos(2.0 * EIGEN_PI * y) * static_cast<double>(exp_f)
		, 0.0);

	jac += 0.1 * jac_m;
	double k = -2.2;
	Eigen::Matrix<double, 3, 2> jac_l;
	jac_l.col(0) = Eigen::Vector3d(
		-std::sin(2.0 * EIGEN_PI * k * x) * std::cos(2.0 * EIGEN_PI * y) * ((4.0 * x) / 5.0 - 2.0 / 5.0) - 2.0 * k * EIGEN_PI * std::cos(2.0 * EIGEN_PI * y) * std::cos(2.0 * EIGEN_PI * k * x) * (std::pow(2.0 * x - 1.0, 2.0) / 10.0 - 1.0 / 10.0),
		-std::sin(2.0 * EIGEN_PI * k * x) * std::sin(2.0 * EIGEN_PI * y) * ((4.0 * x) / 5.0 - 2.0 / 5.0) - 2.0 * k * EIGEN_PI * std::sin(2.0 * EIGEN_PI * y) * std::cos(2.0 * EIGEN_PI * k * x) * (std::pow(2.0 * x - 1.0, 2.0) / 10.0 - 1.0 / 10.0),
		0.0
	);

	jac_l.col(1) = Eigen::Vector3d(
		2.0 * EIGEN_PI * std::sin(2.0 * EIGEN_PI * k * x) * std::sin(2.0 * EIGEN_PI * y) * (std::pow(2.0 * x - 1.0, 2.0) / 10.0 - 1.0 / 10.0),
		-2.0 * EIGEN_PI * std::sin(2.0 * EIGEN_PI * k * x) * std::cos(2.0 * EIGEN_PI * y) * (std::pow(2.0 * x - 1.0, 2.0) / 10.0 - 1.0 / 10.0),
		0.0
	);
	jac += jac_l;
	return jac;
}

Eigen::MatrixXd apply_function_to_uv(const Eigen::MatrixXd& uvs, const std::function<Eigen::Vector3d(Eigen::Vector2d)>& func)
{
	Eigen::MatrixXd func_image(uvs.rows(), 3);

	for (int i = 0; i < uvs.rows(); i++)
	{
		func_image.row(i) = func(uvs.row(i));
	}
	return func_image;
}