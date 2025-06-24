
#include <vector>
#include "math_util.h"
#include "boundary_processing.h"
#include <iostream>
#include <array>
#include <limits>

std::vector<double> linspace(double start, double end, int num_points, bool include_end)
{
	std::vector<double> result;
	if (num_points <= 1)
	{
		if (num_points == 1)
		{
			result.push_back(start);
		}
		return result;
	}

	int divisions = include_end ? num_points - 1 : num_points;
	double step = (end - start) / static_cast<double>(divisions);
	for (int i = 0; i < num_points; ++i) {
		result.push_back(start + i * step);
	}

	return result;
}


Eigen::Vector3d interpolate_curve(double t, const std::vector<Eigen::Vector3d>& curve)
{
	if (std::isnan(t))
		return Eigen::Vector3d(t, t, t);
	int n = curve.size();

	if (t > 1.0)
	{
		return (curve[n - 1] - curve[n - 2]) * (t - 1.0) + curve[n - 1];
	}
	else if (t < 0.0)
	{
		return (curve[0] - curve[1]) * (-t) + curve[0];
	}

	double total_length = 0.0;
	std::vector<double> lengths(curve.size()-1);
	for (int i = 0; i < curve.size() - 1; i++)
	{
		Eigen::Vector3d diff = curve[i + 1] - curve[i];
		lengths[i] = diff.norm();
		total_length += diff.norm();
	}

	double length_to_find = t * total_length;

	double cumulative_length = 0.0;
	for (int i = 0; i < curve.size() - 1; i++)
	{
		double diff_norm = (curve[(i + 1) % curve.size()] - curve[i]).norm();

		if (length_to_find < cumulative_length + diff_norm)
		{
			double rem = (length_to_find - cumulative_length) / lengths[i];
			return  (1. - rem) * curve[i] + rem * curve[i + 1];
		}
		else
		{
			cumulative_length += lengths[i];
		}
	}

	assert(t <= 1.0);
	return curve[curve.size() - 1];
	
}

ray_box_intersection_result ray_box_intersection(Eigen::Vector3d origin, Eigen::Vector3d direction, const struct box& box)
{
	double ox = origin(0);
	double oy = origin(1);
	double oz = origin(2);

	double dx = direction(0);
	double dy = direction(1);
	double dz = direction(2);

	double t_min = std::numeric_limits<double>::lowest();
	double t_max = std::numeric_limits<double>::max();
	double eps = 2.22044604925031e-16;

	ray_box_intersection_result result;

	if (std::abs(dx) < eps)
	{
		if (ox < box.x_min || ox > box.x_max)
		{
			result.intersects = false;
			return result;
		}
	}
	else
	{
		double tx1 = (box.x_min - ox) / dx;
		double tx2 = (box.x_max - ox) / dx;

		t_min = std::max(t_min, std::min(tx1, tx2));
		t_max = std::min(t_max, std::max(tx1, tx2));
	}

	if (std::abs(dy) < eps)
	{
		if (oy < box.y_min || oy > box.y_max)
		{
			result.intersects = false;
			return result;
		}
	}
	else
	{
		double ty1 = (box.y_min - oy) / dy;
		double ty2 = (box.y_max - oy) / dy;

		t_min = std::max(t_min, std::min(ty1, ty2));
		t_max = std::min(t_max, std::max(ty1, ty2));
	}

	if (std::abs(dz) < eps)
	{
		if (oz < box.z_min || oz > box.z_max)
		{
			result.intersects = false;
			return result;
		}
	}
	else
	{
		double tz1 = (box.z_min - oz) / dz;
		double tz2 = (box.z_max - oz) / dz;

		t_min = std::max(t_min, std::min(tz1, tz2));
		t_max = std::min(t_max, std::max(tz1, tz2));
	}

	result.intersects = t_max >= t_min && t_max >= 0;
	result.t_min = t_min;
	result.t_max = t_max;
	return result;
}

box bounding_box(const space_curve_t& patch)
{
	std::vector<double> xs(patch.rows(), 0);
	std::vector<double> ys(patch.rows(), 0);
	std::vector<double> zs(patch.rows(), 0);

	for (int i = 0; i < patch.rows(); i++)
	{
		xs[i] = patch(i, 0);
		ys[i] = patch(i, 1);
		zs[i] = patch(i, 2);
	}

	box box{};
	box.x_min = *std::min_element(xs.begin(), xs.end());
	box.x_max = *std::max_element(xs.begin(), xs.end());
	box.y_min = *std::min_element(ys.begin(), ys.end());
	box.y_max = *std::max_element(ys.begin(), ys.end());
	box.z_min = *std::min_element(zs.begin(), zs.end());
	box.z_max = *std::max_element(zs.begin(), zs.end());

	return box;
}

Eigen::Matrix<double, Eigen::Dynamic, 2> fundamental_solution_double_derivative_laplacian(Eigen::Vector2d p, const boundary_curve_t& boundary_curve, const curve_info_t& curve_info)
{
	int n = boundary_curve.rows();
	if (curve_info.is_open)
	{
		Eigen::Matrix<double, Eigen::Dynamic, 2> res;
		res.resize(n - 1, 2);
		res.setZero();
		return res;
	}

	const boundary_curve_t next = circular_shift(boundary_curve, -1);
	const boundary_curve_t& tau = next - boundary_curve;
	const boundary_curve_t& normals = curve_info.normals;

	const Eigen::VectorXd& dl_v = curve_info.lengths;
	const Eigen::VectorXd dl_sq = dl_v.array().square();

	Eigen::Matrix<double, Eigen::Dynamic, 2> output(n, 2);
	output.setZero();

	for (int i = 0; i < n; i++)
	{
		double x0 = boundary_curve(i, 0);
		double x1 = next(i, 0);

		double y0 = boundary_curve(i, 1);
		double y1 = next(i, 1);

		double p0 = p(0);
		double p1 = p(1);

		double a = dl_sq(i);
		double dl = dl_v(i);
		double nx = normals(i, 0);
		double ny = normals(i, 1);
		for (double X : {0, 1})
		{
			double f_u = 1./(-2. * EIGEN_PI)*(dl*(nx * std::pow(y0,2) - nx * std::pow(x0,2) + nx*p0*x0 - nx*p0*x1 + ny*p1*x0 - ny*p1*x1 - nx*p1*y0 + nx*p1*y1 + ny*p0*y0 - ny*p0*y1 + nx*x0*x1 - 2. * ny*x0*y0 + ny*x0*y1 + ny*x1*y0 - nx*y0*y1) + dl*X*(nx*std::pow(x0,2) - 2. * nx*x0*x1 + 2. * ny*x0*y0 - 2. * ny*x0*y1 + nx*std::pow(x1, 2) - 2. * ny*x1*y0 + 2. * ny*x1*y1 - nx * std::pow(y0,2) + 2. * nx*y0*y1 - nx * std::pow(y1,2))) / ((std::pow(x0,2) - 2. * x0*x1 + std::pow(x1, 2) + std::pow(y0, 2) - 2. * y0*y1 + std::pow(y1, 2))*(std::pow(p0, 2) + 2. * p0*X*x0 - 2. * p0*X*x1 - 2. * p0*x0 + std::pow(p1, 2) + 2. * p1*X*y0 - 2. * p1*X*y1 - 2. * p1*y0 + std::pow(X, 2) * std::pow(x0, 2) - 2. * std::pow(X, 2) * x0*x1 + std::pow(X, 2) * std::pow(x1, 2) + std::pow(X, 2) * std::pow(y0, 2) - 2. * std::pow(X, 2) * y0*y1 + std::pow(X, 2) * std::pow(y1, 2) - 2. * X* std::pow(x0, 2) + 2. * X*x0*x1 - 2. * X*std::pow(y0, 2) + 2. * X*y0*y1 + std::pow(x0, 2) + std::pow(y0, 2)));
			double f_v = 1. / (-2. * EIGEN_PI) * ((dl*(ny* std::pow(x0, 2)- ny * std::pow(y0,2) + nx*p1*x0 - nx*p1*x1 - ny*p0*x0 + ny*p0*x1 + nx*p0*y0 - nx*p0*y1 + ny*p1*y0 - ny*p1*y1 - ny*x0*x1 - 2. * nx*x0*y0 + nx*x0*y1 + nx*x1*y0 + ny*y0*y1) - dl*X*(ny*std::pow(x0,2) - 2. * ny*x0*x1 - 2. * nx*x0*y0 + 2. * nx*x0*y1 + ny*std::pow(x1, 2) + 2. * nx*x1*y0 - 2. * nx*x1*y1 - ny * std::pow(y0,2) + 2. * ny*y0*y1 - ny * std::pow(y1,2))) / ((std::pow(x0,2) - 2. * x0*x1 + std::pow(x1, 2) + std::pow(y0, 2) - 2. * y0*y1 + std::pow(y1, 2))*(std::pow(p0, 2) + 2. * p0*X*x0 - 2. * p0*X*x1 - 2. * p0*x0 + std::pow(p1, 2) + 2. * p1*X*y0 - 2. * p1*X*y1 - 2. * p1*y0 + std::pow(X, 2) * std::pow(x0, 2) - 2. * std::pow(X, 2) * x0*x1 + std::pow(X, 2) * std::pow(x1, 2) + std::pow(X, 2) * std::pow(y0, 2) - 2. * std::pow(X, 2) * y0*y1 + std::pow(X, 2) * std::pow(y1, 2) - 2. * X* std::pow(x0, 2) + 2. * X*x0*x1 - 2. * X*std::pow(y0, 2) + 2. * X*y0*y1 + std::pow(x0, 2) + std::pow(y0, 2))));
			
			double sign = (X < 0.5) ? -1 : 1;
			output(i, 0) = output(i, 0) + f_u * sign;
			output(i, 1) = output(i, 1) + f_v * sign;
		}
	}

	return output;
}

#define SQUARE(X) (X*X)

Eigen::Matrix<double, Eigen::Dynamic, 2> fundamental_solution_derivative_laplacian(Eigen::Vector2d p, const boundary_curve_t& boundary_curve, const curve_info_t& curve_info)
{
	int n = boundary_curve.rows();
	if (curve_info.is_open)
	{
		Eigen::Matrix<double, Eigen::Dynamic, 2> res;
		res.resize(n - 1, 2);
		res.setZero();
		return res;
	}

	const boundary_curve_t next = circular_shift(boundary_curve, -1);
	const boundary_curve_t& tau = next - boundary_curve;
	
	const Eigen::VectorXd& dl_v = curve_info.lengths;
	const Eigen::VectorXd dl_sq = dl_v.array().square();

	Eigen::Matrix<double, Eigen::Dynamic, 2> output(n, 2);
	output.setZero();

	Eigen::Matrix<double, Eigen::Dynamic, 2> F_one(n, 2);
	F_one.setZero();

	Eigen::Matrix<double, Eigen::Dynamic, 2> F_zero(n, 2);
	F_zero.setZero();

	for (int i = 0; i < n; i++)
	{
		for (double X : {0, 1})
		{
			constexpr double pi = EIGEN_PI;
			const double x0 = boundary_curve(i, 0);
			const double x1 = next(i, 0);

			const double y0 = boundary_curve(i, 1);
			const double y1 = next(i, 1);

			const double taux = x1 - x0;
			const double tauy = y1 - y0;

			const double p0 = p(0);
			const double p1 = p(1);

			const double tauxsq = SQUARE(taux);
			const double tauysq = SQUARE(tauy);
			const double norm = 1. / curve_info.lengths[i]; 

			constexpr double a1 = (-1. / 4.) * (1. / pi);
			const double coeff1 = a1 * norm;
			const double a2 = 1. / (p1 * taux + (-1.) * y0 * taux + (-1.) * p0 * tauy + x0 * tauy);
			const double b2 = -a2; // std::pow((-1.) * p1 * taux + y0 * taux + p0 * tauy + (-1.) * x0 * tauy, -1.);
			const double c1 = SQUARE(p0) + SQUARE(p1) + SQUARE(x0) + SQUARE(y0) + 2. * x0 * X * taux + SQUARE(X) * tauxsq;
			const double c2 = (-2.) * p1 * (y0 + X * tauy);
			const double c3 = (-2.) * p0 * (x0 + X * taux) + 2. * y0 * X * tauy + SQUARE(X) * tauysq + c2;
			const double l1 = std::log(c1 + c3);
			const double v1 = b2 * ((-1.) * p0 * taux + x0 * taux + X * tauxsq
				+ (-1.) * p1 * tauy + y0 * tauy + X * tauysq);

			// check if atan is odd
			double f_x = coeff1 * (2. * tauy*std::atan(-v1) + taux *l1);
			double f_y = coeff1 * (2. * taux * std::atan(v1) + tauy*l1);

			double sign = X < 0.5 ? -1 : 1;

			output(i, 0) = output(i, 0) + f_x * sign;
			output(i, 1) = output(i, 1) + f_y * sign;

			if (X > 0.5)
			{
				F_one(i, 0) = f_x * sign;
				F_one(i, 1) = f_y * sign;
			}
			else
			{
				F_zero(i, 0) = f_x * sign;
				F_zero(i, 1) = f_y * sign;
			}
		}
	}

	return output;
}

Eigen::MatrixXd compute_bem_G_from_boundaries(const BoundaryParametrization* boundary_param)
{
	int num_boundaries = boundary_param->get_boundary_curves().size();
	Eigen::MatrixXd output;

	int mat_size = boundary_param->get_total_segments();
	
	output.resize(mat_size, mat_size);// WARNING: DENSE
	int current_row = 0;
	for (int k = 0; k < num_boundaries; k++)
	{
		const boundary_curve_t& boundary_curve = boundary_param->get_boundary_curves()[k];
		const boundary_curve_t next = circular_shift(boundary_curve, -1);
		const boundary_curve_t all_p_i = 0.5 * (next + boundary_curve);
		boundary_curve_t all_p_i_used;

		if (boundary_param->is_open(k))
		{
			all_p_i_used.resize(boundary_curve.rows() - 1, boundary_curve.cols());
			all_p_i_used = all_p_i.block(0, 0, boundary_curve.rows() - 1, boundary_curve.cols());
		}
		else
		{
			all_p_i_used = all_p_i;
		}

		int n = all_p_i_used.rows();

		for (int i = 0; i < n; i++)
		{
			Eigen::VectorXd sol_i = fundamental_solution_laplacian_for_boundaries(Eigen::Vector2d(all_p_i_used(i, 0), all_p_i_used(i, 1)), boundary_param);
			output.row(current_row + i) = sol_i;
		}

		current_row += all_p_i_used.rows();
	}

	return output;
}

Eigen::MatrixXd compute_bem_H_from_boundaries(const BoundaryParametrization* boundary_param)
{
	int num_boundaries = boundary_param->get_boundary_curves().size();
	Eigen::MatrixXd output;
	int mat_size = boundary_param->get_total_segments();

	output.resize(mat_size, mat_size);// WARNING: DENSE
	const Eigen::MatrixXd& normals = boundary_param->get_normals();
	int current_row = 0;

	for (int k = 0; k < num_boundaries; k++)
	{
		const boundary_curve_t& boundary_curve = boundary_param->get_boundary_curves()[k];
		const boundary_curve_t next = circular_shift(boundary_curve, -1);
		const boundary_curve_t all_p_i = 0.5 * (next + boundary_curve);
		boundary_curve_t all_p_i_used;

		if (boundary_param->is_open(k))
		{
			all_p_i_used.resize(boundary_curve.rows() - 1, boundary_curve.cols());
			all_p_i_used = all_p_i.block(0, 0, boundary_curve.rows() - 1, boundary_curve.cols());
		}
		else
		{
			all_p_i_used = all_p_i;
		}

		int n = all_p_i_used.rows();

		for (int i = 0; i < n; i++)
		{
			int l = current_row + i;
			Eigen::MatrixX2d sol_i = fundamental_solution_derivative_laplacian_for_boundaries(Eigen::Vector2d(all_p_i(i, 0), all_p_i(i, 1)), boundary_param);
			const curve_info_t& curve_info = boundary_param->get_curve_infos()[k];
			for (int j = 0; j < sol_i.rows(); j++)
			{
				const double nx = normals.coeff(j, 0);
				const double ny = normals.coeff(j, 1);

				Eigen::Vector2d sol_ij = sol_i.row(j);
				double dG_dn = nx * sol_i(j, 0) + ny * sol_i(j, 1);
				if (l == j)
				{
					if (curve_info.is_open)
						output(l, j) = 1.0;
					else
						output(l, j) = 0.5; // The BEM document adds "+ dG_dn" (H_hat_ii) here;
				}
				else
				{
					output(l, j) = dG_dn;
				}
			}
		}

		current_row += all_p_i_used.rows();
	}

	return output;
}

Eigen::VectorXd df_dn_from_boundary_param(const BoundaryParametrization* boundary_param, const space_curve_t& space_curve)
{
	Eigen::VectorXd boundary_points = space_curve.col(2);


	const Eigen::MatrixXd G = compute_bem_G_from_boundaries(boundary_param);
	const Eigen::MatrixXd H = compute_bem_H_from_boundaries(boundary_param);

	const Eigen::VectorXd Hu = H * boundary_points.head(H.cols());
	const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver = G.colPivHouseholderQr();
	if (solver.info() != Eigen::Success)
	{
		std::cout << "big fail" << std::endl;
	}
	const Eigen::VectorXd output = solver.solve(Hu);

	return output;
}

// space curve here is in UV coordinates that is f(u,v) = z;
Eigen::VectorXd df_dn_from_boundary(const space_curve_t& space_curve, const curve_info_t& curve_info)
{
	const boundary_curve_t boundary_curve = space_curve.leftCols<2>();
	Eigen::VectorXd boundary_points = space_curve.col(2);


	const Eigen::MatrixXd G = compute_bem_G(boundary_curve, curve_info);
	const Eigen::MatrixXd H = compute_bem_H(boundary_curve, curve_info);

	const Eigen::VectorXd Hu = H * boundary_points.head(H.cols());
	const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver = G.colPivHouseholderQr();
	if (solver.info() != Eigen::Success)
	{
		std::cout << "big fail" << std::endl;
	}
	const Eigen::VectorXd output = solver.solve(Hu);

	return output;
}

Eigen::VectorXd df_dn_from_G_and_H(const space_curve_t& space_curve, const Eigen::MatrixXd& G, const Eigen::MatrixXd& H)
{
	Eigen::VectorXd boundary_points = space_curve.col(2);
	const Eigen::VectorXd Hu = H * boundary_points.head(H.cols());
	const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver = G.colPivHouseholderQr();
	if (solver.info() != Eigen::Success)
	{
		std::cout << "big fail" << std::endl;
	}
	const Eigen::VectorXd output = solver.solve(Hu);

	return output;
}

Eigen::MatrixXd exterior_df_dn(const space_curve_t& space_curve, const Eigen::MatrixXd& G, const Eigen::MatrixXd& H, const curve_info_t& curve_info)
{
	const double R = R_AT_INF;
	const Eigen::MatrixXd A = (1.0 / (2.0 * EIGEN_PI)) * std::log(R) * curve_info.lengths.replicate(1, curve_info.lengths.rows()); 
	const Eigen::MatrixXd G_A = G - A;
	double a = A_LOG_COEFF;

	Eigen::Vector3d means = space_curve_means(space_curve);
	Eigen::VectorXd Hu = (H * space_curve.col(0));
	
	Eigen::VectorXd rhs_x = (H * space_curve.col(0)).array() - (a * std::log(R) + means(0));
	Eigen::VectorXd rhs_y = (H * space_curve.col(1)).array() - (a * std::log(R) + means(1));
	Eigen::VectorXd rhs_z = (H * space_curve.col(2)).array() - (a * std::log(R) + means(2));

	const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver = G_A.colPivHouseholderQr();
	if (solver.info() != Eigen::Success)
	{
		std::cout << "big fail" << std::endl;
	}
	Eigen::MatrixXd df_dn_ext;
	df_dn_ext.resize(space_curve.rows(), 3);
	df_dn_ext.col(0) = solver.solve(rhs_x);
	df_dn_ext.col(1) = solver.solve(rhs_y);
	df_dn_ext.col(2) = solver.solve(rhs_z);
	return df_dn_ext;
}

Eigen::Matrix<double, Eigen::Dynamic, 2> fundamental_solution_derivative_laplacian_for_boundaries(Eigen::Vector2d p, const BoundaryParametrization* boundary_param)
{
	int n_bd = boundary_param->get_boundary_curves().size();
	int total_rows = boundary_param->get_total_segments();
	Eigen::MatrixXd out(total_rows, 2);
	out.setZero();

	int current_row = 0;
	for (int i = 0; i < n_bd; i++)
	{
		double sign = boundary_param->get_boundary_sign(i);
		int rows = boundary_param->get_segments(i);
		out.block(current_row, 0, rows, 2) += sign * fundamental_solution_derivative_laplacian(p, boundary_param->get_boundary_curves()[i], boundary_param->get_curve_infos()[i]);
		current_row += rows;
	}

	return out;
}

Eigen::VectorXd fundamental_solution_laplacian_for_boundaries(Eigen::Vector2d p, const BoundaryParametrization* boundary_param)
{
	int n_bd = boundary_param->get_boundary_curves().size();
	int total_rows = boundary_param->get_total_segments();
	Eigen::VectorXd out(total_rows);
	out.setZero();

	int current_row = 0;
	for (int i = 0; i < n_bd; i++)
	{
		double sign = boundary_param->get_boundary_sign(i);

		int rows = boundary_param->get_segments(i);
		out.block(current_row, 0, rows, 1) = sign * fundamental_solution_laplacian(p, boundary_param->get_boundary_curves()[i], boundary_param->get_curve_infos()[i]);
		current_row += rows;
	}

	return out;
}

Eigen::VectorXd fundamental_solution_laplacian(Eigen::Vector2d p, const boundary_curve_t& boundary_curve, const curve_info_t& curve_info)
{
	int n = boundary_curve.rows();
	double open_coeff = 1.0;
	if (curve_info.is_open)
	{
		open_coeff *= 2.0;
		n--;
	}
		
	const boundary_curve_t next = circular_shift(boundary_curve, -1);
	const boundary_curve_t& tau = next - boundary_curve;

	const Eigen::VectorXd& dl = curve_info.lengths;
	const Eigen::VectorXd a = dl.array().square();
	Eigen::VectorXd output(n); 
	output.setZero();

	for (int i = 0; i < n; i++)
	{
		double b = (-2.0 * p(0) * tau(i, 0) + 2.0 * boundary_curve(i, 0) * tau(i, 0)) + (-2.0 * p(1) * tau(i, 1) + 2 * boundary_curve(i, 1) * tau(i, 1));
		double c = p(0) * p(0) + p(1) * p(1) + -2 *p(0) * boundary_curve(i, 0) + std::pow(boundary_curve(i, 0),2.0) + (-2) *p(1) * boundary_curve(i, 1) + std::pow(boundary_curve(i, 1),2.0);
		
		// F(x) = @(x)(-dl/(2*EIGEN_PI))*((1/4)*a.^(-1)*((-4)*a*x+2*((-1)*b.^2+4*a*c).^(1/2)*atan(((-1)*b.^2+4*a*c).^(-1/2)*(b+2*a*x))+(b+2*a*x)*log(c+x*(b+a*x))));
		double X = 1.0;
		double coeff1 = (-dl(i) / (2. * EIGEN_PI));
		double coeff2 = (0.25) * (1/ a(i));
		double det = (-b * b + 4. * a(i) * c);

		double sqrt_det = std::max(1e-20, std::sqrt(det));

		for (int X:{0,1})
		{
			double f = coeff1 * (coeff2 * (-4 * a(i) * X + 2.0 * sqrt_det * std::atan((1.0 / sqrt_det) * (b + 2. * a(i) * X)) + (b + 2. * a(i) * X) * std::log(c + X * (b + a(i) * X))));

			double sign = (X == 0) ? -1 : 1;
			output[i] += sign * f;
		}
	}

	return open_coeff * output;
}

boundary_curve_t circular_shift(const boundary_curve_t& curve, int n)
{
	boundary_curve_t output;
	output.resize(curve.rows(), 2);

	for (int i = 0; i < curve.rows(); i++) {
		int index = (i + n) % curve.rows();
		if (index < 0)
			index += curve.rows();
		output.coeffRef(index, 0) = curve(i, 0);
		output.coeffRef(index, 1) = curve(i, 1);
	}

	return output;
}

Eigen::VectorXd circular_shift(const Eigen::VectorXd& vec, int n)
{
	Eigen::VectorXd output;
	output.resize(vec.rows());

	for (int i = 0; i < output.rows(); i++) {
		int index = (i + n) % output.rows();
		if (index < 0)
			index += output.rows();
		output.coeffRef(index) = vec(i);
	}

	return output;
}

Eigen::MatrixXd compute_bem_H(const boundary_curve_t& boundary_curve, const curve_info_t& curve_info)
{
	const boundary_curve_t next = circular_shift(boundary_curve, -1);
	const boundary_curve_t all_p_i = 0.5*(next + boundary_curve);

	int n = boundary_curve.rows();

	if (curve_info.is_open)
		n--;
	Eigen::MatrixXd output;
	output.resize(n, n); // WARNING: DENSE

	for (int i = 0; i < n; i++)
	{
		Eigen::MatrixX2d sol_i = fundamental_solution_derivative_laplacian(Eigen::Vector2d(all_p_i(i,0), all_p_i(i, 1)), boundary_curve, curve_info);

		for (int j = 0; j < n; j++)
		{
			const double nx = curve_info.normals(j, 0);
			const double ny = curve_info.normals(j, 1);

			Eigen::Vector2d sol_ij = sol_i.row(j);
			double dG_dn = nx * sol_i(j, 0) + ny * sol_i(j, 1);
			if (i == j)
			{
				if (curve_info.is_open)
					output(i, j) = 1.0;
				else
					output(i, j) = 0.5; // The BEM document adds "+ dG_dn" (H_hat_ii) here;
			}
			else
			{
				output(i, j) = dG_dn;
			}
		}
	}

	return output;
}

Eigen::MatrixXd compute_bem_G(const boundary_curve_t& boundary_curve, const curve_info_t& curve_info)
{
	const boundary_curve_t next = circular_shift(boundary_curve, -1);
	const boundary_curve_t all_p_i = 0.5 * (next + boundary_curve);

	int n = boundary_curve.rows();
	Eigen::MatrixXd output;

	if (curve_info.is_open)
		n--;

	output.resize(n, n); // WARNING: DENSE

	for (int i = 0; i < n; i++)
	{
		Eigen::VectorXd sol_i = fundamental_solution_laplacian(Eigen::Vector2d(all_p_i(i, 0), all_p_i(i, 1)), boundary_curve, curve_info);
		for (int j = 0; j < n; j++)
		{
			output(i, j) = sol_i(j);
		}
	}
	return output;
}

Eigen::Matrix<double, 3, 2> jacobian_fd(Eigen::Vector2d point, const std::function<Eigen::Vector3d(Eigen::Vector2d)>& func, double eps)
{
	Eigen::Matrix<double, 3, 2> jac;
	Eigen::Vector2d du = Eigen::Vector2d(eps, 0.0);
	Eigen::Vector2d dv = Eigen::Vector2d(0.0, eps);
	Eigen::Vector3d dp_du = (func(point + du) - func(point - du)) / (2.0 * eps);
	Eigen::Vector3d dp_dv = (func(point + dv) - func(point - dv)) / (2.0 * eps);
	jac.col(0) = dp_du;
	jac.col(1) = dp_dv;
	return jac;
}

Eigen::Vector3d representation_formula_interior(Eigen::Vector2d uv, const BoundaryParametrization* boundary_param, const space_curve_t& space_curve, const Eigen::MatrixXd& df_dn)
{
	double u = uv(0);
	double v = uv(1);
	const Eigen::VectorXd& x_curve = space_curve.col(0);
	const Eigen::VectorXd& y_curve = space_curve.col(1);
	const Eigen::VectorXd& z_curve = space_curve.col(2);

	const Eigen::VectorXd& dx_dn = df_dn.col(0);
	const Eigen::VectorXd& dy_dn = df_dn.col(1);
	const Eigen::VectorXd& dz_dn = df_dn.col(2);

	const Eigen::Vector2d p(u, v);
	double sol_x = 0.0;
	double sol_y = 0.0;
	double sol_z = 0.0;


	bool p_is_inside = boundary_param->is_inside_boundary(Eigen::Vector2d(u, v));

	const Eigen::VectorXd G_i = fundamental_solution_laplacian_for_boundaries(p, boundary_param);

	sol_x += G_i.dot(dx_dn);
	sol_y += G_i.dot(dy_dn);
	sol_z += G_i.dot(dz_dn);

	const Eigen::MatrixX2d dG_dn_int = fundamental_solution_derivative_laplacian_for_boundaries(p, boundary_param);

	const Eigen::VectorXd H_hat_i = dG_dn_int.col(0).cwiseProduct(boundary_param->get_normals().col(0)) + dG_dn_int.col(1).cwiseProduct(boundary_param->get_normals().col(1));
	
	// hack for performance THIS ASSUMES boundary 1 is OPEN and 0 is closed.
	if (boundary_param->get_boundary_curves().size() == 2 && boundary_param->is_open(1))
	{
		int n = H_hat_i.rows();
		assert(x_curve.rows() - 1 == H_hat_i.rows());
		// do hack
		for (int i = 0; i < n; i++)
		{
			sol_x -= x_curve(i) * H_hat_i(i);
			sol_y -= y_curve(i) * H_hat_i(i);
			sol_z -= z_curve(i) * H_hat_i(i);
		}
	}
	else
	{
		sol_x -= H_hat_i.dot(x_curve);
		sol_y -= H_hat_i.dot(y_curve);
		sol_z -= H_hat_i.dot(z_curve);
	}


	return Eigen::Vector3d(sol_x, sol_y, sol_z);
}

Eigen::Vector3d evaluate_F(Eigen::Vector3d func_args, const BoundaryParametrization* boundary_param, const space_curve_t& space_curve, const Eigen::MatrixXd& df_dn, Eigen::Vector3d point, Eigen::Vector3d dir)
{
	double u = func_args(0);
	double v = func_args(1);
	double t = func_args(2);

	Eigen::Vector3d sol = representation_formula_interior(Eigen::Vector2d(u,v), boundary_param, space_curve, df_dn);
	
	sol(0) -= point(0) + dir(0) * t;
	sol(1) -= point(1) + dir(1) * t;
	sol(2) -= point(2) + dir(2) * t;
	return sol;
}

Eigen::Matrix<double, 3, 2> jacobian_F(Eigen::Vector2d point, const space_curve_t& space_curve, const Eigen::MatrixXd& df_dn, const BoundaryParametrization* boundary_param)
{
	const double fd_eps = 1e-8;
	Eigen::Vector3d df_du = (representation_formula_interior(Eigen::Vector2d(point(0) + fd_eps, point(1)), boundary_param, space_curve, df_dn) 
								- representation_formula_interior(Eigen::Vector2d(point(0) - fd_eps, point(1)), boundary_param, space_curve, df_dn))/(2.0 * fd_eps);
	Eigen::Vector3d df_dv = (representation_formula_interior(Eigen::Vector2d(point(0), point(1) + fd_eps), boundary_param, space_curve, df_dn)
		- representation_formula_interior(Eigen::Vector2d(point(0), point(1) - fd_eps), boundary_param, space_curve, df_dn)) / (2.0 * fd_eps);
	
	Eigen::Matrix<double, 3, 2> fd_output;
	fd_output.col(0) = df_du;
	fd_output.col(1) = df_dv;

	return fd_output;
	/*std::cout << "fd" << std::endl;
	std::cout << fd_output << std::endl;

	Eigen::Matrix<double, 3, 2> output;
	
	
	const Eigen::MatrixX2d dG_dn_int = fundamental_solution_derivative_laplacian_for_boundaries(point, boundary_param);
	const Eigen::VectorXd dG_du = dG_dn_int.col(0);
	const Eigen::VectorXd dG_dv = dG_dn_int.col(1);
	
	const Eigen::MatrixX2d dG_dpdn = fundamental_solution_double_derivative_laplacian(point, boundary_param->get_boundary_curves()[0], boundary_param->get_curve_infos()[0]);
	const Eigen::VectorXd ddG_dn_du = dG_dpdn.col(0);
	const Eigen::VectorXd ddG_dn_dv = dG_dpdn.col(1);


	output.col(0) = df_dn.transpose() * dG_du - space_curve.transpose() * ddG_dn_du;
	output.col(1) = df_dn.transpose() * dG_dv - space_curve.transpose() * ddG_dn_dv;

	std::cout << "analytical" << std::endl;
	std::cout << -output << std::endl;

	return -output;*/
}


Eigen::MatrixXd random_shuffle(const Eigen::MatrixXd& v, std::vector<int>& ordering)
{
	Eigen::MatrixXd  output(v.rows(), v.cols());
	std::random_shuffle(ordering.begin(), ordering.end());

	for (int i = 0; i < ordering.size(); i++)
	{
		output.row(i) = v.row(ordering[i]);
	}
	return output;
}

int find_by_row(const Eigen::MatrixXd& mat, const Eigen::RowVector3d& query, double tolerance, int start_idx)
{

	for (int i = start_idx; i < mat.rows(); i++)
	{
		if ((query - mat.row(i)).norm() < tolerance)
			return i;
	}
	return -1;
}


Eigen::MatrixXd rescale_to_unit(const Eigen::MatrixXd& points)
{
	box bb = bounding_box(points);
	double scale_x = bb.x_max - bb.x_min;
	double scale_y = bb.y_max - bb.y_min;
	double scale_z = bb.z_max - bb.z_min;

	double scale_factor = std::max(scale_x, std::max(scale_y, scale_z));

	double center_x = (bb.x_min + bb.x_max) / 2.0;
	double center_y = (bb.y_min + bb.y_max) / 2.0;
	double center_z = (bb.z_min + bb.z_max) / 2.0;
	Eigen::RowVector3d center = Eigen::RowVector3d(center_x, center_y, center_z);

	Eigen::MatrixXd normalized_points = points;
	normalized_points.rowwise() -= center;
	normalized_points /= scale_factor;
	normalized_points.rowwise() += Eigen::RowVector3d(0.5, 0.5, 0.5);
	return normalized_points;
}

Eigen::MatrixXd slice_XY(int row_count)
{
	Eigen::MatrixXd points(row_count * row_count, 3);
	std::vector<double> xs = linspace(0, 1, row_count, true);
	std::vector<double> ys = linspace(0, 1, row_count, true);

	for (int i = 0; i < row_count; i++)
	{
		for (int j = 0; j < row_count; j++)
		{
			points.row(i * row_count + j) = Eigen::Vector3d(xs[i], ys[j], 0.5);
		}
	}

	return points;
}

Eigen::MatrixXd slice_XZ(int row_count)
{
	Eigen::MatrixXd points(row_count * row_count, 3);
	std::vector<double> xs = linspace(0, 1, row_count, true);
	std::vector<double> zs = linspace(0, 1, row_count, true);

	for (int i = 0; i < row_count; i++)
	{
		for (int j = 0; j < row_count; j++)
		{
			points.row(i * row_count + j) = Eigen::Vector3d(xs[i], 0.5, zs[j]);
		}
	}

	return points;
}

std::vector<std::vector<int>> rays_from_slice(const Eigen::MatrixXd& query_points)
{
	int num_rays = std::sqrt<int>(query_points.rows());
	std::vector<std::vector<int>> rays;

	int current = 0;
	for (int i = 0; i < num_rays; i++)
	{
		std::vector<int> vec(num_rays, 0);
		std::iota(vec.begin(), vec.end(), current);
		current += num_rays;		
		rays.push_back(vec);
	}
	return rays;
}

std::vector<Eigen::MatrixXd> rescale_to_unit(const std::vector<Eigen::MatrixXd>& patches)
{
	int total_rows = 0;
	for (int i = 0; i < patches.size(); i++)
		total_rows += patches[i].rows();

	Eigen::MatrixXd points(total_rows, 3);

	int point_id = 0;
	for (int i = 0; i < patches.size(); i++)
	{
		for (int j = 0; j < patches[i].rows(); j++)
			points.row(point_id++) = patches[i].row(j);
	}

	points = rescale_to_unit(points);

	std::vector<Eigen::MatrixXd> new_patches;
	point_id = 0;
	for (int i = 0; i < patches.size(); i++)
	{
		Eigen::MatrixXd p(patches[i].rows(), 3);
		for (int j = 0; j < patches[i].rows(); j++)
			p.row(j) = points.row(point_id++);
		new_patches.push_back(p);
	}
	return new_patches;
}

void triangulate_unit_square(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int N)
{
	int total = N * N;
	V.resize(total, 2);
	std::vector<double> us = linspace(0, 1, N, true);
	std::vector<double> vs = linspace(0, 1, N, true);

	int v_id = 0;
	for (int i = 0; i < us.size(); i++)
	{
		for (int j = 0; j < vs.size(); j++)
		{
			double u = us[i];
			double v = vs[j];
			V.row(v_id++) = Eigen::Vector2d(u, v);
		}
	}

	std::vector<Eigen::Vector3i> faces;
	for (int i = 0; i < N - 1; i++)
	{
		for (int j = 0; j < N - 1; j++)
		{
			faces.push_back(Eigen::Vector3i(i * N + j, (i+1) * N + j, i*N + (j+1)));
			faces.push_back(Eigen::Vector3i((i + 1) * N + j, (i + 1) * N + j + 1, i * N + (j + 1)));
		}
	}

	F.resize(faces.size(), 3);

	for (int i = 0; i < faces.size(); i++)
		F.row(i) = faces[i];
}