#include "adaptive.h"

double bem_double_integral(const Eigen::Vector2d& uv, const space_curve_t& space_curve, const Eigen::MatrixXd& df_dn, const BoundaryParametrization* boundary_param, const Eigen::Vector3d& q)
{
	const Eigen::Vector3d& p = representation_formula_interior(uv, boundary_param, space_curve, df_dn);

	const Eigen::Vector3d r = p - q;

	const Eigen::MatrixXd& jac = jacobian_F(uv, space_curve, df_dn, boundary_param);
	Eigen::Vector3d dp_du = jac.col(0);
	Eigen::Vector3d dp_dv = jac.col(1);
	Eigen::Vector3d normal = dp_du.cross(dp_dv);
	normal.normalize();

	double r_norm = r.norm();
	double r_norm_3 = r_norm * r_norm * r_norm;
	return r.dot(normal) / r_norm_3;
}
