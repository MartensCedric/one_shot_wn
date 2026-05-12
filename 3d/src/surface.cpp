#include "surface.h"

void sample_interior_surface(const precomputed_curve_data& precompute, const curve_net_sampler& sampler, int curve_id)
{
	int n_samples = 50000;
	const BoundaryParametrization* boundary_param = precompute.int_params[curve_id];
	const space_curve_t& space_curve = sampler.patches[curve_id];

	const Eigen::MatrixXd& df_dn = precompute.df_dns[curve_id];
	std::vector<Eigen::Vector2d> uvs = boundary_param->sample_inside_points(n_samples);
	space_curve_t ext_full_values;
	ext_full_values.resize(boundary_param->get_total_points(), 3);
	const boundary_curve_t& bd_large = boundary_param->get_boundary_curves()[0];
	const boundary_curve_t& bd_small = boundary_param->get_boundary_curves()[1];
	int n_large = bd_large.rows();
	int n_small = bd_small.rows();

	space_curve_t values_on_plane = fit_data_to_patch_plane(bd_large, bd_small, space_curve);

	ext_full_values.block(0, 0, n_large, 3) = values_on_plane;
	ext_full_values.block(n_large, 0, n_small, 3) = space_curve;

	space_curve_t patch_ext = values_on_plane;

	std::ofstream outfile("points.m");
	outfile << "pts = [";
	for (int i = 0; i < n_samples; i++)
	{
		Eigen::Vector3d p = representation_formula_interior(uvs[i], boundary_param, ext_full_values, df_dn);

		outfile << p.x() << ", " << p.y() << ", " << p.z() << "; ";
	}

	outfile << "];" << std::endl;

	outfile << "uvs = [";

	for (int i = 0; i < uvs.size(); i++)
	{
		outfile << "[" << uvs[i][0] << ", " << uvs[i][1] << "];";
	}
	outfile << "];" << std::endl;

	outfile << "df_dn = [";
	for (int i = 0; i < df_dn.rows(); i++)
	{
		outfile << "[" << df_dn.row(i)[0] << ", " << df_dn.row(i)[1] << ", " << df_dn.row(i)[2] << "];";
	}

	outfile << "];" << std::endl;

	outfile << "wn = [";
	for (int i = 0; i < uvs.size(); i++)
	{
		double wn = wn_2d_open_square(uvs[i], sampler.insideness[curve_id]);
		outfile << wn << ",";
	}
	outfile << "];" << std::endl;


	outfile << "boundary0 = [";
	const boundary_curve_t& boundary_curve0 = boundary_param->get_boundary_curves()[0];
	for (int i = 0; i < boundary_curve0.rows(); i++)
	{
		outfile << "[" << boundary_curve0.row(i)[0] << ", " << boundary_curve0.row(i)[1] << "];";
	}
	outfile << "];" << std::endl;

	outfile << "boundary1 = [";
	const boundary_curve_t& boundary_curve1 = boundary_param->get_boundary_curves()[1];
	for (int i = 0; i < boundary_curve1.rows(); i++)
	{
		outfile << "[" << boundary_curve1.row(i)[0] << ", " << boundary_curve1.row(i)[1] << "];";
	}
	outfile << "];" << std::endl;



	outfile << "patch = [";
	for (int i = 0; i < space_curve.rows(); i++)
	{
		outfile << "[" << space_curve.row(i)[0] << ", " << space_curve.row(i)[1] << ", " << space_curve.row(i)[2] << "];";
	}
	outfile << "];" << std::endl;


	outfile << "patch_ext = [";
	for (int i = 0; i < patch_ext.rows(); i++)
	{
		outfile << "[" << patch_ext.row(i)[0] << ", " << patch_ext.row(i)[1] << ", " << patch_ext.row(i)[2] << "];";
	}
	outfile << "];" << std::endl;
}

void sample_implicit_surface(implicit_func_t function)
{
	int n_samples = 50000;
	double min_u = -10;
	double min_v = -10;
	double max_u = 10;
	double max_v = 10;

	std::random_device os_seed;
	std::default_random_engine gen(os_seed());

	std::uniform_real_distribution<double> u_dis(min_u, max_u);
	std::uniform_real_distribution<double> v_dis(min_v, max_v);

	std::vector<Eigen::Vector2d> uvs;
	for (int i = 0; i < n_samples; i++)
		uvs.push_back(Eigen::Vector2d(u_dis(gen), v_dis(gen)));

	std::ofstream outfile("implicit_surface_points.m");
	outfile << "pts = [";
	for (int i = 0; i < n_samples; i++)
	{
		Eigen::Vector3d p = function(uvs[i]);

		outfile << p.x() << ", " << p.y() << ", " << p.z() << "; ";
	}

	outfile << "];" << std::endl;

	outfile << "uvs = [";

	for (int i = 0; i < uvs.size(); i++)
	{
		outfile << "[" << uvs[i][0] << ", " << uvs[i][1] << "];";
	}
	outfile << "];" << std::endl;
}

//void sample_surface(const precomputed_curve_data& precompute, const curve_net_sampler& sampler, int curve_id)
//{
//	int n_samples_interior = 1000;
//	int n_samples_exterior = 5000;
//	const BoundaryParametrization* boundary_param = precompute.int_params[curve_id];
//	//const BoundaryParametrization* boundary_param_ext = precompute.ext_params[curve_id];
//	const space_curve_t& space_curve = sampler.patches[curve_id];
//	space_curve_t space_curve_ext;
//	space_curve_ext.resize(space_curve.rows() * 2, 3);
//	space_curve_ext.block(0,0, space_curve.rows(), 3) = dirichlet_at_inf(space_curve, 5.0);
//	space_curve_ext.block(space_curve.rows(), 0, space_curve.rows(), 3) = space_curve;
//	const Eigen::MatrixXd& df_dn = precompute.df_dns[curve_id];
//	//const Eigen::MatrixXd& df_dn_ext = precompute.df_dns_ext[curve_id];
//	std::vector<Eigen::Vector2d> uvs_int = boundary_param->sample_inside_points(n_samples_interior);
//	//std::vector<Eigen::Vector2d> uvs_out = boundary_param_ext->sample_inside_points(n_samples_exterior);
//	std::ofstream outfile("points.m");
//	outfile << "pts_inside = [";
//	for (int i = 0; i < n_samples_interior; i++)
//	{
//		Eigen::Vector3d p = representation_formula_interior(uvs_int[i], boundary_param, space_curve, df_dn);
//
//		outfile << p.x() << ", " << p.y() << ", " << p.z() << "; ";
//	}
//	outfile << "];" << std::endl;
//
//
//	/*outfile << "pts_outside = [";
//	for (int i = 0; i < n_samples_exterior; i++)
//	{
//		Eigen::Vector3d p = representation_formula_interior(uvs_out[i], boundary_param_ext, space_curve_ext, df_dn_ext);
//
//		outfile << p.x() << ", " << p.y() << ", " << p.z() << "; ";
//	}
//	outfile << "];" << std::endl;*/
//
//	outfile << "uvs_inside = [";
//	for (int i = 0; i < uvs_int.size(); i++)
//	{
//		outfile << "[" << uvs_int[i][0] << ", " << uvs_int[i][1] << "];";
//	}
//	outfile << "];" << std::endl;
//
//	//outfile << "uvs_outside = [";
//	//for (int i = 0; i < uvs_out.size(); i++)
//	//{
//	//	outfile << "[" << uvs_out[i][0] << ", " << uvs_out[i][1] << "];";
//	//}
//	//outfile << "];" << std::endl;
//
//	outfile << "boundary = [";
//	const boundary_curve_t& boundary_curve = boundary_param->get_boundary_curves()[0];
//	for (int i = 0; i < boundary_curve.rows(); i++)
//	{
//		outfile << "[" << boundary_curve.row(i)[0] << ", " << boundary_curve.row(i)[1] << "];";
//	}
//	outfile << "];" << std::endl;
//
//	//outfile << "ext_boundary = [";
//
//	//for (const boundary_curve_t& ext_boundary_curve : boundary_param_ext->get_boundary_curves())
//	//{
//	//	for (int i = 0; i < ext_boundary_curve.rows(); i++)
//	//	{
//	//		outfile << "[" << ext_boundary_curve.row(i)[0] << ", " << ext_boundary_curve.row(i)[1] << "];";
//	//	}
//	//}
//	//outfile << "];" << std::endl;
//	
//	outfile << "ext_patch = [";
//	for (int i = 0; i < space_curve_ext.rows(); i++)
//	{
//		outfile << "[" << space_curve_ext.row(i)[0] << ", " << space_curve_ext.row(i)[1] << ", " << space_curve_ext.row(i)[2] << "];";
//	}
//	outfile << "];" << std::endl;
//	
//	outfile << "df_dn = [";
//	for (int i = 0; i < df_dn.rows(); i++)
//	{
//		outfile << "[" << df_dn.row(i)[0] << ", " << df_dn.row(i)[1] << ", " << df_dn.row(i)[2] << "];";
//	}
//
//	outfile << "];" << std::endl;
//
//
//	outfile << "patch = [";
//	for (int i = 0; i < space_curve.rows(); i++)
//	{
//		outfile << "[" << space_curve.row(i)[0] << ", " << space_curve.row(i)[1] << ", " << space_curve.row(i)[2] << "];";
//	}
//	outfile << "];" << std::endl;
//}