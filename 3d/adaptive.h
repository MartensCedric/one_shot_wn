#pragma once

#include <Eigen/Dense>

#include "boundary_processing.h"
#include "math_util.h"
#include "uv_util.h"

double bem_double_integral(const Eigen::Vector2d& uv, const space_curve_t& space_curve, const Eigen::MatrixXd& df_dn, const BoundaryParametrization* boundary_param, const Eigen::Vector3d& q);
