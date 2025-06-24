#pragma once

#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <iostream>
#include "boundary_processing.h"
#include "uv_util.h"
#include "curve_net.h"
#include "curve_net_wn.h"
#include "parametric.h"
#include "open_curve.h"
#include <functional>


void sample_interior_surface(const precomputed_curve_data& precompute, const curve_net_sampler& sampler, int curve_id);
void sample_implicit_surface(implicit_func_t function);
void sample_surface(const precomputed_curve_data& precompute, const curve_net_sampler& sampler, int curve_id);