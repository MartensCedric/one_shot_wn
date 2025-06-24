#pragma once

#include <Eigen/Dense>

void intersect_subd_loop(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces, const Eigen::MatrixXd& query_points);