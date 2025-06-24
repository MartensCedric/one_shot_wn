// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Axom Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/*! \file primal_introduction.cpp
 *  \brief This example code is a demonstration of Axom's primal component.
 *
 *  This file shows how to use Primal to represent geometric primitives, perform
 *  geometric operations, and use a spatial index.  Running the executable from
 *  this file will produce a collection of Asymptote source files.  When
 *  compiled, the Asymptote files produce the figures that accompany primal's
 *  Sphinx documentation.
 */

/* This example code contains snippets used in the Primal Sphinx documentation.
 * They begin and end with comments such as
 *
 * prims_header_start
 * prims_header_end
 * clip_header_start
 * clip_header_end
 * closest_point_header_start
 * closest_point_header_end
 *
 * each prepended with an underscore.
 */

#include "axom/config.hpp"
#include <chrono>
// _prims_header_start
// Axom primitives
#include "axom/primal/geometry/BoundingBox.hpp"
#include "axom/primal/geometry/OrientedBoundingBox.hpp"
#include "axom/primal/geometry/Point.hpp"
#include "axom/primal/geometry/Polygon.hpp"
#include "axom/primal/geometry/Ray.hpp"
#include "axom/primal/geometry/Segment.hpp"
#include "axom/primal/geometry/Triangle.hpp"
#include "axom/primal/geometry/Vector.hpp"
// _prims_header_end

// Axom operations
// Each header is used in its own example, so each one is bracketed for separate
// inclusion.
// _clip_header_start
#include "axom/primal/operators/clip.hpp"
// _clip_header_end
// _closest_point_header_start
#include "axom/primal/operators/closest_point.hpp"
// _closest_point_header_end
// _bbox_header_start
#include "axom/primal/operators/compute_bounding_box.hpp"
// _bbox_header_end
// _intersect_header_start
#include "axom/primal/operators/intersect.hpp"
// _intersect_header_end
// _orient_header_start
#include "axom/primal/operators/orientation.hpp"
// _orient_header_end
// _sqdist_header_start
#include "axom/primal/operators/squared_distance.hpp"
// _sqdist_header_end

// C++ headers
#include <cmath>  // do we need this?
#include <iostream>
#include <fstream>

#include "axom/primal.hpp"                           
#include "axom/core.hpp"
#include "axom/slic.hpp"
#include "axom/fmt.hpp"

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include <cassert>
#include <random>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include "axom/primal/operators/detail/winding_number_impl.hpp"
// _using_start
// "using" directives to simplify code
namespace primal = axom::primal;

constexpr int in2D = 2;

// primitives represented by doubles in 2D
using Point2D = primal::Point<double, in2D>;
using BoundingBoxType = primal::BoundingBox<double, in2D>;
using Bezier = primal::BezierCurve<double, 2>;


using namespace Eigen;
using namespace std;
// _using_end

std::string printPoint(Point2D pt)
{
  return axom::fmt::format("({},{})", pt[0], pt[1]);
}

std::string printPoint(double* pt)
{
  return axom::fmt::format("({},{})", pt[0], pt[1]);
}

void writeToFile(std::string fname, std::string contents)
{
  std::ofstream outfile(fname);
  if(!outfile.good())
  {
    std::cout << "Could not write to " << fname << std::endl;
  }
  else
  {
    outfile << contents;
  }
}

using engine = std::mt19937;



Bezier control_points_to_bezier(const Eigen::MatrixXd& m)
{
	Point2D random_nodes[] = {
  		         Point2D {m(0, 0), m(0, 1)},
			 Point2D {m(1, 0), m(1, 1)},
			 Point2D {m(2, 0), m(2, 1)},
			 Point2D {m(3, 0), m(3, 1)},};
			
		 Bezier curve(random_nodes, 3);
	return curve;
}
constexpr double abs_tol = 1e-7;
constexpr double edge_tol = 1e-7;
constexpr double martens_tol = edge_tol; // same as jacob
constexpr double bezier_box_delta = 1e-2;
constexpr double EPS = primal::PRIMAL_TINY;

MatrixXd evaluate_wn_field_spainhour(const Eigen::MatrixXd& C, int resolution)
{
  double min_x = C.col(0).minCoeff() + bezier_box_delta;
  double max_x = C.col(0).maxCoeff() - bezier_box_delta;
  double min_y = C.col(1).minCoeff() + bezier_box_delta;
  double max_y = C.col(1).maxCoeff() - bezier_box_delta;
  double step_x = (max_x - min_x) / static_cast<double>(resolution);
  double step_y = (max_y - min_y) / static_cast<double>(resolution);
  Bezier bez = control_points_to_bezier(C);
  MatrixXd results(resolution, resolution);
  for(int iteration_x = 0; iteration_x < resolution; iteration_x++)
  {
	double x = min_x + static_cast<double>(iteration_x) * step_x;
	for(int iteration_y = 0; iteration_y < resolution; iteration_y++)
	{
		double y = min_y + static_cast<double>(iteration_y) * step_y;
		Point2D q{x + step_x / 2.0,y  + step_y / 2.0};
		double w = winding_number(q, bez, edge_tol, EPS);
		results(iteration_y,iteration_x) = w;
	}
  }

  return results;
}

// Function declarations
pair<VectorXd, MatrixXd> rayIntersectBezier(const Vector2d& origin, const Vector2d& direction,
                                            const Vector2d& p0, const Vector2d& p1,
                                            const Vector2d& p2, const Vector2d& p3,
                                            double tolerance);
std::vector<MatrixXd> random_bezier_curves(int n);
pair<VectorXd, MatrixXd> bezierClip(const Vector2d& origin, const Vector2d& direction,
                                    const Vector2d& p0, const Vector2d& p1,
                                    const Vector2d& p2, const Vector2d& p3,
                                    double tMin, double tMax, double tolerance);
double quadrature_wn(const std::vector<Eigen::Vector2d>& polyline, const Eigen::Vector2d& q);
bool rayBoxIntersection(const Vector2d& origin, const Vector2d& direction,
                        const Matrix2d& bounds);
std::vector<Eigen::Vector2d> discretize_bezier(const Eigen::Matrix<double, 4, 2>& controlPoints, int numSamples);
void subdivideBezier(const Vector2d& p0, const Vector2d& p1,
                     const Vector2d& p2, const Vector2d& p3, double t,
                     Vector2d& p0_left, Vector2d& p1_left, Vector2d& p2_left, Vector2d& p3_left,
                     Vector2d& p0_right, Vector2d& p1_right, Vector2d& p2_right, Vector2d& p3_right);

pair<Vector2d, Vector2d> evaluateBezier(const Vector2d& p0, const Vector2d& p1,
                                        const Vector2d& p2, const Vector2d& p3, double t);

// Main intersection function
pair<VectorXd, MatrixXd> rayIntersectBezier(const Vector2d& origin, const Vector2d& direction,
                                            const Vector2d& p0, const Vector2d& p1,
                                            const Vector2d& p2, const Vector2d& p3,
                                            double tolerance) {
    return bezierClip(origin, direction, p0, p1, p2, p3, 0.0, 1.0, tolerance);
}

VectorXd windingNumberEndpoints(const MatrixXd& C, const MatrixXd& queryPoints, double tolerance = 1e-6);

bool is_inside_boundingbox(const MatrixXd& bb, const Vector2d& p);

pair<VectorXd, MatrixXd> bezierClip(const Vector2d& origin, const Vector2d& direction,
                                    const Vector2d& p0, const Vector2d& p1,
                                    const Vector2d& p2, const Vector2d& p3,
                                    double tMin, double tMax, double tolerance) {
    // Calculate bounding box
    Matrix2d bounds;
    bounds << p0.cwiseMin(p1).cwiseMin(p2).cwiseMin(p3),
              p0.cwiseMax(p1).cwiseMax(p2).cwiseMax(p3);

    // Check if ray intersects bounding box

    if (!rayBoxIntersection(origin, direction, bounds)) {
        return make_pair(VectorXd(), MatrixXd());
    }
    

    // If the curve is small enough, consider it an intersection
    // but also make sure we're far enough from the endpoints points.
    double tol_sq = tolerance * tolerance;

  
    if ((p3 - p0).squaredNorm() <= tol_sq) {
        double t = (tMin + tMax) / 2.0;
       
        auto [point, tangent] = evaluateBezier(p0, p1, p2, p3, t);

        Vector2d normal(tangent(1), -tangent(0)); // Perpendicular to the tangent
        normal.normalize();
        VectorXd tVec(1);
        tVec << (origin - point).squaredNorm();
        MatrixXd normals(1, 2);
        normals.row(0) = normal;
        return make_pair(tVec, normals);
    }

    // Subdivide the curve
    double midT = (tMin + tMax) / 2.0;
    Vector2d p0_left, p1_left, p2_left, p3_left, p0_right, p1_right, p2_right, p3_right;
    subdivideBezier(p0, p1, p2, p3, midT, p0_left, p1_left, p2_left, p3_left,
                    p0_right, p1_right, p2_right, p3_right);

    // Recursively clip the left and right sub-curves
    auto [tLeft, normalsLeft] = bezierClip(origin, direction, p0_left, p1_left, p2_left, p3_left, tMin, midT, tolerance);
    auto [tRight, normalsRight] = bezierClip(origin, direction, p0_right, p1_right, p2_right, p3_right, midT, tMax, tolerance);

    // Combine results
    VectorXd t(tLeft.size() + tRight.size());
    t << tLeft, tRight;

    MatrixXd normals(normalsLeft.rows() + normalsRight.rows(), 2);
    if (normalsLeft.rows() > 0 && normalsRight.rows() > 0) {
        normals << normalsLeft, normalsRight;
    } else if (normalsLeft.rows() > 0) {
        normals = normalsLeft;
    } else if (normalsRight.rows() > 0) {
        normals = normalsRight;
    } else {
        normals = MatrixXd();
    }

    return make_pair(t, normals);
}

bool is_inside_boundingbox(const MatrixXd& bb, const Vector2d& p)
{
	if(bb(0, 0) > p(0) || bb(0, 1) < p(0))
		return false;
	if(bb(1,0) > p(1) || bb(1,1) < p(1))
		return false;
	return true;
}

bool rayBoxIntersection(const Vector2d& origin, const Vector2d& direction,
                        const Matrix2d& bounds) {
    double tMin = -numeric_limits<double>::infinity();
    double tMax = numeric_limits<double>::infinity();
    
    for (int i = 0; i < 2; ++i) {
        if (direction(i) == 0.0) {
            // Ray is parallel to this axis
            if (origin(i) < bounds(i, 0) || origin(i) > bounds(i, 1)) {
                // Ray is outside the bounds in this dimension
                return false;
            }
        } else {
            // Ray is not parallel to this axis
            double invDir = 1.0 / direction(i);
            double t1 = (bounds(i, 0) - origin(i)) * invDir;
            double t2 = (bounds(i, 1) - origin(i)) * invDir;

            if (invDir < 0.0) {
                swap(t1, t2);
            }

            tMin = max(tMin, t1);
            tMax = min(tMax, t2);

            if (tMin > tMax || tMax < 0) {
                return false;
            }
        }
    }

    return true;
}

// Bezier subdivision function
void subdivideBezier(const Vector2d& p0, const Vector2d& p1,
                     const Vector2d& p2, const Vector2d& p3, double t,
                     Vector2d& p0_left, Vector2d& p1_left, Vector2d& p2_left, Vector2d& p3_left,
                     Vector2d& p0_right, Vector2d& p1_right, Vector2d& p2_right, Vector2d& p3_right) {
    Vector2d q0 = (1 - t) * p0 + t * p1;
    Vector2d q1 = (1 - t) * p1 + t * p2;
    Vector2d q2 = (1 - t) * p2 + t * p3;

    Vector2d r0 = (1 - t) * q0 + t * q1;
    Vector2d r1 = (1 - t) * q1 + t * q2;

    Vector2d s0 = (1 - t) * r0 + t * r1;

    p0_left = p0;
    p1_left = q0;
    p2_left = r0;
    p3_left = s0;

    p0_right = s0;
    p1_right = r1;
    p2_right = q2;
    p3_right = p3;
}


// Bezier evaluation function
pair<Vector2d, Vector2d> evaluateBezier(const Vector2d& p0, const Vector2d& p1,
                                        const Vector2d& p2, const Vector2d& p3, double t) {
    Vector2d point = pow(1 - t, 3) * p0 + 3 * pow(1 - t, 2) * t * p1 + 3 * (1 - t) * pow(t, 2) * p2 + pow(t, 3) * p3;
    Vector2d tangent = 3 * pow(1 - t, 2) * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * pow(t, 2) * (p3 - p2);
    return make_pair(point, tangent);
}


VectorXd windingNumberEndpoints(const MatrixXd& C, const MatrixXd& row_query_points, double tolerance) {
    int wnSize = row_query_points.rows();
    VectorXd wn = VectorXd::Zero(wnSize);

    // Extract control points of the Bézier curve
    Vector2d startPoint = C.row(0);
    Vector2d endPoint = C.row(3);

    // Direction vector for ray intersection (horizontal ray)
    
    Vector2d dir = row_query_points.row(1) - row_query_points.row(0);
    dir.normalize();
    Vector2d p0 = C.row(0).transpose();
    Vector2d p1 = C.row(1).transpose();
    Vector2d p2 = C.row(2).transpose();
    Vector2d p3 = C.row(3).transpose();
    
    Point2D p_start = Point2D{p0(0), p0(1)};
    Point2D p_end = Point2D{p3(0), p3(1)};
      Matrix2d bounds;
	       bounds << p0.cwiseMin(p1).cwiseMin(p2).cwiseMin(p3),
		      p0.cwiseMax(p1).cwiseMax(p2).cwiseMax(p3);

    
    auto [ts_sq, normals] = rayIntersectBezier(row_query_points.row(0).transpose(), dir,
                                                p0, p1, p2, p3, tolerance);
    std::vector<int> sign(ts_sq.size());
    for(int k = 0; k < normals.rows(); k++)
    {
    	bool same_dir = normals.row(k).dot(dir) > 0.0;
    	sign[k] = same_dir ? 1 : -1;
    }
    Vector3d n(0.0, 0.0, 1.0);
    Vector3d dir_3d(dir(0), dir(1), 0.0);
    Vector2d first_q = row_query_points.row(0);  
                                
    for (int i = 0; i < wnSize; ++i) {
        Vector2d q = row_query_points.row(i);
        
        if(!is_inside_boundingbox(bounds, q))
        {
           Point2D p{q(0), q(1)};
           wn(i) = axom::primal::detail::linear_winding_number(p, p_start, p_end, edge_tol);
        }
        else
        {
               //std::cout << "startPoint: \n" << startPoint << std::endl;
               //std::cout << "endPoint: \n" << endPoint << std::endl;
               //std::cout << "q: \n" << q << std::endl;
		Vector2d dir_to_start = (startPoint - q).normalized();
		Vector2d dir_to_end = (endPoint - q).normalized();
	
		double theta_start = acos(dir_to_start.dot(dir));
		double theta_end = acos(dir_to_end.dot(dir));
		double cosTheta = dir_to_start.dot(dir_to_end);
		double thetaRadians = acos(cosTheta);
	 	
		Vector3d dir_to_start_3d = Vector3d(dir_to_start(0), dir_to_start(1), 0.0);
	 	Vector3d dir_to_end_3d = Vector3d(dir_to_end(0), dir_to_end(1), 0.0);
	 	
	 	//std::cout << "dir_to_start: \n" << dir_to_start << std::endl;
	 	//std::cout << "dir_to_end: \n" << dir_to_end << std::endl;
	 	//std::cout << "cosTheta: " << cosTheta << std::endl;
	 	
		//std::cout << "theta_start: " << theta_start << std::endl;
		//std::cout << "theta_end: " << theta_end << std::endl;
		//std::cout << "thetaRadians: " << thetaRadians << std::endl;
	 	
	 	if(dir_to_start_3d.cross(dir_3d).dot(n) > 0.0)
	 		theta_start = 2.0 * M_PI - theta_start;

		if(dir_to_end_3d.cross(dir_3d).dot(n) > 0.0)
	 		theta_end = 2.0 * M_PI  - theta_end; 		
	       // Calculate bounding box
	     
	       if(dir_to_start_3d.cross(dir_to_end_3d).dot(n) < 0.0)
	       {
	    		thetaRadians = 2.0 * M_PI  - thetaRadians;
	       }


	   //std::cout << "theta_start: " << theta_start << std::endl;
	   //std::cout << "theta_end: " << theta_end << std::endl;
	   //std::cout << "thetaRadians: " << thetaRadians << std::endl;

	   double current_t_sq = (q - first_q).squaredNorm();
	   int chi = 0;
	   for(int k = 0; k < ts_sq.rows(); k++)
	   {
		if(current_t_sq <= ts_sq(k))
		{
		 chi += sign[k];
		}
	   }
	   
	   assert(theta_start >= 0.0);
	   assert(theta_end >= 0.0);
	   assert(thetaRadians >= 0.0);
	   
	   int other_chi = chi;
	   if(theta_start < theta_end) // inside
	   {
	    //std::cout << "inside" << std::endl;
	    other_chi++;
	    //std::cout << "wn(i) = (chi * (2.0 * M_PI - thetaRadians) + other_chi * thetaRadians) / (2.0 * M_PI); " << std::endl;
	    //std::cout << chi << " * " << (2.0 * M_PI - thetaRadians) << " + " << other_chi << " * " << thetaRadians << std::endl;
	    wn(i) = (chi * (2.0 * M_PI - thetaRadians) + other_chi * thetaRadians) / (2.0 * M_PI); 
	    
	   }	   
	   else
	   {
	     //std::cout << "outside" << std::endl;
	     other_chi--;
	     //std::cout << "wn(i) = ((chi * thetaRadians) + other_chi * (2.0 * M_PI - thetaRadians)) / (2.0 * M_PI); " << std::endl;
	     //std::cout << chi << " * " << thetaRadians << " + " << other_chi << " * " << (2.0 * M_PI - thetaRadians) << std::endl;
	     wn(i) = ((chi * thetaRadians) + other_chi * (2.0 * M_PI - thetaRadians)) / (2.0 * M_PI); 
	     
	   }
        }	
    }

    return wn;
}

MatrixXd evaluate_wn_field_quadrature(const MatrixXd& C, int resolution)
{
	double min_x = C.col(0).minCoeff() + bezier_box_delta;
  double max_x = C.col(0).maxCoeff() - bezier_box_delta;
  double min_y = C.col(1).minCoeff() + bezier_box_delta;
  double max_y = C.col(1).maxCoeff() - bezier_box_delta;
  double step_x = (max_x - min_x) / static_cast<double>(resolution);
  double step_y = (max_y - min_y) / static_cast<double>(resolution);
  std::vector<Eigen::Vector2d> polyline = discretize_bezier(C, 10000);
  MatrixXd results(resolution, resolution);
  for(int iteration_x = 0; iteration_x < resolution; iteration_x++)
  {
	double x = min_x + static_cast<double>(iteration_x) * step_x;
	for(int iteration_y = 0; iteration_y < resolution; iteration_y++)
	{
		double y = min_y + static_cast<double>(iteration_y) * step_y;
		Vector2d q{x + step_x / 2.0,y  + step_y / 2.0};
		double w = quadrature_wn(polyline, q);
		results(iteration_y,iteration_x) = w;
	}
  }
  
  return results;

}

MatrixXd evaluate_wn_field_martens(const MatrixXd& C, int resolution)
{
  double min_x = C.col(0).minCoeff() + bezier_box_delta;
  double max_x = C.col(0).maxCoeff() - bezier_box_delta;
  double min_y = C.col(1).minCoeff() + bezier_box_delta;
  double max_y = C.col(1).maxCoeff() - bezier_box_delta;
  double step_x = (max_x - min_x) / static_cast<double>(resolution);
  double step_y = (max_y - min_y) / static_cast<double>(resolution);

  std::vector<MatrixXd> row_queries;
  for(int iteration_y = 0; iteration_y < resolution; iteration_y++)
  {
   	double y = min_y + static_cast<double>(iteration_y) * step_y;  
  	MatrixXd row(resolution, 2);
  	for(int iteration_x = 0; iteration_x < resolution; iteration_x++)
  	{
  		double x = min_x + static_cast<double>(iteration_x) * step_x;
  		row.row(iteration_x) = Vector2d(x + step_x / 2.0, y + step_y / 2.0);
  	}
  	
	row_queries.push_back(row);
  }
  
  MatrixXd wnf(resolution, resolution);
  for(int row_idx = 0; row_idx < resolution; row_idx++)
  {
	wnf.row(row_idx) = windingNumberEndpoints(C, row_queries[row_idx], martens_tol);
  }
  
  //Eigen::IOFormat OctaveFmt(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]");
  //std::cout << wnf.format(OctaveFmt) << std::endl;
  
  return wnf;
}

double quadrature_wn(const std::vector<Eigen::Vector2d>& polyline, const Eigen::Vector2d& q) {
    double windingNumber = 0.0;
    int n = polyline.size();

    for (int i = 0; i < n - 1; ++i) {
        Eigen::Vector2d p1 = polyline[i];
        Eigen::Vector2d p2 = polyline[(i + 1)];

        Eigen::Vector2d v1 = p1 - q;
        Eigen::Vector2d v2 = p2 - q;

        double cross = v1.x() * v2.y() - v1.y() * v2.x();
        double dot = v1.x() * v2.x() + v1.y() * v2.y();

        double angle = std::atan2(cross, dot);
        windingNumber += angle;
    }


    windingNumber /= 2.0 * M_PI;

    return windingNumber;
}

std::vector<Eigen::Vector2d> discretize_bezier(const Eigen::Matrix<double, 4, 2>& controlPoints, int numSamples) {
    std::vector<Eigen::Vector2d> points;
    for (int i = 0; i <= numSamples; ++i) {
        double t = static_cast<double>(i) / numSamples;
	Vector2d p0 = controlPoints.row(0);
	Vector2d p1 = controlPoints.row(1);
	Vector2d p2 = controlPoints.row(2);
	Vector2d p3 = controlPoints.row(3);
        points.push_back(evaluateBezier(p0, p1, p2, p3, t).first);
    }

    return points;
}

std::vector<MatrixXd> random_bezier_curves(int n)
{
  std::random_device rd;
  //std::mt19937 gen(rd()); 
  std::mt19937 gen(0xced); 
  std::uniform_real_distribution<double> dis(0.0, 1.0);
  std::vector<MatrixXd> bezier_curves;

  for(int i = 0; i < n; i++)
  {

  MatrixXd curve(4, 2); 
    for(int j = 0; j < 4; j++)
    {
      curve(j, 0) = dis(gen); 
      curve(j, 1) = dis(gen); 
    }

    bezier_curves.push_back(curve);
  }

  return bezier_curves;
}

void dumpMatrixToFile(const Eigen::MatrixXd& matrix, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::out | std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }

    // Write the size of the matrix
    int rows = matrix.rows();
    int cols = matrix.cols();
    outFile.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    outFile.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    // Write the matrix data
    outFile.write(reinterpret_cast<const char*>(matrix.data()), rows * cols * sizeof(double));

    outFile.close();
}

Eigen::MatrixXd loadMatrixFromFile(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::in | std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file for reading!" << std::endl;
        return Eigen::MatrixXd();
    }

    // Read the size of the matrix
    int rows, cols;
    inFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    inFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    // Create a matrix of the appropriate size
    Eigen::MatrixXd matrix(rows, cols);

    // Read the matrix data
    inFile.read(reinterpret_cast<char*>(matrix.data()), rows * cols * sizeof(double));

    inFile.close();

    return matrix;
}

int main(int argc, char** argv)
{
  AXOM_UNUSED_VAR(argc);
  AXOM_UNUSED_VAR(argv);
  
  Eigen::IOFormat OctaveFmt(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]");

  int n = 1000;
  int width = 250;
  
  std::vector<MatrixXd> bezier_curves = random_bezier_curves(n);
std::chrono::steady_clock::time_point begin_spainhour = std::chrono::steady_clock::now();
 
  int martens_misclassified = 0;
  int spainhour_misclassified = 0;

  std::vector<Eigen::MatrixXd> results_spainhour;
  std::vector<Eigen::MatrixXd> results_martens;
  std::vector<Eigen::MatrixXd> results_quadrature;
  for(int curve_idx = 0; curve_idx < n; curve_idx++)
  {
	//std::cout << bezier_curves[curve_idx] << std::endl;
	Eigen::MatrixXd result = evaluate_wn_field_spainhour(bezier_curves[curve_idx], width);
	//std::cout << "wnf_s = " << result.format(OctaveFmt) << std::endl;
	results_spainhour.push_back(result);
  }
  std::chrono::steady_clock::time_point end_spainhour = std::chrono::steady_clock::now();

  //std::cout << std::endl << std::endl;
  std::chrono::steady_clock::time_point begin_martens = std::chrono::steady_clock::now();
  for(int curve_idx = 0; curve_idx < n; curve_idx++)
  {
	//std::cout << curves[curve_idx] << std::endl;
	Eigen::MatrixXd result = evaluate_wn_field_martens(bezier_curves[curve_idx], width);
  	//std::cout << "wnf_m = " << result.format(OctaveFmt) << std::endl;
        results_martens.push_back(result);
  }
  std::chrono::steady_clock::time_point end_martens = std::chrono::steady_clock::now();    
  std::chrono::steady_clock::time_point begin_quad = std::chrono::steady_clock::now();    
  
  
  for(int curve_idx = 0; curve_idx < n; curve_idx++)
  {
  	//std::cout << curve_idx << std::endl;
  	//Eigen::MatrixXd result = evaluate_wn_field_quadrature(bezier_curves[curve_idx], width);
	std::string filename = "gt_10k_250_250_" + std::to_string(curve_idx) + ".bin";
	//dumpMatrixToFile(result, filename);
	Eigen::MatrixXd result = loadMatrixFromFile(filename);
  	//std::cout << "wnf_q = " << result.format(OctaveFmt) << std::endl;
  	results_quadrature.push_back(result);
  }
  std::chrono::steady_clock::time_point end_quad = std::chrono::steady_clock::now();    

  std::vector<double> sse_martens;
  std::vector<double> sse_spainhour;
  int resolution = width;
  for(int curve_idx = 0; curve_idx < n; curve_idx++)
  {
  	Eigen::MatrixXd squared_diff_martens = (results_martens[curve_idx] - results_quadrature[curve_idx]).array().square();
  	sse_martens.push_back(squared_diff_martens.sum());
  	Eigen::MatrixXd squared_diff_spainhour = (results_spainhour[curve_idx] - results_quadrature[curve_idx]).array().square();
  	sse_spainhour.push_back(squared_diff_spainhour.sum());
  	double eps_diff = 1e-10;
  	
	for(int iteration_y = 0; iteration_y < resolution; iteration_y++)
	{
		for(int iteration_x = 0; iteration_x < resolution; iteration_x++)
		{
			
			if(squared_diff_martens(iteration_y, iteration_x) > eps_diff)
		 	{
				martens_misclassified++;
				//std::cout << "Martens misclassfied: (" << curve_idx << ", " << x << ", " << y << ")" << std::endl;
			}
			if(squared_diff_spainhour(iteration_y, iteration_x) > eps_diff)
			{
				spainhour_misclassified++;
				//std::cout << "Spainhour misclassfied: (" << curve_idx << ", " << iteration_y << ", " << iteration_x << ")" << std::endl;
			}
				
			
		}
	}

	//std::cout << curve_idx << ": " << martens_spainhour_ss_diff << std::endl;
  	//Eigen::MatrixXd martens_diff = results_martens[curve_idx] - results_quadrature[curve_idx];
  	//Eigen::MatrixXd spainhour_diff = results_spainhour[curve_idx] - results_quadrature[curve_idx];
  	//double martens_ss = martens_diff.array().square().sum();
  	//double spainhour_ss = spainhour_diff.array().square().sum();
  	//int num_query_points = width * width;

  	//std::cout << "martens_sum_sq: " << martens_ss << std::endl;
  	//std::cout << "spainhour_sum_sq: " << spainhour_ss << std::endl;
  }
 
  
  double total_sse_martens = accumulate(sse_martens.begin(), sse_martens.end(), 0.0);
  double total_sse_spainhour = accumulate(sse_spainhour.begin(), sse_spainhour.end(), 0.0);
  double mse_martens = total_sse_martens / (static_cast<double>(resolution) * static_cast<double>(resolution) * static_cast<double>(n));
  double mse_spainhour = total_sse_spainhour / (static_cast<double>(resolution) * static_cast<double>(resolution) * static_cast<double>(n));

  std::cout << "Edge tolerance used: " << edge_tol << std::endl;
  std::cout << "Spainhour Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_spainhour - begin_spainhour).count() << "[ms]" << std::endl;
  std::cout << "Martens Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_martens - begin_martens).count() << "[ms]" << std::endl;
  std::cout << "Quadrature Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_quad - begin_quad).count() << "[ms]" << std::endl;
  std::cout << "Martens wrong: " << martens_misclassified << std::endl;
  std::cout << "Spainhour wrong: " << spainhour_misclassified << std::endl;
  
  std::cout << "SSE Martens: " << total_sse_martens << std::endl;
  std::cout << "SSE Spainhour: " << total_sse_spainhour << std::endl;
  std::cout << "MSE Martens: " << mse_martens << std::endl;
  std::cout << "MSE Spainhour: " << mse_spainhour << std::endl;
  return 0;
}
