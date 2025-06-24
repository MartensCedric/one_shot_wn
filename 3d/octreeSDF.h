#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <array>
#include <cmath>
#include <utility>
#include <set>

/*
Author: Adrian Szatmari
Date: 2018-03-31
License: MIT
Dependencies: None, just drag and drop octree.h into your project. To visualize use libigl.

Description: this is a small header library to compute the Surface Distance Function (SDF). The SDF measures the penetration distance
in the inward normal direction, until the next surface hit. The following web pages helped for the dev of this code:

WARNING: to save time and memory, this SDF class references the matrices instead of a local copy, therefore do not change their values once
the tree structure is built.

TODO:
-For some reason the code crashes for very big meshes (>500k faces). I suspect it is a stack/heap issue.
-Need to adjust automatically the values that SDF outputs for visualization (using igl::jet).
-Need to multithread the build() routine and and query() routine. The first is non trivia, but the strategy is to manually split
the first 8 children into 8 separate threads, and then to unite the output by hand. The query() can be trivially parallelized.
*/

using namespace std;
double dot(const array<double, 3>& A, const array<double, 3>& B);
array<double, 3> cross(const array<double, 3>& A, const array<double, 3>& B);
bool RayTriangle_t(const array<double, 3>& source, const array<double, 3>& dir, const array<array<double, 3>, 3>& tri, double& t);
bool RayTriangle(const array<double, 3>& source, const array<double, 3>& dir, const array<array<double, 3>, 3>& tri, array<double, 3>& intersection);
bool RayBox(const array<double, 3>& source, const array<double, 3>& dir, double low_t, double high_t, const array<array<double, 3>, 2>& box);

bool TriangleBox(const array<array<double, 3>, 3>& triangle, const array<array<double, 3>, 2>& box);

class SDF {
private:
	//Data

	vector<array<double, 3>>& bary; //of the triangles in F

	//Node
	int leaves;
	array<array<double, 3>, 2> box; //defined by low and high
	vector<int> indices; //of triangles
	array<unique_ptr<SDF>, 8> children;

	//For testing
	bool is_leaf() const;
	bool test1() const; //called only by test()
	int test2() const; //called only by test()
	vector<int> test3() const; //called only by test()
	vector<int> test4() const; //called only by test()
public:
	vector<array<double, 3>>& V;
	vector<array<int, 3>>& F;
	SDF(vector<array<double, 3>>& V, vector<array<int, 3>>& F, vector<array<double, 3>>& bary)
		: V(V), F(F), bary(bary) {}; //passing V, F, bary, by reference
	SDF(const SDF& other) = delete; //non-copy move only semantic
	SDF& operator=(const SDF& rhs) = delete; //non-copy move only semantic

	void init(); //necessary
	void build();
	void test() const;
	vector<array<double, 3>> query(array<double, 3>& source, array<double, 3>& dir) const;
	vector<std::pair<double, int>> query_triangle_index(array<double, 3>& source, array<double, 3>& dir) const;
	vector<double> query(vector<array<double, 3>>& v_normals) const;
};
