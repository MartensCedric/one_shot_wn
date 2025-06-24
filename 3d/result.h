#pragma once

#include <chrono>
#include <iostream>

struct curvenet_timing_info {
	int num_patches;
	int total_patch_samples; // the total number of curve network points
	std::chrono::nanoseconds boundary_processing_result_time = std::chrono::nanoseconds::zero();
	std::chrono::nanoseconds ray_shooting_result_time = std::chrono::nanoseconds::zero();
	int total_rays; // total rays initially planned to be shot
	int ray_length; // length of ray
};

struct winding_number_results {
	curvenet_timing_info timing;
	std::vector<double> gwn;
};
