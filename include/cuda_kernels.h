#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H
#pragma once

#include <vector>
//#include <Eigen/Dense>

struct tmpParticle {
	float pos_x;
	float pos_y;
	float pos_z;
	float vel_x;
	float vel_y;
	float vel_z;
	float force_x;
	float force_y;
	float force_z;
	float density;             // Density value
	float mass;                // Mass value
	int particleType;          // Type of the particle (e.g., fluid, boundary, etc.)
};




void sample_boundary_cuda(std::vector<float>& particle_pos, std::vector<float>& particle_mass, int search_grid_x, int search_grid_y, int search_grid_z, float search_grid_min_x, float search_grid_min_y, float search_grid_min_z, float search_grid_dist, float smoothing_length, float rho_0);
void create_buffers_cuda(std::vector<tmpParticle>& particles, int search_grid_x, int search_grid_y, int search_grid_z);
void create_hash_map_cuda(int search_grid_x, int search_grid_y, int search_grid_z, float search_grid_min_x, float search_grid_min_y, float search_grid_min_z, float search_grid_dist);
void advect_cuda(float delta_t, int num_particles);
void calc_lambda_cuda(int search_grid_x, int search_grid_y, int search_grid_z, float search_grid_min_x, float search_grid_min_y, float search_grid_min_z, float search_grid_dist, float rho_0, float smoothing_length);
void calc_pressures_cuda();
void update_positions_cuda();
void update_velocity_cuda();
void export_data_cuda();
void free_buffers_cuda();

#endif