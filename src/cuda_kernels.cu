#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "../include/cuda_kernels.h"
#include <cooperative_groups.h>

#define BLOCK_SIZE 256

struct SPHConstants {
    float h;       // Smoothing length
    float h2;      // h^2
    float h3;      // h^3
    float sigma;   // Normalization factor (1/pi for 3D)
    float factor;  // Pre-computed factor for the kernel
};

struct cuda_Particle {
    float3 pos;
    float3 pos_old;
    float3 vel;
    float3 force;
    float mass;
    float density;
    int particle_type;
};

cuda_Particle* d_particles;
int* d_grid_hash;
int* d_grid_hash_begin;
int* d_grid_hash_end;
int num_particles;
float3* d_position_correction;
float* d_lambdas;
const float M_PI = 3.14159265358979323846f;

__host__ cuda_Particle eigenToCudaParticle(const tmpParticle& eigenP) {
    cuda_Particle cudaP;
    cudaP.pos = make_float3(eigenP.pos_x, eigenP.pos_y, eigenP.pos_z);
	cudaP.pos_old = make_float3(eigenP.pos_x, eigenP.pos_y, eigenP.pos_z);
    cudaP.vel = make_float3(eigenP.vel_x, eigenP.vel_y, eigenP.vel_z);
    cudaP.force = make_float3(eigenP.force_x, eigenP.force_y, eigenP.force_z);
    cudaP.density = eigenP.density;
    cudaP.mass = eigenP.mass;
    cudaP.particle_type = eigenP.particleType;
    return cudaP;
}

__device__ float sqrdnorm(float3 v) {
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

__device__ float3 fmult(float s, float3 v) {
	return make_float3(v.x * s, v.y * s, v.z * s);
}

__constant__ SPHConstants d_sphConst;


__device__ float kernel(float r_sqrd, float h, float factor) {
    float q = sqrt(r_sqrd) / h;
    float s = 0.0f;
    if (0 <= q <= 1) {
        s += (2.0f / 3.0f - q * q + 0.5f * q * q * q);
    }
    else {
        s += (1.0f / 6.0f * (2.0f - q) * (2.0f - q) * (2.0f - q));
    }
    return factor * s;
}

__device__ float3 kernel_grad(float r_sqrd, float h, float factor, float3 r_dif) {
    float r = sqrt(r_sqrd);
    float q = r / h;
    float f = 0;
    if (0 <= q <= 1) {
        f += (- 2.0f * q + 1.5f * q * q);
    }
    else {
        f += (- 0.5f * (2.0f - q) * (2.0f - q));
    }
    return make_float3(factor * f * r_dif.x / (h * r), factor * f * r_dif.y / (h * r), factor * f * r_dif.z / (h * r));
}

__global__ void calculate_keys_pos(const float3* position, int* keys, int num_particles, int3 grid_res, float3 grid_min, float grid_dist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float x = position[idx].x - grid_min.x;
	float y = position[idx].y - grid_min.y;
	float z = position[idx].z - grid_min.z;
	int3 grid_pos = make_int3(x / grid_dist, y / grid_dist, z / grid_dist);
	keys[idx] = grid_pos.x * grid_res.y * grid_res.z + grid_pos.z * grid_res.y + grid_pos.z;
    //printf("calculate_keys\n");
}

__global__ void computeKeyOffsets(const int* __restrict__ keys,
    int* __restrict__ begin,
    int* __restrict__ end,
    const int num_keys,
    const int max_key) {
    // Shared memory for coalesced reading of keys
    __shared__ int shared_keys[256 + 1];  // +1 for handling boundary conditions

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;

    // Load keys into shared memory
    if (idx < num_keys) {
        shared_keys[tid] = keys[idx];
        // Load one extra element for boundary checking
        if (tid == blockDim.x - 1 && idx + 1 < num_keys) {
            shared_keys[tid + 1] = keys[idx + 1];
        }
    }
    __syncthreads();

    // Process only valid threads
    if (idx < num_keys) {
        const int key = shared_keys[tid];

        // First occurrence check
        bool is_first = (idx == 0) ||
            (idx > 0 && (tid == 0 ? keys[idx - 1] : shared_keys[tid - 1]) != key);

        // Last occurrence check
        bool is_last = (idx == num_keys - 1) ||
            (shared_keys[tid + 1] != key);

        // Use atomic operations only when necessary
        if (is_first) {
            begin[key] = idx;
        }
        if (is_last) {
            end[key] = idx;
        }
    }
}

__global__ void calculate_boundary_volume(
    const float3* position,
	float*  volume,
    const int* cellStart,
    const int* cellEnd,
    const int3 gridDim,
	const float3 grid_min,
    const float grid_dist,
    const SPHConstants sphConst,
    const int numParticles,
    float rho_0
) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= numParticles) return;

	const float3 pos = position[idx];
    float volumeContribution = 0.0f;
    
    float x = position[idx].x - grid_min.x;
    float y = position[idx].y - grid_min.y;
    float z = position[idx].z - grid_min.z;
    int3 cell = make_int3(x / grid_dist, y / grid_dist, z / grid_dist);
    //keys[idx] = grid_pos.x * grid_res.y * grid_res.z + grid_pos.z * grid_res.y + grid_pos.z;

	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			for (int k = -1; k <= 1; k++) {
                int3 neighborCell;
				neighborCell.x = cell.x + i;
				neighborCell.y = cell.y + j;
				neighborCell.z = cell.z + k;
				//calculate boundary volume
                if (neighborCell.x < 0 || neighborCell.x >= gridDim.x ||
                    neighborCell.y < 0 || neighborCell.y >= gridDim.y ||
                    neighborCell.z < 0 || neighborCell.z >= gridDim.z)
                    continue;
				int cellHash = neighborCell.x * gridDim.y * gridDim.z + neighborCell.z * gridDim.y + neighborCell.z;
				int start = cellStart[cellHash];
				if (start == -1) continue;
				int end = cellEnd[cellHash];
				if (end == -1) continue;
				for (int p = start; p <= end; p++) {
					float3 r_dif = make_float3(position[p].x - pos.x, position[p].y - pos.y, position[p].z - pos.z);
					float r_sqrd = r_dif.x * r_dif.x + r_dif.y * r_dif.y + r_dif.z * r_dif.z;
					if (r_sqrd < sphConst.h2)
					    volumeContribution += kernel(r_sqrd, sphConst.h, sphConst.factor);
				}
			}
		}
		volume[idx] = 1.0f/volumeContribution * rho_0;
	}
	//printf("calculate_weights\n");
}

__global__ void calculate_keys(const cuda_Particle* particles, int* keys, int num_particles, int3 grid_res, float3 grid_min, float grid_dist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = particles[idx].pos.x - grid_min.x;
    float y = particles[idx].pos.y - grid_min.y;
    float z = particles[idx].pos.z - grid_min.z;
    int3 grid_pos = make_int3(x / grid_dist, y / grid_dist, z / grid_dist);
    keys[idx] = grid_pos.x * grid_res.y * grid_res.z + grid_pos.z * grid_res.y + grid_pos.z;
    //printf("calculate_key_offsets\n");
}

__global__ void split_data(const char* msg) {
	printf("split_data\n");
}

__global__ void advect_particle(cuda_Particle* particles, float delta_t, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_particles) return;
	cuda_Particle p = particles[idx];
    if (p.particle_type == 1) return;
	p.vel.x += p.force.x / p.mass * delta_t;
	p.vel.y += p.force.y / p.mass * delta_t;
	p.vel.z += p.force.z / p.mass * delta_t;
	p.pos.x += p.vel.x * delta_t;
	p.pos.y += p.vel.y * delta_t;
    p.pos.z += p.vel.z * delta_t;
    //printf("advect_particle\n");
}

__global__ void lambda_particle(const cuda_Particle* particles,
    float* lambdas,
    const int* cellStart,
    const int* cellEnd,
    const int3 gridDim,
    const float3 grid_min,
    const float grid_dist,
    const SPHConstants sphConst,
    const int numParticles,
    float rho_0) 
{
    /**/const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= numParticles) return;

    const float3 pos = particles[idx].pos;
	cuda_Particle particle = particles[idx];
	if (particle.particle_type == 1) return;

    float x = particle.pos.x - grid_min.x;
    float y = particle.pos.y - grid_min.y;
    float z = particle.pos.z - grid_min.z;
    int3 cell = make_int3(x / grid_dist, y / grid_dist, z / grid_dist);

	float rho = 0.0f;
	float3 s_i_part1 = make_float3(0.0f, 0.0f, 0.0f);
	float s_i_part2 = 0.0f;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            for (int k = -1; k <= 1; k++) {
                int3 neighborCell;
                neighborCell.x = cell.x + i;
                neighborCell.y = cell.y + j;
                neighborCell.z = cell.z + k;
                //calculate boundary volume
                if (neighborCell.x < 0 || neighborCell.x >= gridDim.x ||
                    neighborCell.y < 0 || neighborCell.y >= gridDim.y ||
                    neighborCell.z < 0 || neighborCell.z >= gridDim.z)
                    continue;
                int cellHash = neighborCell.x * gridDim.y * gridDim.z + neighborCell.z * gridDim.y + neighborCell.z;
                int start = cellStart[cellHash];
                if (start == -1) continue;
                int end = cellEnd[cellHash];
                if (end == -1) continue;
                for (int p = start; p <= end; p++) {
                    float3 r_dif = make_float3(particles[p].pos.x - pos.x, particles[p].pos.y - pos.y, particles[p].pos.z - pos.z);
                    float r_sqrd = r_dif.x * r_dif.x + r_dif.y * r_dif.y + r_dif.z * r_dif.z;
                    if (r_sqrd < sphConst.h2) {
                        rho += particles[p].mass * kernel(r_sqrd, sphConst.h, sphConst.factor); //calculate density
                        //s_i_part1
                        s_i_part1.x += particles[p].mass / particles[p].density * kernel_grad(r_sqrd, sphConst.h, sphConst.factor, r_dif).x;
                        s_i_part1.y += particles[p].mass / particles[p].density * kernel_grad(r_sqrd, sphConst.h, sphConst.factor, r_dif).y;
                        s_i_part1.z += particles[p].mass / particles[p].density * kernel_grad(r_sqrd, sphConst.h, sphConst.factor, r_dif).z; //s_i_part1
						//you were here. need to add component wise, you idiot.
                        if (particles[p].particle_type > 0) continue;
						s_i_part2 += 1 / particles[p].mass * sqrdnorm(fmult(- 1.0f * particles[p].mass / rho_0, kernel_grad(r_sqrd, sphConst.h, sphConst.factor, r_dif)));
                    }
                }
            }
        }
    }
	float c = rho / rho_0 - 1.0f;
	float s_i = 1.0f/particle.mass * sqrdnorm(s_i_part1) + s_i_part2;
	float lambda = (c>0) ? - c / (s_i + 0.0001f) : 0.0f;
	lambdas[idx] = lambda;
    //printf("lambda_particle\n");
    //figure out, how to do it only for fluid particles
}

__global__ void pressure_particle(const cuda_Particle* particles,
        float* position_corrections,
        const int* cellStart,
        const int* cellEnd,
        const int3 gridDim,
        const float3 grid_min,
        const float grid_dist,
        const SPHConstants sphConst,
        const int numParticles,
        float rho_0)
        {
            const int tid = threadIdx.x;
            const int idx = blockIdx.x * blockDim.x + tid;
            if (idx >= numParticles) return;

            const float3 pos = particles[idx].pos;
            cuda_Particle particle = particles[idx];
            if (particle.particle_type == 1) return;

            float x = particle.pos.x - grid_min.x;
            float y = particle.pos.y - grid_min.y;
            float z = particle.pos.z - grid_min.z;
            int3 cell = make_int3(x / grid_dist, y / grid_dist, z / grid_dist);

            float rho = 0.0f;
            float3 s_i_part1 = make_float3(0.0f, 0.0f, 0.0f);
            float s_i_part2 = 0.0f;

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    for (int k = -1; k <= 1; k++) {

                    }
                }
            }
            


    printf("pressure_particle\n");
}

__global__ void position_update_particle(const char* msg) {
    printf("position_update_particle\n");
}

__global__ void velocity_update_partcile(const char* msg) {
    printf("velocity_update_partcile\n");
}

void sample_boundary_cuda(
    std::vector<float>& particle_pos, 
    std::vector<float>& particle_mass, 
    int search_grid_x, 
    int search_grid_y, 
    int search_grid_z, 
    float search_grid_min_x, 
    float search_grid_min_y, 
    float search_grid_min_z, 
    float search_grid_dist, 
    float smoothing_length, 
    float rho_0
) {
	//create host_variables and device_variables
	std::vector<float3> h_particle_pos;    
	int* h_grid_hash = new int[particle_pos.size()];
	float* h_particle_mass = new float[particle_pos.size()];
	float* d_particle_mass;
    float3* d_particle_pos;
    int* d_grid_hash;
    int* d_grid_hash_begin;
	int* d_grid_hash_end;

	

    void* d_tempStorage = nullptr;
    size_t tempStorageBytes = 0;

	int max_cells = search_grid_x * search_grid_y * search_grid_z;
    std::vector<int>  h_grid_hash_begin(max_cells + 1, -1);
    std::vector<int>  h_grid_hash_end(max_cells + 1, -1);
    std::cout << particle_pos.size() << std::endl;
    std::cout << search_grid_x << ", " << search_grid_y << ", " << search_grid_z << std::endl;
    std::cout << search_grid_min_x << ", " << search_grid_min_y << ", " << search_grid_min_z << std::endl;

    for (int i = 0; i < particle_pos.size(); i+=3) {
        h_particle_pos.push_back(make_float3(particle_pos[i], particle_pos[i + 1], particle_pos[i + 2]));
	}
    //allocate memory
	cudaMalloc(&d_particle_mass, h_particle_pos.size() * sizeof(float));
	cudaMalloc(&d_particle_pos, h_particle_pos.size() * sizeof(float3));
	cudaMalloc(&d_grid_hash, h_particle_pos.size() * sizeof(int));
	cudaMalloc(&d_grid_hash_begin, (max_cells+1) * sizeof(int));
	cudaMalloc(&d_grid_hash_end, (max_cells+1) * sizeof(int));
    //copy data to device
	cudaMemcpy(d_particle_pos, h_particle_pos.data(), h_particle_pos.size() * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_hash_begin, h_grid_hash_begin.data(), (max_cells + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_grid_hash_end, h_grid_hash_end.data(), (max_cells + 1) * sizeof(int), cudaMemcpyHostToDevice);
    //calculate grid indices
	int grid_size = (h_particle_pos.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
	calculate_keys_pos<<<grid_size, BLOCK_SIZE >>> (d_particle_pos, d_grid_hash, h_particle_pos.size(), make_int3(search_grid_x, search_grid_y, search_grid_z), make_float3(search_grid_min_x, search_grid_min_y, search_grid_min_z), search_grid_dist);
    cudaDeviceSynchronize();
    //run search
    cub::DeviceRadixSort::SortPairs(d_tempStorage, tempStorageBytes, d_grid_hash, d_grid_hash, d_particle_pos, d_particle_pos, h_particle_pos.size());
    // Allocate temporary storage
    cudaMalloc(&d_tempStorage, tempStorageBytes);
    // Perform the sorting operation
    cub::DeviceRadixSort::SortPairs(d_tempStorage, tempStorageBytes, d_grid_hash, d_grid_hash, d_particle_pos, d_particle_pos, h_particle_pos.size());
    //create luts
    computeKeyOffsets<<<grid_size, BLOCK_SIZE>>> (d_grid_hash, d_grid_hash_begin, d_grid_hash_end, particle_pos.size(), max_cells);
    //calculate weights
    SPHConstants sphConst;
    sphConst.h = smoothing_length;
    sphConst.h2 = smoothing_length * smoothing_length;
    sphConst.h3 = sphConst.h2 * smoothing_length;
    sphConst.sigma = 1.0f / M_PI;  // 3D normalization
    sphConst.factor = sphConst.sigma / sphConst.h3;

    cudaMemcpyToSymbol(d_sphConst, &sphConst, sizeof(SPHConstants));

    calculate_boundary_volume << <grid_size, BLOCK_SIZE >> > (
        d_particle_pos,
        d_particle_mass,
        d_grid_hash_begin,
        d_grid_hash_end,
        make_int3(search_grid_x, search_grid_y, search_grid_z),
        make_float3(search_grid_min_x, search_grid_min_y, search_grid_min_z),
        search_grid_dist,
        sphConst,
        h_particle_pos.size(),
        rho_0
        );

	//copy data back to host
	//cudaMemcpy(h_grid_hash, d_grid_hash, h_particle_pos.size() * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_grid_hash_begin.data(), d_grid_hash_begin, (max_cells+1) * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_grid_hash_end.data(), d_grid_hash_end, (max_cells+1) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_particle_mass, d_particle_mass, h_particle_pos.size() * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_particle_pos.data(), d_particle_pos, h_particle_pos.size() * sizeof(float3), cudaMemcpyDeviceToHost);

	particle_mass = std::vector<float>(h_particle_mass, h_particle_mass + h_particle_pos.size());
	particle_pos = std::vector<float>(h_particle_pos.size() * 3);
	for (int i = 0; i < h_particle_pos.size(); i++) {
		particle_pos[i * 3] = h_particle_pos[i].x;
		particle_pos[i * 3 + 1] = h_particle_pos[i].y;
		particle_pos[i * 3 + 2] = h_particle_pos[i].z;
	}

    /*
	for (int i = 0; i < 1000; i++) {
		auto tmp = h_particle_pos[i];
		std::cout << i << ": " << tmp.x << ", " << tmp.y << ", " << tmp.z << ", " << h_grid_hash[i] << ", " << h_grid_hash_begin[i] << ", " << h_grid_hash_end[i] << ", " << h_particle_mass[i] << std::endl;
	}*/
	//free memory
	cudaFree(d_particle_mass);
	cudaFree(d_particle_pos);
	cudaFree(d_grid_hash);
	cudaFree(d_grid_hash_begin);
	cudaFree(d_grid_hash_end);
	cudaFree(d_tempStorage);
	delete[] h_grid_hash;
	delete[] h_particle_mass;
}


void create_buffers_cuda(std::vector<tmpParticle>& particles, int search_grid_x, int search_grid_y, int search_grid_z) {
    //SPH params already initialized
	num_particles = particles.size();
    int max_cells = search_grid_x * search_grid_y * search_grid_z;

    //create host data
	std::vector<cuda_Particle> h_particles;
    std::vector<int>  h_grid_hash_begin(max_cells + 1, -1);
    std::vector<int>  h_grid_hash_end(max_cells + 1, -1);
    h_particles.reserve(particles.size());
    for (const auto& p : particles) {
        h_particles.push_back(eigenToCudaParticle(p));
    }

	//create device data


    //allocate mem
	cudaMalloc(&d_particles, num_particles * sizeof(cuda_Particle));
    cudaMalloc(&d_grid_hash, num_particles * sizeof(int));
    cudaMalloc(&d_grid_hash_begin, (max_cells + 1) * sizeof(int));
    cudaMalloc(&d_grid_hash_end, (max_cells + 1) * sizeof(int));
	cudaMalloc(&d_lambdas, num_particles * sizeof(float));
	cudaMolloc(&d_position_correction, num_particles * sizeof(float3));
	//copy data to device

	cudaMemcpy(d_particles, h_particles.data(), num_particles * sizeof(cuda_Particle), cudaMemcpyHostToDevice);
	cudaMemcpy(d_grid_hash_begin, h_grid_hash_begin.data(), (max_cells + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_grid_hash_end, h_grid_hash_end.data(), (max_cells + 1) * sizeof(int), cudaMemcpyHostToDevice);

	std::cout << "create_buffers_cuda" << std::endl;
    cudaDeviceSynchronize();
}

void create_hash_map_cuda(int search_grid_x, int search_grid_y, int search_grid_z, float search_grid_min_x, float search_grid_min_y, float search_grid_min_z, float search_grid_dist) {
	int grid_size = (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    calculate_keys<<<grid_size, BLOCK_SIZE>>> (d_particles, d_grid_hash, num_particles, make_int3(search_grid_x, search_grid_y, search_grid_z), make_float3(search_grid_min_x, search_grid_min_y, search_grid_min_z), search_grid_dist);
	cudaDeviceSynchronize();
    
    void* d_tempStorage = nullptr;
    size_t tempStorageBytes = 0;

    cub::DeviceRadixSort::SortPairs(d_tempStorage, tempStorageBytes, d_grid_hash, d_grid_hash, d_particles, d_particles, num_particles);
    // Allocate temporary storage
    cudaMalloc(&d_tempStorage, tempStorageBytes);
    // Perform the sorting operation
    cub::DeviceRadixSort::SortPairs(d_tempStorage, tempStorageBytes, d_grid_hash, d_grid_hash, d_particles, d_particles, num_particles);
	cudaFree(d_tempStorage);
	computeKeyOffsets <<<grid_size, BLOCK_SIZE>>> (d_grid_hash, d_grid_hash_begin, d_grid_hash_end, num_particles, search_grid_x * search_grid_y * search_grid_z);
    //split_data << <1, 1 >> > ("create_buffers_cuda called");
    cudaDeviceSynchronize();
}

void advect_cuda(float delta_t, int num_particles) {
    int grid_size = (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    advect_particle<<<grid_size, BLOCK_SIZE >>>(d_particles, delta_t, num_particles);
    cudaDeviceSynchronize();
}

void calc_lambda_cuda(//fix inputs
    int search_grid_x, int search_grid_y, int search_grid_z, float search_grid_min_x, float search_grid_min_y, float search_grid_min_z, float search_grid_dist, float rho_0, float smoothing_length) {
	//std::cout << "calc_lambda_cuda" << std::endl;
    SPHConstants sphConst;
    sphConst.h = smoothing_length;
    sphConst.h2 = smoothing_length * smoothing_length;
    sphConst.h3 = sphConst.h2 * smoothing_length;
    sphConst.sigma = 1.0f / M_PI;  // 3D normalization
    sphConst.factor = sphConst.sigma / sphConst.h3;
    int grid_size = (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    lambda_particle<<<grid_size, BLOCK_SIZE >>>(d_particles, d_lambdas, d_grid_hash_begin, d_grid_hash_end, make_int3(search_grid_x, search_grid_y, search_grid_z), make_float3(search_grid_min_x, search_grid_min_y, search_grid_min_z), search_grid_dist, sphConst, num_particles,  rho_0);
    cudaDeviceSynchronize();
}

void calc_pressures_cuda() {
    pressure_particle<<<1, 1>>>("calc_pressures_cuda called");
    cudaDeviceSynchronize();
}

void update_positions_cuda() {
    position_update_particle<<<1, 1>>>("update_positions_cuda called");
    cudaDeviceSynchronize();
}

void update_velocity_cuda() {
    velocity_update_partcile<<<1, 1>>>("update_velocity_cuda called");
    cudaDeviceSynchronize();
}

void export_data_cuda() {
	std::cout << "export_data_cuda" << std::endl;
    cudaDeviceSynchronize();
}

void free_buffers_cuda() {
	std::cout << "free_buffers_cuda" << std::endl;
	cudaDeviceSynchronize();
}