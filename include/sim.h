#ifndef SIM_HPP
#define SIM_HPP
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include "../include/cuda_kernels.h"

struct Particle {
    Eigen::Vector3f pos;       // Position vector
    Eigen::Vector3f vel;       // Velocity vector
    Eigen::Vector3f force;     // Force vector
    float density;             // Density value
    float mass;                // Mass value
    int particleType;          // Type of the particle (e.g., fluid, boundary, etc.)
};



enum class ParticleType {
    Fluid = 0,       // Fluid particles
    Boundary = 1,    // Boundary particles
};

class Simulator {
public:
    Simulator();
    ~Simulator();

    void init(std::string filename, Eigen::AlignedBox3d fluidBox, float particle_radius, float particle_distance, float fluid_density, int frame_rate, int simulation_rate, int iterations, float k);
    void run(int num_iterations);

private:
    float m_particle_radius = 0;
    float m_particle_diameter = 0;
    float m_fluid_sampling_distance = 0;
    float m_boundary_sampling_distance = 0;
    float m_smoothing_length = 0;
	float m_compact_support = 0;
	float m_fluid_density = 0;
    float m_dt = 0;
    int m_frame_rate = 0;
    int m_simulation_rate = 0;
    int m_iterations = 0;
	float m_k = 0; // Stiffness constant
	float m_particleDistance = 0; // Distance between particles
	float m_dist = 0; // Distance between particles
	float m_volume = 0; // Volume of the fluid box
	float m_particle_mass = 0; // Mass of a fluid particle
    std::string m_filename;
    Eigen::AlignedBox3d m_sceneBox; // Bounding box of the scene
    Eigen::AlignedBox3f m_sceneBoxf;
	Eigen::AlignedBox3d m_fluidBox; // Bounding box of the fluid only for init
	std::vector<Eigen::Vector3f> m_positions_fluid; // Vector of positions for fluid particles
	std::vector<Eigen::Vector3f> m_positions_boundary; // Vector of positions for boundary particles
	std::vector<Particle> m_particles; // Vector of particles
	std::vector<tmpParticle> m_tmp_particles; // Vector of temporary particles
	std::vector<float> m_boundary_particle_masses; // Vector of boundary particle masses


    void calc_helper_values();
	//takes in a particle distance and a bounding box and samples the fluid particles and creates a positions vector
    void sample_fluid();
    //takes in a file path to an .obj file and a particle distance to create a positions vector
    void sample_boundary();
	//takes the positions vector of the fluid and boundary particles and creates a vector of particles (and calculates the missing values)
    void create_particles();
	//creates buffers for the particles in cuda
    void create_buffers();
	//creates a hash map for the particles in cuda
    void create_hash_map();
    void step();
    void export_data();

	//helpers for neighborhood_search
    float m_search_grid_dist;
    float m_search_grid_offset;
	Eigen::Vector3f m_search_grid_min;
	Eigen::Vector3i m_search_grid_res;
    Eigen::AlignedBox3f m_search_grid_box;

};

#endif