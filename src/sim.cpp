#include "../include/sim.h"
#include <iostream>
#include "../include/io.h"
#include "../include/sampling.h"
#include <Eigen/Dense>

Simulator::Simulator() {}

Simulator::~Simulator() { free_buffers_cuda(); }

void Simulator::init(std::string filename, Eigen::AlignedBox3d fluidBox, float particle_radius, float particle_distance, float fluid_density, int frame_rate, int simulation_rate, int iterations, float k) {
    std::cout << "Simulator::init() called" << std::endl;
	m_particle_radius = particle_radius;
	m_particle_diameter = 2.0f * m_particle_radius;
    m_fluid_sampling_distance = m_particle_diameter;
	m_boundary_sampling_distance = 0.8 * m_particle_diameter;
	m_smoothing_length = 1.2f * m_particle_radius;
	m_compact_support = 2.0f * m_smoothing_length;
	m_fluid_density = fluid_density;
	m_dt = 1.0f / simulation_rate;
	m_frame_rate = frame_rate;
	m_simulation_rate = simulation_rate;
	m_iterations = iterations;
	m_k = k;
	m_particleDistance = particle_distance;
    m_fluidBox = fluidBox;
	m_filename = filename;
    calc_helper_values();
    sample_fluid();
    sample_boundary();
    create_buffers();
    create_particles();
    create_hash_map();
}

void Simulator::run(int num_iterations) {
    std::cout << "Simulator::run() called" << std::endl;
    int sim_iterations = 100; // Example value
    int frame_rate = 10;      // Example value

    for (int i = 0; i < num_iterations; i++) {
        if (i % frame_rate == 0) {
            export_data();
        }
        step();
    }
}

void Simulator::calc_helper_values() {



    std::cout << "Simulator::calc_helper_values() called" << std::endl;
}

void Simulator::sample_fluid() {
    std::vector<Eigen::Vector3d> tmp_positions;
    learnSPH::sampling::fluid_box(tmp_positions, m_fluidBox.min(), m_fluidBox.max(), m_fluid_sampling_distance);
	std::transform(tmp_positions.begin(), tmp_positions.end(), std::back_inserter(m_positions_fluid), [](const Eigen::Vector3d& pos) { return Eigen::Vector3f(pos.cast<float>()); });
    m_volume = m_fluidBox.sizes().prod();
    m_particle_mass = m_fluid_density * m_volume / m_positions_fluid.size();
    std::cout << "Simulator::sample_fluid() called" << std::endl;
}

void Simulator::sample_boundary() {
    const std::vector<learnSPH::TriMesh> meshes = learnSPH::read_tri_meshes_from_obj(m_filename); //needs to be parametrized into constructor
    const learnSPH::TriMesh& box = meshes[0];
    std::cout << "Calculating boundary box" << std::endl;
    std::vector<Eigen::Vector3d> tmp_boundary_pos;
    learnSPH::sampling::triangle_mesh(tmp_boundary_pos, box.vertices, box.triangles, m_boundary_sampling_distance);
    m_sceneBox = { Eigen::Vector3d(0,0,0), Eigen::Vector3d(0,0,0) };
    for (const auto& v : box.vertices) {
        m_sceneBox.extend(v);
    }
    m_sceneBoxf = { m_sceneBox.min().cast<float>(), m_sceneBox.max().cast<float>() };
    std::transform(tmp_boundary_pos.begin(), tmp_boundary_pos.end(), std::back_inserter(m_positions_boundary), [](const Eigen::Vector3d& pos) { return Eigen::Vector3f(pos.cast<float>()); });
    m_search_grid_dist = 1.5f * m_smoothing_length;
    m_search_grid_offset = 2.0f * m_smoothing_length;
    m_search_grid_box = Eigen::AlignedBox3f(m_sceneBoxf.min(), m_sceneBoxf.max());
    m_search_grid_box.max() += Eigen::Vector3f::Constant(m_search_grid_offset);
    m_search_grid_box.min() -= Eigen::Vector3f::Constant(m_search_grid_offset);
    m_search_grid_res = ((m_search_grid_box.max() - m_search_grid_box.min()) / m_search_grid_dist).cast<int>();
    m_search_grid_min = m_search_grid_box.min();
	std::vector<float> flatVec;
    
    flatVec.reserve(m_positions_boundary.size() * 3); // Reserve space for efficiency
    for (const auto& v : m_positions_boundary) {
        flatVec.push_back(v.x());
        flatVec.push_back(v.y());
        flatVec.push_back(v.z());
    }
    sample_boundary_cuda(flatVec, m_boundary_particle_masses, 
        m_search_grid_res[0], m_search_grid_res[1], m_search_grid_res[2],
       m_search_grid_min[0], m_search_grid_min[1], m_search_grid_min[2], 
        m_search_grid_dist, m_smoothing_length, m_fluid_density);
	for (int i = 0; i < flatVec.size(); i += 3) {
		m_positions_boundary.push_back(Eigen::Vector3f(flatVec[i], flatVec[i + 1], flatVec[i + 2]));
	}
    std::cout << "Simulator::sample_boundary() called" << std::endl;
}

void Simulator::create_particles() {
	m_particles.reserve(m_positions_fluid.size() + m_positions_boundary.size());
	for (const auto& pos : m_positions_fluid) {
		Particle p;
		p.pos = pos;
		p.vel = Eigen::Vector3f::Zero();
		p.force = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		p.density = 0.0f;
		p.mass = m_particle_mass;
		p.particleType = static_cast<int>(ParticleType::Fluid);
		m_particles.push_back(p);
	}
	for (const auto& pos : m_positions_boundary) {
		Particle p;
		p.pos = pos;
		p.vel = Eigen::Vector3f::Zero();
		p.force = Eigen::Vector3f::Zero();
		p.density = 0.0f;
		p.mass = 0.0f;
		p.particleType = static_cast<int>(ParticleType::Boundary);
		m_particles.push_back(p);
	}
	m_tmp_particles.reserve(m_particles.size());
    for (const auto& pos : m_positions_fluid) {
		tmpParticle p;
		p.pos_x = pos.x();
		p.pos_y = pos.y();
		p.pos_z = pos.z();
		p.vel_x = 0.0f;
		p.vel_y = 0.0f;
		p.vel_z = 0.0f;
		p.force_x = 0.0f;
		p.force_y = 0.0f;
		p.force_z = 0.0f;
		p.density = 0.0f;
		p.mass = m_particle_mass;
		p.particleType = static_cast<int>(ParticleType::Fluid);
		m_tmp_particles.push_back(p);
    }
	for (const auto& pos : m_positions_boundary) {
		tmpParticle p;
		p.pos_x = pos.x();
		p.pos_y = pos.y();
		p.pos_z = pos.z();
		p.vel_x = 0.0f;
		p.vel_y = 0.0f;
		p.vel_z = 0.0f;
		p.force_x = 0.0f;
		p.force_y = 0.0f;
		p.force_z = 0.0f;
		p.density = 0.0f;
		p.mass = 0.0f;
		p.particleType = static_cast<int>(ParticleType::Boundary);
		m_tmp_particles.push_back(p);
	}
	create_buffers(); // you stopped here
    /*
    call cuda function to allocate buffers for key and particle arrays as well as allocate memory for the split data
	Do a special run to initialize boundary particles
    

    */
    std::cout << "Simulator::create_particles() called" << std::endl;
}

void Simulator::create_buffers() {
    std::cout << "Simulator::create_buffers() called" << std::endl;
    create_buffers_cuda(m_tmp_particles, m_search_grid_res[0], m_search_grid_res[1], m_search_grid_res[2]);
}

void Simulator::create_hash_map() {
    std::cout << "Simulator::create_hash_map() called" << std::endl;
    create_hash_map_cuda(m_search_grid_res[0], m_search_grid_res[1], m_search_grid_res[2],
        m_search_grid_min[0], m_search_grid_min[1], m_search_grid_min[2],
        m_search_grid_dist);
}

void Simulator::step() {
    std::cout << "Simulator::step() called" << std::endl;
    advect_cuda(m_dt, m_positions_fluid.size());
    create_hash_map_cuda(m_search_grid_res[0], m_search_grid_res[1], m_search_grid_res[2],
        m_search_grid_min[0], m_search_grid_min[1], m_search_grid_min[2],
        m_search_grid_dist);

    for (int i = 0; i < 3; i++) { // Example iteration count
        calc_lambda_cuda(m_search_grid_res[0], m_search_grid_res[1], m_search_grid_res[2],
            m_search_grid_min[0], m_search_grid_min[1], m_search_grid_min[2],
            m_search_grid_dist, m_fluid_density, m_smoothing_length);
        calc_pressures_cuda();
        update_positions_cuda();
    }

    update_velocity_cuda();
}

void Simulator::export_data() {
    std::cout << "Simulator::export_data() called" << std::endl;
    export_data_cuda();
}