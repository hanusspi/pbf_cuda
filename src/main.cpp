#include <iostream>
#include "../include/sim.h"
#include <Eigen/Dense>
#include <vector>
#include <string>

int main() {
	std::string filename = "../res/box.obj";
    Eigen::Vector3d minPoint(0.0, 0.0, 0.0);
    Eigen::Vector3d maxPoint(1.0, 1.0, 1.0);
    Eigen::AlignedBox3d fluidBox(minPoint, maxPoint);
    float particle_radius = 0.01f;
	float particle_distance = 0.01f;
	float fluid_density = 1000.0f;
    int frame_rate = 60.0f;
	int simulation_rate = 1000;
    int iterations = 10;
	float k = 0.1f;


    Simulator sim;
    sim.init(filename, fluidBox, particle_radius, particle_distance, fluid_density, frame_rate, simulation_rate, iterations, k);
    sim.run(100);

    return 0;
}