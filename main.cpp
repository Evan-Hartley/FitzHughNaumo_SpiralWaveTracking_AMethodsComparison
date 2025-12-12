#include "PreCompiledHeader/FHNpch.h"

#include "Structures/StructureManagment.h"
#include "CUDAfiles/ComputeFHN.cuh"

int main() {

    // Adjustable simulation variables
    SimConstraints simulation;
    simulation.nx = 100;
    simulation.ny = 100;
    simulation.dx = 1;
    simulation.dt = 0.2;
    simulation.last_time = 3000;
    simulation.spiral = true;
    simulation.spiral_time = 200;
    simulation.tip_track_JDM = true;
    simulation.tip_track_volt = true;
    simulation.tip_track_phase = true;
    simulation.run_name = "Test";


    // Adjustable model parameters
    Parameters params;
    params.dx = simulation.dx;
    params.dt = simulation.dt;
    params.D_u = 1;
    params.eps = 0.01;
    params.a = 0.1;
    params.b = 0.5;
    params.last_step = static_cast<int>(simulation.last_time / params.dt);
    params.spiral_time = simulation.spiral_time;
    params.utip_pick = 0.40;              // User guess for phase singularity's u value
    params.vtip_pick = 0.07;              // User guess for phase singularity's v value


    // Check for parameter cohesion
    if (params.dt > (params.dx * params.dx) / (4.0 * params.D_u)) {
        std::cerr << "Warning: dt may be too large for stability!" << std::endl;
    }
       
    // Create and fill a data grid of intial u and v conditions
    const int width = simulation.nx, height = simulation.ny;
    GridData grid = GridData(width, height);
    grid = StartSim(grid, 0.0, 0.0);


    // Initialize u and v on host
    std::vector<double> h_u(width * height + 1);
    std::vector<double> h_v(width * height + 1);
    grid.copyToHost(h_u.data(), h_v.data());

    // Evolve the parameters using the FHN model
    evolveFitzHughNagumo(grid, params, simulation, params.last_step);

    // Return data to host
    grid.copyToHost(h_u.data(), h_v.data());


    //End
    return 0;
}