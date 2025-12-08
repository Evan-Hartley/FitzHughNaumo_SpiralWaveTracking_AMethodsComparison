#include "FHNpch.h"

#include "StructureManagment.h"
#include "./ComputeFHN.cuh"

int main() {

    SimConstraints simulation;
    simulation.dx = 1;
    simulation.dt = 0.2;
    simulation.last_time = 3000;
    simulation.spiral = true;
    simulation.spiral_time = 200;
    simulation.tip_track_JDM = true;
    simulation.tip_track_volt = true;
    simulation.tip_track_phase = true;
    simulation.run_name = "TrackingTest";

    Parameters params;
    params.dx = simulation.dx;
    params.dt = simulation.dt;
    params.D_u = 1;
    params.eps = 0.01;
    params.a = 0.1;
    params.b = 0.5;
    params.last_step = static_cast<int>(simulation.last_time / params.dt);
    params.spiral_time = simulation.spiral_time;
    params.utip_pick = 0.3980589461367867;
    params.vtip_pick = 0.0714227261894926;


    if (params.dt > (params.dx * params.dx) / (4.0 * params.D_u)) {
        std::cerr << "Warning: dt may be too large for stability!" << std::endl;
    }

    const int width = 100, height = 100;
    GridData grid = GridData(width, height);
    grid = StartSim(grid, 0.0, 0.0);


    // Initialize u and v on host
    std::vector<double> h_u(width * height + 1);
    std::vector<double> h_v(width * height + 1);
    grid.copyToHost(h_u.data(), h_v.data());
    //int ending_step = static_cast<int>(params.last_step / params.dt);
    evolveFitzHughNagumo(grid, params, simulation, params.last_step);
    grid.copyToHost(h_u.data(), h_v.data());

    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
