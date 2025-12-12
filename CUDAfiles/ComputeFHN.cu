#include "../PreCompiledHeader/FHNpch.h"
#include "../Structures/StructureManagment.h"
#include "./ComputeFHN.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// Allocate space to grid
GridData::GridData(int w, int h) : width(w), height(h) {
    size_t size = w * h * sizeof(double);
    cudaMalloc(&u_vals, size);
    cudaMalloc(&v_vals, size);
}

// Deconstruct grid and free memory
GridData::~GridData() {
    cudaFree(u_vals);
    cudaFree(v_vals);
}

// Construct grid data
GridData::GridData(GridData&& other) noexcept
    : width(other.width), height(other.height),
    u_vals(other.u_vals), v_vals(other.v_vals) {
    other.u_vals = nullptr;
    other.v_vals = nullptr;
}

// Allow the = operator to be used with GridData
GridData& GridData::operator=(GridData&& other) noexcept {
    if (this != &other) {
        cudaFree(u_vals);
        cudaFree(v_vals);
        width = other.width;
        height = other.height;
        u_vals = other.u_vals;
        v_vals = other.v_vals;
        other.u_vals = nullptr;
        other.v_vals = nullptr;
    }
    return *this;
}

// Given i and j coordinates, fid the GridData equivalent index
int GridData::index(int i, int j) const {
    return j * width + i;
}

// Copy grid data from the device using CUDA
void GridData::copyFromHost(double* h_u, double* h_v) {
    cudaMemcpy(u_vals, h_u, width * height * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(v_vals, h_v, width * height * sizeof(double), cudaMemcpyHostToDevice);
}

// Copy grid data to the device using CUDA
void GridData::copyToHost(double* h_u, double* h_v) {
    cudaMemcpy(h_u, u_vals, width * height * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, v_vals, width * height * sizeof(double), cudaMemcpyDeviceToHost);
}

// Initialize the data within GridData
GridData StartSim(GridData& grid, const double u_start, const double v_start) {
    int width = grid.width;
    int height = grid.height;
    int totalSize = width * height;

    // Allocate host memory
    std::vector<double> h_u(totalSize, u_start);
    std::vector<double> h_v(totalSize, v_start);

    // Copy to device
    grid.copyFromHost(h_u.data(), h_v.data());

    // Assumers user that the simulation has been initialized
    printf("Initialized.\n");

    // Returns updated grid
    return std::move(grid);
}

// Equation for linear interpolation -- currently unused
double LinearInterp(double idx1, double idx2, double val1, double val2, double val_tar) {
    double interp_idx = (((val2 - val_tar) / (val2 - val1)) * idx1) + (((val_tar - val1) / (val2 - val1)) * idx2);

    return interp_idx;
}

// Equation for inverse-bilinear interpolation -- currently unused
std::pair<double, double> BilinearInterp(int idx_x1, int idx_x2, int idx_y1, int idx_y2, std::vector<double> val_vec, double val_tar, int width, int iter) {
    // Translate from grid indices provided to linear array index
    int idx11 = idx_y1 * width + idx_x1;
    int idx12 = idx_y2 * width + idx_x1;
    int idx21 = idx_y1 * width + idx_x2;
    int idx22 = idx_y2 * width + idx_x2;

    // User warning
    if ((idx11 > width * width) || (idx12 > width * width) || (idx21 > width * width) || (idx22 > width * width)) {
        std::cerr << "Index too large!" << std::endl;
        printf("idx11: %d, idx12: %d, idx21: %d, idx22: %d\n", idx_x1, idx_x2, idx_y1, idx_y2);
    }

    // User warning
    if ((idx11 < 0) || (idx12 < 0) || (idx21 < 0) || (idx22 < 0)) {
        std::cerr << "Index too small!" << std::endl;
        printf("idx11: %d, idx12: %d, idx21: %d, idx22: %d\n", idx_x1, idx_x2, idx_y1, idx_y2);
    }

    // Call value at each location to be interpolated around
    double val11 = val_vec[idx11];
    double val12 = val_vec[idx12];
    double val21 = val_vec[idx21];
    double val22 = val_vec[idx22];

    // Initialize variables in function scope, default returning the midpoint value for bilinear interpolation
    double x_scale = 0.5;
    double y_scale = 0.5;
    double x_width = idx_x2 - idx_x1;
    double y_width = idx_y2 - idx_y1;
    double x = idx_x1 + x_width * x_scale;
    double y = idx_y1 + y_width * y_scale;

    // Initialize the midpoint as the location guess for inverse bilinear interpolation answer
    int idx_guess = x * width + y;
    double val_guess = val_vec[idx_guess];


    //Predict quadrent of second guess and adjust
    if (val_guess - val11 > 0 && val_guess - val12 > 0) {
        x_scale = 0.25;
    }
    else if (val_guess - val11 < 0 && val_guess - val12 < 0) {
        x_scale = 0.75;
    }

    if (val_guess - val11 > 0 && val_guess - val21 > 0) {
        y_scale = 0.25;
    }
    else if (val_guess - val11 < 0 && val_guess - val12 < 0) {
        y_scale = 0.75;
    }

    // Set second guess
    x = idx_x1 + x_width * x_scale;
    y = idx_y1 + y_width * y_scale;

    // Iterate through number of iterations specified using Newton's method to continue to get a better approximation of inverse-bilinear interpolation values
    for (int k = 0; k < iter; ++k) {
        double estimate_val = (val11 * (idx_x2 - x) * (idx_y2 - y) + val21 * (x - idx_x1) * (idx_y2 - y) + val22 * (x - idx_x1) * (y - idx_y1) + val12 * (idx_x2 - x) * (y - idx_y1)) / (x_width * y_width);
        double residual = estimate_val - val_tar;

        double jacobian_inverse_xscale = (- 1.0 * (val11 * (idx_y2 - y)) + val21 * (idx_y2 - y) + val22 * (y - idx_y1) - val12 * (y - idx_y1)) / (x_width * y_width);
        double jacobian_inverse_yscale = (- 1.0 * val11 * (idx_x2 - x) - val21 * (x - idx_x1) + val22 * (x - idx_x1) + val21 * (idx_x2 - x)) / (x_width * y_width);

        // Apply cap and floor limits to the returned values (safeguard)
        x = x - jacobian_inverse_xscale / residual;
        if (x > idx_x2) {
            x = idx_x2;
        }
        else if (x < idx_x1) {
            x = idx_x1;
        }

        y = y - jacobian_inverse_yscale / residual;
        if (y > idx_y2) {
            y = idx_y2;
        }
        else if (y < idx_y1) {
            y = idx_y1;
        }
    }

    // Return x and y approximate location of tip
    return std::make_pair(x, y);

}

// Keep angle between 0 and 2pi, and alligned with polar axis lines
double AngleWrap(double angle) {

    double diff = std::fmod(angle + M_PI, 2 * M_PI);
    if (diff < 0) {
        diff = diff + 2.0 * M_PI;
    }
    return diff - M_PI;

}

// Using the phase of the spiral wave, detect the tip of the spiral wave (phase method)
std::pair<int, int> PhaseTipDetection(std::vector<double>phase_vals, int width, int height) {
    int total = width * height;
    std::vector<double> wind_vals(total, 1.0);

    // To detect the phase around a singularity (as techinically its phase is undefined) we must calculate the "wind" of the points around our target point
    for (int idx = 0; idx < total; ++idx) {
        int i = idx % width;
        int j = idx / width;

        int right_i = (i < width - 1) ? i + 1 : i;
        int up_j = (j > 0) ? j - 1 : j;


        int right_up_idx = up_j * width + right_i;
        int right_idx = j * width + right_i;
        int up_idx = up_j * width + i;

        double phase_right = AngleWrap(phase_vals[right_idx] - phase_vals[idx]);
        double phase_up = AngleWrap(phase_vals[right_up_idx] - phase_vals[right_idx]);
        double phase_left = AngleWrap(phase_vals[up_idx] - phase_vals[right_up_idx]);
        double phase_down = AngleWrap(phase_vals[idx] - phase_vals[up_idx]);

        wind_vals[idx] = phase_right + phase_up + phase_left + phase_down;
    }

    // Find the wind value that is closest to 2pi, this will be our singularity point
    double targ_wind = 0.0;
    double targ_min = 2.0 * M_PI;
    int idx_targ_last = 0;
    for (int l = 0; l < total; ++l) {
        double tester = abs(abs(wind_vals[l]) - 2.0 * M_PI);
        if (tester < targ_min) {
            targ_wind = abs(wind_vals[l]);
            targ_min = tester;
            idx_targ_last = l;
        }
    }

    // Assign the integer values of the spiral wave tip location coordinates
    int i_targ = idx_targ_last % width;
    int j_targ = idx_targ_last / width;

    // Safe guard
    if (targ_wind > M_PI) {
        return std::make_pair(i_targ, j_targ);
    }

    // Catch NAN values
    return std::make_pair(-1, -1);
}

// Using the voltage and gating variable of the spiral wave, detect the tip of the spiral wave (contour method)
std::pair<int, int> ApproxTip(std::vector<double>val_vec1, std::vector<double>val_vec2, double val_tar1, double val_tar2, int width, int height) {
    int total = width * height;

    // Square distance is used here to make smaller values easier to target
    // Find the squared distance of the grid values from the targeted u and v values for the spiral wave tip
    std::vector<double> dist_sq(total, 1.0);
    for (int ii = 0; ii < total; ++ii) {
        dist_sq[ii] = ((val_vec1[ii] - val_tar1) * (val_vec1[ii] - val_tar1)) + ((val_vec2[ii] - val_tar2) * (val_vec2[ii] - val_tar2));
    }

    // Find the minimum squared distance from the target values
    double dist_sq_min_last = 100.0;
    int idx_min_last = 10;
    for (int idx = 0; idx < total; ++idx) {
        if (dist_sq[idx] < dist_sq_min_last) {
            dist_sq_min_last = dist_sq[idx];
            idx_min_last = idx;
        }
    }
        
    // Assign the integer values of the spiral wave tip location coordinates
    int i = idx_min_last % width;
    int j = idx_min_last / width;

    // Safeguard, catch anything outside of reason (remember we're using the squared distance)
    if (i > 0 && i < width && j > 0 && j < height && dist_sq_min_last < 0.0001) {
        return std::make_pair(i, j);
    }

    // Catch NAN values
    return std::make_pair(-1, -1);
}

// Using the voltage of the spiral wave at two different times, detect the tip of the spiral wave (Jacobian Determinate Method)
std::pair<int, int> Jacobian_Determinate_Method(std::vector<double>val_vec_t1, std::vector<double>val_vec_t2, int width, int height) {
    int total = width * height;

    // Take the determinate of the Jacobian around each point
    std::vector<double> DVx(total);
    for (int idx = 0; idx < total; ++idx) {

        int i = idx % width;
        int j = idx / width;

        int left_i = (i > 0) ? i - 1 : i;
        int right_i = (i < width - 1) ? i + 1 : i;
        int up_j = (j > 0) ? j - 1 : j;
        int down_j = (j < height - 1) ? j +1 : j;

        int left_idx = j * width + left_i;
        int right_idx = j * width + right_i;
        int up_idx = up_j * width + i;
        int down_idx = down_j * width + i;

        double DVx_temp = (val_vec_t1[right_idx] - val_vec_t1[left_idx]) / (right_i - left_i) * (val_vec_t2[up_idx] - val_vec_t2[down_idx]) / (up_j - down_j) - (val_vec_t1[up_idx] - val_vec_t1[down_idx]) / (up_j - down_j) * (val_vec_t2[right_idx] - val_vec_t2[left_idx]) / (right_i - left_i);
        DVx[idx] = DVx_temp;
       
    }


    // Find the greatest value of DVx
    double DVx_max_last = 0.0;
    int idx_max_last = 10;

    for (int idx = 0; idx < total; ++idx) {
        if (DVx[idx] >  DVx_max_last) {
            DVx_max_last = DVx[idx];
            idx_max_last = idx;
        }
    }

    // Assign the integer values of the spiral wave tip location coordinates
    int i = idx_max_last % width;
    int j = idx_max_last / width;

    // Safeguard, check that the spiral wave tip is not at (or close to) the boundaries
    if (i > 5 && i < width - 5 && j > 5 && j < height - 5) {
        return std::make_pair(i, j);
    }

    // Catch NAN values
    return std::make_pair(-1, -1);
}

// CUDA Kernel
//  Apply the first stimulus to the system (a plane wave propogating in the X direction from left to right)
__global__ void applyS1Perp(double* u_in, int width, int height, Parameters params, double time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;


    int col = idx % width;

    if (col <= static_cast<int>(width/10)) {
        u_in[idx] = 1.0;
    }

}

// CUDA Kernel
//  Apply the second stimulus to the system (a plane wave propogating in the Y direction from bottom to top)
__global__ void applyS2Perp(double* u_in, int width, int height, Parameters params, double time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;


    int row = idx / width;

    if (row >= static_cast<int>(height - 0.5 * height)) {
        u_in[idx] = 1.0;
    }

}

// CUDA Kernel
// The standard equations used to calculate the next time step of the FitzHugh-Nagumo equations based on the previous time step
__global__ void fitzhughNagumoKernel(double* u_in, double* v_in, double* u_out, double* v_out, int width, int height, Parameters params, double time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;

    if (idx >= total) return;

    int i = idx % width;
    int j = idx / width;

    // Compute safe neighbor indices
    int left_i = (i > 0) ? i - 1 : i;
    int right_i = (i < width - 1) ? i + 1 : i;
    int up_j = (j > 0) ? j - 1 : j;
    int down_j = (j < height - 1) ? j + 1 : j;

    // Convert to array indexing
    int left_idx = j * width + left_i;
    int right_idx = j * width + right_i;
    int up_idx = up_j * width + i;
    int down_idx = down_j * width + i;


    // Preform laplacian
    double u_center = u_in[idx];
    double u_left = u_in[left_idx];
    double u_right = u_in[right_idx];
    double u_up = u_in[up_idx];
    double u_down = u_in[down_idx];

    double lap_u = (u_left + u_right + u_up + u_down - 4.0 * u_center) / (params.dx * params.dx);

    // FitzHugh Nagumo equations
    double cubic_u = u_center * (1.0 - u_center) * (u_center - params.a);
    double dudt = params.D_u * lap_u + cubic_u - v_in[idx];
    double dvdt = params.eps * (params.b * u_center - v_in[idx]);

    // Update the output
    u_out[idx] = u_center + params.dt * dudt;
    v_out[idx] = v_in[idx] + params.dt * dvdt;

}

/**
 * Wrapper function for the CUDA kernel function.
 */
void evolveFitzHughNagumo(GridData& grid, const Parameters& params, const SimConstraints& simulation, int steps) {
    int total = grid.width * grid.height;
    size_t size = total * sizeof(double);

    // Allocate memory to the current and next step of the function
    double* u1, * v1, * u2, * v2, * u2_2, * v2_2, * u3, * v3;
    cudaMalloc(&u1, size);
    cudaMalloc(&v1, size);
    cudaMalloc(&u2, size);
    cudaMalloc(&v2, size);
    cudaMalloc(&u2_2, size);
    cudaMalloc(&v2_2, size);
    cudaMalloc(&u3, size);
    cudaMalloc(&v3, size);

    // Assign values to the current step of the function
    cudaMemcpy(u1, grid.u_vals, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(v1, grid.v_vals, size, cudaMemcpyDeviceToDevice);

    // Step up for CUDA
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;

    // Set up for tip tracking
    int tip_track_start_step = static_cast<int>(params.spiral_time / params.dt) + static_cast<int>(10 / params.dt);
    int tip_track_steps = (steps + 1) - tip_track_start_step;
    size_t tip_track_size = tip_track_steps * sizeof(double);
    size_t tip_track_size_est = tip_track_steps * sizeof(int);
    std::vector<int> tip_traj_JDM_x(tip_track_steps);
    std::vector<int> tip_traj_JDM_y(tip_track_steps);
    std::vector<double> tip_traj_volt_x(tip_track_steps);
    std::vector<double> tip_traj_volt_y(tip_track_steps);
    std::vector<double> tip_traj_phase_x(tip_track_steps);
    std::vector<double> tip_traj_phase_y(tip_track_steps);


    // For each time step evolve the simulation using the FitzHugh-Nagumo Model
    for(int step = 0; step <= steps; ++step) {
        double t = step * params.dt;

        // Apply S1 stimulus to the simulation
        if (simulation.spiral && step == 0) {
            applyS1Perp CUDA_KERNEL(blocksPerGrid, threadsPerBlock) (
                u1, grid.width, grid.height, params, t
                );
        }

        // Apply S2 stimulus to the simulation at specified time
        if (simulation.spiral && step == params.spiral_time / params.dt) {
            applyS2Perp CUDA_KERNEL(blocksPerGrid, threadsPerBlock) (
                u1, grid.width, grid.height, params, t
                );
        }
        // Calculate next time step
        fitzhughNagumoKernel CUDA_KERNEL(blocksPerGrid, threadsPerBlock) (u1, v1, u2, v2, grid.width, grid.height, params, t);

        // Safeguard
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel 1 launch failed: " << cudaGetErrorString(err) << std::endl;
        }

        // Syncronize calculations using CUDA
        cudaDeviceSynchronize();

        // Update time
        double t_new = t + params.dt;

        // Apply and append to arrays for JDM tip tracking
        if ((simulation.tip_track_JDM) && step >= tip_track_start_step) {
            std::vector<double> host_u_old(total + 1);
            std::vector<double> host_u_new(total + 1);

            cudaMemcpy(host_u_old.data(), u1, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_u_new.data(), u2, size, cudaMemcpyDeviceToHost);

            std::pair<int, int> ij_est = Jacobian_Determinate_Method(host_u_old, host_u_new, grid.width, grid.height);
            int i_est = std::get<0>(ij_est);
            int j_est = std::get<1>(ij_est);

            tip_traj_JDM_x[static_cast<int>(step - tip_track_start_step)] = i_est;
            tip_traj_JDM_y[static_cast<int>(step - tip_track_start_step)] = j_est;
        }

        // Apply and append to arrays for contour tip tracking
        if ((simulation.tip_track_volt) && step >= tip_track_start_step) {

            std::vector<double> host_u_new(total + 1);
            std::vector<double> host_v_new(total + 1);

            cudaMemcpy(host_u_new.data(), u2, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_v_new.data(), v2, size, cudaMemcpyDeviceToHost);

            std::pair<int,int> ij_approx = ApproxTip(host_u_new, host_v_new, params.utip_pick, params.vtip_pick, grid.width, grid.height);
            int i_approx = std::get<0>(ij_approx);
            int j_approx = std::get<1>(ij_approx);

            if (i_approx > 0 && j_approx > 0 && i_approx < grid.width && j_approx < grid.height) {
                int tip_i_min = (i_approx > 0) ? i_approx - 1 : i_approx;
                int tip_i_max = (i_approx < grid.width - 1) ? i_approx + 1 : i_approx;
                int tip_j_min = (j_approx > 0) ? j_approx - 1 : j_approx;
                int tip_j_max = (j_approx < grid.height - 1) ? j_approx + 1 : j_approx;

                tip_traj_volt_x[static_cast<int>(step - tip_track_start_step)] = i_approx;
                tip_traj_volt_y[static_cast<int>(step - tip_track_start_step)] = j_approx;
            }


        }

        // Apply and append to arrays for phase tip tracking
        if ((simulation.tip_track_phase) && step >= tip_track_start_step) {
            std::vector<double> host_u_new(total + 1);
            std::vector<double> host_v_new(total + 1);

            cudaMemcpy(host_u_new.data(), u2, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_v_new.data(), v2, size, cudaMemcpyDeviceToHost);

            std::vector<double> phase_vals(total, 1.0);

            for (int k = 0; k < total; ++k) {
                double uu = host_u_new.data()[k];
                double vv = host_v_new.data()[k];

                phase_vals[k] = atan2(vv - params.vtip_pick, uu - params.utip_pick);
            }

            std::pair<int, int>  ij_approx = PhaseTipDetection(phase_vals, grid.width, grid.height);
            double i_approx = std::get<0>(ij_approx);
            double j_approx = std::get<1>(ij_approx);

            if (i_approx > 0 && j_approx > 0 && i_approx < grid.width && j_approx < grid.height) {
                int tip_i_min = (i_approx > 0) ? i_approx - 1 : i_approx;
                int tip_i_max = (i_approx < grid.width - 1) ? i_approx + 1 : i_approx;
                int tip_j_min = (j_approx > 0) ? j_approx - 1 : j_approx;
                int tip_j_max = (j_approx < grid.height - 1) ? j_approx + 1 : j_approx;

                tip_traj_phase_x[static_cast<int>(step - tip_track_start_step)] = i_approx;
                tip_traj_phase_y[static_cast<int>(step - tip_track_start_step)] = j_approx;
            }

            
        }

        // Every 100ms in the simulation, create an output file for the u and v values at that time
        if ((static_cast<int>(t) % 100 == 0) || (static_cast<int>(t) == simulation.last_time)) {
            std::vector<double> host_u(total);
            std::vector<double> host_v(total);

            // Copy grid values to the device
            cudaMemcpy(host_u.data(), u1, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_v.data(), v1, size, cudaMemcpyDeviceToHost);

            // Create target directory
            std::filesystem::create_directories("Results/" + simulation.run_name);

            // Writing results to rile
            std::string u_filename = "Results/" + simulation.run_name + "/u_vals_t" + std::to_string(static_cast<int>(t)) + ".bin";
            std::string v_filename = "Results/" + simulation.run_name + "/ v_vals_t" + std::to_string(static_cast<int>(t)) + ".bin";

            std::ofstream ufile(u_filename, std::ios::binary);
            std::ofstream vfile(v_filename, std::ios::binary);

            ufile.write(reinterpret_cast<const char*>(host_u.data()), size);
            vfile.write(reinterpret_cast<const char*>(host_v.data()), size);

            ufile.close();
            vfile.close();
        }

        std::swap(u1, u2);
        std::swap(v1, v2);
    }

    // Write JDM tip tacking values to file
    if (simulation.tip_track_JDM) {

        std::string tip_x_JDM_filename = "Results/" + simulation.run_name + "/JDM_tip_x_tracker.bin";
        std::string tip_y_JDM_filename = "Results/" + simulation.run_name + "/JDM_tip_y_tracker.bin";

        std::ofstream tip_x_JDM_file(tip_x_JDM_filename, std::ios::binary);
        std::ofstream tip_y_JDM_file(tip_y_JDM_filename, std::ios::binary);

        tip_x_JDM_file.write(reinterpret_cast<const char*>(&tip_traj_JDM_x[0]), tip_track_size_est);
        tip_y_JDM_file.write(reinterpret_cast<const char*>(&tip_traj_JDM_y[0]), tip_track_size_est);


        tip_x_JDM_file.close();
        tip_y_JDM_file.close();
    }

    // Write contour tip tacking values to file
    if (simulation.tip_track_volt) {

        std::string tip_x_volt_filename = "Results/" + simulation.run_name + "/volt_tip_x_tracker.bin";
        std::string tip_y_volt_filename = "Results/" + simulation.run_name + "/volt_tip_y_tracker.bin";

        std::ofstream tip_x_volt_file(tip_x_volt_filename, std::ios::binary);
        std::ofstream tip_y_volt_file(tip_y_volt_filename, std::ios::binary);

        tip_x_volt_file.write(reinterpret_cast<const char*>(&tip_traj_volt_x[0]), tip_track_size);
        tip_y_volt_file.write(reinterpret_cast<const char*>(&tip_traj_volt_y[0]), tip_track_size);
        

        tip_x_volt_file.close();
        tip_y_volt_file.close();
    }

    // Write phase tip tacking values to file
    if (simulation.tip_track_phase) {

        std::string tip_x_phase_filename = "Results/" + simulation.run_name + "/phase_tip_x_tracker.bin";
        std::string tip_y_phase_filename = "Results/" + simulation.run_name + "/phase_tip_y_tracker.bin";

        std::ofstream tip_x_phase_file(tip_x_phase_filename, std::ios::binary);
        std::ofstream tip_y_phase_file(tip_y_phase_filename, std::ios::binary);

        tip_x_phase_file.write(reinterpret_cast<const char*>(&tip_traj_phase_x[0]), tip_track_size);
        tip_y_phase_file.write(reinterpret_cast<const char*>(&tip_traj_phase_y[0]), tip_track_size);


        tip_x_phase_file.close();
        tip_y_phase_file.close();
    }

    // Copy the  memory of the end step calculated values to the device
    cudaMemcpy(grid.u_vals, u1, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(grid.v_vals, v1, size, cudaMemcpyDeviceToDevice);


    // Clean up memory
    cudaFree(u1); cudaFree(v1);
    cudaFree(u2); cudaFree(v2);
}
