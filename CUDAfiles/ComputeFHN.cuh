#pragma once

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

class GridData {
public:
    int width, height;
    double* u_vals;
    double* v_vals;

    GridData(int w, int h);
    ~GridData();

    // Delete copy constructor and copy assignment
    GridData(const GridData&) = delete;
    GridData& operator=(const GridData&) = delete;

    // Move constructor and move assignment
    GridData(GridData&& other) noexcept;
    GridData& operator=(GridData&& other) noexcept;

    int index(int i, int j) const;
    void copyFromHost(double* h_u, double* h_v);
    void copyToHost(double* h_u, double* h_v);
};

GridData StartSim(GridData& grid, const double u_start, const double v_start);

double LinearInterp(double idx1, double idx2, double val1, double val2, double val_tar);

std::pair<double, double> BilinearInterp(int idx_x1, int idx_x2, int idx_y1, int idx_y2, std::vector<double> val_vec1, double uval_tar, int width, int iter);

double AngleWrap(double angle);

std::pair<int, int> PhaseTipDetection(std::vector<double>phase_vals, int width, int height);

std::pair<int, int> ApproxTip(std::vector<double>val_vec1, std::vector<double>val_vec2, double val_tar1, double val_tar2, int width, int height);

std::pair<int, int> Jacobian_Determinate_Method(std::vector<double>val_vec_t1, std::vector<double>val_vec_t2, int width, int height);

void evolveFitzHughNagumo(GridData& grid, const Parameters& params, const SimConstraints& simulation, int steps);
