#pragma once

struct Parameters {
    int last_step, spiral_time;
    double D_u, eps, a, b, dt, dx;
    double utip_pick, vtip_pick;

    Parameters()
        : D_u(0.0), eps(0.0),
        a(0.0f), b(0.0), dt(0.0), dx(0.0),
        utip_pick(0.0), vtip_pick(0.0),
        last_step(0), spiral_time(0) {}

};

struct SimConstraints {
    int nx, ny, last_time, spiral_time;
    double dt, dx;
    bool spiral, tip_track_JDM, tip_track_volt, tip_track_phase;

    std::string run_name;

    SimConstraints()
        : nx(0), ny(0), dt(0.0), dx(0.0),
        last_time(0), spiral_time(0),
        spiral(false), tip_track_JDM(false), tip_track_volt(false), tip_track_phase(false), run_name("temp") {}

};
