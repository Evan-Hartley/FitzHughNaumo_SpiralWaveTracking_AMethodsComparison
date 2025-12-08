# FitzHughNaumo_SpiralWaveTracking_AMethodsComparison
Compare and Contrast the Contour and Phase Identification Methods for Spiral Wave Identification using the FitzHugh-Nagumo Model

## Code instructions

All togglable variables and adjustable parameters are located in the `main.cpp` file, you will want to adjust variables before compilation
Highlights:
- SimConstraints simulation.last_time: Tells the simulation how many milliseconds to run for
- SimConstraints simulation.spiral: If true the simulation will apply a stimulous at the time specifiec by SimConstraints simulation.spiral_time in order to create a singular spiral wave
- SimConstraints simulation.tip_track_JDM: If true, the program will track the approximate location of the spiral wave tip using the Jacobian Determinate Method
- SimConstraints simulation.tip_track_volt: If true, the program will track the approximate location of the spiral wave tip based on the estimated value of the spiral wave tip in phase space currently stored in Parameters params.utip_pick and Parameters params.vtip_pick.
- SimConstraints simulation.tip_track_JDM: If true, the program will track the approximate location of the spiral wave tip on the estimated value phase angle of the wave about the point Parameters params.utip_pick and Parameters params.vtip_pick.
- SimConstraints.run_name: Stores the title of your simulation, your output files will be stored in a file sharing this name

## Compilation instructions

Run the following commands in the repository directory:
 ```
mkdir build && cd build
cmake ..
cmake --build .
```
 
  The .exe file will then be in a subfolder entitled `FitzHughNagumoModel/run/bin/`
