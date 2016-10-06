#include <iostream>
#include <string>
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <helper_functions.h>
#include <algorithm>
#include "SeaCuda.h"

using namespace std;

int main() {

    // make a sea
    char input_filename[] = "input_file.txt";
    SeaCuda sea(input_filename);

    float * D0 = new float[sea.nlayers*sea.nx*sea.ny];
    float * Sx0 = new float[sea.nlayers*sea.nx*sea.ny];
    float * Sy0 = new float[sea.nlayers*sea.nx*sea.ny];
    float * _Q = new float[sea.nlayers*sea.nx*sea.ny];

    // set initial data
    for (int x = 1; x < (sea.nx - 1); x++) {
        for (int y = 1; y < (sea.ny - 1); y++) {
            D0[(y * sea.nx + x) * sea.nlayers] = 1.0 + 0.4 * exp(-(pow(sea.xs[x-1]-2.0, 2) + pow(sea.ys[y-1]-2.0, 2)) * 2.0);
            D0[(y * sea.nx + x) * sea.nlayers + 1] = 0.8 + 0.2 * exp(-(pow(sea.xs[x-1]-7.0, 2) + pow(sea.ys[y-1]-7.0, 2)) * 2.0);
            for (int l = 0; l < sea.nlayers; l++) {
                Sx0[(y * sea.nx + x) * sea.nlayers + l] = 0.0;
                Sy0[(y * sea.nx + x) * sea.nlayers + l] = 0.0;
                _Q[(y * sea.nx + x) * sea.nlayers + l] = 0.0;
            }
        }
    }

    sea.initial_data(D0, Sx0, Sy0, _Q);

    delete[] D0;
    delete[] Sx0;
    delete[] Sy0;
    delete[] _Q;

    sea.print_inputs();

    // run simulation
    sea.run();

    //sea.output();

    //cout << "Output data to file.\n";

}
