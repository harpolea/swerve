#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <algorithm>
#include "../SeaCuda.h"

using namespace std;

int main() {

    // make a sea
    char input_filename[] = "flat.txt";
    SeaCuda sea(input_filename);

    float * D0 = new float[sea.nlayers*sea.nx*sea.ny];
    float * Sx0 = new float[sea.nlayers*sea.nx*sea.ny];
    float * Sy0 = new float[sea.nlayers*sea.nx*sea.ny];
    float * _Q = new float[sea.nlayers*sea.nx*sea.ny];

    // set initial data
    for (int x = 1; x < (sea.nx - 1); x++) {
        for (int y = 1; y < (sea.ny - 1); y++) {
            D0[(y * sea.nx + x) * sea.nlayers] = 1.0 ;
            D0[(y * sea.nx + x) * sea.nlayers + 1] = 0.8;
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

    // test if output matches input 

}
