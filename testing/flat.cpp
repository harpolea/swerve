#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <algorithm>
#include "../SeaCuda.h"

using namespace std;

/*
This test takes two initially flat layers in flat space and evolves for a few timesteps. It then compares the evolved data against the initial data to check that the system has remained stable and not changed to within a given tolerance.
*/

int main() {

    // make a sea
    char input_filename[] = "flat.txt";
    SeaCuda sea(input_filename);

    float * D0 = new float[sea.nlayers*sea.nx*sea.ny];
    float * Sx0 = new float[sea.nlayers*sea.nx*sea.ny];
    float * Sy0 = new float[sea.nlayers*sea.nx*sea.ny];
    float * zeta0 = new float[sea.nlayers*sea.nx*sea.ny];
    float * _Q = new float[sea.nlayers*sea.nx*sea.ny];
    float * _beta = new float[2*sea.nx*sea.ny];

    // set initial data
    for (int x = 1; x < (sea.nx - 1); x++) {
        for (int y = 1; y < (sea.ny - 1); y++) {
            D0[(y * sea.nx + x) * sea.nlayers] = 1.0 ;
            D0[(y * sea.nx + x) * sea.nlayers + 1] = 0.8;
            _beta[(y * sea.nx + x) * 2] = 0.0;
            _beta[(y * sea.nx + x) * 2 + 1] = 0.0;
            for (int l = 0; l < sea.nlayers; l++) {
                Sx0[(y * sea.nx + x) * sea.nlayers + l] = 0.0;
                Sy0[(y * sea.nx + x) * sea.nlayers + l] = 0.0;
                zeta0[(y * sea.nx + x) * sea.nlayers + l] = 1.0;
                _Q[(y * sea.nx + x) * sea.nlayers + l] = 0.0;
            }
        }
    }

    sea.initial_data(D0, Sx0, Sy0, zeta0, _Q, _beta);

    // copy back to account for the fact that bcs have been enforced.
    for (int i = 0; i < sea.nlayers*sea.nx*sea.ny; i++) {
        D0[i] = sea.U_grid[i*4];
        Sx0[i] = sea.U_grid[i*4 + 1];
        Sy0[i] = sea.U_grid[i*4 + 2];
        zeta0[i] = sea.U_grid[i*4 + 3];
    }

    sea.print_inputs();

    // run simulation
    sea.run();

    // test if output matches input
    float tol = 1.0e-6; // absolute error tolerance

    float * err = new float[sea.nlayers*sea.nx*sea.ny*4];

    bool passed = true;

    for (int i = 0; i < sea.nlayers*sea.nx*sea.ny; i++) {
        //cout << sea.U_grid[i*4] << ' ' << D0[i] << '\n';
        err[i*4] = sea.U_grid[i*4] - D0[i];
        err[i*4+1] = sea.U_grid[i*4 + 1] - Sx0[i];
        err[i*4+2] = sea.U_grid[i*4 + 2] - Sy0[i];
        err[i*4+3] = sea.U_grid[i*4 + 3] - zeta0[i];
        for (int j = 0; j < 4; j++) {
            if (abs(err[i*4 + j]) > tol) {
                cout << "Error for component " << i << ' ' << j << ": " << err[i*4 + j] << " sea.U_grid: " << sea.U_grid[i*4 + 3] << " zeta0: " << zeta0[i] << '\n';
                passed = false;
                break;
            }
        }
    }

    if (passed == true) {
        cout << "Passed flat test!" << '\n';
    } else {
        cout << "Did not pass :(\n";
    }

    // clean up
    delete[] D0;
    delete[] Sx0;
    delete[] Sy0;
    delete[] zeta0;
    delete[] _Q;
    delete[] _beta;
    delete[] err;

    return int(passed);


}
