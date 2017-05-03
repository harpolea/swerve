/**
Includes main function to run mesh cuda simulation.

Compile with 'make mesh'.

Run with `mpirun -np N ./mesh [input file]` where N is the number of processors to use and input file is an optional argument providing the file path to the input file to use. If no input file is provided, will default to use mesh_input.txt.
*/

#include <stdio.h>
#include <cmath>
#include <limits>
#include "../Mesh_cuda.h"
#include <iostream>
#include <string.h>
#include <fstream>
#include <algorithm>
#include "mpi.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../mesh_output.h"

using namespace std;

void acoustic_wave(Sea *sea) {
    // locate index of first compressible level
    int c_in = 0;
    while (sea -> models[c_in] != 'C') c_in += 1;

    float * D0 = new float[sea->nxs[c_in]*sea->nys[c_in]*sea->nzs[c_in]];
    float * Sx0 = new float[sea->nxs[c_in]*sea->nys[c_in]*sea->nzs[c_in]];
    float * Sy0 = new float[sea->nxs[c_in]*sea->nys[c_in]*sea->nzs[c_in]];
    float * Sz0 = new float[sea->nxs[c_in]*sea->nys[c_in]*sea->nzs[c_in]];
    float * tau0 = new float[sea->nxs[c_in]*sea->nys[c_in]*sea->nzs[c_in]];

    // set acoustic wave test initial data
    for (int y = 0; y < sea->nys[c_in]; y++) {
        for (int x = 0; x < sea->nxs[c_in]; x++) {
            D0[y * sea->nxs[c_in] + x] = -0.5 *
                log(1.0 - 2.0 / (sea->zmax+2*sea->dz));// - 0.1 *
                //exp(-(pow(sea->xs[x]-5.0, 2)+pow(sea->ys[y]-5.0, 2)) * 2.0);
            D0[(sea->nys[c_in] + y) * sea->nxs[c_in] + x] = 1.1 + 0.001 * sin(2.0 * sea->xs[x] * M_PI / (sea->xs[sea->nxs[c_in]-1-sea->ng] - sea->xs[sea->ng]));
            D0[(2*sea->nys[c_in] + y) * sea->nxs[c_in] + x] = -0.5 *
                log(1.0 - 2.0 / sea->zmin);

            for (int z = 0; z < sea->nzs[c_in]; z++) {
                Sx0[(z * sea->nys[c_in] + y) * sea->nxs[c_in] + x] = 0.0;
                Sy0[(z * sea->nys[c_in] + y) * sea->nxs[c_in] + x] = 0.0;
            }
        }
    }

    sea->initial_compressible_data(D0, Sx0, Sy0, Sz0, tau0);

    // clean up arrays
    delete[] D0;
    delete[] Sx0;
    delete[] Sy0;
    delete[] Sz0;
    delete[] tau0;
}
