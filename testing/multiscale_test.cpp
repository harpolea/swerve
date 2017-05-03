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

void multiscale_test(Sea *sea) {
    // locate index of first multilayer SWE level
    int m_in = 0;
    while (sea -> models[m_in] != 'M') m_in += 1;

    float * D0 = new float[sea->nxs[m_in]*sea->nys[m_in]*sea->nzs[m_in]];
    float * Sx0 = new float[sea->nxs[m_in]*sea->nys[m_in]*sea->nzs[m_in]];
    float * Sy0 = new float[sea->nxs[m_in]*sea->nys[m_in]*sea->nzs[m_in]];

    // set multiscale test initial data
    for (int y = 0; y < sea->nys[m_in]; y++) {
        for (int x = 0; x < sea->nxs[m_in]; x++) {
            D0[y * sea->nxs[m_in] + x] = -0.5 *
                log(1.0 - 2.0 / (sea->zmax+2*sea->dz));// - 0.1 *
                //exp(-(pow(sea->xs[x]-5.0, 2)+pow(sea->ys[y]-5.0, 2)) * 2.0);
            D0[(sea->nys[m_in] + y) * sea->nxs[m_in] + x] = 1.1 + 0.001 * sin(2.0 * sea->xs[x] * M_PI / (sea->xs[sea->nxs[m_in]-1-sea->ng] - sea->xs[sea->ng]));
            D0[(2*sea->nys[m_in] + y) * sea->nxs[m_in] + x] = -0.5 *
                log(1.0 - 2.0 / sea->zmin);

            for (int z = 0; z < sea->nzs[m_in]; z++) {
                Sx0[(z * sea->nys[m_in] + y) * sea->nxs[m_in] + x] = 0.0;
                Sy0[(z * sea->nys[m_in] + y) * sea->nxs[m_in] + x] = 0.0;
            }
        }
    }

    sea->initial_swe_data(D0, Sx0, Sy0);

    // clean up arrays
    delete[] D0;
    delete[] Sx0;
    delete[] Sy0;
}
