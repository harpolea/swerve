/**
Includes main function to run mesh cuda simulation.

Compile with 'make mesh'.

Run with `mpirun -np N ./mesh [input file]` where N is the number of processors to use and input file is an optional argument providing the file path to the input file to use. If no input file is provided, will default to use mesh_input.txt.
*/

#include <stdio.h>
#include <cmath>
#include <limits>
#include "Mesh_cuda.h"
#include <iostream>
#include <string.h>
#include <fstream>
#include <algorithm>
#include "mpi.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

int main(int argc, char *argv[]) {

    // MPI variables
    MPI_Comm comm;
    MPI_Status status;

    int rank, size;//, source, tag;

    // Initialise MPI and compute number of processes and local rank
    comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        printf("Running on %d process(es)\n", size);
    }

    char input_filename[200];

    if (argc == 1) {
        // no input arguments - default input file.
        string fname = "mesh_input.txt";
        strcpy(input_filename, fname.c_str());
    } else {
        strcpy(input_filename, argv[1]);
    }

    Sea sea(input_filename);

    // locate index of first multilayer SWE level - should be 0 or 1
    int m_in = (sea.models[0] == 'S') ? 1 : 0;

    float * D0 = new float[sea.nxs[m_in]*sea.nys[m_in]*sea.nzs[m_in]];
    float * Sx0 = new float[sea.nxs[m_in]*sea.nys[m_in]*sea.nzs[m_in]];
    float * Sy0 = new float[sea.nxs[m_in]*sea.nys[m_in]*sea.nzs[m_in]];

    // set initial data
    /*for (int y = 0; y < sea.nys[m_in]; y++) {
        for (int x = 0; x < sea.nxs[m_in]; x++) {
            D0[y * sea.nxs[m_in] + x] = -0.5 *
                log(1.0 - 2.0 / (sea.zmax+2*sea.dz));// - 0.1 *
                //exp(-(pow(sea.xs[x]-5.0, 2)+pow(sea.ys[y]-5.0, 2)) * 2.0);
            D0[(sea.nys[m_in] + y) * sea.nxs[m_in] + x] = 1.1 - 0.1 *
                exp(-(pow(sea.xs[x]-5.0, 2)+pow(sea.ys[y]-5.0, 2)) * 2.0);
            D0[(2*sea.nys[m_in] + y) * sea.nxs[m_in] + x] = -0.5 *
                log(1.0 - 2.0 / sea.zmin);

            for (int z = 0; z < sea.nzs[m_in]; z++) {
                Sx0[(z * sea.nys[m_in] + y) * sea.nxs[m_in] + x] = 0.0;
                Sy0[(z * sea.nys[m_in] + y) * sea.nxs[m_in] + x] = 0.0;
            }
        }
    }*/

    // set multiscale test initial data
    for (int y = 0; y < sea.nys[m_in]; y++) {
        for (int x = 0; x < sea.nxs[m_in]; x++) {
            D0[y * sea.nxs[m_in] + x] = -0.5 *
                log(1.0 - 2.0 / (sea.zmax+2*sea.dz));// - 0.1 *
                //exp(-(pow(sea.xs[x]-5.0, 2)+pow(sea.ys[y]-5.0, 2)) * 2.0);
            D0[(sea.nys[m_in] + y) * sea.nxs[m_in] + x] = 1.1  + 0.1 * sin(2.0 * sea.xs[x] * M_PI / (sea.xs[sea.nxs[m_in]-1-sea.ng] - sea.xs[sea.ng]));
            D0[(2*sea.nys[m_in] + y) * sea.nxs[m_in] + x] = -0.5 *
                log(1.0 - 2.0 / sea.zmin);

            for (int z = 0; z < sea.nzs[m_in]; z++) {
                Sx0[(z * sea.nys[m_in] + y) * sea.nxs[m_in] + x] = 0.0;
                Sy0[(z * sea.nys[m_in] + y) * sea.nxs[m_in] + x] = 0.0;
            }
        }
    }

    sea.initial_data(D0, Sx0, Sy0);

    if (rank == 0) {
        sea.print_inputs();
    }

    // clean up arrays
    delete[] D0;
    delete[] Sx0;
    delete[] Sy0;

    sea.run(comm, &status, rank, size);

    MPI_Finalize();
}
