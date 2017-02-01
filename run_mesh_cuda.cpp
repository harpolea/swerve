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

/*
Compile with 'make mesh'

*/


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

    float * D0 = new float[sea.nx*sea.ny*sea.nlayers];
    float * Sx0 = new float[sea.nx*sea.ny*sea.nlayers];
    float * Sy0 = new float[sea.nx*sea.ny*sea.nlayers];
    float * Sz0 = new float[sea.nx*sea.ny*sea.nlayers];
    float * tau = new float[sea.nx*sea.ny*sea.nlayers];

    // set initial data

    for (int y = 0; y < sea.ny; y++) {
        for (int x = 0; x < sea.nx; x++) {
            //D0[y * sea.nx + x] = 1.0 - 0.1 *
              //  exp(-(pow(sea.xs[x]-5.0, 2)+pow(sea.ys[y]-5.0, 2)) * 2.0);
            //D0[(sea.ny + y) * sea.nx + x] = 1.1 - 0.1 *
              //  exp(-(pow(sea.xs[x]-5.0, 2)+pow(sea.ys[y]-5.0, 2)) * 2.0);
            //D0[(2*sea.ny + y) * sea.nx + x] = -0.5 * log(1.0 - 2.0 / sea.zmin);

            // define A at sea floor
            float A_floor = 1.0;

            float ph0 = -0.5 * log(1.0 - 2.0 / sea.zmax);
            float ph1 = -0.5 * log(1.0 - 2.0 / (0.5 * (sea.zmax + sea.zmin)));
            float ph2 = -0.5 * log(1.0 - 2.0 / sea.zmin);

            float p2 = (sea.gamma - 1.0) * (A_floor * exp(sea.gamma * ph2 /
                (sea.gamma - 1.0)) - sea.rho[2]) / sea.gamma;

            float A1 = A_floor * (sea.gamma/(sea.gamma-1.0) * p2 + sea.rho[1]) / (sea.gamma/(sea.gamma-1.0) * p2 + 2.0 * sea.rho[1] - sea.rho[2]);

            float p1 = (sea.gamma - 1.0) * (A1 * exp(sea.gamma * ph1 /
                (sea.gamma - 1.0)) - sea.rho[1]) / sea.gamma;

            float A0 = A1 * (sea.gamma/(sea.gamma-1.0) * p1 + sea.rho[0]) / (sea.gamma/(sea.gamma-1.0) * p1 + 2.0 * sea.rho[0] - sea.rho[1]);

            float p0 = (sea.gamma - 1.0) * (A0 * exp(sea.gamma * ph0 /
                (sea.gamma - 1.0)) - sea.rho[0]) / sea.gamma;

            float rhoh0 = sea.rho[0] + sea.gamma * p0 / (sea.gamma - 1.0);
            float rhoh1 = sea.rho[1] + sea.gamma * p1 / (sea.gamma - 1.0);
            float rhoh2 = sea.rho[2] + sea.gamma * p2 / (sea.gamma - 1.0);

            tau[y * sea.nx + x] = rhoh0 - p0 - sea.rho[0];
            tau[(sea.ny + y) * sea.nx + x] = rhoh1 - p1 - sea.rho[1];
            tau[(2*sea.ny + y) * sea.nx + x] = rhoh2 - p2 - sea.rho[2];

            for (int z = 0; z < sea.nlayers; z++) {
                D0[(z * sea.ny + y) * sea.nx + x] = sea.rho[z];
                Sx0[(z * sea.ny + y) * sea.nx + x] = 0.0;
                Sy0[(z * sea.ny + y) * sea.nx + x] = 0.0;
                Sz0[(z * sea.ny + y) * sea.nx + x] = 0.0;
            }
        }
    }

    sea.initial_data(D0, Sx0, Sy0, Sz0, tau);

    if (rank == 0) {
        sea.print_inputs();
    }

    // clean up arrays
    delete[] D0;
    delete[] Sx0;
    delete[] Sy0;
    delete[] Sz0;
    delete[] tau;

    sea.run(comm, &status, rank, size);

    MPI_Finalize();
}
