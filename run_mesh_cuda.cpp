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
#include "mesh_output.h"

using namespace std;

void multiscale_test(Sea *sea);
void acoustic_wave(Sea *sea);

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

    if (string(input_filename).find("checkpoint") != string::npos) {
        start_from_checkpoint(input_filename, comm, status,
                rank, size);
    } else {
        // input file is a parameter file
        Sea sea(input_filename);

        //multiscale_test(&sea);
        acoustic_wave(&sea);

        if (rank == 0) {
            sea.print_inputs();
        }

        sea.run(comm, &status, rank, size, 0);
    }

    MPI_Finalize();
}

void multiscale_test(Sea *sea) {
    /*
    Initial data for
    */
    // locate index of first multilayer SWE level
    int m_in = 0;
    while (sea -> models[m_in] != 'M') m_in += 1;

    // locate index of first compressible level
    //int c_in = m_in;
    //while (sea -> models[c_in] != 'C') c_in += 1;

    int c_in = sea -> nlevels;
    if (sea -> models[sea -> nlevels-1] == 'C') {
        while(sea -> models[c_in-1] == 'C') c_in -= 1;
    }

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

    float * D0c = new float[sea->nxs[c_in]*sea->nys[c_in]*sea->nzs[c_in]];
    float * Sx0c = new float[sea->nxs[c_in]*sea->nys[c_in]*sea->nzs[c_in]];
    float * Sy0c = new float[sea->nxs[c_in]*sea->nys[c_in]*sea->nzs[c_in]];
    float * Sz0 = new float[sea->nxs[c_in]*sea->nys[c_in]*sea->nzs[c_in]];
    float * tau0 = new float[sea->nxs[c_in]*sea->nys[c_in]*sea->nzs[c_in]];

    for (int z = 0; z < sea->nzs[c_in]; z++) {
        for (int y = 0; y < sea->nys[c_in]; y++) {
            for (int x = 0; x < sea->nxs[c_in]; x++) {
                float max_v = 0.3;
                float r = sqrt(
                    (x - 0.5*sea->nxs[c_in])*(x - 0.5*sea->nxs[c_in]) +
                    (y - 0.5*sea->nys[c_in])*(y - 0.5*sea->nys[c_in]));
                float v = 0.0;
                if (r < 0.05 * sea->nxs[c_in]) {
                    v = 20.0 * max_v * r / sea->nxs[c_in];
                } else if (r < 0.1 * sea->nxs[c_in]) {
                    v = 2.0 * 20.0 * max_v * 0.05 - 20.0 * max_v * r / sea->nxs[c_in];
                }
                float D = sea->Us[c_in][((z*sea->nys[c_in] + y) * sea->nxs[c_in] + x) * sea->vec_dims[c_in]];

                D0c[(z * sea->nys[c_in]+ y) * sea->nxs[c_in] + x] = D;

                if (r > 0.0) {
                    // Sx
                    Sx0c[(z * sea->nys[c_in]+ y) * sea->nxs[c_in] + x]
                        = - D * v * (y - 0.5*sea->nys[c_in]) / r;
                    Sy0c[(z * sea->nys[c_in]+ y) * sea->nxs[c_in] + x]
                        = D * v * (x - 0.5*sea->nxs[c_in]) / r;
                } else {
                    Sx0c[(z * sea->nys[c_in]+ y) * sea->nxs[c_in] + x] = 0.0;
                    Sy0c[(z * sea->nys[c_in] + y) * sea->nxs[c_in] + x] = 0.0;
                }
                Sz0[(z * sea->nys[c_in] + y) * sea->nxs[c_in] + x] = 0.0;
                tau0[(z * sea->nys[c_in] + y) * sea->nxs[c_in] + x] = 0.0;
            }
        }
    }

    sea->initial_compressible_data(D0c, Sx0c, Sy0c, Sz0, tau0);

    // clean up arrays
    delete[] D0c;
    delete[] Sx0c;
    delete[] Sy0c;
    delete[] Sz0;
    delete[] tau0;
}

void acoustic_wave(Sea *sea) {
    /*
    Isentropic smooth flow in 1d, as described in section 6.1.1 of Marti & Muller 15, 4.6 of Zhang & MacFadyen 06.
    */
    // TODO: need to fix main cuda_run script so that it restricts/prolongs out from coarsest compressible grid rather than from coarsest multilayer grid for this example.

    // locate index of first multilayer SWE level
    int m_in = 0;
    while (sea -> models[m_in] != 'M') m_in += 1;

    // locate index of first compressible level
    int c_in = sea -> nlevels;
    if (sea -> models[sea -> nlevels-1] == 'C') {
        while(sea -> models[c_in-1] == 'C') c_in -= 1;
    }

    float * D0 = new float[sea->nxs[c_in]*sea->nys[c_in]*sea->nzs[c_in]];
    float * Sx0 = new float[sea->nxs[c_in]*sea->nys[c_in]*sea->nzs[c_in]];
    float * Sy0 = new float[sea->nxs[c_in]*sea->nys[c_in]*sea->nzs[c_in]];
    float * Sz0 = new float[sea->nxs[c_in]*sea->nys[c_in]*sea->nzs[c_in]];
    float * tau0 = new float[sea->nxs[c_in]*sea->nys[c_in]*sea->nzs[c_in]];

    float L = sea->nxs[c_in] * 0.3;
    float rho_ref = 0.5;
    float alpha = 0.05;
    float K = 1;
    float s = sqrt(sea->gamma - 1.0);

    float rhoh_temp = rho_ref + sea->gamma * K*pow(rho_ref,sea->gamma) / (sea->gamma - 1.0);
    float cs_temp = sqrt(sea->gamma * K*pow(rho_ref,sea->gamma) / rhoh_temp);
    float J = -log((s + cs_temp) / (s - cs_temp)) / s;

    // set acoustic wave test initial data
    for (int z = 0; z < sea->nzs[c_in]; z++) {
        for (int y = 0; y < sea->nys[c_in]; y++) {
            for (int x = 0; x < sea->nxs[c_in]; x++) {
                float r = sqrt((x-0.5*sea->nxs[c_in])*(x-0.5*sea->nxs[c_in]) +
                    (y-0.5*sea->nys[c_in])*(y-0.5*sea->nys[c_in]));
                float f = abs(r) < L ? pow(r*r / (L*L) - 1.0, 4) : 0.0;

                float rho = rho_ref * (1.0 + alpha * f);
                float p = K * pow(rho, sea->gamma);
                float rhoh = rho + sea->gamma * p / (sea->gamma - 1.0);
                float cs = sqrt(sea->gamma * p / rhoh);

                float a = log((s + cs) / (s - cs)) / s;
                float u = (exp(2.0 * (J + a)) - 1.0) / (1.0 + exp(2.0 * (J + a)));

                float W = 1.0 / sqrt(1.0 - u*u);

                D0[(z * sea->nys[c_in] + y) * sea->nxs[c_in] + x] = rho * W;

                if (r < 1.0) {
                    Sx0[(z * sea->nys[c_in] + y) * sea->nxs[c_in] + x] = 0.0;
                    Sy0[(z * sea->nys[c_in] + y) * sea->nxs[c_in] + x] = 0.0;
                } else {
                    Sx0[(z * sea->nys[c_in] + y) * sea->nxs[c_in] + x] =
                        rhoh * u * W * (x-0.5 * sea->nxs[c_in]) / r;

                    Sy0[(z * sea->nys[c_in] + y) * sea->nxs[c_in] + x] =
                        rhoh * u * W * (y-0.5 * sea->nys[c_in]) / r;
                }


                Sz0[(z * sea->nys[c_in] + y) * sea->nxs[c_in] + x] = 0.0;

                tau0[(z * sea->nys[c_in] + y) * sea->nxs[c_in] + x] =
                    rhoh*W*W - p - rho * W;

                //cout << rho * W << ' ' << rhoh * u * W << ' ' << rhoh*W*W - p - rho * W << '\n';
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

    float * D0s = new float[sea->nxs[m_in]*sea->nys[m_in]*sea->nzs[m_in]];
    float * Sx0s = new float[sea->nxs[m_in]*sea->nys[m_in]*sea->nzs[m_in]];
    float * Sy0s = new float[sea->nxs[m_in]*sea->nys[m_in]*sea->nzs[m_in]];

    // set multiscale test initial data
    for (int y = 0; y < sea->nys[m_in]; y++) {
        for (int x = 0; x < sea->nxs[m_in]; x++) {
            D0s[y * sea->nxs[m_in] + x] = -0.5 *
                log(1.0 - 2.0 / (sea->zmax+2*sea->dz));
            D0s[(sea->nys[m_in] + y) * sea->nxs[m_in] + x] = 1.1;
            D0s[(2*sea->nys[m_in] + y) * sea->nxs[m_in] + x] = -0.5 *
                log(1.0 - 2.0 / sea->zmin);

            for (int z = 0; z < sea->nzs[m_in]; z++) {
                Sx0s[(z * sea->nys[m_in] + y) * sea->nxs[m_in] + x] = 0.0;
                Sy0s[(z * sea->nys[m_in] + y) * sea->nxs[m_in] + x] = 0.0;
            }
        }
    }

    sea->initial_swe_data(D0s, Sx0s, Sy0s);

    // clean up arrays
    delete[] D0s;
    delete[] Sx0s;
    delete[] Sy0s;
}
