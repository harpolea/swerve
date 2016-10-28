#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <algorithm>
#include "mpi.h"
#include "SeaCuda.h"

using namespace std;

/*
When the executable is called, if it is given an argument then this shall be used as the input file name. Otherwise it defaults to input_file.txt.
*/

int main(int argc, char *argv[]) {

    // make a sea
    char input_filename[200];

    if (argc == 1) {
        // no input arguments - default input file.
        string fname = "input_file.txt";
        strcpy(input_filename, fname.c_str());
    } else {
        strcpy(input_filename, argv[1]);
    }

    SeaCuda sea(input_filename);

    float * D0 = new float[sea.nlayers*sea.nx*sea.ny];
    float * Sx0 = new float[sea.nlayers*sea.nx*sea.ny];
    float * Sy0 = new float[sea.nlayers*sea.nx*sea.ny];
    float * zeta0 = new float[sea.nlayers*sea.nx*sea.ny];
    float * _Q = new float[sea.nlayers*sea.nx*sea.ny];
    float * _beta = new float[2*sea.nx*sea.ny];

    // set initial data
    for (int x = 0; x < sea.nx; x++) {
        for (int y = 0; y < sea.ny; y++) {
            // set D0 to be two hills of fluid
            //D0[(y * sea.nx + x) * sea.nlayers] = 1.0;// + 0.4 * exp(-(pow(sea.xs[x]-2.0, 2) + pow(sea.ys[y]-2.0, 2)) * 2.0);
            //D0[(y * sea.nx + x) * sea.nlayers + 1] = 0.8;// + 0.2 * exp(-(pow(sea.xs[x]-7.0, 2) + pow(sea.ys[y]-7.0, 2)) * 2.0);
            D0[(y * sea.nx + x) * sea.nlayers] = 1.5 + 0.4 * exp(-(pow(sea.xs[x]-5.0, 2)) * 2.0);

            //-0.1 * exp(-(0.3*pow(sea.xs[x]-5.0, 2) + pow(sea.ys[y]-5.0, 2)) * 2.0);
            D0[(y * sea.nx + x) * sea.nlayers + 1] = 0.8;

            //_Q[(y * sea.nx + x) * sea.nlayers] = 0.0;
            //_Q[(y * sea.nx + x) * sea.nlayers + 1] = 0.1 * exp(-(0.3*pow(sea.xs[x]-5.0, 2) + pow(sea.ys[y]-5.0, 2)) * 2.0);
            //_Q[(y * sea.nx + x) * sea.nlayers + 1] = 0.0;//0.4 * exp(-(pow(sea.xs[x]-5.0, 2) + pow(sea.ys[y]-5.0, 2)) * 2.0);

            _beta[(y * sea.nx + x) * 2] = 0.04 * (sea.ys[y]+5.0);
            _beta[(y * sea.nx + x) * 2 + 1] = 0.0;

            //float r = sqrt(pow(sea.xs[x]-5.0,2) + pow(sea.ys[y]-5.0,2));
            // angular velocity
            //float omega = 0.03;

            // set everything else (Sx, Sy, Q) to be 0
            for (int l = 0; l < sea.nlayers; l++) {
                Sx0[(y * sea.nx + x) * sea.nlayers + l] = 0.0;
                Sy0[(y * sea.nx + x) * sea.nlayers + l] = 0.0;
                _Q[(y * sea.nx + x) * sea.nlayers + l] = 0.0;
                zeta0[(y * sea.nx + x) * sea.nlayers + l] = 1.0;


                // make swirly
                //Sx0[(y * sea.nx + x) * sea.nlayers + l] = -omega * (sea.ys[y]-5.0);
                //Sy0[(y * sea.nx + x) * sea.nlayers + l] = omega * (sea.xs[x]-5.0);

                //if (((sea.xs[x] < 5.5) && (sea.xs[x] > 4.5)) || ((sea.ys[y] < 5.5) && (sea.ys[y] > 4.5))) {
                    //zeta0[(y * sea.nx + x) * sea.nlayers + l] = 1.0;
                //} else {
                    //zeta0[(y * sea.nx + x) * sea.nlayers + l] = 0.0;
                //}


            }
        }
    }


    sea.initial_data(D0, Sx0, Sy0, zeta0, _Q, _beta);

    delete[] D0;
    delete[] Sx0;
    delete[] Sy0;
    delete[] zeta0;
    delete[] _Q;
    delete[] _beta;

    sea.print_inputs();

    // MPI variables

    MPI_Comm comm;
    MPI_Status status;

    int rank, size;//, source, tag;

    // Initialise MPI and compute number of processes and local rank
    comm = MPI_COMM_WORLD;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) printf("Running on %d process(es)\n", size);

    // run simulation
    sea.run(comm, status, rank, size);

    MPI_Finalize();

}
