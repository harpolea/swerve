#include <iostream>
#include <string>
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <algorithm>
#include "SeaCuda.h"

#ifndef H5_NO_NAMESPACE
#ifndef H5_NO_STD
    using std::cout;
    using std::endl;
#endif  // H5_NO_STD
#endif
#include "H5Cpp.h"

using namespace std;

SeaCuda::SeaCuda(int n_layers, int _nx, int _ny, int _nt,
        float xmin, float xmax,
        float ymin, float ymax, float * _rho,
        float * _Q, float _mu,
        float _alpha, float * _beta, float * _gamma,
        bool _periodic, int _dprint)
        : nlayers(n_layers), nx(_nx), ny(_ny), nt(_nt), mu(_mu), alpha(_alpha), periodic(_periodic), dprint(_dprint)
{
    xs = new float[nx-2];
    for (int i = 0; i < (nx - 2); i++) {
        xs[i] = xmin + i * (xmax - xmin) / (nx-2);
    }

    ys = new float[ny-2];
    for (int i = 0; i < (ny - 2); i++) {
        ys[i] = ymin + i * (ymax - ymin) / (ny-2);
    }

    dx = xs[1] - xs[0];
    dy = ys[1] - ys[0];
    dt = 0.1 * min(dx, dy);

    rho = new float[nlayers];
    Q = new float[nlayers];

    for (int i = 0; i < nlayers; i++) {
        rho[i] = _rho[i];
    }

    for (int i = 0; i < nlayers*nx*ny; i++) {
        Q[i] = _Q[i];
    }


    for (int i = 0; i < 2; i++) {
        beta[i] = _beta[i];
        for (int j = 0; j < 2; j++) {
            gamma[i*2+j] = _gamma[i*2+j];
        }
    }

    // find inverse of gamma
    float det = gamma[0] * gamma[1*2+1] - gamma[0*2+1] * gamma[1*2+0];
    //cout << "det = " << det << '\n';
    gamma_up[0] = gamma[1*2+1] / det;
    gamma[0*2+1] = -gamma[0*2+1]/det;
    gamma[1*2+0] = -gamma[1*2+0]/det;
    gamma_up[1*2+1] = gamma[0*2+0]/det;

    //U_grid = new float[nlayers*nx*ny*3 * int(ceil(float((nt+1)/dprint)))];
    U_grid = new float[nlayers*nx*ny*3];

    cout << "Made a Sea.\n";
    //cout << "dt = " << dt << "\tdx = " << dx << "\tdy = " << dy << '\n';

}

SeaCuda::SeaCuda(char * filename)
{
    /*
    Constructor for SeaCuda class using inputs from file.
    */

    // open file
    ifstream inputFile(filename);

    string variableName;
    float value;
    float xmin, xmax, ymin, ymax;

    // read line
    //inputFile >> variableName;

    while (inputFile >> variableName) {

        // need to have nlayers initialised before lots of other stuff, so shall do so here.
        nlayers = 2;

        // mega switch statement of doom
        if (variableName == "nx") {
            inputFile >> value;
            nx = int(value);
        } else if (variableName == "ny") {
            inputFile >> value;
            ny = int(value);
        } else if (variableName == "nlayers") {
            inputFile >> value;
            nlayers = int(value);
            rho = new float[nlayers];
            //Q = new float[nlayers];
        } else if (variableName == "nt") {
            inputFile >> value;
            nt = int(value);
        } else if (variableName == "xmin") {
            inputFile >> xmin;
        } else if (variableName == "xmax") {
            inputFile >> xmax;
        } else if (variableName == "ymin") {
            inputFile >> ymin;
        } else if (variableName == "ymax") {
            inputFile >> ymax;
        } else if (variableName == "rho") {
            for (int i = 0; i < nlayers; i++) {
                inputFile >> rho[i];
            }
        } else if (variableName == "Q") {
            for (int i = 0; i < nlayers; i++) {
                inputFile >> Q[i];
            }
        } else if (variableName == "mu") {
            inputFile >> mu;
        } else if (variableName == "alpha") {
            inputFile >> alpha;
        } else if (variableName == "beta") {
            for (int i = 0; i < 2; i++) {
                inputFile >> beta[i];
            }
        } else if (variableName == "gamma") {
            for (int i = 0; i < 2*2; i++) {
                inputFile >> gamma[i];
            }
        } else if (variableName == "periodic") {
            string tf;
            inputFile >> tf;
            if (tf == "t" || tf == "T") {
                periodic = true;
            } else {
                periodic = false;
            }
        } else if (variableName == "dprint") {
            inputFile >> value;
            dprint = int(value);
        } else if (variableName == "outfile") {
            string f;
            inputFile >> f;
            strncpy(outfile, f.c_str(), sizeof(outfile));
            outfile[sizeof(outfile) - 1] = 0;
        }

    }

    inputFile.close();

    xs = new float[nx-2];
    for (int i = 0; i < (nx - 2); i++) {
        xs[i] = xmin + i * (xmax - xmin) / (nx-2);
    }

    ys = new float[ny-2];
    for (int i = 0; i < (ny - 2); i++) {
        ys[i] = ymin + i * (ymax - ymin) / (ny-2);
    }

    dx = xs[1] - xs[0];
    dy = ys[1] - ys[0];
    dt = 0.1 * min(dx, dy);

    // find inverse of gamma
    float det = gamma[0] * gamma[1*2+1] - gamma[0*2+1] * gamma[1*2+0];
    //cout << "det = " << det << '\n';
    gamma_up[0] = gamma[1*2+1] / det;
    gamma[0*2+1] = -gamma[0*2+1]/det;
    gamma[1*2+0] = -gamma[1*2+0]/det;
    gamma_up[1*2+1] = gamma[0*2+0]/det;

    //U_grid = new float[nlayers*nx*ny*3 * int(ceil(float((nt+1)/dprint)))];
    try {
        Q = new float[int(nlayers*nx*ny)];
        U_grid = new float[int(nlayers*nx*ny*3)];
    } catch (bad_alloc&) {
        cerr << "Could not allocate U_grid - try smaller problem size.\n";
        exit(1);
    }

    cout << "Made a Sea.\n";
    //cout << "dt = " << dt << "\tdx = " << dx << "\tdy = " << dy << '\n';

}

// copy constructor
SeaCuda::SeaCuda(const SeaCuda &seaToCopy)
    : nlayers(seaToCopy.nlayers), nx(seaToCopy.nx), ny(seaToCopy.ny), nt(seaToCopy.nt), dx(seaToCopy.dx), dy(seaToCopy.dy), dt(seaToCopy.dt), mu(seaToCopy.mu), alpha(seaToCopy.alpha), periodic(seaToCopy.periodic), dprint(seaToCopy.dprint)
{

    xs = new float[nx-2];
    for (int i = 0; i < (nx - 2); i++) {
        xs[i] = seaToCopy.xs[i];
    }

    ys = new float[ny-2];
    for (int i = 0; i < (ny - 2); i++) {
        ys[i] = seaToCopy.ys[i];
    }

    rho = new float[nlayers];
    Q = new float[nlayers*nx*ny];

    for (int i = 0; i < nlayers; i++) {
        rho[i] = seaToCopy.rho[i];
    }

    for (int i = 0; i < nlayers*nx*ny; i++) {
        Q[i] = seaToCopy.Q[i];
    }

    U_grid = new float[int(nlayers*nx*ny*3)];// * int(ceil(float((nt+1)/dprint)))];

    for (int i = 0; i < nlayers*nx*ny*3;i++) {// * int(ceil(float((nt+1)/dprint))); i++) {
        U_grid[i] = seaToCopy.U_grid[i];
    }

    for (int i = 0; i < 2; i++) {
        beta[i] = seaToCopy.beta[i];
        for (int j = 0; j < 2; j++) {
            gamma[i*2+j] = seaToCopy.gamma[i*2+j];
            gamma_up[i*2+j] = seaToCopy.gamma_up[i*2+j];
        }
    }

}

// deconstructor
SeaCuda::~SeaCuda() {
    delete[] xs;
    delete[] ys;
    delete[] rho;
    delete[] Q;

    delete[] U_grid;
}


// set the initial data
void SeaCuda::initial_data(float * D0, float * Sx0, float * Sy0, float * _Q) {
    /*
    Initialise D, Sx, Sy and Q.
    */
    for (int i = 0; i < nlayers*nx*ny; i++) {
        U_grid[i*3] = D0[i];
        U_grid[i*3+1] = Sx0[i];
        U_grid[i*3+2] = Sy0[i];
        Q[i] = _Q[i];
    }

    bcs();

    cout << "Set initial data.\n";
}

void SeaCuda::print_inputs() {
    /*
    Print some input and runtime parameters to screen.
    */

    cout << "\nINPUT DATA\n" << "----------\n";
    cout << "(nx, ny, nlayers) \t(" << nx << ',' << ny << ',' << nlayers << ")\n";
    cout << "nt \t\t\t" << nt << '\n';
    cout << "dprint \t\t\t" << dprint << '\n';
    cout << "(dx, dy, dt) \t\t(" << dx << ',' << dy << ',' << dt << ")\n";
    cout << "rho \t\t\t(" << rho[0] << ',' << rho[1] << ")\n";
    cout << "mu \t\t\t" << mu << '\n';
    cout << "alpha \t\t\t" << alpha << '\n';
    cout << "beta \t\t\t(" << beta[0] << ',' << beta[1] << ")\n";
    cout << "gamma \t\t\t((" << gamma[0] << ',' << gamma[1] << "),(" << gamma[2] << ',' << gamma[3] << "))\n";
    cout << "outfile \t\t" << outfile << "\n\n";
}

void SeaCuda::bcs() {
    /*
    Enforce boundary conditions.
    */

    if (periodic) {

        for (int l = 0; l < nlayers; l++) {
            for (int y = 0; y < ny; y++){
                for (int i = 0; i < 3; i++) {
                    U_grid[((y * nx) * nlayers + l)*3+i] = U_grid[((y * nx + (nx-2)) * nlayers + l)*3+i];

                    U_grid[((y * nx + (nx-1)) * nlayers + l)*3+i] = U_grid[((y * nx + 1) * nlayers + l)*3+i];
                }
            }
            for (int x = 0; x < nx; x++){
                for (int i = 0; i < 3; i++) {
                    U_grid[(x * nlayers + l)*3+i] = U_grid[(((ny-2) * nx + x) * nlayers + l)*3+i];

                    U_grid[(((ny-1) * nx + x) * nlayers + l)*3+i] = U_grid[((nx + x) * nlayers + l)*3+i];
                }
            }
        }
    } else { // outflow
        for (int l = 0; l < nlayers; l++) {
            for (int y = 0; y < ny; y++){
                for (int i = 0; i < 3; i++) {
                    U_grid[((y * nx) * nlayers + l)*3+i] = U_grid[((y * nx + 1) * nlayers + l)*3+i];

                    U_grid[((y * nx + (nx-1)) * nlayers + l)*3+i] = U_grid[((y * nx + (nx-2)) * nlayers + l)*3+i];
                }
            }
            for (int x = 0; x < nx; x++){
                for (int i = 0; i < 3; i++) {
                    U_grid[(x * nlayers + l)*3+i] = U_grid[((nx + x) * nlayers + l)*3+i];

                    U_grid[(((ny-1) * nx + x) * nlayers + l)*3+i] = U_grid[(((ny-2) * nx + x) * nlayers + l)*3+i];
                }

            }
        }
    }
}


void SeaCuda::run() {
    /*
    Wrapper for cuda_run function.
    */

    cout << "Beginning evolution.\n";

    cuda_run(beta, gamma_up, U_grid, rho, Q, mu, nx, ny, nlayers, nt,
             alpha, dx, dy, dt, dprint, outfile);
}

// NOTE: this will not work now we don't store everything in U_grid
void SeaCuda::output(char * filename) {
    // open file
    ofstream outFile(filename);

    for (int t = 0; t < (nt+1); t++) {//=dprint) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                for (int l = 0; l < nlayers; l++) {
                    outFile << t << ", " << x << ", " << y << ", " << l;
                    for (int i = 0; i < 3; i++ ) {
                        outFile << ", " << U_grid[(((t * ny + y) * nx + x) * nlayers + l)*3+i];
                    }
                    outFile << '\n';
                }
            }
        }
    }

    outFile.close();
}

void SeaCuda::output_hdf5(char * filename) {
    // create file
    H5::H5File outFile(filename, H5F_ACC_TRUNC);

    hsize_t dims[] = {hsize_t(nt+1), hsize_t(ny), hsize_t(nx), hsize_t(nlayers), 3};

    H5::DataSpace dataspace(5, dims);
    H5::DataSet dataset = outFile.createDataSet("SwerveOutput",
        H5::PredType::NATIVE_FLOAT, dataspace);

    dataset.write(U_grid, H5::PredType::NATIVE_FLOAT);

    outFile.close();

}

void SeaCuda::output() {
    // open file
    output_hdf5(outfile);
}

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
