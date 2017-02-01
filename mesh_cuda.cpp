#include <stdio.h>
#include <cmath>
#include <limits>
#include "Mesh_cuda.h"
#include "mesh_cuda_kernel.h"
#include <iostream>
#include <string.h>
#include <fstream>
#include <algorithm>
#include "mpi.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef H5_NO_NAMESPACE
#ifndef H5_NO_STD
    using std::cout;
    using std::endl;
#endif  // H5_NO_STD
#endif
#include "H5Cpp.h"
using namespace std;

/*
Compile with 'make mesh'

*/

/*
Implement Sea class
*/

Sea::Sea(int _nx, int _ny, int _nz, int _nlayers,
        int _nt, int _ng, int _r, float _df,
        float xmin, float xmax,
        float ymin, float ymax,
        float _zmin, float _zmax, float * _rho, float _p_floor,
        float _Q, float _mu, float _gamma,
        float _alpha, float * _beta, float * _gamma_down,
        bool _periodic, bool _burning, int _dprint)
        : nx(_nx), ny(_ny), nz(_nz), nlayers(_nlayers), ng(_ng), zmin(_zmin), zmax(_zmax), nt(_nt), r(_r), df(_df), p_floor(_p_floor), mu(_mu), gamma(_gamma), alpha(_alpha), periodic(_periodic), burning(_burning), dprint(_dprint)
{
    xs = new float[nx];
    for (int i = 0; i < nx; i++) {
        xs[i] = xmin + (i-ng) * (xmax - xmin) / (nx-2*ng);
    }

    ys = new float[ny];
    for (int i = 0; i < ny; i++) {
        ys[i] = ymin + (i-ng) * (ymax - ymin) / (ny-2*ng);
    }

    dx = xs[1] - xs[0];
    dy = ys[1] - ys[0];
    dz = (zmax - _zmin) / (nz - 1.0);
    dt = 0.1 * min(dx, min(dy, dz));

    rho = new float[nlayers];
    for (int i = 0; i < nlayers; i++) {
        rho[i] = _rho[i];
    }
    Q = _Q;

    for (int i = 0; i < 3; i++) {
        beta[i] = _beta[i];
        for (int j = 0; j < 3; j++) {
            gamma_down[i*3+j] = _gamma_down[i*3+j];
        }
    }

    // find inverse of gamma
    for (int i = 0; i < 3*3; i++) {
        gamma_up[i] = gamma_down[i];
    }
    Sea::invert_mat(gamma_up, 3, 3);

    //float det = gamma_down[0] * gamma_down[1*3+1] - gamma_down[0*3+1] * gamma_down[1*3+0];
    //gamma_up[0] = gamma_down[1*3+1] / det;
    //gamma_up[0*3+1] = -gamma_down[0*3+1]/det;
    //gamma_up[1*3+0] = -gamma_down[1*3+0]/det;
    //gamma_up[1*3+1] = gamma_down[0*3+0]/det;

    nxf = int(r * df * nx);
    nyf = int(r * df * ny);

    // D, Sx, Sy, zeta
    U_coarse = new float[nx*ny*nlayers*5];
    U_fine = new float[nxf*nyf*nz*5];

    matching_indices[0] = int(ceil(nx*0.5*(1-df)));
    matching_indices[1] = int(ceil(nx*0.5*(1+df)));
    matching_indices[2] = int(ceil(ny*0.5*(1-df)));
    matching_indices[3] = int(ceil(ny*0.5*(1+df)));

    cout << "Matching indices: " << matching_indices[0] << ',' << matching_indices[1] << ',' << matching_indices[2] << ',' << matching_indices[3] << '\n';

    cout << "Made a Sea.\n";
}

void Sea::invert_mat(float * M, int m, int n) {
    // invert the m x n matrix A in place using Gaussian elimination
    float * B = new float[m*n*2];
    // initialise augmented matrix
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            B[i*2*n+j] = M[i*n+j];
            B[i*2*n+ n+j] = 0.0;
        }
        B[i*2*n+n+i] = 1.0;
    }

    for (int k = 0; k < min(m,n); k++) {
        // i_max  := argmax (i = k ... m, abs(A[i, k]))
        int i_max = k;
        for (int i = k+1; i < m; i++) {
            if (abs(B[i*n*2+k]) > abs(B[i_max*n*2+k])) {
                i_max = i;
            }
        }
        if (abs(B[i_max*n*2+k]) < 1.0e-12) {
            cout << "Matrix is singular!\n";
        }
        // swap rows(k, i_max)
        for (int i = 0; i < 2*n; i++) {
            float temp = B[k*n*2+i];
            B[k*n*2+i] = B[i_max*n*2+i];
            B[i_max*n*2+i] = temp;
        }


        for (int i = k+1; i < m; i++) {
            float f = B[i*n*2+k] / B[k*n*2+k];
            for (int j = k+1; j < n*2; j++) {
                B[i*n*2+j] -= B[k*n*2+j] * f;
            }
            B[i*n*2+k] = 0.0;
        }
    }

    // back substitution
    for (int k = 0; k < m; k++) {
        for (int i = k+1; i < n; i++) {
            float f = B[k*n*2+i] / B[i*2*n+i];
            for (int j = k+1; j < 2*n; j++) {
                B[k*n*2+j] -= B[i*n*2+j] * f;
            }
        }
        for (int i = k+1; i < 2*n; i++) {
            B[k*n*2+i] /= B[k*n*2+k];
        }
        B[k*n*2+k] = 1.0;
    }

    // put answer back in M
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            M[i*n+j] = B[i*2*n+n+j];

        }
    }

    delete[] B;
}

Sea::Sea(char * filename)
{
    /*
    Constructor for Sea class using inputs from file.
    */

    // open file
    ifstream inputFile(filename);

    string variableName;
    float value;
    float xmin, xmax, ymin, ymax;

    while (inputFile >> variableName) {

        // mega if/else statement of doom
        if (variableName == "nx") {
            inputFile >> value;
            nx = int(value);
        } else if (variableName == "ny") {
            inputFile >> value;
            ny = int(value);
        } else if (variableName == "nz") {
            inputFile >> value;
            nz = int(value);
        } else if (variableName == "nlayers") {
            inputFile >> value;
            nlayers = int(value);
            rho = new float[nlayers];
        } else if (variableName == "ng") {
            inputFile >> value;
            ng = int(value);
        } else if (variableName == "nt") {
            inputFile >> value;
            nt = int(value);
        } else if (variableName == "r") {
            inputFile >> value;
            r = int(value);
        } else if (variableName == "df") {
            inputFile >> df;
        } else if (variableName == "xmin") {
            inputFile >> xmin;
        } else if (variableName == "xmax") {
            inputFile >> xmax;
        } else if (variableName == "ymin") {
            inputFile >> ymin;
        } else if (variableName == "ymax") {
            inputFile >> ymax;
        } else if (variableName == "zmin") {
            inputFile >> zmin;
        } else if (variableName == "zmax") {
            inputFile >> zmax;
        } else if (variableName == "rho") {
            for (int i = 0; i < nlayers; i++) {
                inputFile >> rho[i];
            }
        } else if (variableName == "p_floor") {
            inputFile >> p_floor;
        } else if (variableName == "Q") {
            inputFile >> Q;
        } else if (variableName == "mu") {
            inputFile >> mu;
        } else if (variableName == "gamma") {
            inputFile >> gamma;
        } else if (variableName == "alpha") {
            inputFile >> alpha;
        } else if (variableName == "beta") {
            for (int i = 0; i < 3; i++) {
                inputFile >> beta[i];
            }
        } else if (variableName == "gamma_down") {
            for (int i = 0; i < 3*3; i++) {
                inputFile >> gamma_down[i];
            }
        } else if (variableName == "periodic") {
            string tf;
            inputFile >> tf;
            if (tf == "t" || tf == "T") {
                periodic = true;
            } else {
                periodic = false;
            }
        } else if (variableName == "burning") {
            string tf;
            inputFile >> tf;
            if (tf == "t" || tf == "T") {
                burning = true;
            } else {
                burning = false;
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

    nxf = int(r * df * nx);
    nyf = int(r * df * ny);

    inputFile.close();

    xs = new float[nx];
    for (int i = 0; i < nx; i++) {
        xs[i] = xmin + (i-ng) * (xmax - xmin) / (nx-2*ng);
    }

    ys = new float[ny];
    for (int i = 0; i < ny; i++) {
        ys[i] = ymin + (i-ng) * (ymax - ymin) / (ny-2*ng);
    }

    dx = xs[1] - xs[0];
    dy = ys[1] - ys[0];
    dz = (zmax - zmin) / (nz - 1.0);
    dt = 0.1 * min(dx, min(dy, dz));

    // find inverse of gamma
    //float det = gamma_down[0] * gamma_down[1*2+1] -
                //gamma_down[0*2+1] * gamma_down[1*2+0];
    //gamma_up[0] = gamma_down[1*2+1] / det;
    //gamma_up[0*2+1] = -gamma_down[0*2+1]/det;
    //gamma_up[1*2+0] = -gamma_down[1*2+0]/det;
    //gamma_up[1*2+1] = gamma_down[0*2+0]/det;
    for (int i = 0; i < 3*3; i++) {
        gamma_up[i] = gamma_down[i];
    }
    Sea::invert_mat(gamma_up, 3, 3);

    U_coarse = new float[nx*ny*nlayers*5];
    U_fine = new float[nxf*nyf*nz*5];

    // initialise arrays
    for (int i = 0; i < nx*ny*nlayers*5; i++) {
        U_coarse[i] = 0.0;
    }
    for (int i = 0; i < nxf*nyf*nz*5; i++) {
        U_fine[i] = 0.0;
    }

    matching_indices[0] = int(ceil(nx*0.5*(1-df)));
    matching_indices[1] = int(floor(nx*0.5*(1+df)));
    matching_indices[2] = int(ceil(ny*0.5*(1-df)));
    matching_indices[3] = int(floor(ny*0.5*(1+df)));

    cout << "Matching indices: " << matching_indices[0] << ',' << matching_indices[1] << ',' << matching_indices[2] << ',' << matching_indices[3] << '\n';


    cout << "matching_indices vs nxf: " <<
        matching_indices[1] - matching_indices[0] << ',' << nxf << '\n';
    cout << "Made a Sea.\n";
}

// copy constructor
Sea::Sea(const Sea &seaToCopy)
    : nx(seaToCopy.nx), ny(seaToCopy.ny), nz(seaToCopy.nz), nlayers(seaToCopy.nlayers), ng(seaToCopy.ng), zmin(seaToCopy.zmin), zmax(seaToCopy.zmax), nt(seaToCopy.nt), r(seaToCopy.r), nxf(seaToCopy.nxf), nyf(seaToCopy.nyf), dx(seaToCopy.dx), dy(seaToCopy.dy), dz(seaToCopy.dz), dt(seaToCopy.dt), df(seaToCopy.df), p_floor(seaToCopy.p_floor), mu(seaToCopy.mu), gamma(seaToCopy.gamma), alpha(seaToCopy.alpha), periodic(seaToCopy.periodic), burning(seaToCopy.burning), dprint(seaToCopy.dprint)
{

    xs = new float[nx];
    for (int i = 0; i < nx; i++) {
        xs[i] = seaToCopy.xs[i];
    }

    ys = new float[ny];
    for (int i = 0; i < ny; i++) {
        ys[i] = seaToCopy.ys[i];
    }

    rho = new float[nlayers];
    for (int i = 0; i < nlayers; i++) {
        rho[i] = seaToCopy.rho[i];
    }

    Q = seaToCopy.Q;

    for (int i = 0; i < 3*nx*ny; i++) {
        beta[i] = seaToCopy.beta[i];
    }

    U_coarse = new float[int(nx*ny*nlayers*5)];
    U_fine = new float[nxf*nyf*nz*5];

    for (int i = 0; i < nx*ny*nlayers*5;i++) {
        U_coarse[i] = seaToCopy.U_coarse[i];
    }

    for (int i = 0; i < nxf*nyf*nz*5;i++) {
        U_fine[i] = seaToCopy.U_fine[i];
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            gamma_down[i*3+j] = seaToCopy.gamma_down[i*3+j];
            gamma_up[i*3+j] = seaToCopy.gamma_up[i*3+j];
        }
    }

    for (int i = 0; i < 2*2; i++) {
        matching_indices[i] = seaToCopy.matching_indices[i];
    }
}

// deconstructor
Sea::~Sea() {
    delete[] xs;
    delete[] ys;
    delete[] rho;

    delete[] U_coarse;
    delete[] U_fine;
}

// set the initial data
void Sea::initial_data(float * D0, float * Sx0, float * Sy0, float * Sz0, float * tau) {
    /*
    Initialise D, Sx, Sy and Q.
    */
    for (int i = 0; i < nx*ny*nlayers; i++) {
        U_coarse[i*5] = D0[i];
        U_coarse[i*5+1] = Sx0[i];
        U_coarse[i*5+2] = Sy0[i];
        U_coarse[i*5+3] = Sz0[i];
        U_coarse[i*5+4] = tau[i];
    }

    bcs(U_coarse, nx, ny, nlayers, 5);

    cout << "Set initial data.\n";
}

void Sea::print_inputs() {
    /*
    Print some input and runtime parameters to screen.
    */

    cout << "\nINPUT DATA\n" << "----------\n";
    cout << "(nx, ny, nlayers, ng) \t(" << nx << ',' << ny << ',' << nlayers << ',' << ng << ")\n";
    cout << "nt \t\t\t" << nt << '\n';
    cout << "(nxf, nyf, nz, r, df) \t(" << nxf << ',' << nyf << ',' << nz << ',' << r << ',' << df << ")\n";
    cout << "dprint \t\t\t" << dprint << '\n';
    cout << "(dx, dy, dz, dt) \t(" << dx << ',' << dy << ',' << dz << ',' << dt << ")\n";
    cout << "rho \t\t\t" << rho[0] << ',' << rho[1]<< "\n";
    cout << "mu \t\t\t" << mu << '\n';
    cout << "alpha \t\t\t" << alpha << '\n';
    cout << "beta \t\t\t(" << beta[0] << ',' << beta[1] << ',' << beta[2] << ")\n";
    cout << "gamma_down \t\t((" << gamma_down[0] << ',' << gamma_down[1] << ',' << gamma_down[2] << "),(" << gamma_down[3] << ',' << gamma_down[4] << ',' << gamma_down[5] << "),(" << gamma_down[6] << ',' << gamma_down[7] << ',' << gamma_down[8] << "))\n";
    cout << "burning \t\t" << burning << '\n';
    cout << "outfile \t\t" << outfile << "\n\n";
}

void Sea::bcs(float * grid, int n_x, int n_y, int n_z, int vec_dim) {
    /*
    Enforce boundary conditions on grid of quantities with dimension vec_dim.
    */

    if (periodic) {
        for (int z = 0; z < n_z; z++) {
            for (int y = 0; y < n_y; y++){
                for (int g = 0; g < ng; g++) {
                    for (int l = 0; l < vec_dim; l++) {
                        grid[((z * n_z + y) * n_x + g) * vec_dim + l] =
                            grid[((z * n_z + y) * n_x + (n_x-2*ng+g)) * vec_dim + l];

                        grid[((z * n_z + y) * n_x + (n_x-ng+g)) * vec_dim + l] =
                            grid[((z * n_z + y) * n_x + ng+g) * vec_dim + l];
                    }
                }
            }
            for (int g = 0; g < ng; g++) {
                for (int x = 0; x < n_x; x++){
                    for (int l = 0; l < vec_dim; l++) {
                        grid[((z * n_z + g) * n_x + x) * vec_dim + l] =
                            grid[((z * n_z + n_y-ng-1) * n_x + x) * vec_dim + l];

                        grid[((z * n_z + n_y-ng+g) * n_x + x) * vec_dim + l] =
                            grid[((z * n_z + ng) * n_x + x) * vec_dim + l];
                    }
                }
            }
        }
    } else { // outflow
        for (int z = 0; z < n_z; z++) {
            for (int y = 0; y < n_y; y++){
                for (int g = 0; g < ng; g++) {
                    for (int l = 0; l < vec_dim; l++) {
                        grid[((z * n_y + y) * n_x + g) * vec_dim + l] =
                            grid[((z * n_y + y) * n_x + ng) * vec_dim + l];

                        grid[((z * n_y + y) * n_x + (n_x-1-g)) * vec_dim + l] =
                            grid[((z * n_y + y) * n_x + (n_x-1-ng)) * vec_dim + l];
                    }
                }
            }
            for (int g = 0; g < ng; g++) {
                for (int x = 0; x < n_x; x++){
                    for (int l = 0; l < vec_dim; l++) {
                        grid[((z * n_y + g) * n_x + x) * vec_dim + l] =
                            grid[((z * n_y + ng) * n_x + x) * vec_dim + l];

                        grid[((z * n_y + n_y-1-g) * n_x + x) * vec_dim + l] =
                            grid[((z * n_y + n_y-1-ng) * n_x + x) * vec_dim + l];
                    }
                }
            }
        }
    }
}

void Sea::run(MPI_Comm comm, MPI_Status * status, int rank, int size) {
    /*
    run code
    */
    cuda_run(beta, gamma_up, U_coarse, U_fine, rho, p_floor, mu,
             nx, ny, nlayers, nxf, nyf, nz, ng, nt,
             alpha, gamma, zmin, zmax, dx, dy, dz, dt, burning, dprint, outfile, comm, *status, rank, size, matching_indices);
}
