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

#ifndef H5_NO_NAMESPACE
#ifndef H5_NO_STD
    using std::cout;
    using std::endl;
#endif  // H5_NO_STD
#endif
#include "H5Cpp.h"
using namespace std;

/*
Compile with

g++ mesh_cuda.cpp -I/usr/include/hdf5/serial -I/usr/include/hdf5 -lhdf5_cpp -lhdf5 -L/usr/lib/x86_64-linux-gnu/hdf5/serial -o mesh

*/

bool nan_check(float a) {
    // check to see whether float a is a nan
    if (a != a) {
        return true;
    } else {
        return false;
    }
}



/*
Implement Sea class
*/

Sea::Sea(int _nx, int _ny, int _nt, int _ng, int _r, float _df,
        float xmin, float xmax,
        float ymin, float ymax, float  _rho,
        float _Q, float _mu, float _gamma,
        float _alpha, float * _beta, float * _gamma_down,
        bool _periodic, bool _burning, int _dprint)
        : nx(_nx), ny(_ny), ng(_ng), nt(_nt), r(_r), df(_df), mu(_mu), gamma(_gamma), alpha(_alpha), periodic(_periodic), burning(_burning), dprint(_dprint)
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
    dt = 0.1 * min(dx, dy);

    rho = _rho;
    Q = _Q;

    for (int i = 0; i < 2; i++) {
        beta[i] = _beta[i];
        for (int j = 0; j < 2; j++) {
            gamma_down[i*2+j] = _gamma_down[i*2+j];
        }
    }

    // find inverse of gamma
    float det = gamma_down[0] * gamma_down[1*2+1] - gamma_down[0*2+1] * gamma_down[1*2+0];
    gamma_up[0] = gamma_down[1*2+1] / det;
    gamma_up[0*2+1] = -gamma_down[0*2+1]/det;
    gamma_up[1*2+0] = -gamma_down[1*2+0]/det;
    gamma_up[1*2+1] = gamma_down[0*2+0]/det;

    nxf = int(r * df * nx);
    nyf = int(r * df * ny);

    // D, Sx, Sy, zeta
    U_coarse = new float[nx*ny*3];
    U_fine = new float[nxf*nyf*4];

    matching_indices[0] = int(ceil(nx*0.5*(1-df)));
    matching_indices[1] = int(ceil(nx*0.5*(1+df)));
    matching_indices[2] = int(ceil(ny*0.5*(1-df)));
    matching_indices[3] = int(ceil(ny*0.5*(1+df)));


    cout << "Made a Sea.\n";
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
        } else if (variableName == "rho") {
            inputFile >> rho;
        } else if (variableName == "Q") {
            inputFile >> Q;
        } else if (variableName == "mu") {
            inputFile >> mu;
        } else if (variableName == "gamma") {
            inputFile >> gamma;
        } else if (variableName == "alpha") {
            inputFile >> alpha;
        } else if (variableName == "beta") {
            for (int i = 0; i < 2; i++) {
                inputFile >> beta[i];
            }
        } else if (variableName == "gamma_down") {
            for (int i = 0; i < 2*2; i++) {
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
    dt = 0.1 * min(dx, dy);

    // find inverse of gamma
    float det = gamma_down[0] * gamma_down[1*2+1] -
                gamma_down[0*2+1] * gamma_down[1*2+0];
    gamma_up[0] = gamma_down[1*2+1] / det;
    gamma_up[0*2+1] = -gamma_down[0*2+1]/det;
    gamma_up[1*2+0] = -gamma_down[1*2+0]/det;
    gamma_up[1*2+1] = gamma_down[0*2+0]/det;

    cout << "gamma_up: " << gamma_up[0] << ',' << gamma_up[1] << ',' <<
        gamma_up[2] << ',' << gamma_up[3] << '\n';

    try {
        U_coarse = new float[int(nx*ny*3)];
        U_fine = new float[int(nxf*nyf*4)];
        //beta = new float[int(2*nx*ny)];
    } catch (bad_alloc&) {
        cerr << "Could not allocate U_grid - try smaller problem size.\n";
        exit(1);
    }

    // initialise arrays
    for (int i = 0; i < nx*ny*3; i++) {
        U_coarse[i] = 0.0;
    }
    for (int i = 0; i < nxf*nyf*4; i++) {
        U_fine[i] = 0.0;
    }

    matching_indices[0] = int(ceil(nx*0.5*(1-df)));
    matching_indices[1] = int(ceil(nx*0.5*(1+df)));
    matching_indices[2] = int(ceil(ny*0.5*(1-df)));
    matching_indices[3] = int(ceil(ny*0.5*(1+df)));

    cout << "matching_indices vs nxf: " <<
        matching_indices[1] - matching_indices[0] << ',' << nxf << '\n';
    cout << "Made a Sea.\n";
}

// copy constructor
Sea::Sea(const Sea &seaToCopy)
    : nx(seaToCopy.nx), ny(seaToCopy.ny), ng(seaToCopy.ng), nt(seaToCopy.nt), r(seaToCopy.r), nxf(seaToCopy.nxf), nyf(seaToCopy.nyf), dx(seaToCopy.dx), dy(seaToCopy.dy), dt(seaToCopy.dt), df(seaToCopy.df), mu(seaToCopy.mu), gamma(seaToCopy.gamma), alpha(seaToCopy.alpha), periodic(seaToCopy.periodic), burning(seaToCopy.burning), dprint(seaToCopy.dprint)
{

    xs = new float[nx];
    for (int i = 0; i < nx; i++) {
        xs[i] = seaToCopy.xs[i];
    }

    ys = new float[ny];
    for (int i = 0; i < ny; i++) {
        ys[i] = seaToCopy.ys[i];
    }

    //beta = new float[2*nx*ny];

    rho = seaToCopy.rho;

    Q = seaToCopy.Q;

    for (int i = 0; i < 2*nx*ny; i++) {
        beta[i] = seaToCopy.beta[i];
    }

    U_coarse = new float[int(nx*ny*3)];
    U_fine = new float[nxf*nyf*4];

    for (int i = 0; i < nx*ny*3;i++) {
        U_coarse[i] = seaToCopy.U_coarse[i];
    }

    for (int i = 0; i < nxf*nyf*4;i++) {
        U_fine[i] = seaToCopy.U_fine[i];
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            gamma_down[i*2+j] = seaToCopy.gamma_down[i*2+j];
            gamma_up[i*2+j] = seaToCopy.gamma_up[i*2+j];
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
    //delete[] beta;

    delete[] U_coarse;
    delete[] U_fine;
}


// set the initial data
void Sea::initial_data(float * D0, float * Sx0, float * Sy0) {
    /*
    Initialise D, Sx, Sy and Q.
    */
    for (int i = 0; i < nx*ny; i++) {
        U_coarse[i*3] = D0[i];
        U_coarse[i*3+1] = Sx0[i];
        U_coarse[i*3+2] = Sy0[i];
    }

    bcs(U_coarse, nx, ny, 3);

    cout << "Set initial data.\n";
}

void Sea::print_inputs() {
    /*
    Print some input and runtime parameters to screen.
    */

    cout << "\nINPUT DATA\n" << "----------\n";
    cout << "(nx, ny, ng) \t\t(" << nx << ',' << ny << ',' << ng << ")\n";
    cout << "nt \t\t\t" << nt << '\n';
    cout << "dprint \t\t\t" << dprint << '\n';
    cout << "(dx, dy, dt) \t\t(" << dx << ',' << dy << ',' << dt << ")\n";
    cout << "rho \t\t\t" << rho << "\n";
    cout << "mu \t\t\t" << mu << '\n';
    cout << "alpha \t\t\t" << alpha << '\n';
    cout << "beta \t\t\t(" << beta[0] << ',' << beta[1] << ")\n";
    cout << "gamma_down \t\t((" << gamma_down[0] << ',' << gamma_down[1] << "),(" << gamma_down[2] << ',' << gamma_down[3] << "))\n";
    cout << "burning \t\t" << burning << '\n';
    cout << "outfile \t\t" << outfile << "\n\n";
}

void Sea::bcs(float * grid, int n_x, int n_y, int vec_dim) {
    /*
    Enforce boundary conditions on grid of quantities with dimension vec_dim.
    */

    if (periodic) {
        for (int l = 0; l < vec_dim; l++) {
            for (int y = 0; y < n_y; y++){
                for (int g = 0; g < ng; g++) {
                    grid[(y * n_x + g) * vec_dim + l] =
                        grid[(y * n_x + (n_x-2*ng+g)) * vec_dim + l];

                    grid[(y * n_x + (n_x-ng+g)) * vec_dim + l] =
                        grid[(y * n_x + ng+g) * vec_dim + l];
                }
            }
            for (int x = 0; x < n_x; x++){
                for (int g = 0; g < ng; g++) {
                    grid[(g * n_x + x) * vec_dim + l] =
                        grid[((n_y-ng-1) * n_x + x) * vec_dim + l];

                    grid[((n_y-ng+g) * n_x + x) * vec_dim + l] =
                        grid[(ng * n_x + x) * vec_dim + l];
                }
            }
        }
    } else { // outflow
        for (int l = 0; l < vec_dim; l++) {
            for (int y = 0; y < n_y; y++){
                for (int g = 0; g < ng; g++) {
                    grid[(y * n_x + g) * vec_dim + l] =
                        grid[(y * n_x + ng) * vec_dim + l];

                    grid[(y * n_x + (n_x-1-g)) * vec_dim + l] =
                        grid[(y * n_x + (n_x-1-ng)) * vec_dim + l];
                }
            }
            for (int x = 0; x < n_x; x++){
                for (int g = 0; g < ng; g++) {
                    grid[(g * n_x + x) * vec_dim + l] =
                        grid[(ng * n_x + x) * vec_dim + l];

                    grid[((n_y-1-g) * n_x + x) * vec_dim + l] =
                        grid[((n_y-1-ng) * n_x + x) * vec_dim + l];
                }
            }
        }
    }
}

float Sea::phi(float r) {
    // calculate superbee slope limiter Phi(r)
    float ph = 0.0;
    if (r >= 1.0) {
        ph = min(float(2.0), min(r, float(2.0 / (1.0 + r))));
    } else if (r >= 0.5) {
        ph = 1.0;
    } else if (r > 0.0) {
        ph = 2.0 * r;
    }
    return ph;
}

float Sea::rhoh_from_p(float p) {
    // calculate rhoh using p for gamma law equation of state
    return rho + gamma * p / (gamma - 1.0);
}

float Sea::p_from_rhoh(float rhoh) {
    // calculate p using rhoh for gamma law equation of state
    return (rhoh - rho) * (gamma - 1.0) / gamma;
}

float Sea::phi_from_p(float p) {
    // calculate the metric potential Phi given p for gamma law equation of
    // state
    return 1.0 + (gamma - 1.0) / gamma *
        log(1.0 + gamma * p / ((gamma - 1.0) * rho));
}

void Sea::prolong_grid(float * q_c, float * q_f) {
    // prolong coarse grid to fine one
    float * qc_comp = new float[int(nx*ny*4)];
    float * Sx = new float[int(nx*ny*4)];
    float * Sy = new float[int(nx*ny*4)];
    float * p = new float[int(nx*ny)];

    p_from_swe(q_c, p);

    // first calculate the compressible conserved variables on the coarse grid
    for (int i = 0; i < nx*ny; i++) {
        float rhoh = rhoh_from_p(p[i]);
        float W = sqrt((q_c[i*3+1] * q_c[i*3+1] * gamma_up[0] +
                2.0 * q_c[i*3+1] * q_c[i*3+2] * gamma_up[1] +
                q_c[i*3+2] * q_c[i*3+2] * gamma_up[3]) /
                (q_c[i*3] * q_c[i*3]) + 1.0);

        qc_comp[i*4] = rho * W;
        qc_comp[i*4+1] = rhoh * W * q_c[i*3+1] / q_c[i*3];
        qc_comp[i*4+2] = rhoh * W * q_c[i*3+2] / q_c[i*3];
        qc_comp[i*4+3] = rhoh*W*W - p[i] - qc_comp[i*4];

        // NOTE: hack?
        if (qc_comp[i*4+3] < 0.0) qc_comp[i*4+3] = 0.0;
    }

    /*cout << "compressible coarse grid: \n";
    for (int j = 0; j < ny; j++) {
        for(int i = 0; i < nx; i++) {
            //cout << p[j*nx + i] <<' ';
            cout << qc_comp[(j*nx + i)*4] << ' ';
        }
        cout << '\n';
    }
    cout << '\n';*/

    // do some slope limiting
    for (int j = matching_indices[2]; j < matching_indices[3]+1; j++) {
        for (int i = matching_indices[0]; i < matching_indices[1]+1; i++) {
            for (int n = 0; n < 4; n++) {

                // x-dir
                float S_upwind = (qc_comp[(j * nx + i+1) * 4 + n] -
                    qc_comp[(j * nx + i) * 4 + n]) / dx;
                float S_downwind = (qc_comp[(j * nx + i) * 4 + n] -
                    qc_comp[(j * nx + i-1) * 4 + n]) / dx;

                Sx[(j * nx + i) * 4 + n] = 0.5 * (S_upwind + S_downwind);

                float r = 1.0e6;
                if (abs(S_downwind) > 1.0e-10) {
                    r = S_upwind / S_downwind;
                }

                Sx[(j * nx + i) * 4 + n] *= phi(r);

                // y-dir
                S_upwind = (qc_comp[((j+1) * nx + i) * 4 + n] -
                    qc_comp[(j * nx + i) * 4 + n]) / dy;
                S_downwind = (qc_comp[(j * nx + i) * 4 + n] -
                    qc_comp[((j-1) * nx + i) * 4 + n]) / dy;

                Sy[(j * nx + i) * 4 + n] = 0.5 * (S_upwind + S_downwind);

                r = 1.0e6;
                if (abs(S_downwind) > 1.0e-10) {
                    r = S_upwind / S_downwind;
                }

                Sy[(j * nx + i) * 4 + n] *= phi(r);
            }
        }
    }

    // reconstruct values at fine grid cell centres
    for (int j = 0; j < matching_indices[3] - matching_indices[2]+1; j++) {
        for (int i = 0; i < matching_indices[1] - matching_indices[0]+1; i++) {
            for (int n = 0; n < 4; n++) {
                int coarse_index = ((j + matching_indices[2]) * nx + i +
                    matching_indices[0]) * 4 + n;

                q_f[(2*j * nxf + 2*i) * 4 + n] = qc_comp[coarse_index] -
                    0.25 * (dx * Sx[coarse_index] + dy * Sy[coarse_index]);

                q_f[(2*j * nxf + 2*i+1) * 4 + n] = qc_comp[coarse_index] +
                    0.25 * (dx * Sx[coarse_index] - dy * Sy[coarse_index]);

                q_f[((2*j+1) * nxf + 2*i) * 4 + n] = qc_comp[coarse_index] +
                    0.25 * (-dx * Sx[coarse_index] + dy * Sy[coarse_index]);

                q_f[((2*j+1) * nxf + 2*i+1) * 4 + n] = qc_comp[coarse_index] +
                    0.25 * (dx * Sx[coarse_index] + dy * Sy[coarse_index]);
            }
        }
    }

    delete[] qc_comp;
    delete[] Sx;
    delete[] Sy;
    delete[] p;
}

void Sea::restrict_grid(float * q_c, float * q_f) {
    // restrict fine grid to coarse grid

    float * q_prim = new float[nxf*nyf*4];
    float * qf_sw = new float[nxf*nyf*3];

    // find primitive variables
    cons_to_prim_comp(q_f, q_prim, nxf, nyf, gamma, gamma_up);

    // calculate SWE conserved variables on fine grid
    for (int i = 0; i < nxf*nyf; i++) {
        float p = p_from_rho_eps(q_prim[i*4], q_prim[i*4+3], gamma);
        float phi = phi_from_p(p);

        float u = q_prim[i*4+1];
        float v = q_prim[i*4+2];

        float W = 1.0 / sqrt(1.0 -
                u*u*gamma_up[0] - 2.0 * u*v * gamma_up[1] - v*v*gamma_up[3]);

        qf_sw[i*3] = phi * W;
        qf_sw[i*3+1] = phi * W * W * u;
        qf_sw[i*3+2] = phi * W * W * v;
    }

    // interpolate fine grid to coarse grid
    for (int j = 1; j < matching_indices[3] - matching_indices[2]; j++) {
        for (int i = 1; i < matching_indices[1] - matching_indices[0]; i++) {
            for (int n = 0; n < 3; n++) {
                q_c[((j+matching_indices[2]) * nx +
                      i+matching_indices[0]) * 3+n] =
                      0.25 * (qf_sw[(j*2 * nxf + i*2) * 3 + n] +
                              qf_sw[(j*2 * nxf + i*2+1) * 3 + n] +
                              qf_sw[((j*2+1) * nxf + i*2) * 3 + n] +
                              qf_sw[((j*2+1) * nxf + i*2+1) * 3 + n]);
            }
        }
    }

    delete[] q_prim;
    delete[] qf_sw;
}

void Sea::p_from_swe(float * q, float * p) {
    // calculate p using SWE conserved variables

    // only use on coarse grid
    for (int i = 0; i < nx*ny; i++) {
        float W = sqrt((q[i*3+1]*q[i*3+1] * gamma_up[0] +
                2.0 * q[i*3+1] * q[i*3+2] * gamma_up[1] +
                q[i*3+2] * q[i*3+2] * gamma_up[3]) / (q[i*3]*q[i*3]) + 1.0);

        float ph = q[i*3] / W;

        p[i] = rho * (gamma - 1.0) * (exp(gamma * (ph - 1.0) /
            (gamma - 1.0)) - 1.0) / gamma;
    }
}


void Sea::evolve(float * q, int n_x, int n_y, int vec_dim, float * F,
                 flux_func_ptr flux_func, float d_x, float d_y) {
    // find Lax-Friedrichs flux using finite volume methods

    int grid_size = n_x * n_y * vec_dim;
    float * qx_p = new float[grid_size];
    float * qx_m = new float[grid_size];
    float * qy_p = new float[grid_size];
    float * qy_m = new float[grid_size];
    float * fx_p = new float[grid_size];
    float * fx_m = new float[grid_size];
    float * fy_p = new float[grid_size];
    float * fy_m = new float[grid_size];

    for (int j = 1; j < n_y-1; j++) {
        for (int i = 1; i < n_x-1; i++) {
            for (int n = 0; n < vec_dim; n++) {
                // x-dir
                float S_upwind = (q[(j * n_x + i+1) * vec_dim + n] -
                    q[(j * n_x + i) * vec_dim + n]);
                float S_downwind = (q[(j * n_x + i) * vec_dim + n] -
                    q[(j * n_x + i-1) * vec_dim + n]);
                float r = 1.0e6;
                if (S_downwind > 1.0e-7) {
                    r = S_upwind / S_downwind;
                }
                float S = 0.5 * (S_upwind + S_downwind);
                S *= phi(r);

                qx_p[(j * n_x + i) * vec_dim + n] =
                    q[(j * n_x + i) * vec_dim + n] + S * 0.5 * d_x;
                qx_m[(j * n_x + i) * vec_dim + n] =
                    q[(j * n_x + i) * vec_dim + n] - S * 0.5 * d_x;

                // y-dir
                S_upwind = (q[((j+1) * n_x + i) * vec_dim + n] -
                    q[(j * n_x + i) * vec_dim + n]);
                S_downwind = (q[(j * n_x + i) * vec_dim + n] -
                    q[((j-1) * n_x + i) * vec_dim + n]);
                r = 1.0e6;
                if (S_downwind > 1.0e-7) {
                    r = S_upwind / S_downwind;
                }
                S = 0.5 * (S_upwind + S_downwind);
                S *= phi(r);

                qy_p[(j * n_x + i) * vec_dim + n] =
                    q[(j * n_x + i) * vec_dim + n] + S * 0.5 * d_y;
                qy_m[(j * n_x + i) * vec_dim + n] =
                    q[(j * n_x + i) * vec_dim + n] - S * 0.5 * d_y;
            }
        }
    }

    bcs(qx_p, n_x, n_y, vec_dim);
    bcs(qx_m, n_x, n_y, vec_dim);
    bcs(qy_p, n_x, n_y, vec_dim);
    bcs(qy_m, n_x, n_y, vec_dim);

    // calculate fluxes at cell boundaries
    flux_func(qx_p, fx_p, true, n_x, n_y, gamma_up, alpha, beta, gamma);
    flux_func(qx_m, fx_m, true, n_x, n_y, gamma_up, alpha, beta, gamma);
    flux_func(qy_p, fy_p, false, n_x, n_y, gamma_up, alpha, beta, gamma);
    flux_func(qy_m, fy_m, false, n_x, n_y, gamma_up, alpha, beta, gamma);

    float a = 0.1 * min(d_x, d_y) / dt;

    // Lax-Friedrichs flux

    for (int j = 2; j < n_y-2; j++) {
        for (int i = 2; i < n_x-2; i++) {
            for (int n = 0; n < vec_dim; n++) {
                float Fx_m = 0.5 * (
                    fx_p[(j * n_x + i-1) * vec_dim + n] +
                    fx_m[(j * n_x + i) * vec_dim + n] + a *
                    (qx_p[(j * n_x + i-1) * vec_dim + n] -
                    qx_m[(j * n_x + i) * vec_dim + n]));

                float Fx_p = 0.5 * (
                    fx_p[(j * n_x + i) * vec_dim + n] +
                    fx_m[(j * n_x + i+1) * vec_dim + n] + a *
                    (qx_p[(j * n_x + i) * vec_dim + n] -
                    qx_m[(j * n_x + i+1) * vec_dim + n]));

                float Fy_m = 0.5 * (
                    fy_p[((j-1) * n_x + i) * vec_dim + n] +
                    fy_m[(j * n_x + i) * vec_dim + n] + a *
                    (qy_p[((j-1) * n_x + i) * vec_dim + n] -
                    qy_m[(j * n_x + i) * vec_dim + n]));

                float Fy_p = 0.5 * (
                    fy_p[(j * n_x + i) * vec_dim + n] +
                    fy_m[((j+1) * n_x + i) * vec_dim + n] + a *
                    (qy_p[(j * n_x + i) * vec_dim + n] -
                    qy_m[((j+1) * n_x + i) * vec_dim + n]));

                F[(j * n_x + i) * vec_dim + n] = -a * alpha * (
                    (Fx_p - Fx_m) / d_x + (Fy_p - Fy_m) / d_y);
            }
        }
    }

    bcs(F, n_x, n_y, vec_dim);

    delete[] qx_p;
    delete[] qx_m;
    delete[] qy_p;
    delete[] qy_m;
    delete[] fx_p;
    delete[] fx_m;
    delete[] fy_p;
    delete[] fy_m;
}

void Sea::rk3(float * q, int n_x, int n_y, int vec_dim, float * F,
              flux_func_ptr flux_func, float d_x, float d_y, float _dt) {
    // implement third-order Runge-Kutta algorithm to evolve through single
    // timestep

    int grid_size = n_x * n_y * vec_dim;

    float * q_temp = new float[grid_size];

    evolve(q, n_x, n_y, vec_dim, F, flux_func, d_x, d_y);

    for (int i = 0; i < grid_size; i++) {
        q_temp[i] = q[i] + _dt * F[i];
    }

    evolve(q_temp, n_x, n_y, vec_dim, F, flux_func, d_x, d_y);

    for (int i = 0; i < grid_size; i++) {
        q_temp[i] = 0.25 * (3.0 * q[i] + q_temp[i] + _dt * F[i]);
    }

    evolve(q_temp, n_x, n_y, vec_dim, F, flux_func, d_x, d_y);

    for (int i = 0; i < grid_size; i++) {
        q[i] = (q[i] + 2.0 * q_temp[i] + 2.0 * _dt * F[i]) / 3.0;
    }

    delete[] q_temp;

}

void Sea::run(MPI_Comm comm, MPI_Status * status, int rank, int size) {
    /*
    run code
    */

    cuda_run(beta, gamma_up, U_coarse, U_fine, rho, mu,
             nx, ny, nxf, nyf, ng, nt,
             alpha, gamma, dx, dy, dt, burning, dprint, outfile, comm, *status, rank, size, matching_indices);
}

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

    float * D0 = new float[sea.nx*sea.ny];
    float * Sx0 = new float[sea.nx*sea.ny];
    float * Sy0 = new float[sea.nx*sea.ny];

    // set initial data
    for (int x = 0; x < sea.nx; x++) {
        for (int y = 0; y < sea.ny; y++) {
            D0[y * sea.nx + x] = 1.0 + 0.4 *
                exp(-(pow(sea.xs[x]-5.0, 2)+pow(sea.ys[y]-5.0, 2)) * 2.0);

            Sx0[y * sea.nx + x] = 0.0;
            Sy0[y * sea.nx + x] = 0.0;
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
