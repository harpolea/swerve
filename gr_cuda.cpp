#include <iostream>
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <algorithm>
#include "SeaCuda.h"

using namespace std;


// TODO: GET RID OF THIS
//void __syncthreads() {}


SeaCuda::SeaCuda(int n_layers, int _nx, int _ny, int _nt,
        float xmin, float xmax,
        float ymin, float ymax, float * _rho,
        float * _Q,
        float _alpha, float * _beta, float * _gamma,
        bool _periodic)
        : nlayers(n_layers), nx(_nx), ny(_ny), nt(_nt), alpha(_alpha), periodic(_periodic)
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

    U_grid = new float[nlayers*nx*ny*(nt+1)*3];

    cout << "Made a Sea.\n";
    //cout << "dt = " << dt << "\tdx = " << dx << "\tdy = " << dy << '\n';

}

SeaCuda::SeaCuda(const SeaCuda &seaToCopy)
    : nlayers(seaToCopy.nlayers), nx(seaToCopy.nx), ny(seaToCopy.ny), nt(seaToCopy.nt), dx(seaToCopy.dx), dy(seaToCopy.dy), dt(seaToCopy.dt), alpha(seaToCopy.alpha), periodic(seaToCopy.periodic)
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
    Q = new float[nlayers];

    for (int i = 0; i < nlayers; i++) {
        rho[i] = seaToCopy.rho[i];
        Q[i] = seaToCopy.Q[i];
    }

    U_grid = new float[nlayers*nx*ny*(nt+1)*3];

    for (int i = 0; i < nlayers*nx*ny*(nt+1)*3; i++) {
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

// return U state vector
void SeaCuda::U(float * grid, int l, int x, int y, int t, float * u) {
    for (int i=0; i < 3; i++) {
        u[i] = grid[(((t * ny + y) * nx + x) * nlayers + l)*3+i];
    }
}

// set the initial data
void SeaCuda::initial_data(float * D0, float * Sx0, float * Sy0) {
    for (int i = 0; i < nlayers*nx*ny; i++) {
        U_grid[i*3] = D0[i];
        U_grid[i*3+1] = Sx0[i];
        U_grid[i*3+2] = Sy0[i];
    }

    bcs(0);

    cout << "Set initial data.\n";
}

void SeaCuda::bcs(int t) {
    if (periodic) {

        for (int l = 0; l < nlayers; l++) {
            for (int y = 0; y < ny; y++){
                for (int i = 0; i < 3; i++) {
                    U_grid[(((t * ny + y) * nx) * nlayers + l)*3+i] = U_grid[(((t * ny + y) * nx + (nx-2)) * nlayers + l)*3+i];

                    U_grid[(((t * ny + y) * nx + (nx-1)) * nlayers + l)*3+i] = U_grid[(((t * ny + y) * nx + 1) * nlayers + l)*3+i];
                }
            }
            for (int x = 0; x < nx; x++){
                for (int i = 0; i < 3; i++) {
                    U_grid[(((t * ny) * nx + x) * nlayers + l)*3+i] = U_grid[(((t * ny + (ny-2)) * nx + x) * nlayers + l)*3+i];

                    U_grid[(((t * ny + (ny-1)) * nx + x) * nlayers + l)*3+i] = U_grid[(((t * ny + 1) * nx + x) * nlayers + l)*3+i];
                }
            }
        }
    } else { // outflow
        for (int l = 0; l < nlayers; l++) {
            for (int y = 0; y < ny; y++){
                for (int i = 0; i < 3; i++) {
                    U_grid[(((t * ny + y) * nx) * nlayers + l)*3+i] = U_grid[(((t * ny + y) * nx + 1) * nlayers + l)*3+i];

                    U_grid[(((t * ny + y) * nx + (nx-1)) * nlayers + l)*3+i] = U_grid[(((t * ny + y) * nx + (nx-2)) * nlayers + l)*3+i];
                }
            }
            for (int x = 0; x < nx; x++){
                for (int i = 0; i < 3; i++) {
                    U_grid[(((t * ny) * nx + x) * nlayers + l)*3+i] = U_grid[(((t * ny + 1) * nx + x) * nlayers + l)*3+i];

                    U_grid[(((t * ny + (ny-1)) * nx + x) * nlayers + l)*3+i] = U_grid[(((t * ny + (ny-2)) * nx + x) * nlayers + l)*3+i];
                }

            }
        }
    }
}



void SeaCuda::run() {

    cout << "Beginning evolution.\n";

    cuda_run(beta, gamma_up, U_grid, rho, Q, nx, ny, nlayers, nt,
             alpha, dx, dy, dt);
}

void SeaCuda::output(char * filename) {
    // open file
    ofstream outFile(filename);

    // only going to output every 10 because file size is ridiculous
    for (int t = 0; t < (nt+1); t+=10) {
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
}

int main() {

    // initialise parameters
    static const int nlayers = 2;
    int nx = 200;
    int ny = 200;
    int nt = 600;
    float xmin = 0.0;
    float xmax = 10.0;
    float ymin = 0.0;
    float ymax = 10.0;
    float rho[nlayers];
    float Q[nlayers];
    float alpha = 0.9;
    float beta[2];
    float gamma[2*2];
    bool periodic = false;

    float D0[nlayers*nx*ny];
    float Sx0[nlayers*nx*ny];
    float Sy0[nlayers*nx*ny];

    for (int i =0; i < nlayers; i++) {
        rho[i] = 1.0;
        Q[i] = 0.0;
    }

    for (int i = 0; i < 2; i++) {
        beta[i] = 0.0;
        //gamma[i] = new float[2];
        for (int j = 0; j < 2; j++) {
            gamma[i*2+j] = 0.0;
        }
        gamma[i*2+i] = 1.0 / (alpha*alpha);
    }

    // make a sea
    SeaCuda sea(nlayers, nx, ny, nt, xmin, xmax, ymin, ymax, rho, Q, alpha, beta, gamma, periodic);

    // set initial data
    for (int x = 1; x < (nx - 1); x++) {
        for (int y = 1; y < (ny - 1); y++) {
            D0[(y * nx + x) * nlayers] = 1.0 + 0.4 * exp(-(pow(sea.xs[x-1]-2.0, 2) + pow(sea.ys[y-1]-2.0, 2)) * 2.0);
            D0[(y * nx + x) * nlayers + 1] = 0.8 + 0.2 * exp(-(pow(sea.xs[x-1]-7.0, 2) + pow(sea.ys[y-1]-7.0, 2)) * 2.0);
            for (int l = 0; l < nlayers; l++) {
                Sx0[(y * nx + x) * nlayers + l] = 0.0;
                Sy0[(y * nx + x) * nlayers + l] = 0.0;
            }
        }
    }

    sea.initial_data(D0, Sx0, Sy0);

    // run simulation
    sea.run();

    char filename[] = "out.dat";
    sea.output(filename);

    cout << "Output data to file.\n";

}
