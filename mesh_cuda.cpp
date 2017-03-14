#include <stdio.h>
#include <stdlib.h>
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

Sea::Sea(int _nx, int _ny,
        int _nt, int _ng, int _r, float _df,
        float xmin, float xmax,
        float ymin, float ymax,
        float _zmin, float _zmax, float * _rho,
        float _Q, float _gamma, float _E_He, float _Cv,
        float _alpha, float * _beta, float * _gamma_down,
        bool _periodic, bool _burning, int _dprint)
        : nx(_nx), ny(_ny), ng(_ng), zmin(_zmin), zmax(_zmax), nt(_nt), r(_r), df(_df), gamma(_gamma), E_He(_E_He), Cv(_Cv), alpha(_alpha), periodic(_periodic), burning(_burning), dprint(_dprint)
{
    /**
    Implement Sea class
    */
    xs = new float[nx];
    for (int i = 0; i < nx; i++) {
        xs[i] = xmin + (i-ng) * (xmax - xmin) / (nx-2*ng);
    }

    ys = new float[ny];
    for (int i = 0; i < ny; i++) {
        ys[i] = ymin + (i-ng) * (ymax - ymin) / (ny-2*ng);
    }

    nxs[0] = nx;
    nys[0] = ny;

    for (int i = 1; i < nlevels; i++) {
        nxs[i] = int(r * df * nxs[i-1]);
        nys[i] = int(r * df * nys[i-1]);
    }

    dx = xs[1] - xs[0];
    dy = ys[1] - ys[0];
    // NOTE: need to define this in such a way that it is calculated using layer separation on first compressible grid
    for (int i = 0; i < nlevels; i++) {
        // first compressible layer
        if (models[i] == 'C') {
            dz = (zmax - zmin) / (nzs[i] - 1.0);
            dz *= pow(r, i);
            break;
        }
    }
    float cfl = 0.5; // cfl number?
    dt = cfl * min(dx, min(dy, dz));

    rho = new float[nzs[0]];
    for (int i = 0; i < nzs[0]; i++) {
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

    matching_indices = new int[4 * (nlevels-1)];
    matching_indices[0] = int(ceil(nx*0.5*(1-df)));
    matching_indices[1] = int(ceil(nx*0.5*(1+df)));
    matching_indices[2] = int(ceil(ny*0.5*(1-df)));
    matching_indices[3] = int(ceil(ny*0.5*(1+df)));

    cout << "Matching indices: " << matching_indices[0] << ',' << matching_indices[1] << ',' << matching_indices[2] << ',' << matching_indices[3] << '\n';

    cout << "Made a Sea.\n";
}

void Sea::invert_mat(float * M, int m, int n) {
    /**
    Invert the m x n matrix M in place using Gaussian elimination
    */
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
    /**
    Constructor for Sea class using inputs from file.
    Data is validated: an error will be thrown and the program terminated if any of the inputs are found to be invalid.
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
        } else if (variableName == "nlevels") {
            inputFile >> value;
            nlevels = int(value);
            models = new char[nlevels];
            nxs = new int[nlevels];
            nys = new int[nlevels];
            nzs = new int[nlevels];
            vec_dims = new int[nlevels];
        } else if (variableName == "models") {
            for (int i = 0; i < nlevels; i++) {
                inputFile >> models[i];
                if (models[i] == 'S' || models[i] == 'M') {
                    vec_dims[i] = 4;
                } else {
                    vec_dims[i] = 6;
                }
            }
        } else if (variableName == "nzs") {
            for (int i = 0; i < nlevels; i++) {
                inputFile >> value;
                nzs[i] = int(value);
            }
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
            int m_in = (models[0] == 'S') ? 1 : 0;
            rho = new float[nzs[m_in]];
            for (int i = 0; i < nzs[m_in]; i++) {
                inputFile >> rho[i];
            }
        } else if (variableName == "Q") {
            inputFile >> Q;
        } else if (variableName == "gamma") {
            inputFile >> gamma;
        } else if (variableName == "E_He") {
            inputFile >> E_He;
        } else if (variableName == "Cv") {
            inputFile >> Cv;
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

    // data validation
    if (nx < 0 || nx > 1e5) {
        printf("Invalid nx: %d\n", nx);
        exit(EXIT_FAILURE);
    }
    if (ny < 0 || ny > 1e5) {
        printf("Invalid ny: %d\n", ny);
        exit(EXIT_FAILURE);
    }
    if (nlevels < 0 || nlevels > 1e2) {
        printf("Invalid nlevels: %d\n", nlevels);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < nlevels; i++) {
        if (nzs[i] < 0 || nzs[i] > nzs[nlevels-1]) {
            printf("Invalid nzs[%d]: %d\n", i, nzs[i]);
            exit(EXIT_FAILURE);
        }
    }
    for (int i = 0; i < nlevels; i++) {
        if (models[i] != 'S' && models[i] != 'M' && models[i] != 'C' && models[i] != 'L') {
            printf("Invalid model[%d]: %c\n", i, models[i]);
            exit(EXIT_FAILURE);
        }
    }
    if (ng < 0 || ng > 1e2) {
        printf("Invalid ng: %d\n", ng);
        exit(EXIT_FAILURE);
    }
    if (nt < 0 || nt > 1e8) {
        printf("Invalid nt: %d\n", nt);
        exit(EXIT_FAILURE);
    }
    if (r < 0 || r > 1e2) {
        printf("Invalid r: %d\n", r);
        exit(EXIT_FAILURE);
    }
    if (df < 0.0 || df > 1.0) {
        printf("Invalid df: %f\n", df);
        exit(EXIT_FAILURE);
    }
    if (xmin < -1.0e5 || xmin > 1.0e5) {
        printf("Invalid xmin: %f\n", xmin);
        exit(EXIT_FAILURE);
    }
    if (xmax < -1.0e5 || xmax > 1.0e5 || xmax < xmin) {
        printf("Invalid xmax: %f\n", xmax);
        exit(EXIT_FAILURE);
    }
    if (ymin < -1.0e5 || ymin > 1.0e5) {
        printf("Invalid ymin: %f\n", ymin);
        exit(EXIT_FAILURE);
    }
    if (ymax < -1.0e5 || ymax > 1.0e5 || ymax < ymin) {
        printf("Invalid ymax: %f\n", ymax);
        exit(EXIT_FAILURE);
    }
    if (zmin < -1.0e5 || zmin > 1.0e5) {
        printf("Invalid zmin: %f\n", zmin);
        exit(EXIT_FAILURE);
    }
    if (zmax < -1.0e5 || zmax > 1.0e5 || zmax < zmin) {
        printf("Invalid zmax: %f\n", zmax);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < nzs[0]; i++) {
        if (rho[i] < 0 || rho[i] > 1.0e8) {
            printf("Invalid rho[%d]: %f\n", i, rho[i]);
            exit(EXIT_FAILURE);
        }
    }
    if (Q < -1.0e8 || Q > 1.0e8) {
        printf("Invalid Q: %f\n", Q);
        exit(EXIT_FAILURE);
    }
    if (gamma <  0.0 || gamma > 1.0e2) {
        printf("Invalid gamma: %f\n", gamma);
        exit(EXIT_FAILURE);
    }
    if (E_He <  0.0 || E_He > 1.0e8) {
        printf("Invalid E_He: %f\n", E_He);
        exit(EXIT_FAILURE);
    }
    if (Cv <  0.0 || Cv > 1.0e2) {
        printf("Invalid Cv: %f\n", Cv);
        exit(EXIT_FAILURE);
    }
    if (alpha <  0.0 || alpha > 1.0) {
        printf("Invalid alpha: %f\n", alpha);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < 3; i++) {
        if (beta[i] < -1.0 || beta[i] > 1.0) {
            printf("Invalid beta[%d]: %f\n", i, beta[i]);
            exit(EXIT_FAILURE);
        }
    }
    for (int i = 0; i < 3*3; i++) {
        if (gamma_down[i] < -1.0e2 || gamma_down[i] > 1.0e2) {
            printf("Invalid gamma_down[%d]: %f\n", i, gamma_down[i]);
            exit(EXIT_FAILURE);
        }
    }
    if (dprint < 0 || dprint > 1e9) {
        printf("Invalid dprint: %d\n", dprint);
        exit(EXIT_FAILURE);
    }

    inputFile.close();

    nxs[0] = nx;
    nys[0] = ny;

    for (int i = 1; i < nlevels; i++) {
        nxs[i] = int(r * df * nxs[i-1]);
        nys[i] = int(r * df * nys[i-1]);
    }

    // index of first multilayer SWE level
    int m_in = (models[0] == 'S') ? 1 : 0;

    xs = new float[nxs[m_in]];
    for (int i = 0; i < nxs[m_in]; i++) {
        xs[i] = xmin + (i-ng) * (xmax - xmin) / (nxs[m_in]-2*ng);
    }

    ys = new float[nys[m_in]];
    for (int i = 0; i < nys[m_in]; i++) {
        ys[i] = ymin + (i-ng) * (ymax - ymin) / (nys[m_in]-2*ng);
    }

    dx = (xmax - xmin) / (nxs[0]-2*ng);
    dy = (ymax - ymin) / (nys[0]-2*ng);

    // need to define this in such a way that it is calculated using layer separation on first compressible grid
    for (int i = 0; i < nlevels; i++) {
        // first compressible layer
        if (models[i] == 'C') {
            dz = (zmax - zmin) / (nzs[i] - 1.0);
            dz *= pow(r, i);
            break;
        }
    }

    float cfl = 0.5; // cfl number?
    dt = cfl * min(dx, min(dy, dz));

    // find inverse of gamma
    for (int i = 0; i < 3*3; i++) {
        gamma_up[i] = gamma_down[i];
    }
    Sea::invert_mat(gamma_up, 3, 3);

    cout << "nxs, nys, nzs, vec_dims:\n";
    for (int i = 0; i < nlevels; i++) {
        cout << nxs[i] << ' ' << nys[i] << ' ' << nzs[i] << ' ' << vec_dims[i] << '\n';
    }

    Us = new float*[nlevels];

    for (int i = 0; i < nlevels; i++) {
        Us[i] = new float[nxs[i]*nys[i]*nzs[i]*vec_dims[i]];
        for (int j = 0; j < nxs[i]*nys[i]*nzs[i]*vec_dims[i]; j++) {
            Us[i][j] = 0.0;
        }
    }

    matching_indices = new int[4 * (nlevels-1)];

    for (int i = 0; i < nlevels-1; i++) {
        matching_indices[i*4] = int(ceil(nxs[i]*0.5*(1-df)));
        matching_indices[i*4+1] = int(floor(nxs[i]*0.5*(1+df)));
        matching_indices[i*4+2] = int(ceil(nys[i]*0.5*(1-df)));
        matching_indices[i*4+3] = int(floor(nys[i]*0.5*(1+df)));

        cout << "Matching indices: " << matching_indices[i*4] << ',' << matching_indices[i*4+1] << ',' << matching_indices[i*4+2] << ',' << matching_indices[i*4+3] << '\n';

        cout << "matching_indices vs nxf: " <<
            matching_indices[i*4+1] - matching_indices[i*4] << ',' << nxs[i+1] << '\n';

    }
    cout << "Made a Sea.\n";
}

// copy constructor
Sea::Sea(const Sea &seaToCopy)
    : nx(seaToCopy.nx), ny(seaToCopy.ny), ng(seaToCopy.ng), zmin(seaToCopy.zmin), zmax(seaToCopy.zmax), nt(seaToCopy.nt), r(seaToCopy.r), dx(seaToCopy.dx), dy(seaToCopy.dy), dz(seaToCopy.dz), dt(seaToCopy.dt), df(seaToCopy.df), gamma(seaToCopy.gamma), E_He(seaToCopy.E_He), Cv(seaToCopy.Cv), alpha(seaToCopy.alpha), periodic(seaToCopy.periodic), burning(seaToCopy.burning), dprint(seaToCopy.dprint)
{
    /**
    copy constructor
    */

    xs = new float[nx];
    for (int i = 0; i < nx; i++) {
        xs[i] = seaToCopy.xs[i];
    }

    ys = new float[ny];
    for (int i = 0; i < ny; i++) {
        ys[i] = seaToCopy.ys[i];
    }

    rho = new float[nzs[0]];
    for (int i = 0; i < nzs[0]; i++) {
        rho[i] = seaToCopy.rho[i];
    }

    Q = seaToCopy.Q;

    for (int i = 0; i < 3*nx*ny; i++) {
        beta[i] = seaToCopy.beta[i];
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

Sea::~Sea() {
    /**
    Deconstructor
    */
    delete[] xs;
    delete[] ys;
    delete[] rho;
    delete[] models;
    delete[] nxs;
    delete[] nys;
    delete[] nzs;
    delete[] vec_dims;
    delete[] matching_indices;

    for (int i = 0; i < nlevels; i++) {
        delete[] Us[i];
    }
}

void Sea::initial_data(float * D0, float * Sx0, float * Sy0) {
    /**
    Initialise D, Sx, Sy and Q on coarsest multilayer SWE grid.
    */
    // find coarsest multilayer SWE grid
    // TODO: make sure ensure this exists when initialise object
    int grid_index = (models[0] == 'S') ? 1 : 0;

    for (int i = 0; i < nxs[grid_index]*nys[grid_index]*nzs[grid_index]; i++) {
        // it's on a SWE grid so know vec_dim = 4
        Us[grid_index][i*4] = D0[i];
        Us[grid_index][i*4+1] = Sx0[i];
        Us[grid_index][i*4+2] = Sy0[i];
        Us[grid_index][i*4+3] =
            0.9 * float(i) / (nxs[grid_index]*nys[grid_index]*nzs[grid_index]);
    }

    bcs(Us[grid_index], nxs[grid_index], nys[grid_index], nzs[grid_index],
        vec_dims[grid_index]);

    cout << "Set initial data.\n";
}

void Sea::print_inputs() {
    /**
    Print some input and runtime parameters to screen.
    */
    cout << "\nINPUT DATA\n" << "----------\n";
    cout << "(nx, ny, nz, ng) \t(" << nxs[0] << ',' << nys[0] << ',' << nzs[0] << ',' << ng << ")\n";
    cout << "nt \t\t\t" << nt << '\n';
    cout << "(r, df, nlevels) \t(" << r << ',' << df << ',' << nlevels << ")\n";
    cout << "dprint \t\t\t" << dprint << '\n';
    cout << "(dx, dy, dz, dt) \t(" << dx << ',' << dy << ',' << dz << ',' << dt << ")\n";
    cout << "rho \t\t\t" << rho[0] << ',' << rho[1]<< ',' << rho[2] << "\n";
    cout << "Q \t\t\t" << Q << '\n';
    cout << "E_He \t\t\t" << E_He << '\n';
    cout << "Cv \t\t\t" << Cv << '\n';
    cout << "alpha \t\t\t" << alpha << '\n';
    cout << "beta \t\t\t(" << beta[0] << ',' << beta[1] << ',' << beta[2] << ")\n";
    cout << "gamma_down \t\t((" << gamma_down[0] << ',' << gamma_down[1] << ',' << gamma_down[2] << "),(" << gamma_down[3] << ',' << gamma_down[4] << ',' << gamma_down[5] << "),(" << gamma_down[6] << ',' << gamma_down[7] << ',' << gamma_down[8] << "))\n";
    cout << "burning \t\t" << burning << '\n';
    cout << "outfile \t\t" << outfile << "\n\n";
}

void Sea::bcs(float * grid, int n_x, int n_y, int n_z, int vec_dim) {
    /**
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
    /**
    run code
    */
    int m_in = (models[0] == 'S') ? 1 : 0;
    // hack for now
    float * Qs = new float[nzs[m_in]];
    for (int i = 0; i < nzs[m_in]; i++) {
        Qs[i] = Q;
    }

    cuda_run(beta, gamma_up, Us, rho, Qs,
             nxs, nys, nzs, nlevels, models, vec_dims,
             ng, nt,
             alpha, gamma, E_He, Cv, zmin, dx, dy, dz, dt, burning, dprint,
             outfile, comm, *status, rank, size, matching_indices, r);

    delete[] Qs;
}
