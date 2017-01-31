#ifndef MESH_CUDA_H
#define MESH_CUDA_H

#include "cuda_runtime.h"
#include "mpi.h"

class Sea {
public:
    Sea(int _nx, int _ny, int _nz, int _nlayers, int _nt, int _ng,
            int _r, float _df,
            float xmin, float xmax,
            float ymin, float ymax,
            float zmin, float zmax, float  * _rho, float p_floor,
            float  _Q, float _mu, float _gamma,
            float _alpha, float * _beta, float * _gamma_down,
            bool _periodic, bool _burning, int _dprint);

    Sea(char * filename); // constructor which takes input from file

    Sea(const Sea &); // copy constructor

    void initial_data(float * D0, float * Sx0, float * Sy0, float * Sz0, float * tau);

    void bcs(float * grid, int n_x, int n_y, int n_z, int vec_dim);

    void print_inputs();

    static void invert_mat(float * A, int m, int n);

    void run(MPI_Comm comm, MPI_Status * status, int rank, int size);

    ~Sea();

    // these need to be public
    //int nlayers;
    int nx;
    int ny;
    int nz;
    int nlayers;
    int ng;

    float zmin;
    float *xs;
    float *ys;
    float *U_coarse;
    float *U_fine;
    float *rho;
    float gamma;

private:

    int nt;
    int r; // refinement ratio
    int nxf;
    int nyf;
    int matching_indices[2*2];

    float dx;
    float dy;
    float dz;
    float dt;
    float df;

    float p_floor;
    float Q;
    float mu; // friction

    float alpha;
    float beta[3];
    float gamma_down[3*3];
    float gamma_up[3*3];

    bool periodic;
    bool burning;

    int dprint; // number of timesteps between printouts

    char outfile[200];
};

#endif
