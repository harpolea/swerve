#ifndef MESH_CUDA_H
#define MESH_CUDA_H

#include "cuda_runtime.h"
#include "mpi.h"

void check_mpi_error(int mpi_err);

void getNumKernels(int nx, int ny, int nlayers, int ng, int n_processes, int *maxBlocks, int *maxThreads, dim3 *kernels, int *cumulative_kernels);

void getNumBlocksAndThreads(int nx, int ny, int nlayers, int ng, int maxBlocks, int maxThreads, int n_processes, dim3 *kernels, dim3 *blocks, dim3 *threads);

unsigned int nextPow2(unsigned int x);

void bcs_fv(float * grid, int nx, int ny, int nlayers, int ng);

void bcs_mpi(float * grid, int nx, int ny, int nlayers, int ng, MPI_Comm comm, MPI_Status status, int rank, int n_processes, int y_size);

typedef void (* flux_func_ptr)(float * q, float * f, bool x_dir,
                                float * gamma_up,
                                float alpha, float * beta, float gamma);

typedef float (* fptr)(float p, float D, float Sx, float Sy, float tau,
                       float gamma, float * gamma_up);

float zbrent(fptr func, const float x1, const float x2, const float tol,
             float D, float Sx, float Sy, float tau, float gamma,
             float * gamma_up);

// the typedef/function pointer thing sadly does not work well with
// member functions :(
float f_of_p(float p, float D, float Sx, float Sy, float tau, float gamma,
             float * gamma_up);

float p_from_rho_eps(float rho, float eps, float gamma);

void cons_to_prim_comp(float * q_cons, float * q_prim, int nx, int ny,
                       float gamma, float * gamma_up);

void prolong_grid(float * q_c, float * q_f,
                      int nx, int ny, int nxf, int nyf, float dx, float dy,
                      float * gamma_up, float rho, float gamma,
                      int * matching_indices);
void restrict_grid(float * q_c, float * q_f,
                       int nx, int ny, int nxf, int nyf,
                       int * matching_indices,
                       float rho, float gamma, float * gamma_up);

void cuda_run(float * beta, float * gamma_up, float * Uc_h, float * Uf_h,
         float rho, float mu, int nx, int ny,
         int nxf, int nyf, int ng,
         int nt, float alpha, float gamma, float dx, float dy, float dt, bool burning,
         int dprint, char * filename,
         MPI_Comm comm, MPI_Status status, int rank, int n_processes,
         int * matching_indices);

class Sea {
public:
    Sea(int _nx, int _ny, int _nt, int _ng, int _r, float _df,
            float xmin, float xmax,
            float ymin, float ymax, float  _rho,
            float  _Q, float _mu, float _gamma,
            float _alpha, float * _beta, float * _gamma_down,
            bool _periodic, bool _burning, int _dprint);

    Sea(char * filename); // constructor which takes input from file

    Sea(const Sea &); // copy constructor

    void initial_data(float * D0, float * Sx0, float * Sy0);

    void bcs(float * grid, int n_x, int n_y, int vec_dim);

    void print_inputs();

    void run(MPI_Comm comm, MPI_Status * status, int rank, int size);

    float phi(float r); // MC limiter

    void prolong_grid(float * q_c, float * q_f);
    void restrict_grid(float * q_c, float * q_f);
    void p_from_swe(float * q, float * p);

    void evolve(float * q, int n_x, int n_y, int vec_dim, float * F,
                flux_func_ptr flux_func, float d_x, float d_y);

    void rk3(float * q, int n_x, int n_y, int vec_dim, float * F,
             flux_func_ptr flux_func, float d_x, float d_y, float _dt);

    float rhoh_from_p(float p);
    float p_from_rhoh(float rhoh);

    float phi_from_p(float p);

    ~Sea();

    // these need to be public
    //int nlayers;
    int nx;
    int ny;
    int ng;
    float *xs;
    float *ys;
    float *U_coarse;
    float *U_fine;

private:

    int nt;
    int r; // refinement ratio
    int nxf;
    int nyf;
    int matching_indices[2*2];

    float dx;
    float dy;
    float dt;
    float df;

    float rho;
    float Q;
    float mu; // friction
    float gamma;

    float alpha;
    float beta[2];
    float gamma_down[2*2];
    float gamma_up[2*2];

    bool periodic;
    bool burning;

    int dprint; // number of timesteps between printouts

    char outfile[200];
};

#endif
