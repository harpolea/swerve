#ifndef SEA_CUDA_H
#define SEA_CUDA_H

#include "cuda_runtime.h"
#include "mpi.h"

void getNumKernels(int nx, int ny, int nlayers, int ng, int n_processes, int *maxBlocks, int *maxThreads, dim3 *kernels, int *cumulative_kernels);

void getNumBlocksAndThreads(int nx, int ny, int nlayers, int ng, int maxBlocks, int maxThreads, int n_processes, dim3 *kernels, dim3 *blocks, dim3 *threads);

unsigned int nextPow2(unsigned int x);

void bcs_fv(float * grid, int nx, int ny, int nlayers, int ng);

void bcs_mpi(float * grid, int nx, int ny, int nlayers, int ng, MPI_Comm comm, MPI_Status status, int rank, int n_processes);

void check_mpi_error(int mpi_err);

void homogeneuous_fv(dim3 * kernels, dim3 * threads, dim3 * blocks, float * beta_d, float * gamma_up_d,
       float * Un_d, float * F_d,
       float * qx_p_d, float * qx_m_d, float * qy_p_d, float * qy_m_d,
       float * fx_p_d, float * fx_m_d, float * fy_p_d, float * fy_m_d,
       int nx, int ny, int nlayers, float alpha,
       float dx, float dy, float dt, int rank);

void rk3_fv(dim3 * kernels, dim3 * threads, dim3 * blocks,
      float * beta_d, float * gamma_up_d, float * Un_d,
      float * F_d, float * Up_d,
      float * qx_p_d, float * qx_m_d, float * qy_p_d, float * qy_m_d,
      float * fx_p_d, float * fx_m_d, float * fy_p_d, float * fy_m_d,
      int nx, int ny, int nlayers, int ng, float alpha,
      float dx, float dy, float dt,
      float * Up_h, float * F_h, float * Un_h);

void cuda_run(float * beta, float * gamma_up, float * U_grid,
         float * rho, float * Q, float mu,
         int nx, int ny, int nlayers, int ng,
         int nt, float alpha, float dx, float dy, float dt, bool burning,
         int dprint, char * filename, MPI_Comm comm, MPI_Status status, int rank, int size);


class SeaCuda {
public:
    SeaCuda(int n_layers, int _nx, int _ny, int _nt, int _ng,
            float xmin, float xmax,
            float ymin, float ymax, float * _rho,
            float * _Q, float mu,
            float _alpha, float * _beta, float * _gamma,
            bool _periodic, bool _burning, int _dprint);

    SeaCuda(char * filename); // constructor which takes input from file

    SeaCuda(const SeaCuda &); // copy constructor

    void initial_data(float * D0, float * Sx0, float * Sy0, float * zeta0, float * _Q, float * _beta);

    void bcs(float * grid, int vec_dim);

    void print_inputs();

    void run(MPI_Comm comm, MPI_Status status, int rank, int size);

    void output(char * filename);
    void output_hdf5(char * filename);
    void output();

    ~SeaCuda();

    // these need to be public
    int nlayers;
    int nx;
    int ny;
    int ng;
    float *xs;
    float *ys;
    float *U_grid;

private:

    int nt;

    float dx;
    float dy;
    float dt;

    float *rho;
    float *Q;
    float mu; // friction

    float alpha;
    float *beta;
    float gamma[2*2];
    float gamma_up[2*2];

    bool periodic;
    bool burning;

    int dprint; // number of timesteps between printouts

    char outfile[200];
};

#endif
