#ifndef MESH_CUDA_KERNEL_H
#define MESH_CUDA_KERNEL_H

#include "cuda_runtime.h"
#include "mpi.h"

typedef void (* flux_func_ptr)(float * q, float * f, int dir,
                                float * gamma_up,
                                float alpha, float * beta, float gamma);

typedef float (* fptr)(float p, float D, float Sx, float Sy, float Sz,
                       float tau, float gamma, float * gamma_up);

unsigned int nextPow2(unsigned int x);

__host__ __device__ bool nan_check(float a);

__host__ __device__ float zbrent(fptr func, const float x1, const float x2,
             const float tol,
             float D, float Sx, float Sy, float Sz, float tau, float gamma,
             float * gamma_up);

void check_mpi_error(int mpi_err);

void getNumKernels(int nx, int ny, int nz, int ng, int n_processes, int *maxBlocks, int *maxThreads, dim3 *kernels, int *cumulative_kernels);

void getNumBlocksAndThreads(int nx, int ny, int nz, int ng, int maxBlocks, int maxThreads, int n_processes, dim3 *kernels, dim3 *blocks, dim3 *threads);

void bcs_fv(float * grid, int nx, int ny, int nz, int ng, int vec_dim);

void bcs_mpi(float * grid, int nx, int ny, int nz, int vec_dim, int ng, MPI_Comm comm, MPI_Status status, int rank, int n_processes, int y_size);

__host__ __device__ float W_swe(float * q, float * gamma_up);

__host__ __device__ float phi(float r);

__device__ float find_height(float ph);

__device__ float find_pot(float r);

__device__ float rhoh_from_p(float p, float rho, float gamma);

__device__ float p_from_rhoh(float rhoh, float rho, float gamma);

__device__ __host__ float p_from_rho_eps(float rho, float eps, float gamma);

__device__ __host__ float phi_from_p(float p, float rho, float gamma, float A);

__device__ __host__ float f_of_p(float p, float D, float Sx, float Sy,
                                 float Sz, float tau, float gamma,
                                 float * gamma_up);

__device__ float h_dot(float phi, float old_phi, float dt);

__device__ void calc_As(float * rhos, float * ps, float * A,
                        float p_floor, int nlayers, float gamma);

__device__ void cons_to_prim_comp_d(float * q_cons, float * q_prim,
                       float gamma, float * gamma_up);

void cons_to_prim_comp(float * q_cons, float * q_prim, int nx, int ny, int nz,
                       float gamma, float * gamma_up);

__device__ void shallow_water_fluxes(float * q, float * f, int dir,
                         float * gamma_up, float alpha, float * beta,
                         float gamma);

__device__ void compressible_fluxes(float * q, float * f, int dir,
                      float * gamma_up, float alpha, float * beta,
                      float gamma);

void p_from_swe(float * q, float * p, int nx, int ny, int nz,
               float * gamma_up, float rho, float gamma, float A);

__device__ float p_from_swe(float * q, float * gamma_up, float rho,
                           float gamma, float W, float A);

__global__ void compressible_from_swe(float * q, float * q_comp,
                          int nx, int ny, int nz,
                          float * gamma_up, float * rho, float gamma,
                          int kx_offset, int ky_offset, float dt,
                          float * old_phi, float p_floor);

__global__ void prolong_reconstruct(float * q_comp, float * q_f, float * q_c,
                  int nx, int ny, int nlayers, int nxf, int nyf, int nz, float dx, float dy, float dz, float zmin,
                  int * matching_indices_d, float * gamma_up,
                  int kx_offset, int ky_offset);

void prolong_grid(dim3 * kernels, dim3 * threads, dim3 * blocks,
                int * cumulative_kernels, float * q_cd, float * q_fd,
                int nx, int ny, int nlayers, int nxf, int nyf, int nz,
                float dx, float dy, float dz, float dt, float zmin,
                float * gamma_up_d, float * rho, float gamma,
                int * matching_indices_d, int ng, int rank, float * qc_comp,
                float * old_phi_d, float p_floor);

__global__ void swe_from_compressible(float * q, float * q_swe,
                                      int nxf, int nyf, int nz,
                                      float * gamma_up, float * rho,
                                      float gamma,
                                      int kx_offset, int ky_offset,
                                      float p_floor);

__device__ float height_err(float * q_c_new, float * qf_sw, float zmin,
                          int nxf, int nyf, int nz, float dz,
                          float * gamma_up, int x, int y,
                          float height_guess);

__global__ void restrict_interpolate(float * qf_sw, float * q_c,
                                   int nx, int ny, int nlayers,
                                   int nxf, int nyf, int nz,
                                   float dz, float zmin,
                                   int * matching_indices,
                                   float * gamma_up,
                                   int kx_offset, int ky_offset);

void restrict_grid(dim3 * kernels, dim3 * threads, dim3 * blocks,
                   int * cumulative_kernels, float * q_cd, float * q_fd,
                   int nx, int ny, int nlayers, int nxf, int nyf, int nz,
                   float dz, float zmin, int * matching_indices,
                   float * rho, float gamma, float * gamma_up,
                   int ng, int rank, float * qf_swe, float p_floor);

__global__ void evolve_fv(float * beta_d, float * gamma_up_d,
                    float * Un_d, flux_func_ptr flux_func,
                    float * qx_plus_half, float * qx_minus_half,
                    float * qy_plus_half, float * qy_minus_half,
                    float * fx_plus_half, float * fx_minus_half,
                    float * fy_plus_half, float * fy_minus_half,
                    int nx, int ny, int nz, int vec_dim, float alpha, float gamma,
                    float dx, float dy, float dt,
                    int kx_offset, int ky_offset);

__global__ void evolve_z(float * beta_d, float * gamma_up_d,
                     float * Un_d, flux_func_ptr flux_func,
                     float * qz_plus_half, float * qz_minus_half,
                     float * fz_plus_half, float * fz_minus_half,
                     int nx, int ny, int nz, int vec_dim, float alpha, float gamma,
                     float dz, float dt,
                     int kx_offset, int ky_offset);

__global__ void evolve_fv_fluxes(float * F,
                  float * qx_plus_half, float * qx_minus_half,
                  float * qy_plus_half, float * qy_minus_half,
                  float * fx_plus_half, float * fx_minus_half,
                  float * fy_plus_half, float * fy_minus_half,
                  int nx, int ny, int nz, int vec_dim, float alpha,
                  float dx, float dy, float dt,
                  int kx_offset, int ky_offset);

__global__ void evolve_z_fluxes(float * F,
                   float * qz_plus_half, float * qz_minus_half,
                   float * fz_plus_half, float * fz_minus_half,
                   int nx, int ny, int nz, int vec_dim, float alpha,
                   float dz, float dt,
                   int kx_offset, int ky_offset);

__global__ void evolve_fv_heating(float * gamma_up_d,
                    float * Un_d, float * Up, float * U_half,
                    float * qx_plus_half, float * qx_minus_half,
                    float * qy_plus_half, float * qy_minus_half,
                    float * fx_plus_half, float * fx_minus_half,
                    float * fy_plus_half, float * fy_minus_half,
                    float * sum_phs, float * rho_d, float * Q_d,
                    float mu,
                    int nx, int ny, int nlayers, float alpha,
                    float dx, float dy, float dt,
                    bool burning,
                    int kx_offset, int ky_offset);

__global__ void evolve2(float * gamma_up_d,
                     float * Un_d, float * Up, float * U_half,
                     float * sum_phs, float * rho_d, float * Q_d,
                     float mu,
                     int nx, int ny, int nlayers, int ng, float alpha,
                     float dx, float dy, float dt,
                     int kx_offset, int ky_offset);

void homogeneuous_fv(dim3 * kernels, dim3 * threads, dim3 * blocks,
    int * cumulative_kernels, float * beta_d, float * gamma_up_d,
    float * Un_d, float * F_d,
    float * qx_p_d, float * qx_m_d, float * qy_p_d, float * qy_m_d,
    float * qz_p_d, float * qz_m_d,
    float * fx_p_d, float * fx_m_d, float * fy_p_d, float * fy_m_d,
    float * fz_p_d, float * fz_m_d,
    int nx, int ny, int nz, int vec_dim, int ng, float alpha, float gamma,
    float dx, float dy, float dz, float dt, int rank,
    flux_func_ptr h_flux_func, bool do_z);

void rk3(dim3 * kernels, dim3 * threads, dim3 * blocks,
       int * cumulative_kernels,
       float * beta_d, float * gamma_up_d, float * Un_d,
       float * F_d, float * Up_d,
       float * qx_p_d, float * qx_m_d, float * qy_p_d, float * qy_m_d,
       float * qz_p_d, float * qz_m_d,
       float * fx_p_d, float * fx_m_d, float * fy_p_d, float * fy_m_d,
       float * fz_p_d, float * fz_m_d,
       int nx, int ny, int nz, int vec_dim, int ng, float alpha, float gamma,
       float dx, float dy, float dz, float dt,
       float * Up_h, float * F_h, float * Un_h,
       MPI_Comm comm, MPI_Status status, int rank, int n_processes,
       flux_func_ptr h_flux_func, bool do_z);

void cuda_run(float * beta, float * gamma_up, float * Uc_h, float * Uf_h,
         float * rho, float p_floor, float mu, int nx, int ny, int nlayers,
         int nxf, int nyf, int nz, int ng,
         int nt, float alpha, float gamma, float zmin,
         float dx, float dy, float dz, float dt, bool burning,
         int dprint, char * filename,
         MPI_Comm comm, MPI_Status status, int rank, int n_processes,
         int * matching_indices);

__global__ void test_find_height(bool * passed);
__global__ void test_find_pot(bool * passed);
__global__ void test_rhoh_from_p(bool * passed);
__global__ void test_p_from_rhoh(bool * passed);
__global__ void test_p_from_rho_eps(bool * passed);
__global__ void test_hdot(bool * passed);
__global__ void test_calc_As(bool * passed);
__global__ void test_cons_to_prim_comp_d(bool * passed, float * q_prims);
__global__ void test_shallow_water_fluxes(bool * passed);
__global__ void test_compressible_fluxes(bool * passed);
__global__ void test_p_from_swe(bool * passed);
__global__ void test_height_err(bool * passed);

#endif
