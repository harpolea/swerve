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

/** check to see whether float a is a nan
*/
__host__ __device__ bool nan_check(float a);

/**
Using Brent's method, return the root of a function or functor func known
to lie between x1 and x2. The root will be regined until its accuracy is
tol.

\param func
    function pointer to shallow water or compressible flux function.
\param x1, x2
    limits of root
\param tol
    tolerance to which root shall be calculated to
\param D, Sx, Sy, Sz, tau
    conserved variables
\param gamma
    adiabatic index
\param gamma_up
    spatial metric
*/
__host__ __device__ float zbrent(fptr func, const float x1, const float x2,
             const float tol,
             float D, float Sx, float Sy, float Sz, float tau, float gamma,
             float * gamma_up);

/**
Checks to see if the integer returned by an mpi function, mpi_err, is an MPI error. If so, it prints out some useful stuff to screen.
*/
void check_mpi_error(int mpi_err);

/**
Return the number of kernels needed to run the problem given its size and the constraints of the GPU.

\param nx, ny, nz
    dimensions of problem
\param ng
    number of ghost cells
\param maxBlocks, maxThreads
    maximum number of blocks and threads possible for device(s)
\param n_processes
    number of MPI processes
\param kernels
    number of kernels per process
\param cumulative_kernels
    cumulative total of kernels per process
*/
void getNumKernels(int nx, int ny, int nz, int ng, int n_processes, int *maxBlocks, int *maxThreads, dim3 *kernels, int *cumulative_kernels);

/**
Returns the number of blocks and threads required for each kernel given the size of the problem and the constraints of the device.

\param nx, ny, nz
    dimensions of problem
\param ng
    number of ghost cells
\param maxBlocks, maxThreads
    maximum number of blocks and threads possible for device(s)
\param n_processes
    number of MPI processes
\param kernels, blocks, threads
    number of kernels, blocks and threads per process / kernel
*/
void getNumBlocksAndThreads(int nx, int ny, int nz, int ng, int maxBlocks, int maxThreads, int n_processes, dim3 *kernels, dim3 *blocks, dim3 *threads);

/**
Enforce boundary conditions on section of grid.

\param grid
    grid of data
\param nx, ny, nz
    dimensions of grid
\param ng
    number of ghost cells
\param vec_dim
    dimension of state vector
*/
void bcs_fv(float * grid, int nx, int ny, int nz, int ng, int vec_dim);

/**
Enforce boundary conditions across processes / at edges of grid.

Loops have been ordered in a way so as to try and keep memory accesses as contiguous as possible.

Need to do non-blocking send, blocking receive then wait.

\param grid
    grid of data
\param nx, ny, nz
    dimensions of grid
\param vec_dim
    dimension of state vector
\param ng
    number of ghost cells
\param comm
    MPI communicator
\param status
    status of MPI processes
\param rank, n_processes
    rank of MPI process and total number of MPI processes
\param y_size
    size of grid in y direction running on each process (except the last one)
\param do_z
    true if need to implement bcs in vertical direction as well
*/
void bcs_mpi(float * grid, int nx, int ny, int nz, int vec_dim, int ng, MPI_Comm comm, MPI_Status status, int rank, int n_processes, int y_size, bool do_z);

__host__ __device__ float W_swe(float * q, float * gamma_up); /**< calculate Lorentz factor for conserved swe state vector */

/**
calculate superbee slope limiter Phi(r)
*/
__host__ __device__ float phi(float r);

/**
Finds r given Phi.
*/
__device__ float find_height(float ph);

/**
Finds Phi given r.
*/
__device__ float find_pot(float r);

/**
calculate rhoh using p for gamma law equation of state
*/
__device__ float rhoh_from_p(float p, float rho, float gamma);

/**
calculate p using rhoh for gamma law equation of state
*/
__device__ float p_from_rhoh(float rhoh, float rho, float gamma);

/**
calculate p using rho and epsilon for gamma law equation of state
*/
__device__ __host__ float p_from_rho_eps(float rho, float eps, float gamma);

/**
Calculate the metric potential Phi given p for gamma law equation of
state

\param p, rho
    pressure and density
\param gamma
    adiabatic index
\param A
    constant used in Phi to p conversion
*/
__device__ __host__ float phi_from_p(float p, float rho, float gamma, float A);

/**
Function of p whose root is to be found when doing conserved to
primitive variable conversion

\param p
    pressure
\param D, Sx, Sy, Sz, tau
    components of conserved state vector
\param gamma
    adiabatic index
\param gamma_up
    spatial metric
*/
__device__ __host__ float f_of_p(float p, float D, float Sx, float Sy,
                                 float Sz, float tau, float gamma,
                                 float * gamma_up);

/**
Calculates the time derivative of the height given the shallow water
variable phi at current time and previous timestep
NOTE: this is an upwinded approximation of hdot - there may be a better
way to do this which will more accurately give hdot at current time.

\param phi
    Phi at current timestep
\param old_phi
    Phi at previous timestep
\param dt
    timestep
*/

__device__ float h_dot(float phi, float old_phi, float dt);

/**
Calculate the heating rate per unit mass from the shallow water variables

\param rho
    densities of layers
\param p
    pressure
\param gamma
    adiabatic index
\param Y
    species fraction
\param Cv
    specific heat in constant volume
*/
__device__ float calc_Q_swe(float rho, float p, float gamma, float Y, float Cv);

/**
Calculate the heating rate per unit mass.

\param rho
    densities of layers
\param q_cons
    conservative state vector
\param nx, ny, nz
    dimensions of grid
\param gamma
    adiabatic index
\param gamma_up
    contravariant spatial metric
\param Q
    array that shall contain heating rate per unit mass
\param Cv
    specific heat in constant volume
*/
void calc_Q(float * rho, float * q_cons, int nx, int ny, int nz,
            float gamma, float * gamma_up, float * Q, float Cv);

/**
Calculates the As used to calculate the pressure given Phi, given
the pressure at the sea floor

\param rhos
    densities of layers
\param phis
    Vector of Phi for different layers
\param A
    vector of As for layers
\param nlayers
    number of layers
\param gamma
    adiabatic index
\param surface_phi
    Phi at surface
\param surface_rho
    density at surface
*/
__device__ void calc_As(float * rhos, float * phis, float * A,
                        int nlayers, float gamma,
                        float surface_phi, float surface_rho);

/**
Convert compressible conserved variables to primitive variables

\param q_cons
    state vector of conserved variables
\param q_prim
    state vector of primitive variables
\param gamma
    adiabatic index
\param gamma_up
    spatial metric
*/
__device__ void cons_to_prim_comp_d(float * q_cons, float * q_prim,
                       float gamma, float * gamma_up);

/**
Convert compressible conserved variables to primitive variables

\param q_cons
   grid of conserved variables
\param q_prim
   grid where shall put the primitive variables
\param nxf, nyf, nz
   grid dimensions
\param gamma
   adiabatic index
\param gamma_up
   contravariant spatial metric
*/
void cons_to_prim_comp(float * q_cons, float * q_prim, int nxf, int nyf, int nz,
                       float gamma, float * gamma_up);

/**
Calculate the flux vector of the shallow water equations

\param q
   state vector
\param f
   grid where fluxes shall be stored
\param dir
   0 if calculating flux in x-direction, 1 if in y-direction
\param gamma_up
   spatial metric
\param alpha
   lapse function
\param beta
   shift vector
\param gamma
   adiabatic index
*/
__device__ void shallow_water_fluxes(float * q, float * f, int dir,
                         float * gamma_up, float alpha, float * beta,
                         float gamma);

/**
Calculate the flux vector of the compressible GR hydrodynamics equations

\param q
    state vector
\param f
    grid where fluxes shall be stored
\param dir
    0 if calculating flux in x-direction, 1 if in y-direction,
    2 if in z-direction
\param gamma_up
    spatial metric
\param alpha
    lapse function
\param beta
    shift vector
\param gamma
    adiabatic index
*/
__device__ void compressible_fluxes(float * q, float * f, int dir,
                      float * gamma_up, float alpha, float * beta,
                      float gamma);

/**
Calculate p using SWE conserved variables

\param q
  state vector
\param p
  grid where pressure shall be stored
\param nx, ny, nz
  grid dimensions
\param gamma_up
  spatial metric
\param rho
  density
\param gamma
  adiabatic index
\param A
  variable required in p(Phi) calculation
*/
void p_from_swe(float * q, float * p, int nx, int ny, int nz,
               float * gamma_up, float rho, float gamma, float A);

/**
Calculates p and returns using SWE conserved variables

\param q
   state vector
\param gamma_up
   spatial metric
\param rho
   density
\param gamma
   adiabatic index
\param W
   Lorentz factor
\param A
   variable required in p(Phi) calculation
*/
__device__ float p_from_swe(float * q, float * gamma_up, float rho,
                           float gamma, float W, float A);

/**
Calculates the compressible state vector from the SWE variables.

\param q
   grid of SWE state vector
\param q_comp
   grid where compressible state vector to be stored
\param nxs, nys, nzs
   grid dimensions
\param gamma_up
   spatial metric
\param rho, gamma
   density and adiabatic index
\param kx_offset, ky_offset
   kernel offsets in the x and y directions
\param dt
   timestep
\param old_phi
   Phi at previous timestep
\param level
    index of level
*/
__global__ void compressible_from_swe(float * q, float * q_comp,
                          int * nxs, int * nys, int * nzs,
                          float * gamma_up, float * rho, float gamma,
                          int kx_offset, int ky_offset, float dt,
                          float * old_phi, int level);

/**
Calculates slope limited verticle gradient at layer_frac between middle and amiddle.
Left, middle and right are from row n, aleft, amiddle and aright are from row above it (n-1)
*/
__device__ float slope_limit(float layer_frac, float left, float middle, float right, float aleft, float amiddle, float aright);

/**
Reconstruct fine grid variables from compressible variables on coarse grid

\param q_comp
    compressible variables on coarse grid
\param q_f
    fine grid state vector
\param q_c
    coarse grid swe state vector
\param nxs, nys, nzs
    grid dimensions
\param dx, dy, dz
    coarse grid spacings
\param matching_indices_d
    position of fine grid wrt coarse grid
\param gamma_up
    spatial metric
\param kx_offset, ky_offset
    kernel offsets in the x and y directions
\param coarse_level
  index of coarser level
\param nlevels
    total number of levels
*/
__global__ void prolong_reconstruct_comp_from_swe(float * q_comp, float * q_f, float * q_c,
                  int * nxs, int * nys, int * nzs, float dx, float dy, float dz, float zmin,
                  int * matching_indices_d, float * gamma_up,
                  int kx_offset, int ky_offset, int coarse_level, int nlevels);

/**
Prolong coarse grid data to fine grid

\param kernels, threads, blocks
  number of kernels, threads and blocks for each process/kernel
\param cumulative_kernels
  cumulative number of kernels in mpi processes of r < rank
\param q_cd, q_fd
  coarse and fine grids of state vectors
\param nxs, nys, nzs
  dimensions of grids
\param dx, dy, dz
  coarse grid cell spacings
\param dt
  timestep
\param zmin
  height of sea floor
\param gamma_up_d
  spatial metric
\param rho, gamma
  density and adiabatic index
\param matching_indices_d
  position of fine grid wrt coarse grid
\param ng
  number of ghost cells
\param rank
  rank of MPI process
\param qc_comp
  grid of compressible variables on coarse grid
\param old_phi_d
  Phi at previous timstep
\param coarse_level
    index of coarser level
\param nlevels
  total number of levels
*/
void prolong_swe_to_comp(dim3 * kernels, dim3 * threads, dim3 * blocks,
                int * cumulative_kernels, float * q_cd, float * q_fd,
                int * nxs, int * nys, int * nzs,
                float dx, float dy, float dz, float dt, float zmin,
                float * gamma_up_d, float * rho, float gamma,
                int * matching_indices_d, int ng, int rank, float * qc_comp,
                float * old_phi_d, int coarse_level, int nlevels);

/**
Reconstruct fine grid variables from compressible variables on coarse grid

\param q_comp
    compressible variables on coarse grid
\param q_f
    fine grid state vector
\param q_c
    coarse grid swe state vector
\param nxs, nys, nzs
    grid dimensions
\param matching_indices_d
    position of fine grid wrt coarse grid
\param kx_offset, ky_offset
    kernel offsets in the x and y directions
\param clevel
  index of coarser level
\param nlevels
    total number of levels
*/
__global__ void prolong_reconstruct_comp(float * q_f, float * q_c,
                    int * nxs, int * nys, int * nzs,
                    int * matching_indices_d,
                    int kx_offset, int ky_offset, int clevel, int nlevels);

/**
Prolong coarse grid data to fine grid

\param kernels, threads, blocks
  number of kernels, threads and blocks for each process/kernel
\param cumulative_kernels
  cumulative number of kernels in mpi processes of r < rank
\param q_cd, q_fd
  coarse and fine grids of state vectors
\param nxs, nys, nzs
  dimensions of grids
\param matching_indices_d
  position of fine grid wrt coarse grid
\param ng
  number of ghost cells
\param rank
  rank of MPI process
\param coarse_level
    index of coarser level
\param nlevels
  total number of levels
*/
void prolong_comp_to_comp(dim3 * kernels, dim3 * threads, dim3 * blocks,
                  int * cumulative_kernels, float * q_cd, float * q_fd,
                  int * nxs, int * nys, int * nzs,
                  int * matching_indices_d, int ng, int rank, int coarse_level, int nlevels);

/**
Reconstruct multilayer swe fine grid variables from single layer swe variables on coarse grid

\param q_comp
  compressible variables on coarse grid
\param q_f
  fine grid state vector
\param q_c
  coarse grid swe state vector
\param nxs, nys, nzs
  grid dimensions
\param matching_indices_d
  position of fine grid wrt coarse grid
\param kx_offset, ky_offset
  kernel offsets in the x and y directions
\param clevel
index of coarser level
\param nlevels
  total number of levels
*/
__global__ void prolong_reconstruct_swe_from_swe(float * qf, float * qc,
                  int * nxs, int * nys, int * nzs,
                  float zmin,
                  int * matching_indices_d, float * gamma_up,
                  int kx_offset, int ky_offset, int clevel, int nlevels);

/**
Prolong coarse grid single layer swe data to fine multilayer swe grid.

\param kernels, threads, blocks
    number of kernels, threads and blocks for each process/kernel
\param cumulative_kernels
    cumulative number of kernels in mpi processes of r < rank
\param q_cd, q_fd
    coarse and fine grids of state vectors
\param nxs, nys, nzs
    dimensions of grids
\param matching_indices_d
    position of fine grid wrt coarse grid
\param ng
    number of ghost cells
\param rank
    rank of MPI process
\param coarse_level
  index of coarser level
\param nlevels
    total number of levels
*/
void prolong_swe_to_swe(dim3 * kernels, dim3 * threads, dim3 * blocks,
                int * cumulative_kernels, float * q_cd, float * q_fd,
                int * nxs, int * nys, int * nzs,
                float dx, float dy, float dz, float dt, float zmin,
                float * gamma_up_d, float * rho, float gamma,
                int * matching_indices_d, int ng, int rank,
                int coarse_level, int nlevels);

/**
Calculates the SWE state vector from the compressible variables.

\param q
    grid of compressible state vector
\param q_swe
    grid where SWE state vector to be stored
\param nxs, nys, nzs
    grid dimensions
\param gamma_up
    spatial metric
\param rho, gamma
    density and adiabatic index
\param kx_offset, ky_offset
    kernel offsets in the x and y directions
\param qc
    coarse grid
\param matching_indices
    indices of fine grid wrt coarse grid
\param coarse_level, nlevels
    index of coarser grid and total number of grid levels
*/
__global__ void swe_from_compressible(float * q, float * q_swe,
                                      int * nxs, int * nys, int * nzs,
                                      float * gamma_up, float * rho,
                                      float gamma,
                                      int kx_offset, int ky_offset,
                                      float * qc,
                                      int * matching_indices,
                                      int coarse_level, int nlevels);

/**
Interpolate SWE variables on fine grid to get them on coarse grid.

\param qf_swe
  SWE variables on fine grid
\param q_c
  coarse grid state vector
\param nxs, nys, nzs
  grid dimensions
\param matching_indices
  position of fine grid wrt coarse grid
\param gamma_up
  spatial metric
\param kx_offset, ky_offset
  kernel offsets in the x and y directions
\param coarse_level, nlevels
    index of coarser level and total number of levels
*/
__global__ void restrict_interpolate_swe(float * qf_sw, float * q_c,
                                   int * nxs, int * nys, int * nzs,
                                   float dz, float zmin,
                                   int * matching_indices,
                                   float * gamma_up,
                                   int kx_offset, int ky_offset,
                                   int coarse_level, int nlevels);

/**
Restrict fine grid data to coarse grid

\param kernels, threads, blocks
   number of kernels, threads and blocks for each process/kernel
\param cumulative_kernels
   cumulative number of kernels in mpi processes of r < rank
\param q_cd, q_fd
   coarse and fine grids of state vectors
\param nxs, nys, nzs
   dimensions of grids
\param matching_indices
   position of fine grid wrt coarse grid
\param rho, gamma
   density and adiabatic index
\param gamma_up
   spatial metric
\param ng
   number of ghost cells
\param rank
   rank of MPI process
\param qf_swe
   grid of SWE variables on fine grid
\param coarse_level, nlevels
    index of coarser level and total number of levels
*/
void restrict_comp_to_swe(dim3 * kernels, dim3 * threads, dim3 * blocks,
                   int * cumulative_kernels, float * q_cd, float * q_fd,
                   int * nxs, int * nys, int * nzs,
                   float dz, float zmin, int * matching_indices,
                   float * rho, float gamma, float * gamma_up,
                   int ng, int rank, float * qf_swe,
                   int coarse_level, int nlevels);

/**
Interpolate fine grid compressible variables to get them on coarser compressible grid.

\param qf
 variables on fine grid
\param qc
 coarse grid state vector
\param nxs, nys, nzs
 grid dimensions
\param matching_indices
 position of fine grid wrt coarse grid
\param kx_offset, ky_offset
 kernel offsets in the x and y directions
\param clevel, nlevels
   index of coarser level and total number of levels
*/
__global__ void restrict_interpolate_comp(float * qf, float * qc,
                                    int * nxs, int * nys, int * nzs,
                                    int * matching_indices,
                                    int kx_offset, int ky_offset,
                                    int clevel, int nlevels);

/**
Restrict fine compressible grid data to coarse compressible grid.

\param kernels, threads, blocks
   number of kernels, threads and blocks for each process/kernel
\param cumulative_kernels
   cumulative number of kernels in mpi processes of r < rank
\param q_cd, q_fd
   coarse and fine grids of state vectors
\param nxs, nys, nzs
   dimensions of grids
\param matching_indices
   position of fine grid wrt coarse grid
\param ng
   number of ghost cells
\param rank
   rank of MPI process
\param coarse_level, nlevels
    index of coarser level and total number of levels
*/
void restrict_comp_to_comp(dim3 * kernels, dim3 * threads, dim3 * blocks,
                    int * cumulative_kernels, float * q_cd, float * q_fd,
                    int * nxs, int * nys, int * nzs,
                    int * matching_indices,
                    int ng, int rank,
                    int coarse_level, int nlevels);

/**
Interpolate multilayer SWE variables on fine grid to get them on single layer SWE coarse grid.

\param qf
    variables on fine grid
\param qc
    coarse grid state vector
\param nxs, nys, nzs
    grid dimensions
\param matching_indices
    position of fine grid wrt coarse grid
\param kx_offset, ky_offset
    kernel offsets in the x and y directions
\param clevel, nlevels
   index of coarser level and total number of levels
*/
__global__ void restrict_interpolate_swe_to_swe(float * qf, float * qc,
                                     int * nxs, int * nys, int * nzs,
                                     int * matching_indices,
                                     int kx_offset, int ky_offset,
                                     int clevel, int nlevels);

/**
Restrict fine multilayer swe grid data to coarse single layer swe grid.

\param kernels, threads, blocks
    number of kernels, threads and blocks for each process/kernel
\param cumulative_kernels
    cumulative number of kernels in mpi processes of r < rank
\param q_cd, q_fd
    coarse and fine grids of state vectors
\param nxs, nys, nzs
    dimensions of grids
\param matching_indices
    position of fine grid wrt coarse grid
\param ng
    number of ghost cells
\param rank
    rank of MPI process
\param coarse_level, nlevels
    index of coarser level and total number of levels
*/
void restrict_swe_to_swe(dim3 * kernels, dim3 * threads, dim3 * blocks,
                 int * cumulative_kernels, float * q_cd, float * q_fd,
                 int * nxs, int * nys, int * nzs,
                 int * matching_indices,
                 int ng, int rank,
                 int coarse_level, int nlevels);

/**
First part of evolution through one timestep using finite volume methods.
Reconstructs state vector to cell boundaries using slope limiter
and calculates fluxes there.

NOTE: we assume that beta is smooth so can get value at cell boundaries with simple averaging

\param beta_d
   shift vector at each grid point.
\param gamma_up_d
   gamma matrix at each grid point
\param Un_d
   state vector at each grid point in each layer
\param flux_func
   pointer to function to be used to calulate fluxes
\param qx_plus_half, qx_minus_half
   state vector reconstructed at right and left boundaries
\param qy_plus_half, qy_minus_half
   state vector reconstructed at top and bottom boundaries
\param fx_plus_half, fx_minus_half
   flux vector at right and left boundaries
\param fy_plus_half, fy_minus_half
   flux vector at top and bottom boundaries
\param nx, ny, nz
   dimensions of grid
\param alpha, gamma
   lapse function and adiabatic index
\param dx, dy, dt
   grid dimensions and timestep
\param kx_offset, ky_offset
   x, y offset for current kernel
*/
__global__ void evolve_fv(float * beta_d, float * gamma_up_d,
                    float * Un_d, flux_func_ptr flux_func,
                    float * qx_plus_half, float * qx_minus_half,
                    float * qy_plus_half, float * qy_minus_half,
                    float * fx_plus_half, float * fx_minus_half,
                    float * fy_plus_half, float * fy_minus_half,
                    int nx, int ny, int nz, int vec_dim, float alpha, float gamma,
                    float dx, float dy, float dt,
                    int kx_offset, int ky_offset);

/**
First part of evolution through one timestep using finite volume methods.
Reconstructs state vector to cell boundaries using slope limiter
and calculates fluxes there.

NOTE: we assume that beta is smooth so can get value at cell boundaries with simple averaging

\param beta_d
    shift vector at each grid point.
\param gamma_up_d
    gamma matrix at each grid point
\param Un_d
    state vector at each grid point in each layer
\param flux_func
    pointer to function to be used to calculate fluxes
\param qz_plus_half, qz_minus_half
    state vector reconstructed at top and bottom boundaries
\param fz_plus_half, fz_minus_half
    flux vector at top and bottom boundaries
\param nx, ny, nz
    dimensions of grid
\param vec_dim
    dimension of state vector
\param alpha, gamma
    lapse function and adiabatic index
\param dz, dt
    vertical grid spacing and timestep
\param kx_offset, ky_offset
    x, y offset for current kernel
*/
__global__ void evolve_z(float * beta_d, float * gamma_up_d,
                     float * Un_d, flux_func_ptr flux_func,
                     float * qz_plus_half, float * qz_minus_half,
                     float * fz_plus_half, float * fz_minus_half,
                     int nx, int ny, int nz, int vec_dim, float alpha, float gamma,
                     float dz, float dt,
                     int kx_offset, int ky_offset);

/**
Calculates fluxes in finite volume evolution by solving the Riemann
problem at the cell boundaries.

\param F
    flux vector at each point in grid and each layer
\param qx_plus_half, qx_minus_half
    state vector reconstructed at right and left boundaries
\param qy_plus_half, qy_minus_half
    state vector reconstructed at top and bottom boundaries
\param fx_plus_half, fx_minus_half
    flux vector at right and left boundaries
\param fy_plus_half, fy_minus_half
    flux vector at top and bottom boundaries
\param nx, ny, nz
    dimensions of grid
\param vec_dim
    dimension of state vector
\param alpha
    lapse function
\param dx, dy, dt
    gridpoint spacing and timestep spacing
\param kx_offset, ky_offset
    x, y offset for current kernel
*/
__global__ void evolve_fv_fluxes(float * F,
                  float * qx_plus_half, float * qx_minus_half,
                  float * qy_plus_half, float * qy_minus_half,
                  float * fx_plus_half, float * fx_minus_half,
                  float * fy_plus_half, float * fy_minus_half,
                  int nx, int ny, int nz, int vec_dim, float alpha,
                  float dx, float dy, float dt,
                  int kx_offset, int ky_offset);

/**
Calculates fluxes in finite volume evolution by solving the Riemann
problem at the cell boundaries in z direction.

\param F
  flux vector at each point in grid and each layer
\param qz_plus_half, qz_minus_half
  state vector reconstructed at right and left boundaries
\param fz_plus_half, fz_minus_half
  flux vector at top and bottom boundaries
\param nx, ny, nz
  dimensions of grid
\param vec_dim
  dimension of state vector
\param alpha
  lapse function
\param dz, dt
  gridpoint spacing and timestep spacing
\param kx_offset, ky_offset
  x, y offset for current kernel
*/
__global__ void evolve_z_fluxes(float * F,
                   float * qz_plus_half, float * qz_minus_half,
                   float * fz_plus_half, float * fz_minus_half,
                   int nx, int ny, int nz, int vec_dim, float alpha,
                   float dz, float dt,
                   int kx_offset, int ky_offset);

/**
Does the heating part of the evolution.

\param gamma_up_d
   gamma matrix at each grid point
\param Up
   state vector at next timestep
\param U_half
   state vector at half timestep
\param qx_plus_half, qx_minus_half
   state vector reconstructed at right and left boundaries
\param qy_plus_half, qy_minus_half
   state vector reconstructed at top and bottom boundaries
\param fx_plus_half, fx_minus_half
   flux vector at right and left boundaries
\param fy_plus_half, fy_minus_half
   flux vector at top and bottom boundaries
\param sum_phs
   sum of Phi in different layers
\param rho_d
   list of densities in different layers
\param Q_d
   heating rate in each layer
\param nx, ny, nlayers
   dimensions of grid
\param alpha, gamma
   lapse function and adibatic index
\param dx, dy, dt
   gridpoint spacing and timestep spacing
\param burning
   is burning present in this system?
\param Cv
    specific heat in constant volume
\param E_He
    energy release per unit mass of helium
\param kx_offset, ky_offset
   x, y offset for current kernel
*/
__global__ void evolve_fv_heating(float * gamma_up_d,
                    float * Up, float * U_half,
                    float * qx_plus_half, float * qx_minus_half,
                    float * qy_plus_half, float * qy_minus_half,
                    float * fx_plus_half, float * fx_minus_half,
                    float * fy_plus_half, float * fy_minus_half,
                    float * sum_phs, float * rho_d, float * Q_d,
                    int nx, int ny, int nlayers, float alpha, float gamma,
                    float dx, float dy, float dt,
                    bool burning, float Cv, float E_He,
                    int kx_offset, int ky_offset);

/**
Adds buoyancy terms.

\param Un_d
    state vector at each grid point in each layer at current timestep
\param Up
    state vector at next timestep
\param U_half
    state vector at half timestep
\param sum_phs
    sum of Phi in different layers
\param nx, ny, nlayers
    dimensions of grid
\param ng
    number of ghost cells
\param alpha
    lapse function
\param dx, dy, dt
    gridpoint spacing and timestep spacing
\param kx_offset, ky_offset
    x, y offset for current kernel
*/
__global__ void evolve2(float * Un_d, float * Up, float * U_half,
                     float * sum_phs,
                     int nx, int ny, int nlayers, int ng, float alpha,
                     float dx, float dy, float dt,
                     int kx_offset, int ky_offset);

/**
Solves the homogeneous part of the equation (ie the bit without source terms).

\param kernels, threads, blocks
    number of kernels, threads and blocks for each process/kernel
\param cumulative_kernels
    Cumulative total of kernels in ranks < rank of current MPI process
\param beta_d
    shift vector at each grid point
\param gamma_up_d
    gamma matrix at each grid point
\param Un_d
    state vector at each grid point in each layer at current timestep
\param F_d
    flux vector
\param qx_p_d, qx_m_d
    state vector reconstructed at right and left boundaries
\param qy_p_d, qy_m_d
    state vector reconstructed at top and bottom boundaries
\param fx_p_d, fx_m_d
    flux vector at right and left boundaries
\param fy_p_d, fy_m_d
    flux vector at top and bottom boundaries
\param nx, ny, nz
    dimensions of grid
\param alpha, gamma
    lapse function and adiabatic index
\param dx, dy, dz, dt
    gridpoint spacing and timestep spacing
\param rank
    rank of MPI process
\param flux_func
    pointer to function to be used to calculate fluxes
\param do_z
    should we evolve in the z direction?
*/
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

/**
Integrates the homogeneous part of the ODE in time using RK3.

\param kernels, threads, blocks
    number of kernels, threads and blocks for each process/kernel
\param cumulative_kernels
    Cumulative total of kernels in ranks < rank of current MPI process
\param beta_d
    shift vector at each grid point
\param gamma_up_d
    gamma matrix at each grid point
\param Un_d
    state vector at each grid point in each layer at current timestep on device
\param F_d
    flux vector on device
\param Up_d
    state vector at next timestep on device
\param qx_p_d, qx_m_d
    state vector reconstructed at right and left boundaries
\param qy_p_d, qy_m_d
    state vector reconstructed at top and bottom boundaries
\param fx_p_d, fx_m_d
    flux vector at right and left boundaries
\param fy_p_d, fy_m_d
    flux vector at top and bottom boundaries
\param nx, ny, nz
    dimensions of grid
\param vec_dim
    dimension of state vector
\param ng
    number of ghost cells
\param alpha, gamma
    lapse function and adiabatic index
\param dx, dy, dz, dt
    gridpoint spacing and timestep spacing
\param Up_h, F_h, Un_h
    state vector at next timestep, flux vector and state vector at current timestep on host
\param comm
    MPI communicator
\param status
    status of MPI processes
\param rank, n_processes
    rank of current MPI process and total number of MPI processes
\param flux_func
    pointer to function to be used to calculate fluxes
\param do_z
    should we evolve in the z direction?
*/
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

/**
Evolve system through nt timesteps, saving data to filename every dprint timesteps.

\param beta
   shift vector at each grid point
\param gamma_up
   gamma matrix at each grid point
\param rho
   densities in each layer
\param Q
   heating rate at each point and in each layer
\param nxs, nys, nzs
   dimensions of grids
\param nlevels
    number of levels of mesh refinement
\param models
    Array describing the physical model to use on each level. S = single layer SWE, M = multilayer SWE, C = compressible, L = Low Mach
\param vec_dims
    Dimensions of state vectors on each grid
\parm Us_h
    Array of pointers to grids.
\param ng
   number of ghost cells
\param nt
   total number of timesteps
\param alpha
   lapse function
\param gamma
   adiabatic index
\param E_He
    energy release per unit mass of helium burning
\param Cv
    specific heat per unit volume
\param zmin
   height of sea floor
\param dx, dy, dz, dt
   gridpoint spacing and timestep spacing
\param burning
   is burning included in this system?
\param dprint
   number of timesteps between each printout
\param filename
   name of file to which output is printed
\param comm
   MPI communicator
\param status
   status of MPI processes
\param rank, n_processes
   rank of current MPI process and total number of MPI processes
\param matching_indices
   position of fine grid wrt coarse grid
\param r
    ratio of grid resolutions
*/
void cuda_run(float * beta, float * gamma_up,
         float ** Us_h, float * rho, float * Q,
         int * nxs, int * nys, int * nzs, int nlevels, char * models,
         int * vec_dims, int ng,
         int nt, float alpha, float gamma, float E_He, float Cv,
         float zmin,
         float dx, float dy, float dz, float dt, bool burning,
         int dprint, char * filename,
         MPI_Comm comm, MPI_Status status, int rank, int n_processes,
         int * matching_indices, int r);

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

#endif
