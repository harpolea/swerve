#ifndef _GR_CUDA_KERNEL_H_
#define _GR_CUDA_KERNEL_H_

#include <stdio.h>
#include <mpi.h>
#include "H5Cpp.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_functions.h>

using namespace std;

/*
TODO: cuda_run is a beast, so split up into multiple functions to e.g. keep MPI stuff away from other stuff

TODO: at the moment, every processor and every kernel has all the data. Change this so that processors only have the data they need.

TODO: Change bcs etc so less copying of data to CPU - instead can use e.g. cudaMemcpyPeerAsync to copy between GPUs on the same node.

NOTE: debug memory leaks using
    mpirun -np 4 valgrind --leak-check=full ./gr_cuda

NOTE: run using
    mpirun -np 4 ./gr_cuda

*/

// prototypes

void getNumKernels(int nx, int ny, int nlayers, int ng, int n_processes, int *maxBlocks, int *maxThreads, dim3 *kernels, int *cumulative_kernels);

void getNumBlocksAndThreads(int nx, int ny, int nlayers, int ng, int maxBlocks, int maxThreads, int n_processes, dim3 *kernels, dim3 *blocks, dim3 *threads);

unsigned int nextPow2(unsigned int x);

void bcs_fv(float * grid, int nx, int ny, int nlayers, int ng);

void bcs_mpi(float * grid, int nx, int ny, int nlayers, int ng, MPI_Comm comm, MPI_Status status, int rank, int n_processes);

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

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

void getNumKernels(int nx, int ny, int nlayers, int ng, int n_processes, int *maxBlocks, int *maxThreads, dim3 * kernels, int * cumulative_kernels) {
    /*
    Return the number of kernels needed to run the problem given its size and the constraints of the GPU.
    */
    // won't actually use maxThreads - fix to account for the fact we want something square
    *maxThreads = nlayers * int(sqrt(float(*maxThreads)/nlayers)) * int(sqrt(*maxThreads/nlayers));
    *maxBlocks = int(sqrt(float(*maxBlocks))) * int(sqrt(float(*maxBlocks)));

    // calculate number of kernels needed

    if (nx*ny*nlayers > *maxBlocks * *maxThreads) {
        int kernels_x = int(ceil(float(nx-2*ng) / (sqrt(float(*maxThreads * *maxBlocks)/nlayers) - 2.0*ng)));
        int kernels_y = int(ceil(float(ny-2*ng) / (sqrt(float(*maxThreads * *maxBlocks)/nlayers) - 2.0*ng)));

        // easiest (but maybe not most balanced way) is to split kernels into strips
        // not enough kernels to fill all processes. This would be inefficient if ny > nx, so need to fix this to look in the y direction if this is the case.
        if (kernels_y < n_processes) {
            for (int i = 0; i < kernels_y; i++) {
                kernels[i].x = kernels_x;
                kernels[i].y = 1;
            }
            // blank out the other ones
            for (int i = kernels_y; i < n_processes; i++) {
                kernels[i].x = 0;
                kernels[i].y = 0;
            }
        } else {

            // split up in the y direction to keep stuff contiguous in memory
            int strip_width = int(floor(float(kernels_y) / float(n_processes)));


            for (int i = 0; i < n_processes; i++) {
                kernels[i].y = strip_width;
                kernels[i].x = kernels_x;
            }
            // give the last one the remainders
            kernels[n_processes-1].y += kernels_y - n_processes * strip_width;
        }

    } else {

        kernels[0].x = 1;
        kernels[0].y = 1;

        for (int i = 1; i < n_processes; i++) {
            kernels[i].x = 0;
            kernels[i].y = 0;
        }
    }

    cumulative_kernels[0] = kernels[0].x * kernels[0].y;

    for (int i = 1; i < n_processes; i++) {
      cumulative_kernels[i] = cumulative_kernels[i-1] + kernels[i].x * kernels[i].y;
    }
}

void getNumBlocksAndThreads(int nx, int ny, int nlayers, int ng, int maxBlocks, int maxThreads, int n_processes, dim3 *kernels, dim3 *blocks, dim3 *threads)
{
    /*
    Returns the number of blocks and threads required for each kernel given the size of the problem and the constraints of the device.
    */

    //get device capability, to avoid block/grid size exceed the upper bound
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    int total = (nx - 2*ng) * (ny - 2*ng) * nlayers;

    int kernels_x = 0;
    int kernels_y = 0;

    for (int i = 0; i < n_processes; i++) {
        kernels_y += kernels[i].y;
    }
    kernels_x = kernels[0].x;

    if ((kernels_x > 1) || (kernels_y > 1)) {
        // initialise
        threads[0].x = 0;
        threads[0].y = 0;
        blocks[0].x = 0;
        blocks[0].y = 0;


        for (int j = 0; j < (kernels_y-1); j++) {
            for (int i = 0; i < (kernels_x-1); i++) {
                threads[j*kernels_x + i].x = int(sqrt(float(maxThreads)/nlayers));
                threads[j*kernels_x + i].y = int(sqrt(float(maxThreads)/nlayers));
                threads[j*kernels_x + i].z = nlayers;

                blocks[j*kernels_x + i].x = int(sqrt(float(maxBlocks)));
                blocks[j*kernels_x + i].y = int(sqrt(float(maxBlocks)));
                blocks[j*kernels_x + i].z = 1;
            }

        }
        // kernels_x-1
        int nx_remaining = nx - threads[0].x * blocks[0].x * (kernels_x - 1) + 2 * ng;

        printf("nx_remaining: %i\n", nx_remaining);

        for (int j = 0; j < (kernels_y-1); j++) {

            threads[j*kernels_x + kernels_x-1].y =
                int(sqrt(float(maxThreads)/nlayers));
            threads[j*kernels_x + kernels_x-1].z = nlayers;

            threads[j*kernels_x + kernels_x-1].x =
                (nx_remaining < threads[j*kernels_x + kernels_x-1].y) ? nx_remaining : threads[j*kernels_x + kernels_x-1].y;

            blocks[j*kernels_x + kernels_x-1].x = int(ceil(float(nx_remaining) /
                float(threads[j*kernels_x + kernels_x-1].x)));
            blocks[j*kernels_x + kernels_x-1].y = int(sqrt(float(maxBlocks)));
            blocks[j*kernels_x + kernels_x-1].z = 1;
        }

        // kernels_y-1
        int ny_remaining = ny - threads[0].y * blocks[0].y * (kernels_y - 1) + 2 * ng;
        printf("ny_remaining: %i\n", ny_remaining);
        for (int i = 0; i < (kernels_x-1); i++) {

            threads[(kernels_y-1)*kernels_x + i].x =
                int(sqrt(float(maxThreads)/nlayers));
            threads[(kernels_y-1)*kernels_x + i].y =
                (ny_remaining < threads[(kernels_y-1)*kernels_x + i].x) ? ny_remaining : threads[(kernels_y-1)*kernels_x + i].x;
            threads[(kernels_y-1)*kernels_x + i].z = nlayers;

            blocks[(kernels_y-1)*kernels_x + i].x = int(sqrt(float(maxBlocks)));
            blocks[(kernels_y-1)*kernels_x + i].y = int(ceil(float(ny_remaining) /
                float(threads[(kernels_y-1)*kernels_x + i].y)));
            blocks[(kernels_y-1)*kernels_x + i].z = 1;
        }

        // recalculate
        nx_remaining = nx - threads[0].x * blocks[0].x * (kernels_x - 1) + 2 * ng;
        ny_remaining = ny - threads[0].y * blocks[0].y * (kernels_y - 1) + 2 * ng;
        printf("nx_remaining: %i\n", nx_remaining);
        printf("ny_remaining: %i\n", ny_remaining);

        // (kernels_x-1, kernels_y-1)
        threads[(kernels_y-1)*kernels_x + kernels_x-1].x =
            (nx_remaining < int(sqrt(float(maxThreads)/nlayers))) ? nx_remaining : int(sqrt(float(maxThreads/nlayers)));
        threads[(kernels_y-1)*kernels_x + kernels_x-1].y =
            (ny_remaining < int(sqrt(float(maxThreads)/nlayers))) ? ny_remaining : int(sqrt(float(maxThreads)/nlayers));
        threads[(kernels_y-1)*kernels_x + kernels_x-1].z = nlayers;

        blocks[(kernels_y-1)*kernels_x + kernels_x-1].x =
            int(ceil(float(nx_remaining) /
            float(threads[(kernels_y-1)*kernels_x + kernels_x-1].x)));
        blocks[(kernels_y-1)*kernels_x + kernels_x-1].y =
            int(ceil(float(ny_remaining) /
            float(threads[(kernels_y-1)*kernels_x + kernels_x-1].y)));
        blocks[(kernels_y-1)*kernels_x + kernels_x-1].z = 1;

    } else {

        int total_threads = (total < maxThreads*2) ? nextPow2((total + 1)/ 2) : maxThreads;
        threads[0].x = int(floor(sqrt(float(total_threads)/float(nlayers))));
        threads[0].y = int(floor(sqrt(float(total_threads)/float(nlayers))));
        threads[0].z = nlayers;
        total_threads = threads[0].x * threads[0].y * threads[0].z;
        int total_blocks = int(ceil(float(total) / float(total_threads)));

        blocks[0].x = int(ceil(sqrt(float(total_blocks)/float(nx*ny))*nx));
        blocks[0].y = int(ceil(sqrt(float(total_blocks)/float(nx*ny))*ny));
        blocks[0].z = 1;

        total_blocks = blocks[0].x * blocks[0].y;

        if ((float)total_threads*total_blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
        {
            printf("n is too large, please choose a smaller number!\n");
        }

        if (total_blocks > prop.maxGridSize[0])
        {
            printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
                   total_blocks, prop.maxGridSize[0], total_threads*2, total_threads);

            blocks[0].x /= 2;
            blocks[0].y /= 2;
            threads[0].x *= 2;
            threads[0].y *= 2;
        }

    }
}


__device__ void bcs(float * grid, int nx, int ny, int nlayers, int kx_offset, int ky_offset) {
    /*
    Enforce boundary conditions on section of grid.
    */
    // outflow
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int l = threadIdx.z;

    if ((l < nlayers) && (y < ny) && (x < nx) ) {
        for (int i = 0; i < 4; i++) {
            if (x == 0) {
                grid[((y * nx) * nlayers + l)*4+i] = grid[((y * nx + 1) * nlayers + l)*4+i];
            } else if (x == (nx-1)) {
                grid[((y * nx + (nx-1)) * nlayers + l)*4+i] = grid[((y * nx + (nx-2)) * nlayers + l)*4+i];
            } else if (y == 0) {
                grid[(x * nlayers + l)*4+i] = grid[((nx + x) * nlayers + l)*4+i];
            } else if (y == (ny-1)) {
                grid[(((ny-1) * nx + x) * nlayers + l)*4+i] = grid[(((ny-2) * nx + x) * nlayers + l)*4+i];
            }
        }
    }
}

__device__ void bcs_fv(float * grid, int nx, int ny, int nlayers, int ng, int kx_offset, int ky_offset) {
    /*
    Enforce boundary conditions on section of grid.
    */
    // outflow
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int l = threadIdx.z;

    if ((l < nlayers) && (y < ny) && (x < nx) ) {
        for (int i = 0; i < 4; i++) {
            if (x < ng) {
                grid[((y * nx + x) * nlayers + l)*4+i] = grid[((y * nx + ng) * nlayers + l)*4+i];
            } else if (x > (nx-ng-1)) {
                grid[((y * nx + x) * nlayers + l)*4+i] = grid[((y * nx + (nx-ng-1)) * nlayers + l)*4+i];
            } else if (y < ng) {
                grid[((y * nx + x) * nlayers + l)*4+i] = grid[(((ng * nx + x) *  + x) * nlayers + l)*4+i];
            } else if (y > (ny-ng-1)) {
                grid[((y * nx + x) * nlayers + l)*4+i] = grid[(((ny-ng-1) * nx + x) * nlayers + l)*4+i];
            }
        }
    }
}

void bcs_fv(float * grid, int nx, int ny, int nlayers, int ng) {
    /*
    Enforce boundary conditions on section of grid.
    */
    // outflow

    for (int l = 0; l < nlayers; l++) {
        for (int y = 0; y < ny; y++){
            for (int i = 0; i < 4; i++) {
                for (int g = 0; g < ng; g++) {
                    grid[((y * nx + g) * nlayers + l)*4+i] = grid[((y * nx + ng) * nlayers + l)*4+i];

                    grid[((y * nx + (nx-1-g)) * nlayers + l)*4+i] = grid[((y * nx + (nx-1-ng)) * nlayers + l)*4+i];
                }
            }
        }
        for (int x = 0; x < nx; x++){
            for (int i = 0; i < 4; i++) {
                for (int g = 0; g < ng; g++) {
                    grid[((g * nx + x) * nlayers + l)*4+i] = grid[((ng * nx + x) * nlayers + l)*4+i];

                    grid[(((ny-1-g) * nx + x) * nlayers + l)*4+i] = grid[(((ny-1-ng) * nx + x) * nlayers + l)*4+i];
                }
            }

        }
    }
}

void bcs_mpi(float * grid, int nx, int ny, int nlayers, int ng, MPI_Comm comm, MPI_Status status, int rank, int n_processes) {
    /*
    Enforce boundary conditions across processes / at edges of grid.

    Loops have been ordered in a way so as to try and keep memory accesses as contiguous as possible.

    Need to do non-blocking send, blocking receive then wait.

    NOTE: this assumes each process only has the data it works on which is not true - change this (eg by including kernel offsets)
    */

    // interior cells between processes

    // make some buffers for sending and receiving
    float * ysbuf = new float[nlayers*nx*ng*4];
    float * yrbuf = new float[nlayers*nx*ng*4];

    int tag = 1;
    MPI_Request request;

    // if there are process above and below, send/receive
    if ((rank > 0) && (rank < n_processes-1)) {
        // send to below, receive from above
        // copy stuff to buffer
        for (int g = 0; g < ng; g++){
            for (int x = 0; x < nx; x++) {
                for (int i = 0; i < 4; i++) {
                    for (int l = 0; l < nlayers; l++) {
                        ysbuf[((g * nx + x) * nlayers + l)*4+i] = grid[((g * nx + x) * nlayers + l)*4+i];
                    }
                }
            }
        }
        MPI_Issend(ysbuf, nlayers*nx*ng*4, MPI_FLOAT, rank-1, tag, comm, &request);
        MPI_Recv(yrbuf, nlayers*nx*ng*4, MPI_FLOAT, rank+1, tag, comm, &status);
        MPI_Wait(&request, &status);

        // copy received data back to grid
        for (int g = 0; g < ng; g++){
            for (int x = 0; x < nx; x++) {
                for (int l = 0; l < nlayers; l++) {
                    for (int i = 0; i < 4; i++) {
                        grid[(((ny-2*ng+g) * nx + x) * nlayers + l)*4+i] = yrbuf[((g * nx + x) * nlayers + l)*4+i];
                    }
                }
            }
        }
        // send to above, receive from below
        // copy stuff to buffer
        for (int g = 0; g < ng; g++){
            for (int x = 0; x < nx; x++) {
                for (int i = 0; i < 4; i++) {
                    for (int l = 0; l < nlayers; l++) {
                        ysbuf[((g * nx + x) * nlayers + l)*4+i] = grid[(((ny-2*ng+g) * nx + x) * nlayers + l)*4+i];
                    }
                }
            }
        }
        MPI_Issend(ysbuf, nlayers*nx*ng*4, MPI_FLOAT, rank+1, tag, comm, &request);
        MPI_Recv(yrbuf, nlayers*nx*ng*4, MPI_FLOAT, rank-1, tag, comm, &status);
        MPI_Wait(&request, &status);

        // copy received data back to grid
        for (int g = 0; g < ng; g++){
            for (int x = 0; x < nx; x++) {
                for (int l = 0; l < nlayers; l++) {
                    for (int i = 0; i < 4; i++) {
                        grid[((g * nx + x) * nlayers + l)*4+i] = yrbuf[((g * nx + x) * nlayers + l)*4+i];
                    }
                }
            }
        }

    } else if (rank == 0) {
        // do outflow for top boundary
        // copy stuff to buffer
        for (int g = 0; g < ng; g++){
            for (int x = 0; x < nx; x++) {
                for (int l = 0; l < nlayers; l++) {
                    for (int i = 0; i < 4; i++) {
                        ysbuf[((g * nx + x) * nlayers + l)*4+i] = grid[(((ny-2*ng+g) * nx + x) * nlayers + l)*4+i];
                    }
                }
            }
        }

        MPI_Issend(ysbuf, nlayers*nx*ng*4, MPI_FLOAT, 1, tag, comm, &request);
        MPI_Recv(yrbuf, nlayers*nx*ng*4, MPI_FLOAT, 1, tag, comm, &status);
        MPI_Wait(&request, &status);

        // copy received data back to grid
        for (int g = 0; g < ng; g++){
            for (int x = 0; x < nx; x++) {
                for (int l = 0; l < nlayers; l++) {
                    for (int i = 0; i < 4; i++) {
                        grid[(((ny-2*ng+g) * nx + x) * nlayers + l)*4+i] = yrbuf[((g * nx + x) * nlayers + l)*4+i];
                    }
                }
            }
        }

        // outflow stuff on top boundary
        for (int g = 0; g < ng; g++){
            for (int x = 0; x < nx; x++) {
                for (int i = 0; i < 4; i++) {
                    for (int l = 0; l < nlayers; l++) {

                        grid[((g * nx + x) * nlayers + l)*4+i] = grid[((ng * nx + x) * nlayers + l)*4+i];
                    }
                }
            }
        }

    } else {
        // copy stuff to buffer
        for (int g = 0; g < ng; g++){
            for (int x = 0; x < nx; x++) {
                for (int i = 0; i < 4; i++) {
                    for (int l = 0; l < nlayers; l++) {
                        ysbuf[((g * nx + x) * nlayers + l)*4+i] = grid[((g * nx + x) * nlayers + l)*4+i];
                    }
                }
            }
        }
        // bottom-most process
        MPI_Issend(ysbuf, nlayers*nx*ng*4, MPI_FLOAT, rank-1, tag, comm, &request);
        MPI_Recv(yrbuf, nlayers*nx*ng*4, MPI_FLOAT, rank-1, tag, comm, &status);
        MPI_Wait(&request, &status);

        // copy received data back to grid
        for (int g = 0; g < ng; g++){
            for (int x = 0; x < nx; x++) {
                for (int i = 0; i < 4; i++) {
                    for (int l = 0; l < nlayers; l++) {
                        grid[((g * nx + x) * nlayers + l)*4+i] = yrbuf[((g * nx + x) * nlayers + l)*4+i];
                    }
                }
            }
        }

        // outflow for bottom boundary
        for (int g = 0; g < ng; g++){
            for (int x = 0; x < nx; x++) {
                for (int i = 0; i < 4; i++) {
                    for (int l = 0; l < nlayers; l++) {

                        grid[(((ny-1-g) * nx + x) * nlayers + l)*4+i] = grid[(((ny-1-ng) * nx + x) * nlayers + l)*4+i];
                    }
                }
            }
        }
    }

    delete[] ysbuf;
    delete[] yrbuf;
}


__device__ void calc_Q(float * U, float * rho_d, float * Q_d,
                       int nx, int ny, int nlayers,
                       int kx_offset, int ky_offset, bool burning) {
    /*
    Calculate heating rate using equation 64 of Spitkovsky+ 2002.
    */

    if (burning) {
        int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
        int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
        int l = threadIdx.z;

        // set some constants
        //float kappa = 0.03; // opacity, constant
        //float column_depth = 5.4; // y
        float Y = 1.0; // for simplicity as they do just have eps_3alpha = 0 so that helium abundance remains constant.

        // in this model the scale height represents the temperature

        if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1)) && (l < nlayers)) {
            // changed to e^-35 to try and help GPU
            Q_d[(y * nx + x) * nlayers + l] = 3.0e13 * rho_d[l]*rho_d[l] * pow(Y, 3) * exp(-35.0/U[((y * nx + x) * nlayers + l)*4]) / pow(U[((y * nx + x) * nlayers + l)*4], 3); //- 0.4622811 * pow(U[((y * nx + x) * nlayers + l)*4], 4) / (3.0 * kappa * column_depth * column_depth);
        }
    }
}


__global__ void evolve_fv(float * beta_d, float * gamma_up_d,
                     float * Un_d,
                     float * qx_plus_half, float * qx_minus_half,
                     float * qy_plus_half, float * qy_minus_half,
                     float * fx_plus_half, float * fx_minus_half,
                     float * fy_plus_half, float * fy_minus_half,
                     int nx, int ny, int nlayers, float alpha,
                     float dx, float dy, float dt,
                     int kx_offset, int ky_offset) {
    /*
    First part of evolution through one timestep using finite volume methods.
    Reconstructs state vector to cell boundaries using slope limiter
    and calculates fluxes there.

    NOTE: we assume that beta is smooth so can get value at cell boundaries with simple averaging
    */

    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int l = threadIdx.z;

    int offset = ((y * nx + x) * nlayers + l) * 4;

    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1)) && (l < nlayers)) {

        // x-direction
        for (int i = 0; i < 4; i++) {
            float S_upwind = (Un_d[((y * nx + x+1) * nlayers + l) * 4 + i] -
                Un_d[((y * nx + x) * nlayers + l) * 4 + i]) / dx;
            float S_downwind = (Un_d[((y * nx + x) * nlayers + l) * 4 + i] -
                Un_d[((y * nx + x-1) * nlayers + l) * 4 + i]) / dx;
            float S = 0.5 * (S_upwind + S_downwind); // S_av

            float r = 1.0e5;

            // make sure don't divide by zero
            if (abs(S_downwind) > 1.0e-5) {
                r = S_upwind / S_downwind;
            }

            // MC
            //float phi = max(float(0.0), min(float((2.0 * r) / (1.0 + r)), float(2.0 / (1.0 + r))));
            // superbee
            float phi = 0.0;
            if (r >= 1.0) {
                phi = min(float(2.0), min(r, float(2.0 / (1.0 + r))));
            } else if (r >= 0.5) {
                phi = 1.0;
            } else if (r > 0.0) {
                phi = 2.0 * r;
            }

            S *= phi;

            qx_plus_half[offset + i] = Un_d[offset + i] + S * 0.5 * dx;
            qx_minus_half[offset + i] = Un_d[offset + i] - S * 0.5 * dx;

            // initialise
            fx_plus_half[offset + i] = 0.0;
            fx_minus_half[offset + i] = 0.0;
        }

        // plus half stuff

        float W = sqrt(
            float(qx_plus_half[offset + 1] * qx_plus_half[offset + 1] *
            gamma_up_d[0] +
            2.0 * qx_plus_half[offset + 1] * qx_plus_half[offset + 2] *
            gamma_up_d[1] +
            qx_plus_half[offset + 2] * qx_plus_half[offset + 2] *
            gamma_up_d[3]) /
            (qx_plus_half[offset] * qx_plus_half[offset]) + 1.0);

        float u = qx_plus_half[offset + 1] / (qx_plus_half[offset] * W);
        float v = qx_plus_half[offset + 2] / (qx_plus_half[offset] * W);
        // beta[0] at i+1/2, j
        float beta = 0.5 * (beta_d[(y * nx + x) * 2] + beta_d[(y * nx + x+1) * 2]);
        float qx = u * gamma_up_d[0] + v * gamma_up_d[1] - beta / alpha;

        fx_plus_half[offset] = qx_plus_half[offset] * qx;

        fx_plus_half[offset + 1] = qx_plus_half[offset + 1] * qx +
            0.5 * qx_plus_half[offset] * qx_plus_half[offset] / (W*W);

        fx_plus_half[offset + 2] = qx_plus_half[offset + 2] * qx;

        fx_plus_half[offset + 3] = qx_plus_half[offset + 3] * qx;

        // minus half stuff
        W = sqrt(
            float(qx_minus_half[offset + 1] * qx_minus_half[offset + 1] *
            gamma_up_d[0] +
            2.0 * qx_minus_half[offset + 1] * qx_minus_half[offset + 2] *
            gamma_up_d[1] +
            qx_minus_half[offset + 2] * qx_minus_half[offset + 2] *
            gamma_up_d[3]) /
            (qx_minus_half[offset] * qx_minus_half[offset]) + 1.0);

        u = qx_minus_half[offset + 1] / (qx_minus_half[offset] * W);
        v = qx_minus_half[offset + 2] / (qx_minus_half[offset] * W);
        // beta[0] at i-1/2, j
        beta = 0.5 * (beta_d[(y * nx + x-1) * 2] + beta_d[(y * nx + x) * 2]);
        qx = u * gamma_up_d[0] + v * gamma_up_d[1] - beta / alpha;

        fx_minus_half[offset] = qx_minus_half[offset] * qx;
        fx_minus_half[offset + 1] = qx_minus_half[offset + 1] * qx +
            0.5 * qx_minus_half[offset] * qx_minus_half[offset] / (W*W);
        fx_minus_half[offset + 2] = qx_minus_half[offset + 2] * qx;
        fx_minus_half[offset + 3] = qx_minus_half[offset + 3] * qx;

        // y-direction
        for (int i = 0; i < 4; i++) {
            float S_upwind = (Un_d[(((y+1) * nx + x) * nlayers + l) * 4 + i] -
                Un_d[((y * nx + x) * nlayers + l) * 4 + i]) / dy;
            float S_downwind = (Un_d[((y * nx + x) * nlayers + l) * 4 + i] -
                Un_d[(((y-1) * nx + x) * nlayers + l) * 4 + i]) / dy;
            float S = 0.5 * (S_upwind + S_downwind); // S_av

            float r = 1.0e5;

            // make sure don't divide by zero
            if (abs(S_downwind) > 1.0e-5) {
                r = S_upwind / S_downwind;
            }

            // MC
            //float phi = max(float(0.0), min(float((2.0 * r) / (1.0 + r)), float(2.0 / (1.0 + r))));
            // superbee
            float phi = 0.0;
            if (r >= 1.0) {
                phi = min(float(2.0), min(r, float(2.0 / (1.0 + r))));
            } else if (r >= 0.5) {
                phi = 1.0;
            } else if (r > 0.0) {
                phi = 2.0 * r;
            }

            S *= phi;

            qy_plus_half[offset + i] = Un_d[offset + i] + S * 0.5 * dy;
            qy_minus_half[offset + i] = Un_d[offset + i] - S * 0.5 * dy;

            // initialise
            fy_plus_half[offset + i] = 0.0;
            fy_minus_half[offset + i] = 0.0;
        }

        // plus half stuff

        W = sqrt(
            float(qy_plus_half[offset + 1] * qy_plus_half[offset + 1] *
            gamma_up_d[0] +
            2.0 * qy_plus_half[offset + 1] * qy_plus_half[offset + 2] *
            gamma_up_d[1] +
            qy_plus_half[offset + 2] * qy_plus_half[offset + 2] *
            gamma_up_d[3]) /
            (qy_plus_half[offset] * qy_plus_half[offset]) + 1.0);

        u = qy_plus_half[offset + 1] / (qy_plus_half[offset] * W);
        v = qy_plus_half[offset + 2] / (qy_plus_half[offset] * W);
        // beta[1] at i, j+1/2
        beta = 0.5 * (beta_d[((y+1) * nx + x) * 2 + 1] + beta_d[(y * nx + x) * 2 + 1]);
        float qy = v * gamma_up_d[3] + u * gamma_up_d[1] - beta / alpha;

        fy_plus_half[offset] = qy_plus_half[offset] * qy;
        fy_plus_half[offset + 1] = qy_plus_half[offset + 1] * qy;
        fy_plus_half[offset + 2] = qy_plus_half[offset + 2] * qy +
            0.5 * qy_plus_half[offset] * qy_plus_half[offset] / (W*W);
        fy_plus_half[offset + 3] = qy_plus_half[offset + 3] * qy;

        // minus half stuff
        W = sqrt(
            float(qy_minus_half[offset+1] * qy_minus_half[offset+1] *
            gamma_up_d[0] +
            2.0 * qy_minus_half[offset + 1] * qy_minus_half[offset + 2] *
            gamma_up_d[1] +
            qy_minus_half[offset+2] * qy_minus_half[offset + 2] *
            gamma_up_d[3]) /
            (qy_minus_half[offset]*qy_minus_half[offset]) + 1.0);

        u = qy_minus_half[offset + 1] / (qy_minus_half[offset] * W);
        v = qy_minus_half[offset + 2] / (qy_minus_half[offset] * W);
        // beta[1] at i, j-1/2
        beta = 0.5 * (beta_d[((y-1) * nx + x) * 2 + 1] + beta_d[(y * nx + x) * 2 + 1]);
        qy = v * gamma_up_d[3] + u * gamma_up_d[1] - beta / alpha;

        fy_minus_half[offset] = qy_minus_half[offset] * qy;
        fy_minus_half[offset + 1] = qy_minus_half[offset + 1] * qy;
        fy_minus_half[offset + 2] = qy_minus_half[offset + 2] * qy +
            0.5 * qy_minus_half[offset] * qy_minus_half[offset] / (W*W);
        fy_minus_half[offset + 3] = qy_minus_half[offset + 3] * qy;
    }
}

__global__ void evolve_fv_fluxes(float * F,
                     float * qx_plus_half, float * qx_minus_half,
                     float * qy_plus_half, float * qy_minus_half,
                     float * fx_plus_half, float * fx_minus_half,
                     float * fy_plus_half, float * fy_minus_half,
                     int nx, int ny, int nlayers, float alpha,
                     float dx, float dy, float dt,
                     int kx_offset, int ky_offset) {
    /*
    Calculates fluxes in finite volume evolution by solving the Riemann
    problem at the cell boundaries.
    */
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int l = threadIdx.z;

    // do fluxes
    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1)) && (l < nlayers)) {
        for (int i = 0; i < 4; i++) {
            // x-boundary
            // from i-1
            float fx_m = 0.5 * (
                fx_plus_half[((y * nx + x-1) * nlayers + l) * 4 + i] +
                fx_minus_half[((y * nx + x) * nlayers + l) * 4 + i] +
                qx_plus_half[((y * nx + x-1) * nlayers + l) * 4 + i] -
                qx_minus_half[((y * nx + x) * nlayers + l) * 4 + i]);
            // from i+1
            float fx_p = 0.5 * (
                fx_plus_half[((y * nx + x) * nlayers + l) * 4 + i] +
                fx_minus_half[((y * nx + x+1) * nlayers + l) * 4 + i] +
                qx_plus_half[((y * nx + x) * nlayers + l) * 4 + i] -
                qx_minus_half[((y * nx + x+1) * nlayers + l) * 4 + i]);

            // y-boundary
            // from j-1
            float fy_m = 0.5 * (
                fy_plus_half[(((y-1) * nx + x) * nlayers + l) * 4 + i] +
                fy_minus_half[((y * nx + x) * nlayers + l) * 4 + i] +
                qy_plus_half[(((y-1) * nx + x) * nlayers + l) * 4 + i] -
                qy_minus_half[((y * nx + x) * nlayers + l) * 4 + i]);
            // from j+1
            float fy_p = 0.5 * (
                fy_plus_half[((y * nx + x) * nlayers + l) * 4 + i] +
                fy_minus_half[(((y+1) * nx + x) * nlayers + l) * 4 + i] +
                qy_plus_half[((y * nx + x) * nlayers + l) * 4 + i] -
                qy_minus_half[(((y+1) * nx + x) * nlayers + l) * 4 + i]);

            F[((y * nx + x) * nlayers + l)*4 + i] =
                -((1.0/dx) * alpha * (fx_p - fx_m) +
                (1.0/dy) * alpha * (fy_p - fy_m));
        }
    }
}

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
                     int kx_offset, int ky_offset) {
    /*
    Does the heating part of the evolution.
    */
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int l = threadIdx.z;


    // copy to U_half
    if ((x < nx) && (y < ny) && (l < nlayers)) {
        for (int i = 0; i < 4; i++) {
            U_half[((y * nx + x) * nlayers + l)*4+i] =
                Up[((y * nx + x) * nlayers + l)*4+i];
        }
    }

    // calculate Q
    calc_Q(Up, rho_d, Q_d, nx, ny, nlayers, kx_offset, ky_offset, burning);

    float W = 1.0;

    // do source terms
    if ((x < nx) && (y < ny) && (l < nlayers)) {

        W = sqrt(float((U_half[((y * nx + x) * nlayers + l)*4+1] *
            U_half[((y * nx + x) * nlayers + l)*4+1] * gamma_up_d[0] +
            2.0 * U_half[((y * nx + x) * nlayers + l)*4+1] *
            U_half[((y * nx + x) * nlayers + l)*4+2] *
            gamma_up_d[1] +
            U_half[((y * nx + x) * nlayers + l)*4+2] *
            U_half[((y * nx + x) * nlayers + l)*4+2] *
            gamma_up_d[3]) /
            (U_half[((y * nx + x) * nlayers + l)*4] *
            U_half[((y * nx + x) * nlayers + l)*4]) + 1.0));

        U_half[((y * nx + x) * nlayers + l)*4] /= W;

    }

    __syncthreads();

    if ((x < nx) && (y < ny) && (l < nlayers)) {

        sum_phs[(y * nx + x) * nlayers + l] = 0.0;


        float sum_qs = 0.0;
        float deltaQx = 0.0;
        float deltaQy = 0.0;

        if (l < (nlayers - 1)) {
            sum_qs += (Q_d[(y * nx + x) * nlayers + l+1] - Q_d[(y * nx + x) * nlayers + l]);
            deltaQx = (Q_d[(y * nx + x) * nlayers + l] + mu) *
                (U_half[((y * nx + x) * nlayers + l)*4+1] -
                 U_half[((y * nx + x) * nlayers + (l+1))*4+1]) /
                 (W*U_half[((y * nx + x) * nlayers + l)*4]);
            deltaQy = (Q_d[(y * nx + x) * nlayers + l] + mu) *
                (U_half[((y * nx + x) * nlayers + l)*4+2] -
                 U_half[((y * nx + x) * nlayers + (l+1))*4+2]) /
                 (W*U_half[((y * nx + x) * nlayers + l)*4]);
        }
        if (l > 0) {
            sum_qs += -rho_d[l-1] / rho_d[l] * (Q_d[(y * nx + x) * nlayers + l] - Q_d[(y * nx + x) * nlayers + l-1]);
            deltaQx = rho_d[l-1] / rho_d[l] *
                (Q_d[(y * nx + x) * nlayers + l] + mu) *
                (U_half[((y * nx + x) * nlayers + l)*4+1] -
                 U_half[((y * nx + x) * nlayers + l-1)*4+1]) /
                 (W*U_half[((y * nx + x) * nlayers + l)*4]);
            deltaQy = rho_d[l-1] / rho_d[l] *
                (Q_d[(y * nx + x) * nlayers + l] + mu) *
                (U_half[((y * nx + x) * nlayers + l)*4+2] -
                 U_half[((y * nx + x) * nlayers + l-1)*4+2]) /
                 (W*U_half[((y * nx + x) * nlayers + l)*4]);
        }

        for (int j = 0; j < l; j++) {
            sum_phs[(y * nx + x) * nlayers + l] += rho_d[j] / rho_d[l] *
                U_half[((y * nx + x) * nlayers + j)*4];
        }
        for (int j = l+1; j < nlayers; j++) {
            sum_phs[(y * nx + x) * nlayers + l] = sum_phs[(y * nx + x) * nlayers + l] +
                U_half[((y * nx + x) * nlayers + j)*4];
        }

        // D
        Up[((y * nx + x) * nlayers + l)*4] += dt * alpha * sum_qs;

        // Sx
        Up[((y * nx + x) * nlayers + l)*4+1] += dt * alpha * (-deltaQx);

        // Sy
        Up[((y * nx + x) * nlayers + l)*4+2] += dt * alpha * (-deltaQy);

        // zeta
        Up[((y * nx + x) * nlayers + l)*4+3] += -dt * alpha * Q_d[(y * nx + x) * nlayers + l] * rho_d[l];

    }
}


__global__ void evolve2(float * gamma_up_d,
                     float * Un_d, float * Up, float * U_half,
                     float * sum_phs, float * rho_d, float * Q_d,
                     float mu,
                     int nx, int ny, int nlayers, int ng, float alpha,
                     float dx, float dy, float dt,
                     int kx_offset, int ky_offset) {
    /*
    Adds buoyancy terms.
    */
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int l = threadIdx.z;

    //printf("kx_offset: %i\n", kx_offset);

    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1)) && (l < nlayers)) {

        float a = dt * alpha *
            U_half[((y * nx + x) * nlayers + l)*4] * (0.5 / dx) * (sum_phs[(y * nx + x+1) * nlayers + l] -
            sum_phs[(y * nx + x-1) * nlayers + l]);

        if (abs(a) < 0.9 * dx / dt) {
            Up[((y * nx + x) * nlayers + l)*4+1] = Up[((y * nx + x) * nlayers + l)*4+1] - a;
        }

        a = dt * alpha *
            U_half[((y * nx + x) * nlayers + l)*4] * (0.5 / dy) *
            (sum_phs[((y+1) * nx + x) * nlayers + l] -
             sum_phs[((y-1) * nx + x) * nlayers + l]);

        if (abs(a) < 0.9 * dy / dt) {
            Up[((y * nx + x) * nlayers + l)*4+2] = Up[((y * nx + x) * nlayers + l)*4+2] - a;
        }

    }

    __syncthreads();

    bcs_fv(Up, nx, ny, nlayers, ng-1, kx_offset, ky_offset);

    // copy back to grid
    if ((x < nx) && (y < ny) && (l < nlayers)) {
        for (int i = 0; i < 4; i++) {
            Un_d[((y * nx + x) * nlayers + l)*4+i] =
                Up[((y * nx + x) * nlayers + l)*4+i];
        }
    }
}

void homogeneuous_fv(dim3 * kernels, dim3 * threads, dim3 * blocks, float * beta_d, float * gamma_up_d,
       float * Un_d, float * F_d,
       float * qx_p_d, float * qx_m_d, float * qy_p_d, float * qy_m_d,
       float * fx_p_d, float * fx_m_d, float * fy_p_d, float * fy_m_d,
       int nx, int ny, int nlayers, float alpha,
       float dx, float dy, float dt, int rank) {
    /*
    Solves the homogeneous part of the equation (ie the bit without source terms).
    */

    int kx_offset = 0;
    int ky_offset = kernels[0].y * rank * blocks[0].y * threads[0].y;

    for (int j = 0; j < kernels[rank].y; j++) {
       kx_offset = 0;
       for (int i = 0; i < kernels[rank].x; i++) {
           evolve_fv<<<blocks[j * kernels[rank].x + i], threads[j * kernels[rank].x + i]>>>(beta_d, gamma_up_d, Un_d,
                  qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                  fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                  nx, ny, nlayers, alpha,
                  dx, dy, dt, kx_offset, ky_offset);
          kx_offset += blocks[j * kernels[rank].x + i].x * threads[j * kernels[rank].x + i].x;
       }
       ky_offset += blocks[j * kernels[rank].x].y * threads[j * kernels[rank].x].y;
    }

    ky_offset = kernels[0].y * rank * blocks[0].y * threads[0].y;

    for (int j = 0; j < kernels[rank].y; j++) {
       kx_offset = 0;
       for (int i = 0; i < kernels[rank].x; i++) {
           evolve_fv_fluxes<<<blocks[j * kernels[rank].x + i],
                              threads[j * kernels[rank].x + i]>>>(
                  F_d,
                  qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                  fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                  nx, ny, nlayers, alpha,
                  dx, dy, dt, kx_offset, ky_offset);

           kx_offset += blocks[j * kernels[rank].x + i].x * threads[j * kernels[rank].x + i].x;
       }
       ky_offset += blocks[j * kernels[rank].x].y * threads[j * kernels[rank].x].y;
    }
}

void rk3_fv(dim3 * kernels, dim3 * threads, dim3 * blocks,
       float * beta_d, float * gamma_up_d, float * Un_d,
       float * F_d, float * Up_d,
       float * qx_p_d, float * qx_m_d, float * qy_p_d, float * qy_m_d,
       float * fx_p_d, float * fx_m_d, float * fy_p_d, float * fy_m_d,
       int nx, int ny, int nlayers, int ng, float alpha,
       float dx, float dy, float dt,
       float * Up_h, float * F_h, float * Un_h,
       MPI_Comm comm, MPI_Status status, int rank, int n_processes) {
    /*
    Integrates the homogeneous part of the ODE in time using RK3.
    */

    // u1 = un + dt * F(un)
    homogeneuous_fv(kernels, threads, blocks,
          beta_d, gamma_up_d, Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d,
          nx, ny, nlayers, alpha,
          dx, dy, dt, rank);

    // copy back flux
    cudaMemcpy(F_h, F_d, nx*ny*nlayers*4*sizeof(float), cudaMemcpyDeviceToHost);
    //bcs_fv(F_h, nx, ny, nlayers, ng);
    if (n_processes == 1) {
        bcs_fv(F_h, nx, ny, nlayers, ng);
    } else {
        bcs_mpi(F_h, nx, ny, nlayers, ng, comm, status, rank, n_processes);
    }

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            for (int l = 0; l < nlayers; l++) {
                for (int i = 0; i < 4; i++) {
                    Up_h[((y * nx + x) * nlayers + l) * 4 + i] = Un_h[((y * nx + x) * nlayers + l) * 4 + i] + dt * F_h[((y * nx + x) * nlayers + l) * 4 + i];
                }
            }
        }
    }

    // enforce boundaries and copy back
    //bcs_fv(Up_h, nx, ny, nlayers, ng);
    if (n_processes == 1) {
        bcs_fv(Up_h, nx, ny, nlayers, ng);
    } else {
        bcs_mpi(Up_h, nx, ny, nlayers, ng, comm, status, rank, n_processes);
    }
    cudaMemcpy(Un_d, Up_h, nx*ny*nlayers*4*sizeof(float), cudaMemcpyHostToDevice);

    // u2 = 0.25 * (3*un + u1 + dt*F(u1))
    homogeneuous_fv(kernels, threads, blocks,
          beta_d, gamma_up_d, Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d,
          nx, ny, nlayers, alpha,
          dx, dy, dt, rank);

    // copy back flux
    cudaMemcpy(F_h, F_d, nx*ny*nlayers*4*sizeof(float), cudaMemcpyDeviceToHost);
    //bcs_fv(F_h, nx, ny, nlayers, ng);
    if (n_processes == 1) {
        bcs_fv(F_h, nx, ny, nlayers, ng);
    } else {
        bcs_mpi(F_h, nx, ny, nlayers, ng, comm, status, rank, n_processes);
    }

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            for (int l = 0; l < nlayers; l++) {
                for (int i = 0; i < 4; i++) {
                    Up_h[((y * nx + x) * nlayers + l) * 4 + i] = 0.25 * (
                        3.0 * Un_h[((y * nx + x) * nlayers + l) * 4 + i] +
                        Up_h[((y * nx + x) * nlayers + l) * 4 + i] +
                        dt * F_h[((y * nx + x) * nlayers + l) * 4 + i]);
                }
            }
        }
    }

    // enforce boundaries and copy back
    //bcs_fv(Up_h, nx, ny, nlayers, ng);
    if (n_processes == 1) {
        bcs_fv(Up_h, nx, ny, nlayers, ng);
    } else {
        bcs_mpi(Up_h, nx, ny, nlayers, ng, comm, status, rank, n_processes);
    }
    cudaMemcpy(Un_d, Up_h, nx*ny*nlayers*4*sizeof(float), cudaMemcpyHostToDevice);

    // un+1 = (1/3) * (un + 2*u2 + 2*dt*F(u2))
    homogeneuous_fv(kernels, threads, blocks,
          beta_d, gamma_up_d, Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d,
          nx, ny, nlayers, alpha,
          dx, dy, dt, rank);

    // copy back flux
    cudaMemcpy(F_h, F_d, nx*ny*nlayers*4*sizeof(float), cudaMemcpyDeviceToHost);
    //bcs_fv(F_h, nx, ny, nlayers, ng);
    if (n_processes == 1) {
        bcs_fv(F_h, nx, ny, nlayers, ng);
    } else {
        bcs_mpi(F_h, nx, ny, nlayers, ng, comm, status, rank, n_processes);
    }

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            for (int l = 0; l < nlayers; l++) {
                for (int i = 0; i < 4; i++) {
                    Up_h[((y * nx + x) * nlayers + l) * 4 + i] = (1/3.0) * (
                        Un_h[((y * nx + x) * nlayers + l) * 4 + i] +
                        2.0*Up_h[((y * nx + x) * nlayers + l) * 4 + i] +
                        2.0*dt * F_h[((y * nx + x) * nlayers + l) * 4 + i]);
                }
            }
        }
    }

    // enforce boundaries
    //bcs_fv(Up_h, nx, ny, nlayers, ng);
    if (n_processes == 1) {
        bcs_fv(F_h, nx, ny, nlayers, ng);
    } else {
        bcs_mpi(Up_h, nx, ny, nlayers, ng, comm, status, rank, n_processes);
    }

    cudaMemcpy(Up_d, Up_h, nx*ny*nlayers*4*sizeof(float), cudaMemcpyHostToDevice);

}


void cuda_run(float * beta, float * gamma_up, float * Un_h,
         float * rho, float * Q, float mu, int nx, int ny, int nlayers, int ng,
         int nt, float alpha, float dx, float dy, float dt, bool burning,
         int dprint, char * filename,
         MPI_Comm comm, MPI_Status status, int rank, int n_processes) {
    /*
    Evolve system through nt timesteps, saving data to filename every dprint timesteps.
    */

    // set up GPU stuff
    int count;
    cudaGetDeviceCount(&count);

    if (rank == 0) {
        cudaError_t err = cudaGetLastError();
        // check that we actually have some GPUS
        if (err != cudaSuccess) {
            printf("Error: %s\n", cudaGetErrorString(err));
            printf("Aborting program.\n");
            return;
        }

        printf("Found %i CUDA devices\n", count);

    }

    // if rank > number of GPUs, exit now
    if (rank >= count) {
        return;
    }

    // redefine - we only want to run on as many cores as we have GPUs
    n_processes = count;

    if (rank == 0) {
        printf("Running on %i processor(s)\n", n_processes);
    }

    int maxThreads = 256;
    int maxBlocks = 256; //64;

    dim3 *kernels = new dim3[n_processes];
    int *cumulative_kernels = new int[n_processes];

    getNumKernels(nx, ny, nlayers, ng, n_processes, &maxBlocks, &maxThreads, kernels, cumulative_kernels);

    int total_kernels = cumulative_kernels[n_processes-1];

    //int kernels_y = kernels[0].y;

    dim3 *blocks = new dim3[total_kernels];
    dim3 *threads = new dim3[total_kernels];

    getNumBlocksAndThreads(nx, ny, nlayers, ng, maxBlocks, maxThreads, n_processes, kernels, blocks, threads);

    printf("rank: %i\n", rank);
    printf("kernels: (%i, %i)\n", kernels[rank].x, kernels[rank].y);
    printf("cumulative kernels: %i\n", cumulative_kernels[rank]);

    int k_offset = 0;
    if (rank > 0) {
      k_offset = cumulative_kernels[rank-1];
    }

    for (int i = k_offset; i < cumulative_kernels[rank]; i++) {
        printf("blocks: (%i, %i, %i) , threads: (%i, %i, %i)\n",
               blocks[i].x, blocks[i].y, blocks[i].z,
               threads[i].x, threads[i].y, threads[i].z);
    }

    // gpu variables
    float * beta_d;
    float * gamma_up_d;
    float * Un_d;
    float * rho_d;
    float * Q_d;

    // set device
    cudaSetDevice(rank);

    // allocate memory on device
    cudaMalloc((void**)&beta_d, 2*nx*ny*sizeof(float));
    cudaMalloc((void**)&gamma_up_d, 4*sizeof(float));
    cudaMalloc((void**)&Un_d, nx*ny*nlayers*4*sizeof(float));
    cudaMalloc((void**)&rho_d, nlayers*sizeof(float));
    cudaMalloc((void**)&Q_d, nlayers*nx*ny*sizeof(float));

    // copy stuff to GPU
    cudaMemcpy(beta_d, beta, 2*nx*ny*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma_up_d, gamma_up, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Un_d, Un_h, nx*ny*nlayers*4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rho_d, rho, nlayers*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Q_d, Q, nlayers*nx*ny*sizeof(float), cudaMemcpyHostToDevice);

    float *Up_d, *U_half_d, *sum_phs_d;
    cudaMalloc((void**)&Up_d, nlayers*nx*ny*4*sizeof(float));
    cudaMalloc((void**)&U_half_d, nlayers*nx*ny*4*sizeof(float));
    cudaMalloc((void**)&sum_phs_d, nlayers*nx*ny*sizeof(float));

    float *qx_p_d, *qx_m_d, *qy_p_d, *qy_m_d, *fx_p_d, *fx_m_d, *fy_p_d, *fy_m_d;
    float *Up_h = new float[nlayers*nx*ny*4];
    float *F_h = new float[nlayers*nx*ny*4];

    cudaMalloc((void**)&qx_p_d, nlayers*nx*ny*4*sizeof(float));
    cudaMalloc((void**)&qx_m_d, nlayers*nx*ny*4*sizeof(float));
    cudaMalloc((void**)&qy_p_d, nlayers*nx*ny*4*sizeof(float));
    cudaMalloc((void**)&qy_m_d, nlayers*nx*ny*4*sizeof(float));
    cudaMalloc((void**)&fx_p_d, nlayers*nx*ny*4*sizeof(float));
    cudaMalloc((void**)&fx_m_d, nlayers*nx*ny*4*sizeof(float));
    cudaMalloc((void**)&fy_p_d, nlayers*nx*ny*4*sizeof(float));
    cudaMalloc((void**)&fy_m_d, nlayers*nx*ny*4*sizeof(float));

    if (strcmp(filename, "na") != 0) {
        hid_t outFile, dset, mem_space, file_space;

        if (rank == 0) {
            // create file
            outFile = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

            // create dataspace
            int ndims = 5;
            hsize_t dims[] = {hsize_t((nt+1)/dprint+1), hsize_t(ny), hsize_t(nx), hsize_t(nlayers), 4};
            file_space = H5Screate_simple(ndims, dims, NULL);

            hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_layout(plist, H5D_CHUNKED);
            hsize_t chunk_dims[] = {1, hsize_t(ny), hsize_t(nx), hsize_t(nlayers), 4};
            H5Pset_chunk(plist, ndims, chunk_dims);

            // create dataset
            dset = H5Dcreate(outFile, "SwerveOutput", H5T_NATIVE_FLOAT, file_space, H5P_DEFAULT, plist, H5P_DEFAULT);

            H5Pclose(plist);

            // make a memory dataspace
            mem_space = H5Screate_simple(ndims, chunk_dims, NULL);

            // select a hyperslab
            file_space = H5Dget_space(dset);
            hsize_t start[] = {0, 0, 0, 0, 0};
            hsize_t hcount[] = {1, hsize_t(ny), hsize_t(nx), hsize_t(nlayers), 4};
            H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, hcount, NULL);
            // write to dataset
            printf("Printing t = %i\n", 0);
            H5Dwrite(dset, H5T_NATIVE_FLOAT, mem_space, file_space, H5P_DEFAULT, Un_h);
            // close file dataspace
            H5Sclose(file_space);
        }

        // main loop
        for (int t = 0; t < nt; t++) {
            //printf("t = %i\n", t);
            // offset by kernels in previous
            int kx_offset = 0;
            int ky_offset = kernels[0].y * rank * blocks[0].y * threads[0].y;

            rk3_fv(kernels, threads, blocks,
                beta_d, gamma_up_d, Un_d, U_half_d, Up_d,
                qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                nx, ny, nlayers, ng, alpha,
                dx, dy, dt, Up_h, F_h, Un_h,
                comm, status, rank, n_processes);

            for (int j = 0; j < kernels[rank].y; j++) {
                kx_offset = 0;
                for (int i = 0; i < kernels[rank].x; i++) {
                    evolve_fv_heating<<<blocks[j * kernels[rank].x + i], threads[j * kernels[rank].x + i]>>>(
                           gamma_up_d, Un_d,
                           Up_d, U_half_d,
                           qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                           fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                           sum_phs_d, rho_d, Q_d, mu,
                           nx, ny, nlayers, alpha,
                           dx, dy, dt, burning, kx_offset, ky_offset);
                    kx_offset += blocks[j * kernels[rank].x + i].x * threads[j * kernels[rank].x + i].x;
                }
                ky_offset += blocks[j * kernels[rank].x].y * threads[j * kernels[rank].x].y;
            }

            kx_offset = 0;
            ky_offset = kernels[0].y * rank * blocks[0].y * threads[0].y;

            for (int j = 0; j < kernels[rank].y; j++) {
                kx_offset = 0;
                for (int i = 0; i < kernels[rank].x; i++) {
                    evolve2<<<blocks[j * kernels[rank].x + i], threads[j * kernels[rank].x + i]>>>(gamma_up_d, Un_d,
                           Up_d, U_half_d, sum_phs_d, rho_d, Q_d, mu,
                           nx, ny, nlayers, ng, alpha,
                           dx, dy, dt, kx_offset, ky_offset);
                    kx_offset += blocks[j * kernels[rank].x + i].x * threads[j * kernels[rank].x + i].x;
                }
                ky_offset += blocks[j * kernels[rank].x].y * threads[j * kernels[rank].x].y;
            }

            cudaDeviceSynchronize();

            cudaError_t err = cudaGetLastError();

            if (err != cudaSuccess)
                printf("Error: %s\n", cudaGetErrorString(err));

            // boundaries
            cudaMemcpy(Un_h, Un_d, nx*ny*nlayers*4*sizeof(float), cudaMemcpyDeviceToHost);
            //bcs_fv(Un_h, nx, ny, nlayers, ng);
            if (n_processes == 1) {
                bcs_fv(Un_h, nx, ny, nlayers, ng);
            } else {
                bcs_mpi(Un_h, nx, ny, nlayers, ng, comm, status, rank, n_processes);
            }
            cudaMemcpy(Un_d, Un_h, nx*ny*nlayers*4*sizeof(float), cudaMemcpyHostToDevice);


            if ((t+1) % dprint == 0) {
                if (rank == 0) {
                    printf("Printing t = %i\n", t+1);

                    float * buf = new float[nlayers*nx*ny*4];
                    int tag = 0;
                    for (int source = 1; source < n_processes; source++) {
                        printf("Receiving from rank %i\n", source);
                        MPI_Recv(buf, nlayers*nx*ny*4, MPI_FLOAT, source, tag, comm, &status);

                        // copy data back to grid
                        ky_offset = kernels[0].y * rank * blocks[0].y * threads[0].y;
                        // cheating slightly and using the fact that are moving from  top to bottom to make calculations a bit easier.
                        for (int y = ky_offset; y < ny; y++) {
                            for (int x = 0; x < nx; x++) {
                                for (int l = 0; l < nlayers; l++) {
                                    for (int i = 0; i < 4; i++) {
                                        Un_h[((y * nx + x) * nlayers + l) * 4 + i] = buf[((y * nx + x) * nlayers + l) * 4 + i];
                                    }
                                }
                            }
                        }
                    }

                    delete[] buf;

                    // receive data from other processes and copy to grid

                    // select a hyperslab
                    file_space = H5Dget_space(dset);
                    hsize_t start[] = {hsize_t((t+1)/dprint), 0, 0, 0, 0};
                    hsize_t hcount[] = {1, hsize_t(ny), hsize_t(nx), hsize_t(nlayers), 4};
                    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, hcount, NULL);
                    // write to dataset
                    H5Dwrite(dset, H5T_NATIVE_FLOAT, mem_space, file_space, H5P_DEFAULT, Un_h);
                    // close file dataspae
                    H5Sclose(file_space);
                } else { // send data to rank 0
                    printf("Rank %i sending\n", rank);
                    int tag = 0;
                    MPI_Ssend(Un_h, ny*nx*nlayers*4, MPI_FLOAT, 0, tag, comm);
                }

            }
        }

        if (rank == 0) {
            H5Sclose(mem_space);
            H5Fclose(outFile);
        }

    } else { // don't print
        for (int t = 0; t < nt; t++) {

            int kx_offset = 0;
            int ky_offset = kernels[0].y * rank * blocks[0].y * threads[0].y;

            rk3_fv(kernels, threads, blocks,
                beta_d, gamma_up_d, Un_d, U_half_d, Up_d,
                qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                nx, ny, nlayers, ng, alpha,
                dx, dy, dt, Up_h, F_h, Un_h,
                comm, status, rank, n_processes);

            for (int j = 0; j < kernels[rank].y; j++) {
                kx_offset = 0;
                for (int i = 0; i < kernels[rank].x; i++) {
                    evolve_fv_heating<<<blocks[j * kernels[rank].x + i], threads[j * kernels[rank].x + i]>>>(
                           gamma_up_d, Un_d,
                           Up_d, U_half_d,
                           qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                           fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                           sum_phs_d, rho_d, Q_d, mu,
                           nx, ny, nlayers, alpha,
                           dx, dy, dt, burning, kx_offset, ky_offset);
                    kx_offset += blocks[j * kernels[rank].x + i].x * threads[j * kernels[rank].x + i].x;
                }
                ky_offset += blocks[j * kernels[rank].x].y * threads[j * kernels[rank].x].y;
            }


            kx_offset = 0;
            ky_offset = kernels[0].y * rank * blocks[0].y * threads[0].y;

            for (int j = 0; j < kernels[rank].y; j++) {
                kx_offset = 0;
                for (int i = 0; i < kernels[rank].x; i++) {
                    evolve2<<<blocks[j * kernels[rank].x + i], threads[j * kernels[rank].x + i]>>>(gamma_up_d, Un_d,
                           Up_d, U_half_d, sum_phs_d, rho_d, Q_d, mu,
                           nx, ny, nlayers, ng, alpha,
                           dx, dy, dt, kx_offset, ky_offset);
                    kx_offset += blocks[j * kernels[rank].x + i].x * threads[j * kernels[rank].x + i].x;
                }
                ky_offset += blocks[j * kernels[rank].x].y * threads[j * kernels[rank].x].y;
            }

            cudaDeviceSynchronize();

            // boundaries
            cudaMemcpy(Un_h, Un_d, nx*ny*nlayers*4*sizeof(float), cudaMemcpyDeviceToHost);
            //bcs_fv(Un_h, nx, ny, nlayers, ng);
            if (n_processes == 1) {
                bcs_fv(Un_h, nx, ny, nlayers, ng);
            } else {
                bcs_mpi(Un_h, nx, ny, nlayers, ng, comm, status, rank, n_processes);
            }
            cudaMemcpy(Un_d, Un_h, nx*ny*nlayers*4*sizeof(float), cudaMemcpyHostToDevice);

            cudaError_t err = cudaGetLastError();

            if (err != cudaSuccess)
                printf("Error: %s\n", cudaGetErrorString(err));

        }
    }

    // delete some stuff
    cudaFree(beta_d);
    cudaFree(gamma_up_d);
    cudaFree(Un_d);
    cudaFree(rho_d);
    cudaFree(Q_d);
    cudaFree(Up_d);
    cudaFree(U_half_d);
    cudaFree(sum_phs_d);

    cudaFree(qx_p_d);
    cudaFree(qx_m_d);
    cudaFree(qy_p_d);
    cudaFree(qy_m_d);
    cudaFree(fx_p_d);
    cudaFree(fx_m_d);
    cudaFree(fy_p_d);
    cudaFree(fy_m_d);

    delete[] kernels;
    delete[] cumulative_kernels;
    delete[] threads;
    delete[] blocks;

    delete[] Up_h;
    delete[] F_h;
}


#endif
