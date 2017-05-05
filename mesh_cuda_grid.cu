/*
File containing routines which deal with multigrid operations.

TODO: need to interpolate rho onto fine grid at start (probably in mesh constructor)
*/

#include <stdio.h>
#include <mpi.h>
#include "H5Cpp.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_functions.h>
#include "Mesh_cuda.h"
#include "mesh_cuda_kernel.h"
#include "mesh_output.h"

using namespace std;

void getNumKernels(int nx, int ny, int nz, int ng, int n_processes,
                   int *maxBlocks, int *maxThreads, dim3 * kernels,
                   int * cumulative_kernels) {
    /**
    Return the number of kernels needed to run the problem given its size and the constraints of the GPU.

    Parameters
    ----------
    nx, ny, nz : int
        dimensions of problem
    ng : int
        number of ghost cells
    maxBlocks, maxThreads : int
        maximum number of blocks and threads possible for device(s)
    n_processes : int
        number of MPI processes
    kernels : dim3 *
        number of kernels per process
    cumulative_kernels : int *
        cumulative total of kernels per process
    */
    // won't actually use maxThreads - fix to account for the fact we want something square
    *maxThreads = nz * int(sqrt(float(*maxThreads)/float(nz))) *
                  int(sqrt(float(*maxThreads)/float(nz)));

    // calculate number of kernels needed

    if (nx*ny*nz > *maxBlocks * *maxThreads) {
        int kernels_x = int(ceil(float(nx-2*ng) /
                        (sqrt(float(*maxThreads * *maxBlocks)/nz)-2*ng)));
        int kernels_y = int(ceil(float(ny-2*ng) /
                        (sqrt(float(*maxThreads * *maxBlocks)/nz)-2*ng)));

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
            int strip_width = int(floor(float(kernels_y) /
                              float(n_processes)));

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
      cumulative_kernels[i] = cumulative_kernels[i-1] + kernels[i].x *
                              kernels[i].y;
    }
}

void getNumBlocksAndThreads(int nx, int ny, int nz, int ng, int maxBlocks,
                            int maxThreads, int n_processes,
                            dim3 *kernels, dim3 *blocks, dim3 *threads)
{
    /**
    Returns the number of blocks and threads required for each kernel given the size of the problem and the constraints of the device.

    Parameters
    ----------
    nx, ny, nz : int
        dimensions of problem
    ng : int
        number of ghost cells
    maxBlocks, maxThreads : int
        maximum number of blocks and threads possible for device(s)
    n_processes : int
        number of MPI processes
    kernels, blocks, threads : dim3 *
        number of kernels, blocks and threads per process / kernel
    */

    //get device capability, to avoid block/grid size exceed the upper bound
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    int total = nx * ny * nz;

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
                threads[j*kernels_x + i].x = int(sqrt(float(maxThreads)/nz));
                threads[j*kernels_x + i].y = int(sqrt(float(maxThreads)/nz));
                threads[j*kernels_x + i].z = nz;

                blocks[j*kernels_x + i].x = int(sqrt(float(maxBlocks)));
                blocks[j*kernels_x + i].y = int(sqrt(float(maxBlocks)));
                blocks[j*kernels_x + i].z = 1;
            }
        }
        // kernels_x-1
        int nx_remaining = nx-(threads[0].x*blocks[0].x-2*ng)*(kernels_x - 1);

        for (int j = 0; j < (kernels_y-1); j++) {

            threads[j*kernels_x + kernels_x-1].y =
                int(sqrt(float(maxThreads)/nz));
            threads[j*kernels_x + kernels_x-1].z = nz;

            threads[j*kernels_x + kernels_x-1].x =
                (nx_remaining < threads[j*kernels_x + kernels_x-1].y) ?
                 nx_remaining : threads[j*kernels_x + kernels_x-1].y;

            blocks[j*kernels_x + kernels_x-1].x =
                int(ceil(float(nx_remaining) /
                float(threads[j*kernels_x + kernels_x-1].x)));
            blocks[j*kernels_x + kernels_x-1].y = int(sqrt(float(maxBlocks)));
            blocks[j*kernels_x + kernels_x-1].z = 1;
        }

        // kernels_y-1
        int ny_remaining = ny-(threads[0].y*blocks[0].y-2*ng)*(kernels_y - 1);

        for (int i = 0; i < (kernels_x-1); i++) {

            threads[(kernels_y-1)*kernels_x + i].x =
                int(sqrt(float(maxThreads)/nz));
            threads[(kernels_y-1)*kernels_x + i].y =
                (ny_remaining < threads[(kernels_y-1)*kernels_x + i].x) ?
                 ny_remaining : threads[(kernels_y-1)*kernels_x + i].x;
            threads[(kernels_y-1)*kernels_x + i].z = nz;

            blocks[(kernels_y-1)*kernels_x + i].x =
                int(sqrt(float(maxBlocks)));
            blocks[(kernels_y-1)*kernels_x + i].y =
                int(ceil(float(ny_remaining) /
                float(threads[(kernels_y-1)*kernels_x + i].y)));
            blocks[(kernels_y-1)*kernels_x + i].z = 1;
        }

        // recalculate
        nx_remaining = nx-(threads[0].x * blocks[0].x - 2*ng)*(kernels_x - 1);
        ny_remaining = ny-(threads[0].y * blocks[0].y - 2*ng)*(kernels_y - 1);

        // (kernels_x-1, kernels_y-1)
        threads[(kernels_y-1)*kernels_x + kernels_x-1].x =
            (nx_remaining < int(sqrt(float(maxThreads)/nz))) ?
            nx_remaining : int(sqrt(float(maxThreads)/nz));
        threads[(kernels_y-1)*kernels_x + kernels_x-1].y =
            (ny_remaining < int(sqrt(float(maxThreads)/nz))) ?
            ny_remaining : int(sqrt(float(maxThreads)/nz));
        threads[(kernels_y-1)*kernels_x + kernels_x-1].z = nz;

        blocks[(kernels_y-1)*kernels_x + kernels_x-1].x =
            int(ceil(float(nx_remaining) /
            float(threads[(kernels_y-1)*kernels_x + kernels_x-1].x)));
        blocks[(kernels_y-1)*kernels_x + kernels_x-1].y =
            int(ceil(float(ny_remaining) /
            float(threads[(kernels_y-1)*kernels_x + kernels_x-1].y)));
        blocks[(kernels_y-1)*kernels_x + kernels_x-1].z = 1;

    } else {

        int total_threads = (total < maxThreads*2) ?
                            nextPow2((total + 1)/ 2) : maxThreads;
        threads[0].x = int(floor(sqrt(float(total_threads)/nz)));
        threads[0].y = int(floor(sqrt(float(total_threads)/nz)));
        threads[0].z = nz;
        total_threads = threads[0].x * threads[0].y * threads[0].z;
        int total_blocks = int(ceil(float(total) / float(total_threads)));

        blocks[0].x = int(ceil(sqrt(float(total_blocks)/float(nx*ny))*nx));
        blocks[0].y = int(ceil(sqrt(float(total_blocks)/float(nx*ny))*ny));
        blocks[0].z = 1;

        total_blocks = blocks[0].x * blocks[0].y;

        if ((float)total_threads*total_blocks >
            (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
            printf("n is too large, please choose a smaller number!\n");
        }

        if (total_blocks > prop.maxGridSize[0]) {
            printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
                   total_blocks, prop.maxGridSize[0],
                   total_threads*2, total_threads);

            blocks[0].x /= 2;
            blocks[0].y /= 2;
            threads[0].x *= 2;
            threads[0].y *= 2;
        }
    }
}

void bcs_fv(float * grid, int nx, int ny, int nz, int ng, int vec_dim,
            bool periodic, bool do_z) {
    /**
    Enforce boundary conditions on section of grid.

    Parameters
    ----------
    grid : float *
        grid of data
    nx, ny, nz : int
        dimensions of grid
    ng : int
        number of ghost cells
    vec_dim : int
        dimension of state vector
    periodic : bool
        do we use periodic or outflow boudary conditions?
    do_z : bool
        true if need to implement bcs in vertical direction as well
    */

    if (periodic) {
        // periodic
        for (int z = 0; z < nz; z++) {
            for (int y = 0; y < ny; y++){
                for (int g = 0; g < ng; g++) {
                    for (int i = 0; i < vec_dim; i++) {
                        grid[((z * ny + y) * nx + g) * vec_dim+i] =
                            grid[((z * ny + y) * nx + (nx-2*ng+g))*vec_dim+i];

                        grid[((z * ny + y) * nx + (nx-ng+g))*vec_dim+i] =
                            grid[((z * ny + y) * nx + ng+g) * vec_dim+i];
                    }
                }
            }
            for (int g = 0; g < ng; g++) {
                for (int x = 0; x < nx; x++){
                    for (int i = 0; i < vec_dim; i++) {
                        grid[((z * ny + g) * nx + x)*vec_dim+i] =
                            grid[((z * ny + ny-2*ng+g) * nx + x)*vec_dim+i];

                        grid[((z * ny + ny-ng+g) * nx + x)*vec_dim+i] =
                            grid[((z * ny + ng + g) * nx + x)*vec_dim+i];
                    }
                }
            }
        }
    } else {
        // outflow
        for (int z = 0; z < nz; z++) {
            for (int y = 0; y < ny; y++){
                for (int g = 0; g < ng; g++) {
                    for (int i = 0; i < vec_dim; i++) {
                        grid[((z * ny + y) * nx + g) * vec_dim+i] =
                            grid[((z * ny + y) * nx + ng)*vec_dim+i];

                        grid[((z * ny + y) * nx + (nx-1-g))*vec_dim+i] =
                            grid[((z * ny + y) * nx + (nx-1-ng))*vec_dim+i];
                    }
                }
            }
            for (int g = 0; g < ng; g++) {
                for (int x = 0; x < nx; x++){
                    for (int i = 0; i < vec_dim; i++) {
                        grid[((z * ny + g) * nx + x)*vec_dim+i] =
                            grid[((z * ny + ng) * nx + x)*vec_dim+i];

                        grid[((z * ny + ny-1-g) * nx + x)*vec_dim+i] =
                            grid[((z * ny + ny-1-ng) * nx + x)*vec_dim+i];
                    }
                }
            }
        }
    }

    if (do_z) { // outflow
        for (int g = 0; g < ng; g++) {
            for (int y = 0; y < ny; y++){
                for (int x = 0; x < nx; x++) {
                    for (int i = 0; i < vec_dim; i++) {
                        grid[((g * ny + y) * nx + x) * vec_dim+i] =
                            grid[((ng * ny + y) * nx + x)*vec_dim+i];

                        grid[(((nz-1-g) * ny + y) * nx + x)*vec_dim+i] =
                            grid[(((nz-1-ng) * ny + y) * nx + x)*vec_dim+i];
                    }
                }
            }
        }
    }
}

void bcs_mpi(float * grid, int nx, int ny, int nz, int vec_dim, int ng,
             MPI_Comm comm, MPI_Status status, int rank, int n_processes,
             int y_size, bool do_z, bool periodic) {
    /**
    Enforce boundary conditions across processes / at edges of grid.

    Loops have been ordered in a way so as to try and keep memory accesses as contiguous as possible.

    Need to do non-blocking send, blocking receive then wait.

    Parameters
    ----------
    grid : float *
        grid of data
    nx, ny, nz : int
        dimensions of grid
    vec_dim : int
        dimension of state vector
    ng : int
        number of ghost cells
    comm : MPI_Comm
        MPI communicator
    status : MPI_Status
        status of MPI processes
    rank, n_processes : int
        rank of MPI process and total number of MPI processes
    y_size : int
        size of grid in y direction running on each process (except the last one)
    do_z : bool
        true if need to implement bcs in vertical direction as well
    periodic : bool
        do we use periodic or outflow boudary conditions?
    */

    if (periodic) {
        // x boundaries
        for (int z = 0; z < nz; z++) {
            for (int y = 0; y < ny; y++) {
                for (int g = 0; g < ng; g++) {
                    for (int i = 0; i < vec_dim; i++) {
                        grid[((z * ny + y) * nx + g) *vec_dim+i] =
                            grid[((z * ny + y) * nx + (nx-2*ng+g))*vec_dim+i];

                        grid[((z * ny + y) * nx + (nx-ng+g))*vec_dim+i] =
                            grid[((z * ny + y) * nx + ng + g) *vec_dim+i];
                    }
                }
            }
        }
    } else {

        // x boundaries
        for (int z = 0; z < nz; z++) {
            for (int y = 0; y < ny; y++) {
                for (int g = 0; g < ng; g++) {
                    for (int i = 0; i < vec_dim; i++) {
                        grid[((z * ny + y) * nx + g) *vec_dim+i] =
                            grid[((z * ny + y) * nx + ng) *vec_dim+i];

                        grid[((z * ny + y) * nx + (nx-1-g))*vec_dim+i] =
                            grid[((z * ny + y) * nx + (nx-1-ng))*vec_dim+i];
                    }
                }
            }
        }

        // z boundaries
        /*if (do_z) {
            for (int g = 0; g < ng; g++) {
                for (int y = 0; y < ny; y++) {
                    for (int x = 0; x < nx; x++) {
                        for (int i = 0; i < vec_dim; i++) {
                            grid[((g * ny + y) * nx + x) *vec_dim+i] = grid[((ng * ny + y) * nx + x) *vec_dim+i];

                            grid[(((nz-1-g) * ny + y) * nx + x)*vec_dim+i] = grid[(((nz-1-ng) * ny + y) * nx + x)*vec_dim+i];
                        }
                    }
                }
            }
        }*/

    }

    // interior cells between processes

    // make some buffers for sending and receiving
    float * ysbuf = new float[nx*ng*nz*vec_dim];
    float * yrbuf = new float[nx*ng*nz*vec_dim];

    int tag = 1;
    int mpi_err;
    MPI_Request request;

    // if there are process above and below, send/receive
    if ((rank > 0) && (rank < n_processes-1)) {
        // send to below, receive from above
        // copy stuff to buffer
        for (int z = 0; z < nz; z++) {
            for (int g = 0; g < ng; g++) {
                for (int x = 0; x < nx; x++) {
                    for (int i = 0; i < vec_dim; i++) {
                        ysbuf[((z * ng + g) * nx + x)*vec_dim+i] =
                            grid[((z*ny + y_size*rank+ng+g)*nx + x)*vec_dim+i];
                    }
                }
            }
        }
        mpi_err = MPI_Issend(ysbuf,nx*ng*nz*vec_dim, MPI_FLOAT, rank-1, tag,
                             comm, &request);
        check_mpi_error(mpi_err);
        mpi_err = MPI_Recv(yrbuf, nx*ng*nz*vec_dim, MPI_FLOAT, rank+1, tag,
                           comm, &status);
        check_mpi_error(mpi_err);
        MPI_Wait(&request, &status);

        // copy received data back to grid
        for (int z = 0; z < nz; z++) {
            for (int g = 0; g < ng; g++) {
                for (int x = 0; x < nx; x++) {
                    for (int i = 0; i < vec_dim; i++) {
                        grid[((z*ny + y_size*rank+ny-ng+g)*nx + x)*vec_dim+i] =
                            yrbuf[((z * ng + g) * nx + x)*vec_dim+i];
                    }
                }
            }
        }
        // send to above, receive from below
        // copy stuff to buffer
        for (int z = 0; z < nz; z++) {
            for (int g = 0; g < ng; g++) {
                for (int x = 0; x < nx; x++) {
                    for (int i = 0; i < vec_dim; i++) {
                        ysbuf[((z * ng + g) * nx + x)*vec_dim+i] =
                            grid[((z*ny+y_size*rank+ny-2*ng+g)*nx+x)*vec_dim+i];
                    }
                }
            }
        }
        MPI_Issend(ysbuf,nx*ng*nz*vec_dim, MPI_FLOAT, rank+1, tag, comm,
                   &request);
        MPI_Recv(yrbuf, nx*ng*nz*vec_dim, MPI_FLOAT, rank-1, tag, comm,
                 &status);
        MPI_Wait(&request, &status);

        // copy received data back to grid
        for (int z = 0; z < nz; z++) {
            for (int g = 0; g < ng; g++) {
                for (int x = 0; x < nx; x++) {
                    for (int i = 0; i < vec_dim; i++) {
                        grid[((z * ny + y_size*rank+g) * nx + x)*vec_dim+i] =
                            yrbuf[((z * ng + g) * nx + x)*vec_dim+i];
                    }
                }
            }
        }
    } else if (rank == 0) {
        // do outflow for top boundary
        // copy stuff to buffer
        for (int z = 0; z < nz; z++) {
            for (int g = 0; g < ng; g++) {
                for (int x = 0; x < nx; x++) {
                    for (int i = 0; i < vec_dim; i++) {
                        ysbuf[((z * ng + g) * nx + x)*vec_dim+i] =
                            grid[((z * ny + ny-2*ng+g) * nx + x)*vec_dim+i];
                    }
                }
            }
        }

        MPI_Issend(ysbuf, nx*ng*nz*vec_dim, MPI_FLOAT, 1, tag, comm, &request);
        MPI_Recv(yrbuf, nx*ng*nz*vec_dim, MPI_FLOAT, 1, tag, comm, &status);
        MPI_Wait(&request, &status);

        // copy received data back to grid
        for (int z = 0; z < nz; z++) {
            for (int g = 0; g < ng; g++) {
                for (int x = 0; x < nx; x++) {
                    for (int i = 0; i < vec_dim; i++) {
                        grid[((z * ny + ny-ng+g) * nx + x)*vec_dim+i] =
                            yrbuf[((z * ng + g) * nx + x)*vec_dim+i];
                    }
                }
            }
        }

        if (periodic) {

            // periodic stuff on top boundary
            for (int z = 0; z < nz; z++) {
                for (int g = 0; g < ng; g++) {
                    for (int x = 0; x < nx; x++) {
                        for (int i = 0; i < vec_dim; i++) {
                            grid[((z * ny + g) * nx + x)*vec_dim+i] =
                                grid[((z * ny + ny-2*ng+g) * nx + x)*vec_dim+i];
                        }
                    }
                }
            }

        } else {

            // outflow stuff on top boundary
            for (int z = 0; z < nz; z++) {
                for (int g = 0; g < ng; g++) {
                    for (int x = 0; x < nx; x++) {
                        for (int i = 0; i < vec_dim; i++) {
                            grid[((z * ny + g) * nx + x)*vec_dim+i] =
                                grid[((z * ny + ng) * nx + x)*vec_dim+i];
                        }
                    }
                }
            }
        }
    } else {
        // copy stuff to buffer
        for (int z = 0; z < nz; z++) {
            for (int g = 0; g < ng; g++) {
                for (int x = 0; x < nx; x++) {
                    for (int i = 0; i < vec_dim; i++) {
                        ysbuf[((z * ng + g) * nx + x)*vec_dim+i] =
                            grid[((z * ny + y_size*rank+ng+g) * nx + x)*vec_dim+i];
                    }
                }
            }
        }
        // bottom-most process
        MPI_Issend(ysbuf, nx*ng*nz*vec_dim, MPI_FLOAT, rank-1, tag, comm,
                   &request);
        MPI_Recv(yrbuf, nx*ng*nz*vec_dim, MPI_FLOAT, rank-1, tag, comm,
                 &status);
        MPI_Wait(&request, &status);

        // copy received data back to grid
        for (int z = 0; z < nz; z++) {
            for (int g = 0; g < ng; g++) {
                for (int x = 0; x < nx; x++) {
                    for (int i = 0; i < vec_dim; i++) {
                        grid[((z * ny + y_size*rank+g) * nx + x)*vec_dim+i] =
                            yrbuf[((z * ng + g) * nx + x)*vec_dim+i];
                    }
                }
            }

            if (periodic) {

                // periodic for bottom boundary
                for (int g = 0; g < ng; g++) {
                    for (int x = 0; x < nx; x++) {
                        for (int i = 0; i < vec_dim; i++) {
                            grid[((z * ny + ny-ng+g) * nx + x)*vec_dim+i] =
                                grid[((z * ny + ng+g) * nx + x)*vec_dim+i];
                        }
                    }
                }

            } else {

                // outflow for bottom boundary
                for (int g = 0; g < ng; g++) {
                    for (int x = 0; x < nx; x++) {
                        for (int i = 0; i < vec_dim; i++) {
                            grid[((z * ny + ny-1-g) * nx + x)*vec_dim+i] =
                                grid[((z * ny + ny-1-ng) * nx + x)*vec_dim+i];
                        }
                    }
                }
            }
        }
    }

    delete[] ysbuf;
    delete[] yrbuf;

}

__global__ void prolong_reconstruct_comp_from_swe(float * q_comp,
                    float * q_f, float * q_c,
                    int * nxs, int * nys, int * nzs, int ng,
                    float dz, float zmin,
                    int * matching_indices_d,
                    int kx_offset, int ky_offset, int clevel, bool boundaries) {
    /**
    Reconstruct fine grid variables from compressible variables on coarse grid

    Parameters
    ----------
    q_comp : float *
        compressible variables on coarse grid
    q_f : float *
        fine grid state vector
    q_c : float *
        coarse grid swe state vector
    nxs, nys, nzs : int *
        grid dimensions
    ng: int
        number of ghost cells
    dz : float
        coarse grid vertical spacing
    matching_indices_d : int *
        position of fine grid wrt coarse grid
    kx_offset, ky_offset : int
        kernel offsets in the x and y directions
    clevel : int
        index of coarser level
    */

    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    bool prolong;

    if (boundaries) {
        prolong = ((x < ng) ||
            ((x > int(round(nxs[clevel+1]*0.5)) - (ng)) &&
            (x < int(round(nxs[clevel+1]*0.5)))) ||
            (y < ng) ||
            ((y > int(round(nys[clevel+1]*0.5)) - (ng)) &&
            (y < int(round(nys[clevel+1]*0.5))))) &&
            (z < nzs[clevel+1]);
    } else {
        prolong = (x < int(round(nxs[clevel+1]*0.5))) &&
            (y < int(round(nys[clevel+1]*0.5))) &&
            (z < nzs[clevel+1]);
    }

    prolong = prolong && 2*x < nxs[clevel+1] && 2*y < nys[clevel+1];

    if (prolong) {
        // corresponding x and y on the coarse grid
        int c_x = x + matching_indices_d[clevel*4];
        int c_y = y + matching_indices_d[clevel*4+2];

        // height of this layer
        float height = zmin + dz * (nzs[clevel+1] - z - 1.0);
        float * q_swe;
        q_swe = (float *)malloc(4 * sizeof(float));
        for (int i = 0; i < 4; i++) {
            q_swe[i] = q_c[(c_y*nxs[clevel]+c_x)*4+i];
        }
        float W = W_swe(q_swe);
        float r = find_height(q_c[(c_y * nxs[clevel] + c_x) * 4]/W);
        // Heights are sane here?
        //printf("z = %i, heights = %f, %f\n", z, height, r);
        float prev_r = r;

        int neighbour_layer = nzs[clevel]; // SWE layer just below compressible layer
        float layer_frac = 0.0; // fraction of distance between SWE layers that compressible is at

        if (height > r) { // compressible layer above top SWE layer
            neighbour_layer = 1;
            for (int i = 0; i < 4; i++) {
                q_swe[i] = q_c[((nys[clevel]+c_y)*nxs[clevel]+c_x)*4+i];
            }
            W = W_swe(q_swe);
            r = find_height(q_c[((nys[clevel]+c_y)*nxs[clevel]+c_x)*4] / W);
            layer_frac = (height - prev_r) / (r - prev_r);
        } else {

            // find heights of SWE layers - if height of SWE layer is above it, stop.
            for (int l = 1; l < nzs[clevel]-1; l++) {
                prev_r = r;
                for (int i = 0; i < 4; i++) {
                    q_swe[i] = q_c[((l*nys[clevel]+c_y)*nxs[clevel]+c_x)*4+i];
                }
                W = W_swe(q_swe);
                r = find_height(q_c[((l * nys[clevel] + c_y) * nxs[clevel] + c_x) * 4] / W);
                if (height > r) {
                    neighbour_layer = l;
                    layer_frac = (height - prev_r)/ (r - prev_r);
                    break;
                }
            }

            if (neighbour_layer == nzs[clevel]) {
                // lowest compressible beneath lowest SWE layer
                neighbour_layer = nzs[clevel] - 1;
                if (z == (nzs[clevel+1]-1)) {
                    layer_frac = 1.0;
                } else {
                    prev_r = r;
                    int l = neighbour_layer;
                    for (int i = 0; i < 4; i++) {
                        q_swe[i] =
                            q_c[((l*nys[clevel]+c_y)*nxs[clevel]+c_x)*4+i];
                    }
                    W = W_swe(q_swe);
                    r = find_height(q_c[((l * nys[clevel] + c_y) * nxs[clevel] + c_x) * 4] / W);
                    layer_frac = (height - prev_r) / (r - prev_r);
                    //printf("Lower layer frac: %f  ", layer_frac);
                }
            }
        }

        free(q_swe);

        for (int n = 0; n < 6; n++) {
            // do some slope limiting
            // x-dir
            float S_upwind = (layer_frac *
                (q_comp[((neighbour_layer * nys[clevel] + c_y) * nxs[clevel] + c_x+1) * 6 + n] -
                q_comp[((neighbour_layer * nys[clevel] + c_y) * nxs[clevel] + c_x) * 6 + n]) +
                (1.0 - layer_frac) *
                (q_comp[(((neighbour_layer-1)*nys[clevel] + c_y) * nxs[clevel] + c_x+1)*6 + n] -
                q_comp[(((neighbour_layer-1)*nys[clevel]+c_y)*nxs[clevel]+c_x)*6 + n]));
            float S_downwind = (layer_frac *
                (q_comp[((neighbour_layer * nys[clevel] + c_y) * nxs[clevel] + c_x) * 6 + n] -
                q_comp[((neighbour_layer * nys[clevel] + c_y) * nxs[clevel] + c_x-1) * 6 + n])
                + (1.0 - layer_frac) *
                (q_comp[(((neighbour_layer-1)*nys[clevel] + c_y) * nxs[clevel] + c_x)*6 + n] -
                q_comp[(((neighbour_layer-1)*nys[clevel] + c_y)*nxs[clevel]+c_x-1)*6+n]));

            float Sx = 0.5 * (S_upwind + S_downwind);

            float r = 1.0e6;
            if (abs(S_downwind) > 1.0e-10) {
                r = S_upwind / S_downwind;
            }

            Sx *= phi(r);

            // y-dir
            S_upwind = (layer_frac *
                (q_comp[((neighbour_layer * nys[clevel] + c_y+1) * nxs[clevel] + c_x) * 6 + n] -
                q_comp[((neighbour_layer * nys[clevel] + c_y) * nxs[clevel] + c_x) * 6 + n]) +
                (1.0 - layer_frac) *
                (q_comp[(((neighbour_layer-1)*nys[clevel] + c_y+1) * nxs[clevel] + c_x)*6 + n] -
                q_comp[(((neighbour_layer-1)*nys[clevel]+c_y)*nxs[clevel]+c_x)*6 + n]));
            S_downwind = (layer_frac *
                (q_comp[((neighbour_layer * nys[clevel] + c_y) * nxs[clevel] + c_x) * 6 + n] -
                q_comp[((neighbour_layer * nys[clevel] + c_y-1) * nxs[clevel] + c_x) * 6 + n])
                + (1.0 - layer_frac) *
                (q_comp[(((neighbour_layer-1)*nys[clevel] + c_y) * nxs[clevel] + c_x)*6 + n] -
                q_comp[(((neighbour_layer-1)*nys[clevel] + c_y-1)*nxs[clevel]+c_x)*6+n]));

            float Sy = 0.5 * (S_upwind + S_downwind);

            r = 1.0e6;
            if (abs(S_downwind) > 1.0e-10) {
                r = S_upwind / S_downwind;
            }

            Sy *= phi(r);

            // vertically interpolated component of q_comp
            float interp_q_comp = layer_frac *
                q_comp[((neighbour_layer * nys[clevel] + c_y) * nxs[clevel] + c_x) * 6 + n] +
                (1.0 - layer_frac) *
                q_comp[(((neighbour_layer-1) * nys[clevel] + c_y) * nxs[clevel] + c_x) * 6 + n];

            q_f[((z * nys[clevel+1] + 2*y)*nxs[clevel+1] + 2*x) * 6 + n] =
                    interp_q_comp - 0.25 * (Sx + Sy);

            if (2*x+1 < nxs[clevel+1] && 2*y < nys[clevel+1]) {
                q_f[((z * nys[clevel+1] + 2*y)*nxs[clevel+1] + 2*x+1)*6 + n] =
                    interp_q_comp + 0.25 * (Sx - Sy);
            }

            if (2*x < nxs[clevel+1] && 2*y+1 < nys[clevel+1]) {
                q_f[((z * nys[clevel+1] + 2*y+1)*nxs[clevel+1] + 2*x)*6 + n] =
                    interp_q_comp + 0.25 * (-Sx + Sy);
            }

            if (2*x+1 < nxs[clevel+1] && 2*y+1 < nys[clevel+1]) {
                q_f[((z * nys[clevel+1]+2*y+1)*nxs[clevel+1] + 2*x+1)*6 + n] =
                    interp_q_comp + 0.25 * (Sx + Sy);
            }

        }

        //printf("(%d, %d, %d): %f, \n", 2*x, 2*y, z, q_f[((z * nyf + 2*y+1) * nxf + 2*x) * 6+4]);
        //printf("(%d, %d, %d): %f, \n", 2*x, 2*y+1, z, q_f[((z * nyf + 2*y+1) * nxf + 2*x) * 6]);
        //printf("(%d, %d, %d): %f, \n", 2*x, 2*y+1, z, q_f[((z * nyf + 2*y+1) * nxf + 2*x) * 6]);
        //printf("(%d, %d, %d): %f, \n", 2*x+1, 2*y+1, z, q_f[((z * nyf + 2*y+1) * nxf + 2*x+1) * 6]);
    }
}

void prolong_swe_to_comp(dim3 * kernels, dim3 * threads, dim3 * blocks,
                  int * cumulative_kernels, float * q_cd, float * q_fd,
                  int * nxs, int * nys, int * nzs,
                  float dz, float dt, float zmin,
                  float * rho, float gamma,
                  int * matching_indices_d, int ng, int rank, float * qc_comp,
                  float * old_phi_d, int clevel, bool boundaries) {
    /**
    Prolong coarse grid data to fine grid

    Parameters
    ----------
    kernels, threads, blocks : dim3 *
        number of kernels, threads and blocks for each process/kernel
    cumulative_kernels : int *
        cumulative number of kernels in mpi processes of r < rank
    q_cd, q_fd : float *
        coarse and fine grids of state vectors
    nxs, nys, nzs : int *
        dimensions of grids
    dz : float
        coarse grid cell vertical spacings
    dt : float
        timestep
    zmin : float
        height of sea floor
    rho, gamma : float
        density and adiabatic index
    matching_indices_d : int *
        position of fine grid wrt coarse grid
    ng : int
        number of ghost cells
    rank : int
        rank of MPI process
    qc_comp : float *
        grid of compressible variables on coarse grid
    old_phi_d : float *
        Phi at previous timstep
    clevel : int
        index of coarser level
    */

    int kx_offset = 0;
    int ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

    int k_offset = 0;
    if (rank > 0) {
        k_offset = cumulative_kernels[rank - 1];
    }

    for (int j = 0; j < kernels[rank].y; j++) {
       kx_offset = 0;
       for (int i = 0; i < kernels[rank].x; i++) {
            compressible_from_swe<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(q_cd, qc_comp, nxs, nys, nzs, rho, gamma, kx_offset, ky_offset, dt, old_phi_d, clevel);
            kx_offset += blocks[k_offset + j * kernels[rank].x + i].x *
                threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
       }
       ky_offset += blocks[k_offset + j * kernels[rank].x].y *
            threads[k_offset + j * kernels[rank].x].y - 2*ng;
    }

    ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

    for (int j = 0; j < kernels[rank].y; j++) {
       kx_offset = 0;
       for (int i = 0; i < kernels[rank].x; i++) {
           prolong_reconstruct_comp_from_swe<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(qc_comp, q_fd, q_cd, nxs, nys, nzs, ng, dz, zmin, matching_indices_d, kx_offset, ky_offset, clevel, boundaries);

           kx_offset += blocks[k_offset + j * kernels[rank].x + i].x *
                threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
       }
       ky_offset += blocks[k_offset + j * kernels[rank].x].y *
            threads[k_offset + j * kernels[rank].x].y - 2*ng;
    }
}

__global__ void prolong_reconstruct_comp(float * q_f, float * q_c,
                    int * nxs, int * nys, int * nzs, int ng,
                    int * matching_indices_d,
                    int kx_offset, int ky_offset, int clevel, bool boundaries) {
    /**
    Reconstruct fine grid variables from compressible variables on coarse grid

    Parameters
    ----------
    q_f : float *
        fine grid state vector
    q_c : float *
        coarse grid swe state vector
    nxs, nys, nzs : int *
        grid dimensions
    ng : int
        number of ghost cells
    matching_indices_d : int *
        position of fine grid wrt coarse grid
    kx_offset, ky_offset : int
        kernel offsets in the x and y directions
    clevel : int
        index of coarser level
    */

    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    bool prolong;

    if (boundaries) {
        prolong = ((x < ng) ||
            ((x > int(round(nxs[clevel+1]*0.5)) - ng) &&
            (x < int(round(nxs[clevel+1]*0.5)))) ||
            (y < ng) ||
            ((y > int(round(nys[clevel+1]*0.5)) - ng) &&
            (y < int(round(nys[clevel+1]*0.5))))) &&
            (z < nzs[clevel+1]-1);
    } else {
        prolong = (x >= 0) && (x < int(round(nxs[clevel+1]*0.5)+1)) &&
            (y >= 0) && (y < int(round(nys[clevel+1]*0.5)+1)) &&
            (z >= 0) && (z < (nzs[clevel]-1));
    }

    if (prolong) {
        // corresponding x and y on the coarse grid
        int c_x = x + matching_indices_d[clevel*4];
        int c_y = y + matching_indices_d[clevel*4+2];

        for (int n = 0; n < 6; n++) {
            // do some slope limiting
            // x-dir above
            float S_upwind = 0.75 * (
                q_c[(((z+1) * nys[clevel] + c_y) * nxs[clevel] + c_x+1)*6+n] -
                q_c[(((z+1) * nys[clevel] + c_y) * nxs[clevel] + c_x)*6 + n]) +
                0.25 *
                (q_c[((z*nys[clevel] + c_y) * nxs[clevel] + c_x+1)*6 + n] -
                q_c[((z*nys[clevel]+c_y) * nxs[clevel]+c_x)*6 + n]);
            float S_downwind = 0.75 *
                (q_c[(((z+1) * nys[clevel] + c_y) * nxs[clevel] + c_x)*6+ n] -
                q_c[(((z+1) * nys[clevel] + c_y) * nxs[clevel] + c_x-1)*6+n]) +
                0.25 *
                (q_c[((z*nys[clevel] + c_y) * nxs[clevel] + c_x)*6 + n] -
                q_c[((z*nys[clevel] + c_y) * nxs[clevel]+c_x-1)*6+n]);

            float Sxp = 0.5 * (S_upwind + S_downwind);

            float r = 1.0e6;
            if (abs(S_downwind) > 1.0e-10) {
                r = S_upwind / S_downwind;
            }

            Sxp *= phi(r);

            // x-dir below
            S_upwind = 0.25 * (
                q_c[(((z+1) * nys[clevel] + c_y) * nxs[clevel] + c_x+1)*6+n] -
                q_c[(((z+1) * nys[clevel] + c_y) * nxs[clevel] + c_x)*6 + n]) +
                0.75 *
                (q_c[((z*nys[clevel] + c_y) * nxs[clevel] + c_x+1)*6 + n] -
                q_c[((z*nys[clevel]+c_y) * nxs[clevel]+c_x)*6 + n]);
            S_downwind = 0.25 *
                (q_c[(((z+1) * nys[clevel] + c_y) * nxs[clevel] + c_x)*6+ n] -
                q_c[(((z+1) * nys[clevel] + c_y) * nxs[clevel] + c_x-1)*6+n]) +
                0.75 *
                (q_c[((z*nys[clevel] + c_y) * nxs[clevel] + c_x)*6 + n] -
                q_c[((z*nys[clevel] + c_y) * nxs[clevel]+c_x-1)*6+n]);

            float Sxm = 0.5 * (S_upwind + S_downwind);

            r = 1.0e6;
            if (abs(S_downwind) > 1.0e-10) {
                r = S_upwind / S_downwind;
            }

            Sxm *= phi(r);

            // y-dir above
            S_upwind = 0.75 *
                (q_c[(((z+1) * nys[clevel] + c_y+1) * nxs[clevel] + c_x)*6+n] -
                q_c[(((z+1) * nys[clevel] + c_y) * nxs[clevel] + c_x)*6 + n]) +
                0.25 *
                (q_c[((z*nys[clevel] + c_y+1) * nxs[clevel] + c_x)*6 + n] -
                q_c[((z*nys[clevel]+c_y) * nxs[clevel]+c_x)*6 + n]);
            S_downwind = 0.75 *
                (q_c[(((z+1) * nys[clevel] + c_y) * nxs[clevel] + c_x)*6+n] -
                q_c[(((z+1) * nys[clevel] + c_y-1) * nxs[clevel] + c_x)*6+n]) +
                0.25 *
                (q_c[((z*nys[clevel] + c_y) * nxs[clevel] + c_x)*6 + n] -
                q_c[((z*nys[clevel] + c_y-1) * nxs[clevel]+c_x)*6+n]);

            float Syp = 0.5 * (S_upwind + S_downwind);

            r = 1.0e6;
            if (abs(S_downwind) > 1.0e-10) {
                r = S_upwind / S_downwind;
            }

            Syp *= phi(r);

            // y-dir below
            S_upwind = 0.25 *
                (q_c[(((z+1) * nys[clevel] + c_y+1) * nxs[clevel] + c_x)*6+n] -
                q_c[(((z+1) * nys[clevel] + c_y) * nxs[clevel] + c_x)*6 + n]) +
                0.75 *
                (q_c[((z*nys[clevel] + c_y+1) * nxs[clevel] + c_x)*6 + n] -
                q_c[((z*nys[clevel]+c_y) * nxs[clevel]+c_x)*6 + n]);
            S_downwind = 0.25 *
                (q_c[(((z+1) * nys[clevel] + c_y) * nxs[clevel] + c_x)*6+n] -
                q_c[(((z+1) * nys[clevel] + c_y-1) * nxs[clevel] + c_x)*6+n]) +
                0.75 *
                (q_c[((z*nys[clevel] + c_y) * nxs[clevel] + c_x)*6 + n] -
                q_c[((z*nys[clevel] + c_y-1) * nxs[clevel]+c_x)*6+n]);

            float Sym = 0.5 * (S_upwind + S_downwind);

            r = 1.0e6;
            if (abs(S_downwind) > 1.0e-10) {
                r = S_upwind / S_downwind;
            }

            Sym *= phi(r);

            // vertically interpolated component of q_c
            float interp_q_cp = 0.75 *
                q_c[(((z+1) * nys[clevel] + c_y) * nxs[clevel] + c_x)*6+n] +
                0.25 *
                q_c[((z * nys[clevel] + c_y) * nxs[clevel] + c_x) * 6 + n];

            float interp_q_cm = 0.25 *
                q_c[(((z+1) * nys[clevel] + c_y) * nxs[clevel] + c_x)*6+n] +
                0.75 *
                q_c[((z * nys[clevel] + c_y) * nxs[clevel] + c_x) * 6 + n];

            if (2*z+2 <  nzs[clevel+1] && 2*x < nxs[clevel+1] &&
                2*y < nys[clevel+1])
                q_f[(((2*z+2)*nys[clevel+1] + 2*y)*nxs[clevel+1] + 2*x)*6+n] =
                interp_q_cp - 0.25 * (Sxp + Syp);

            if (2*z+2 <  nzs[clevel+1] && 2*x+1 < nxs[clevel+1] &&
                2*y < nys[clevel+1])
                q_f[(((2*z+2)*nys[clevel+1]+ 2*y)*nxs[clevel+1] + 2*x+1)*6+n] =
                interp_q_cp + 0.25 * (Sxp - Syp);

            if (2*z+2 <  nzs[clevel+1] && 2*x < nxs[clevel+1] &&
                2*y+1 < nys[clevel+1])
                q_f[(((2*z+2)*nys[clevel+1]+2*y+1)* nxs[clevel+1] + 2*x)*6+n] =
                interp_q_cp + 0.25 * (-Sxp + Syp);

            if (2*z+2 <  nzs[clevel+1] && 2*x+1 < nxs[clevel+1] &&
                2*y+1 < nys[clevel+1])
                q_f[(((2*z+2)*nys[clevel+1]+2*y+1)*nxs[clevel+1]+ 2*x+1)*6+n] =
                interp_q_cp + 0.25 * (Sxp + Syp);

            if (2*z+1 <  nzs[clevel+1] && 2*x < nxs[clevel+1] &&
                2*y < nys[clevel+1])
                q_f[(((2*z+1)*nys[clevel+1]+2*y)*nxs[clevel+1] + 2*x) *6 + n] =
                interp_q_cm - 0.25 * (Sxm + Sym);

            if (2*z+1 <  nzs[clevel+1] && 2*x+1 < nxs[clevel+1] &&
                2*y < nys[clevel+1])
                q_f[(((2*z+1)*nys[clevel+1]+2*y)*nxs[clevel+1]+ 2*x+1)*6 + n] =
                interp_q_cm + 0.25 * (Sxm - Sym);

            if (2*z+1 <  nzs[clevel+1] && 2*x < nxs[clevel+1] &&
                2*y+1 < nys[clevel+1])
                q_f[(((2*z+1)*nys[clevel+1]+2*y+1)*nxs[clevel+1]+ 2*x)*6 + n] =
                interp_q_cm + 0.25 * (-Sxm + Sym);

            if (2*z+1 <  nzs[clevel+1] && 2*x+1 < nxs[clevel+1] &&
                2*y+1 < nys[clevel+1])
                q_f[(((2*z+1)*nys[clevel+1]+2*y+1)*nxs[clevel+1]+ 2*x+1)*6+n] =
                interp_q_cm + 0.25 * (Sxm + Sym);
        }

    }
}

void prolong_comp_to_comp(dim3 * kernels, dim3 * threads, dim3 * blocks,
                  int * cumulative_kernels, float * q_cd, float * q_fd,
                  int * nxs, int * nys, int * nzs,
                  int * matching_indices_d, int ng, int rank, int clevel, bool boundaries) {
    /**
    Prolong coarse grid data to fine grid

    Parameters
    ----------
    kernels, threads, blocks : dim3 *
        number of kernels, threads and blocks for each process/kernel
    cumulative_kernels : int *
        cumulative number of kernels in mpi processes of r < rank
    q_cd, q_fd : float *
        coarse and fine grids of state vectors
    nxs, nys, nzs : int *
        dimensions of grids
    ng : int
        number of ghost cells
    matching_indices_d : int *
        position of fine grid wrt coarse grid
    ng : int
        number of ghost cells
    rank : int
        rank of MPI process
    clevel : int
        index of coarser level
    */

    int kx_offset = 0;
    int ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

    int k_offset = 0;
    if (rank > 0) {
        k_offset = cumulative_kernels[rank - 1];
    }

    for (int j = 0; j < kernels[rank].y; j++) {
       kx_offset = 0;
       for (int i = 0; i < kernels[rank].x; i++) {
           prolong_reconstruct_comp<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(q_fd, q_cd, nxs, nys, nzs, ng, matching_indices_d, kx_offset, ky_offset, clevel, boundaries);

           kx_offset += blocks[k_offset + j * kernels[rank].x + i].x *
                threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
       }
       ky_offset += blocks[k_offset + j * kernels[rank].x].y *
            threads[k_offset + j * kernels[rank].x].y - 2*ng;
    }
}

__global__ void prolong_reconstruct_swe_from_swe(float * qf, float * qc,
                    int * nxs, int * nys, int * nzs, int ng,
                    int * matching_indices_d,
                    int kx_offset, int ky_offset, int clevel, bool boundaries) {
    /**
    Reconstruct multilayer swe fine grid variables from single layer swe variables on coarse grid

    Parameters
    ----------
    qf : float *
        fine grid state vector
    qc : float *
        coarse grid swe state vector
    nxs, nys, nzs : int *
        grid dimensions
    ng : int
        number of ghost cells
    matching_indices_d : int *
        position of fine grid wrt coarse grid
    kx_offset, ky_offset : int
        kernel offsets in the x and y directions
    clevel : int
        index of coarser level
    */

    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    bool prolong;

    if (boundaries) {
        prolong = ((x < ng) ||
            ((x > int(round(nxs[clevel+1]*0.5)) - ng) &&
            (x < int(round(nxs[clevel+1]*0.5)))) ||
            (y < ng) ||
            ((y > int(round(nys[clevel+1]*0.5)) - ng) &&
            (y < int(round(nys[clevel+1]*0.5))))) &&
            (z == 0);
    } else {
        prolong = (x >=0) && (x < int(round(nxs[clevel+1]*0.5)+1)) &&
            (y >= 0) && (y < int(round(nys[clevel+1]*0.5)+1)) &&
            (z == 0);
    }

    if (prolong) {
        // corresponding x and y on the coarse grid
        int c_x = x + matching_indices_d[clevel*4];
        int c_y = y + matching_indices_d[clevel*4+2];

        for (int n = 0; n < 4; n++) {
            // do some slope limiting
            // x-dir
            float S_upwind =
                qc[(c_y * nxs[clevel] + c_x+1) * 4 + n] -
                qc[(c_y * nxs[clevel] + c_x) * 4 + n];
            float S_downwind =
                qc[(c_y * nxs[clevel] + c_x) * 4 + n] -
                qc[(c_y * nxs[clevel] + c_x-1) * 4 + n];

            float Sx = 0.5 * (S_upwind + S_downwind);

            float r = 1.0e6;
            if (abs(S_downwind) > 1.0e-10) {
                r = S_upwind / S_downwind;
            }

            Sx *= phi(r);

            // y-dir
            S_upwind =
                qc[((c_y+1) * nxs[clevel] + c_x) * 4 + n] -
                qc[(c_y * nxs[clevel] + c_x) * 4 + n];
            S_downwind =
                qc[(c_y * nxs[clevel] + c_x) * 4 + n] -
                qc[((c_y-1) * nxs[clevel] + c_x) * 4 + n];

            float Sy = 0.5 * (S_upwind + S_downwind);

            r = 1.0e6;
            if (abs(S_downwind) > 1.0e-10) {
                r = S_upwind / S_downwind;
            }

            Sy *= phi(r);

            // vertically interpolated component of q_comp
            float interp_q_comp =
                qc[(c_y * nxs[clevel] + c_x) * 4 + n];

            if (2*x < nxs[clevel+1] && 2*y < nys[clevel+1])
                qf[(2*y * nxs[clevel+1] + 2*x) * 4 + n] =
                    interp_q_comp - 0.25 * (Sx + Sy);

            if (2*x+1 < nxs[clevel+1] && 2*y < nys[clevel+1])
                qf[(2*y * nxs[clevel+1] + 2*x+1) * 4 + n] =
                    interp_q_comp + 0.25 * (Sx - Sy);

            if (2*x < nxs[clevel+1] && 2*y+1 < nys[clevel+1])
                qf[((2*y+1) * nxs[clevel+1] + 2*x) * 4 + n] =
                    interp_q_comp + 0.25 * (-Sx + Sy);

            if (2*x+1 < nxs[clevel+1] && 2*y+1 < nys[clevel+1])
                qf[((2*y+1) * nxs[clevel+1] + 2*x+1) * 4 + n] =
                    interp_q_comp + 0.25 * (Sx + Sy);
        }
    }
}

void prolong_swe_to_swe(dim3 * kernels, dim3 * threads, dim3 * blocks,
                  int * cumulative_kernels, float * q_cd, float * q_fd,
                  int * nxs, int * nys, int * nzs,
                  int * matching_indices_d, int ng, int rank,
                  int clevel, bool boundaries) {
    /**
    Prolong coarse grid single layer swe data to fine multilayer swe grid.

    Parameters
    ----------
    kernels, threads, blocks : dim3 *
        number of kernels, threads and blocks for each process/kernel
    cumulative_kernels : int *
        cumulative number of kernels in mpi processes of r < rank
    q_cd, q_fd : float *
        coarse and fine grids of state vectors
    nxs, nys, nzs : int *
        dimensions of grids
    matching_indices_d : int *
        position of fine grid wrt coarse grid
    ng : int
        number of ghost cells
    rank : int
        rank of MPI process
    clevel : int
        index of coarser level
    */

    int kx_offset = 0;
    int ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

    int k_offset = 0;
    if (rank > 0) {
        k_offset = cumulative_kernels[rank - 1];
    }

    for (int j = 0; j < kernels[rank].y; j++) {
       kx_offset = 0;
       for (int i = 0; i < kernels[rank].x; i++) {
           prolong_reconstruct_swe_from_swe<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(q_fd, q_cd, nxs, nys, nzs, ng, matching_indices_d, kx_offset, ky_offset, clevel, boundaries);

           kx_offset += blocks[k_offset + j * kernels[rank].x + i].x *
                threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
       }
       ky_offset += blocks[k_offset + j * kernels[rank].x].y *
            threads[k_offset + j * kernels[rank].x].y - 2*ng;
    }
}

__global__ void prolong_reconstruct_multiswe_from_multiswe(float * qf, float * qc,
                    int * nxs, int * nys, int * nzs, int ng,
                    int * matching_indices_d,
                    int kx_offset, int ky_offset, int clevel, bool boundaries) {
    /**
    Reconstruct multilayer swe fine grid variables from multilayer swe variables on coarse grid

    Parameters
    ----------
    qf : float *
        fine grid state vector
    qc : float *
        coarse grid swe state vector
    nxs, nys, nzs : int *
        grid dimensions
    ng : int
        number of ghost cells
    matching_indices_d : int *
        position of fine grid wrt coarse grid
    kx_offset, ky_offset : int
        kernel offsets in the x and y directions
    clevel : int
        index of coarser level
    */

    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    bool prolong;

    if (boundaries) {
        prolong = ((x < ng) ||
            ((x > int(round(nxs[clevel+1]*0.5)) - ng) &&
            (x < int(round(nxs[clevel+1]*0.5)))) ||
            (y < ng) ||
            ((y > int(round(nys[clevel+1]*0.5)) - ng) &&
            (y < int(round(nys[clevel+1]*0.5))))) &&
            (z < nzs[clevel]);
    } else {
        prolong = (x >=0) && (x < int(round(nxs[clevel+1]*0.5)+1)) &&
            (y >= 0) && (y < int(round(nys[clevel+1]*0.5)+1)) &&
            (z < nzs[clevel]);
    }

    if (prolong) {
        // corresponding x and y on the coarse grid
        int c_x = x + matching_indices_d[clevel*4];
        int c_y = y + matching_indices_d[clevel*4+2];

        for (int n = 0; n < 4; n++) {
            // do some slope limiting
            // x-dir
            float S_upwind =
                qc[((z * nys[clevel] + c_y) * nxs[clevel] + c_x+1) * 4 + n] -
                qc[((z * nys[clevel] + c_y) * nxs[clevel] + c_x) * 4 + n];
            float S_downwind =
                qc[((z * nys[clevel] + c_y) * nxs[clevel] + c_x) * 4 + n] -
                qc[((z * nys[clevel] + c_y) * nxs[clevel] + c_x-1) * 4 + n];

            float Sx = 0.5 * (S_upwind + S_downwind);

            float r = 1.0e6;
            if (abs(S_downwind) > 1.0e-10) {
                r = S_upwind / S_downwind;
            }

            Sx *= phi(r);

            // y-dir
            S_upwind =
                qc[((z * nys[clevel] + c_y+1) * nxs[clevel] + c_x) * 4 + n] -
                qc[((z * nys[clevel] + c_y) * nxs[clevel] + c_x) * 4 + n];
            S_downwind =
                qc[((z * nys[clevel] + c_y) * nxs[clevel] + c_x) * 4 + n] -
                qc[((z * nys[clevel] + c_y-1) * nxs[clevel] + c_x) * 4 + n];

            float Sy = 0.5 * (S_upwind + S_downwind);

            r = 1.0e6;
            if (abs(S_downwind) > 1.0e-10) {
                r = S_upwind / S_downwind;
            }

            Sy *= phi(r);

            // vertically interpolated component of q_comp
            float interp_q_comp =
                qc[((z * nys[clevel] + c_y) * nxs[clevel] + c_x) * 4 + n];

            if (2*x < nxs[clevel+1] && 2*y < nys[clevel+1])
                qf[((z * nys[clevel+1] + 2*y) * nxs[clevel+1] + 2*x) * 4 + n] =
                    interp_q_comp - 0.25 * (Sx + Sy);

            if (2*x+1 < nxs[clevel+1] && 2*y < nys[clevel+1])
                qf[((z*nys[clevel+1] + 2*y) * nxs[clevel+1] + 2*x+1) * 4 + n] =
                    interp_q_comp + 0.25 * (Sx - Sy);

            if (2*x < nxs[clevel+1] && 2*y+1 < nys[clevel+1])
                qf[((z*nys[clevel+1] + 2*y+1) * nxs[clevel+1] + 2*x) * 4 + n] =
                    interp_q_comp + 0.25 * (-Sx + Sy);

            if (2*x+1 < nxs[clevel+1] && 2*y+1 < nys[clevel+1])
                qf[((z*nys[clevel+1]+2*y+1) * nxs[clevel+1] + 2*x+1) * 4 + n] =
                    interp_q_comp + 0.25 * (Sx + Sy);
        }
    }
}

void prolong_multiswe_to_multiswe(dim3 * kernels, dim3 * threads, dim3 * blocks,
                  int * cumulative_kernels, float * q_cd, float * q_fd,
                  int * nxs, int * nys, int * nzs,
                  int * matching_indices_d, int ng, int rank,
                  int clevel, bool boundaries) {
    /**
    Prolong coarse grid multilayer swe data to fine multilayer swe grid.

    Parameters
    ----------
    kernels, threads, blocks : dim3 *
        number of kernels, threads and blocks for each process/kernel
    cumulative_kernels : int *
        cumulative number of kernels in mpi processes of r < rank
    q_cd, q_fd : float *
        coarse and fine grids of state vectors
    nxs, nys, nzs : int *
        dimensions of grids
    matching_indices_d : int *
        position of fine grid wrt coarse grid
    ng : int
        number of ghost cells
    rank : int
        rank of MPI process
    clevel : int
        index of coarser level
    */

    int kx_offset = 0;
    int ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

    int k_offset = 0;
    if (rank > 0) {
        k_offset = cumulative_kernels[rank - 1];
    }

    for (int j = 0; j < kernels[rank].y; j++) {
       kx_offset = 0;
       for (int i = 0; i < kernels[rank].x; i++) {
           prolong_reconstruct_multiswe_from_multiswe<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(q_fd, q_cd, nxs, nys, nzs, ng, matching_indices_d, kx_offset, ky_offset, clevel, boundaries);

           kx_offset += blocks[k_offset + j * kernels[rank].x + i].x *
                threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
       }
       ky_offset += blocks[k_offset + j * kernels[rank].x].y *
            threads[k_offset + j * kernels[rank].x].y - 2*ng;
    }
}


__global__ void restrict_interpolate_swe(float * qf_prim, float * q_c,
                                     int * nxs, int * nys, int * nzs, int ng,
                                     float dz, float zmin,
                                     int * matching_indices,
                                     int kx_offset, int ky_offset,
                                     int clevel) {

    /**
    Interpolate SWE variables on fine grid to get them on coarse grid.

    Parameters
    ----------
    qf_swe : float *
        SWE variables on fine grid
    q_c : float *
        coarse grid state vector
    nxs, nys, nzs : int *
        coarse grid dimensions
    ng : int
        number of ghost cells
    matching_indices : int *
        position of fine grid wrt coarse grid
    kx_offset, ky_offset : int
        kernel offsets in the x and y directions
    clevel : int
        index of coarser grid level
    */
    // interpolate fine grid to coarse grid
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    // note we're not going to restrict the top layer
    if (((x > 1) && (x < int(round(nxs[clevel+1]*0.5))-1)) &&
        ((y > 1) && (y < int(round(nys[clevel+1]*0.5))-1)) &&
        (z > 0) && (z < nzs[clevel]-1)) {
        // first find position of layers relative to fine grid
        int coarse_index = ((z * nys[clevel] +
            y + matching_indices[clevel*4+2]) *
            nxs[clevel] +
            x + matching_indices[clevel*4]) * 4;

        float * q_c_new;
        q_c_new = (float *)malloc(4 * sizeof(float));
        for (int i = 0; i < 4; i++) {
            q_c_new[i] = q_c[coarse_index+i];
        }

        float W = W_swe(q_c_new);
        float r = find_height(q_c[coarse_index] / W);
        float height_guess = find_height(q_c[coarse_index] / W);

        int z_index = nzs[clevel+1];
        float z_frac = 0.0;

        if (height_guess > (zmin + (nzs[clevel+1] - 1.0) * dz)) { // SWE layer above top compressible layer
            z_index = 1;
            float height = zmin + (nzs[clevel+1] - 1 - 1) * dz;
            z_frac = -(height_guess - (height+dz)) / dz;
        } else {

            for (int i = 1; i < (nzs[clevel+1]-1); i++) {
                float height = zmin + (nzs[clevel+1] - 1 - i) * dz;
                if (height_guess > height) {
                    z_index = i;
                    z_frac = -(height_guess - (height+dz)) / dz;
                    break;
                }
            }

            if (z_index == nzs[clevel+1]) {
                z_index = nzs[clevel+1] - 1;
                z_frac = 1.0;
            }
        }

        int l = z_index;
        float Ww[8];
        for (int j = 0; j < 2; j++) {
            for (int i = 0; i < 2; i++) {
                float u, v, w;
                u=qf_prim[((l*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+1];
                v=qf_prim[((l*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+2];
                w=qf_prim[((l*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+3];
                Ww[j*2+i] = 1.0 / sqrt(1.0 -
                        u*u*gamma_up_d[0] - 2.0 * u*v * gamma_up_d[1] -
                        2.0 * u*w * gamma_up_d[2] - v*v*gamma_up_d[4] -
                        2.0 * v*w*gamma_up_d[5] - w*w*gamma_up_d[8]);

                u=qf_prim[(((l-1)*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+1];
                v=qf_prim[(((l-1)*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+2];
                w=qf_prim[(((l-1)*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+3];
                Ww[(2+j)*2+i] = 1.0 / sqrt(1.0 -
                        u*u*gamma_up_d[0] - 2.0 * u*v * gamma_up_d[1] -
                        2.0 * u*w * gamma_up_d[2] - v*v*gamma_up_d[4] -
                        2.0 * v*w*gamma_up_d[5] - w*w*gamma_up_d[8]);

            }
        }

        float interp_W = z_frac * 0.25 * (Ww[0] + Ww[1] + Ww[2] + Ww[3]) +
            (1.0 - z_frac) * 0.25 * (Ww[4] + Ww[5] + Ww[6] + Ww[7]);

        float Phi = find_pot(height_guess);
        // now need to do linear interpolation thing on u, v
        float u[8];
        float v[8];
        float WX[8];
        for (int j = 0; j < 2; j++) {
            for (int i = 0; i < 2; i++) {
                u[j*2+i] =
                qf_prim[((l*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+1];

                v[j*2+i] =
                qf_prim[((l*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+2];

                WX[j*2+i] = Ww[j*2+i] *
                qf_prim[((l*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+5];

                u[(2+j)*2+i] =
                qf_prim[(((l-1)*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+1];
                v[(2+j)*2+i] =
                qf_prim[(((l-1)*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+2];
                WX[(2+j)*2+i] = Ww[(2+j)*2+i] *
                qf_prim[(((l-1)*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+5];
            }
        }

        float interp_u = z_frac * 0.25 * (u[0] + u[1] + u[2] + u[3]) +
            (1.0 - z_frac) * 0.25 * (u[4] + u[5] + u[6] + u[7]);
        float interp_v = z_frac * 0.25 * (v[0] + v[1] + v[2] + v[3]) +
            (1.0 - z_frac) * 0.25 * (v[4] + v[5] + v[6] + v[7]);
        float interp_WX = z_frac * 0.25 * (WX[0] + WX[1] + WX[2] + WX[3]) +
            (1.0 - z_frac) * 0.25 * (WX[4] + WX[5] + WX[6] + WX[7]);

        q_c[coarse_index] = Phi * interp_W;
        q_c[coarse_index + 1] = q_c[coarse_index] * interp_W * interp_u;
        q_c[coarse_index + 2] = q_c[coarse_index] * interp_W * interp_v;
        q_c[coarse_index + 3] = Phi * interp_WX ;

        free(q_c_new);
    } /*else if ((x > 0) && (x < int(round(nxf*0.5))) && (y > 0) && (y < int(round(nyf*0.5))) && (z == nlayers-1)) { // sea floor
        int coarse_index = ((z * ny + y+matching_indices[2]) * nx +
              x+matching_indices[0]) * 4;
        int z_index = nz-1;
        for (int n = 0; n < 3; n++) {
            q_c[coarse_index + n] = 0.25 *
                (qf_sw[((z_index * nyf + y*2) * nxf + x*2) * 4 + n] +
                qf_sw[((z_index * nyf + y*2) * nxf + x*2+1) * 4 + n] +
                qf_sw[((z_index * nyf + y*2+1) * nxf + x*2) * 4 + n] +
                qf_sw[((z_index * nyf + y*2+1) * nxf + x*2+1) * 4 + n]);
        }
    }*/
}

void restrict_comp_to_swe(dim3 * kernels, dim3 * threads, dim3 * blocks,
                    int * cumulative_kernels, float * q_cd, float * q_fd,
                    int * nxs, int * nys, int * nzs,
                    float dz, float zmin, int * matching_indices,
                    float * rho, float gamma,
                    int ng, int rank, float * qf_swe,
                    int clevel) {
    /**
    Restrict fine compressible grid data to coarse swe grid

    Parameters
    ----------
    kernels, threads, blocks : dim3 *
        number of kernels, threads and blocks for each process/kernel
    cumulative_kernels : int *
        cumulative number of kernels in mpi processes of r < rank
    q_cd, q_fd : float *
        coarse and fine grids of state vectors
    nxs, nys, nzs : int *
        dimensions of grids
    matching_indices : int *
        position of fine grid wrt coarse grid
    rho, gamma : float
        density and adiabatic index
    ng : int
        number of ghost cells
    rank : int
        rank of MPI process
    qf_swe : float *
        grid of SWE variables on fine grid
    clevel : int
        index of coarser grid
    */

    int kx_offset = 0;
    int ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

    int k_offset = 0;
    if (rank > 0) {
        k_offset = cumulative_kernels[rank - 1];
    }

    for (int j = 0; j < kernels[rank].y; j++) {
       kx_offset = 0;
       for (int i = 0; i < kernels[rank].x; i++) {
            calc_comp_prim<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(q_fd, nxs, nys, nzs, gamma, kx_offset, ky_offset, clevel);

            kx_offset += blocks[k_offset + j * kernels[rank].x + i].x *
                threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
       }
       ky_offset += blocks[k_offset + j * kernels[rank].x].y *
            threads[k_offset + j * kernels[rank].x].y - 2*ng;
    }

    ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

    for (int j = 0; j < kernels[rank].y; j++) {
       kx_offset = 0;
       for (int i = 0; i < kernels[rank].x; i++) {
           restrict_interpolate_swe<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(q_fd ,q_cd, nxs, nys, nzs, ng, dz, zmin, matching_indices, kx_offset, ky_offset, clevel);

           kx_offset += blocks[k_offset + j * kernels[rank].x + i].x *
                threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
       }
       ky_offset += blocks[k_offset + j * kernels[rank].x].y *
            threads[k_offset + j * kernels[rank].x].y - 2*ng;
    }
}

__global__ void restrict_interpolate_comp(float * qf, float * qc,
                                     int * nxs, int * nys, int * nzs, int ng,
                                     int * matching_indices,
                                     int kx_offset, int ky_offset,
                                     int clevel) {

    /**
    Interpolate fine grid compressible variables to get them on coarser compressible grid.

    Parameters
    ----------
    qf : float *
        variables on fine grid
    qc : float *
        coarse grid state vector
    nxs, nys, nzs : int *
        coarse grid dimensions
    ng : int
        number of ghost cells
    matching_indices : int *
        position of fine grid wrt coarse grid
    kx_offset, ky_offset : int
        kernel offsets in the x and y directions
    clevel : int
        index of coarser grid level
    */
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    if (((x > 1) && (x < int(round(nxs[clevel+1]*0.5))-1)) &&
        ((y > 1) && (y < int(round(nys[clevel+1]*0.5))-1)) &&
        (z > 0) && (z < nzs[clevel]-1)) {
        // first find position of layers relative to fine grid
        int coarse_index = ((z * nys[clevel] +
            y + matching_indices[clevel*4+2]) *
            nxs[clevel] +
            x + matching_indices[clevel*4]) * 6;

        for (int i = 0; i < 6; i++) {
            // average in x-direction to get xtp (top above), xbp (bottom above), xtm (top below), xbm (bottom below)
            float xtp = 0.5 * (
                qf[(((z*2+1)*nys[clevel+1]+y*2+1)*nxs[clevel+1]+x*2)*6+i] +
                qf[(((z*2+1)*nys[clevel+1]+y*2+1)*nxs[clevel+1]+x*2+1)*6+i]);
            float xbp = 0.5 * (
                qf[(((z*2+1)*nys[clevel+1]+y*2)*nxs[clevel+1]+x*2)*6+i] +
                qf[(((z*2+1)*nys[clevel+1]+y*2)*nxs[clevel+1]+x*2+1)*6+i]);
            float xtm = 0.5 * (
                qf[(((z*2)*nys[clevel+1]+y*2+1)*nxs[clevel+1]+x*2)*6+i] +
                qf[(((z*2)*nys[clevel+1]+y*2+1)*nxs[clevel+1]+x*2+1)*6+i]);
            float xbm = 0.5 * (
                qf[(((z*2)*nys[clevel+1]+y*2)*nxs[clevel+1]+x*2)*6+i] +
                qf[(((z*2)*nys[clevel+1]+y*2)*nxs[clevel+1]+x*2+1)*6+i]);

            // average in y-direction to get yp (above) and ym (below)
            float yp = 0.5 * (xtp + xbp);
            float ym = 0.5 * (xtm + xbm);

            // average in z-direction
            qc[coarse_index+i] = 0.5 * (yp + ym);
        }
    }
}

void restrict_comp_to_comp(dim3 * kernels, dim3 * threads, dim3 * blocks,
                    int * cumulative_kernels, float * q_cd, float * q_fd,
                    int * nxs, int * nys, int * nzs,
                    int * matching_indices,
                    int ng, int rank, int clevel) {
    /**
    Restrict fine compressible grid data to coarse compressible grid.

    Parameters
    ----------
    kernels, threads, blocks : dim3 *
        number of kernels, threads and blocks for each process/kernel
    cumulative_kernels : int *
        cumulative number of kernels in mpi processes of r < rank
    q_cd, q_fd : float *
        coarse and fine grids of state vectors
    nxs, nys, nzs : int *
        dimensions of grids
    matching_indices : int *
        position of fine grid wrt coarse grid
    ng : int
        number of ghost cells
    rank : int
        rank of MPI process
    clevel : int
        index of coarser grid
    */

    int kx_offset = 0;
    int ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

    int k_offset = 0;
    if (rank > 0) {
        k_offset = cumulative_kernels[rank - 1];
    }

    for (int j = 0; j < kernels[rank].y; j++) {
       kx_offset = 0;
       for (int i = 0; i < kernels[rank].x; i++) {
           restrict_interpolate_comp<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(q_fd, q_cd, nxs, nys, nzs, ng, matching_indices, kx_offset, ky_offset, clevel);

           kx_offset += blocks[k_offset + j * kernels[rank].x + i].x *
                threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
       }
       ky_offset += blocks[k_offset + j * kernels[rank].x].y *
            threads[k_offset + j * kernels[rank].x].y - 2*ng;
    }
}

__global__ void restrict_interpolate_swe_to_swe(float * qf, float * qc,
                                     int * nxs, int * nys, int * nzs, int ng,
                                     int * matching_indices,
                                     int kx_offset, int ky_offset,
                                     int clevel) {

    /**
    Interpolate multilayer SWE variables on fine grid to get them on single layer SWE coarse grid.

    Parameters
    ----------
    qf : float *
        multilayer SWE variables on fine grid
    qc : float *
        coarse grid state vector
    nxs, nys, nzs : int *
        coarse grid dimensions
    ng : int
        number of ghost cells
    matching_indices : int *
        position of fine grid wrt coarse grid
    gamma_up : float *
        spatial metric
    kx_offset, ky_offset : int
        kernel offsets in the x and y directions
    clevel : int
        index of coarser grid level
    */
    // interpolate fine grid to coarse grid
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    if (((x > 1) && (x < int(round(nxs[clevel+1]*0.5))-1)) &&
        ((y > 1) && (y < int(round(nys[clevel+1]*0.5))-1)) &&
        (z==0)) {
        // first find position of layers relative to fine grid
        int coarse_index = ((y + matching_indices[clevel*4+2]) * nxs[clevel] +
            x + matching_indices[clevel*4]) * 4;

        for (int i = 0; i < 4; i++) {
            qc[coarse_index + i] = 0.25 *
                (qf[(y * 2 * nxs[clevel+1] + x * 2) * 4 + i] +
                 qf[((y * 2 + 1) * nxs[clevel+1] + x * 2) * 4 + i] +
                 qf[(y * 2 * nxs[clevel+1] + x * 2 + 1) * 4 + i] +
                 qf[((y * 2 + 1) * nxs[clevel+1] + x * 2 + 1) * 4 + i]);
        }
    }
}

void restrict_swe_to_swe(dim3 * kernels, dim3 * threads, dim3 * blocks,
                    int * cumulative_kernels, float * q_cd, float * q_fd,
                    int * nxs, int * nys, int * nzs,
                    int * matching_indices,
                    int ng, int rank, int clevel) {
    /**
    Restrict fine multilayer swe grid data to coarse single layer swe grid.

    Parameters
    ----------
    kernels, threads, blocks : dim3 *
        number of kernels, threads and blocks for each process/kernel
    cumulative_kernels : int *
        cumulative number of kernels in mpi processes of r < rank
    q_cd, q_fd : float *
        coarse and fine grids of state vectors
    nxs, nys, nzs : int *
        dimensions of grids
    matching_indices : int *
        position of fine grid wrt coarse grid
    ng : int
        number of ghost cells
    rank : int
        rank of MPI process
    clevel : int
        index of coarser grid
    */

    int kx_offset = 0;
    int ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

    int k_offset = 0;
    if (rank > 0) {
        k_offset = cumulative_kernels[rank - 1];
    }

    for (int j = 0; j < kernels[rank].y; j++) {
       kx_offset = 0;
       for (int i = 0; i < kernels[rank].x; i++) {
           restrict_interpolate_swe_to_swe<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(q_fd, q_cd, nxs, nys, nzs, ng, matching_indices, kx_offset, ky_offset, clevel);

           kx_offset += blocks[k_offset + j * kernels[rank].x + i].x *
                threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
       }
       ky_offset += blocks[k_offset + j * kernels[rank].x].y *
            threads[k_offset + j * kernels[rank].x].y - 2*ng;
    }
}

__global__ void restrict_interpolate_multiswe_to_multiswe(float * qf, float * qc,
                                     int * nxs, int * nys, int * nzs, int ng,
                                     int * matching_indices,
                                     int kx_offset, int ky_offset,
                                     int clevel) {

    /**
    Interpolate multilayer SWE variables on fine grid to get them on multilayer SWE coarse grid.

    Parameters
    ----------
    qf : float *
        multilayer SWE variables on fine grid
    qc : float *
        coarse grid state vector
    nxs, nys, nzs : int *
        coarse grid dimensions
    ng : int
        number of ghost cells
    matching_indices : int *
        position of fine grid wrt coarse grid
    gamma_up : float *
        spatial metric
    kx_offset, ky_offset : int
        kernel offsets in the x and y directions
    clevel : int
        index of coarser grid level
    */
    // interpolate fine grid to coarse grid
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    if (((x > 1) && (x < int(round(nxs[clevel+1]*0.5))-1)) &&
        ((y > 1) && (y < int(round(nys[clevel+1]*0.5))-1)) &&
        (z < nzs[clevel])) {
        // first find position of layers relative to fine grid
        int coarse_index = ((z * nys[clevel] +
            y + matching_indices[clevel*4+2]) * nxs[clevel] +
            x + matching_indices[clevel*4]) * 4;

        for (int i = 0; i < 4; i++) {
            qc[coarse_index + i] = 0.25 *
                (qf[((z*nys[clevel+1]+y*2) * nxs[clevel+1] + x * 2) * 4 + i] +
                 qf[((z*nys[clevel+1]+y*2+1)* nxs[clevel+1] + x * 2) * 4 + i] +
                 qf[((z*nys[clevel+1]+y*2)*nxs[clevel+1]+ x * 2 + 1) * 4 + i] +
                 qf[((z*nys[clevel+1]+y*2+1)*nxs[clevel+1]+x*2 + 1) * 4 + i]);
        }
    }
}

void restrict_multiswe_to_multiswe(dim3 * kernels, dim3 * threads, dim3 * blocks,
                    int * cumulative_kernels, float * q_cd, float * q_fd,
                    int * nxs, int * nys, int * nzs,
                    int * matching_indices,
                    int ng, int rank, int clevel) {
    /**
    Restrict fine multilayer swe grid data to coarse multilayer swe grid.

    Parameters
    ----------
    kernels, threads, blocks : dim3 *
        number of kernels, threads and blocks for each process/kernel
    cumulative_kernels : int *
        cumulative number of kernels in mpi processes of r < rank
    q_cd, q_fd : float *
        coarse and fine grids of state vectors
    nxs, nys, nzs : int *
        dimensions of grids
    matching_indices : int *
        position of fine grid wrt coarse grid
    ng : int
        number of ghost cells
    rank : int
        rank of MPI process
    clevel : int
        index of coarser grid
    */

    int kx_offset = 0;
    int ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

    int k_offset = 0;
    if (rank > 0) {
        k_offset = cumulative_kernels[rank - 1];
    }

    for (int j = 0; j < kernels[rank].y; j++) {
       kx_offset = 0;
       for (int i = 0; i < kernels[rank].x; i++) {
           restrict_interpolate_multiswe_to_multiswe<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(q_fd, q_cd, nxs, nys, nzs, ng, matching_indices, kx_offset, ky_offset, clevel);

           kx_offset += blocks[k_offset + j * kernels[rank].x + i].x *
                threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
       }
       ky_offset += blocks[k_offset + j * kernels[rank].x].y *
            threads[k_offset + j * kernels[rank].x].y - 2*ng;
    }
}

void interpolate_rhos(float * rho_column, float * rho_grid, float zmin, float zmax, float dz, float * phs, int nx, int ny, int nz) {
    /*
    When implemented, this will calculate rho on the compressible grid given
    the rhos on each of the SW layers and the heights of the layers (via phi).
    */
    for (int i = 0; i < nx * ny * nz; i++) {
        rho_grid[i] = rho_column[0];
    }
}
