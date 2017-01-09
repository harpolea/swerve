#ifndef _MESH_CUDA_KERNEL_H_
#define _MESH_CUDA_KERNEL_H_

#include <stdio.h>
#include <mpi.h>
#include "H5Cpp.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_functions.h>
#include "Mesh_cuda.h"

using namespace std;

// TODO: This file is becoming way too long - move a load of the functions into other files.

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

__host__ __device__ bool nan_check(float a) {
    // check to see whether float a is a nan
    if (a != a || (abs(a) > 1.0e13)) {
        return true;
    } else {
        return false;
    }
}

__host__ __device__ float zbrent(fptr func, const float x1, const float x2,
             const float tol,
             float D, float Sx, float Sy, float Sz, float tau, float gamma,
             float * gamma_up) {
    /*
    Using Brent's method, return the root of a function or functor func known
    to lie between x1 and x2. The root will be regined until its accuracy is
    tol.

    Parameters
    ----------
    func : fptr
        function pointer to shallow water or compressible flux function.
    x1, x2 : const float
        limits of root
    tol : const float
        tolerance to which root shall be calculated to
    D, Sx, Sy, tau: float
        conserved variables
    gamma : float
        adiabatic index
    gamma_up : float *
        spatial metric
    */

    const int ITMAX = 300;

    float a = x1, b = x2;
    float c, d=0.0, e=0.0;
    float fa = func(a, D, Sx, Sy, Sz, tau, gamma, gamma_up);
    float fb = func(b, D, Sx, Sy, Sz, tau, gamma, gamma_up);
    float fc=0.0, fs, s;

    if (fa * fb >= 0.0) {
        //cout << "Root must be bracketed in zbrent.\n";
        //printf("Root must be bracketed in zbrent.\n");
        return x2;
    }

    if (abs(fa) < abs(fb)) {
        // swap a, b
        d = a;
        a = b;
        b = d;

        d = fa;
        fa = fb;
        fb = d;
    }

    c = a;
    fc = fa;

    bool mflag = true;

    for (int i = 0; i < ITMAX; i++) {
        if (fa != fc && fb != fc) {
            s = a*fb*fc / ((fa-fb) * (fa-fc)) + b*fa*fc / ((fb-fa)*(fb-fc)) +
                c*fa*fb / ((fc-fa)*(fc-fb));
        } else {
            s = b - fb * (b-a) / (fb-fa);
        }

        // list of conditions
        bool con1 = false;
        if (0.25*(3.0 * a + b) < b) {
            if (s < 0.25*(3.0 * a + b) || s > b) {
                con1 = true;
            }
        } else if (s < b || s > 0.25*(3.0 * a + b)) {
            con1 = true;
        }
        bool con2 = false;
        if (mflag && abs(s-b) >= 0.5*abs(b-c)) {
            con2 = true;
        }
        bool con3 = false;
        if (!(mflag) && abs(s-b) >= 0.5 * abs(c-d)) {
            con3 = true;
        }
        bool con4 = false;
        if (mflag && abs(b-c) < tol) {
            con4 = true;
        }
        bool con5 = false;
        if (!(mflag) && abs(c-d) < tol) {
            con5 = true;
        }

        if (con1 || con2 || con3 || con4 || con5) {
            s = 0.5 * (a+b);
            mflag = true;
        } else {
            mflag = false;
        }

        fs = func(s, D, Sx, Sy, Sz, tau, gamma, gamma_up);

        d = c;
        c = b;
        fc = fb;

        if (fa * fs < 0.0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if (abs(fa) < abs(fb)) {
            e = a;
            a = b;
            b = e;

            e = fa;
            fa = fb;
            fb = e;
        }

        // test for convegence
        if (fb == 0.0 || fs == 0.0 || abs(b-a) < tol) {
            return b;
        }
    }
    //cout << "Maximum number of iterations exceeded in zbrent.\n";
    //printf("Maximum number of iterations exceeded in zbrent.\n");
    return x1;
}

void check_mpi_error(int mpi_err) {
    /*
    Checks to see if the integer returned by an mpi function, mpi_err, is an MPI error. If so, it prints out some useful stuff to screen.
    */

    int errclass, resultlen;
    char err_buffer[MPI_MAX_ERROR_STRING];

    if (mpi_err != MPI_SUCCESS) {
        MPI_Error_class(mpi_err, &errclass);
        if (errclass == MPI_ERR_RANK) {
            fprintf(stderr,"%s","Invalid rank used in MPI send call\n");
            MPI_Error_string(mpi_err,err_buffer,&resultlen);
            fprintf(stderr,"%s",err_buffer);
            MPI_Finalize();
        } else {
            fprintf(stderr, "%s","Other MPI error\n");
            MPI_Error_string(mpi_err,err_buffer,&resultlen);
            fprintf(stderr,"%s",err_buffer);
            MPI_Finalize();
        }
    }
}

void getNumKernels(int nx, int ny, int nz, int ng, int n_processes, int *maxBlocks, int *maxThreads, dim3 * kernels, int * cumulative_kernels) {
    /*
    Return the number of kernels needed to run the problem given its size and the constraints of the GPU.

    Parameters
    ----------
    nx, ny : int
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
    *maxThreads = nz * int(sqrt(float(*maxThreads)/float(nz))) * int(sqrt(float(*maxThreads)/float(nz)));

    // calculate number of kernels needed

    if (nx*ny*nz > *maxBlocks * *maxThreads) {
        int kernels_x = int(ceil(float(nx) / (sqrt(float(*maxThreads * *maxBlocks)/nz))));
        int kernels_y = int(ceil(float(ny) / (sqrt(float(*maxThreads * *maxBlocks)/nz))));

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

void getNumBlocksAndThreads(int nx, int ny, int nz, int ng, int maxBlocks, int maxThreads, int n_processes, dim3 *kernels, dim3 *blocks, dim3 *threads)
{
    /*
    Returns the number of blocks and threads required for each kernel given the size of the problem and the constraints of the device.

    Parameters
    ----------
    nx, ny : int
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
        int nx_remaining = nx - (threads[0].x * blocks[0].x) * (kernels_x - 1);

        for (int j = 0; j < (kernels_y-1); j++) {

            threads[j*kernels_x + kernels_x-1].y =
                int(sqrt(float(maxThreads)/nz));
            threads[j*kernels_x + kernels_x-1].z = nz;

            threads[j*kernels_x + kernels_x-1].x =
                (nx_remaining < threads[j*kernels_x + kernels_x-1].y) ? nx_remaining : threads[j*kernels_x + kernels_x-1].y;

            blocks[j*kernels_x + kernels_x-1].x = int(ceil(float(nx_remaining) /
                float(threads[j*kernels_x + kernels_x-1].x)));
            blocks[j*kernels_x + kernels_x-1].y = int(sqrt(float(maxBlocks)));
            blocks[j*kernels_x + kernels_x-1].z = 1;
        }

        // kernels_y-1
        int ny_remaining = ny - (threads[0].y * blocks[0].y) * (kernels_y - 1);

        for (int i = 0; i < (kernels_x-1); i++) {

            threads[(kernels_y-1)*kernels_x + i].x =
                int(sqrt(float(maxThreads)/nz));
            threads[(kernels_y-1)*kernels_x + i].y =
                (ny_remaining < threads[(kernels_y-1)*kernels_x + i].x) ? ny_remaining : threads[(kernels_y-1)*kernels_x + i].x;
            threads[(kernels_y-1)*kernels_x + i].z = nz;

            blocks[(kernels_y-1)*kernels_x + i].x = int(sqrt(float(maxBlocks)));
            blocks[(kernels_y-1)*kernels_x + i].y = int(ceil(float(ny_remaining) /
                float(threads[(kernels_y-1)*kernels_x + i].y)));
            blocks[(kernels_y-1)*kernels_x + i].z = 1;
        }

        // recalculate
        nx_remaining = nx - (threads[0].x * blocks[0].x) * (kernels_x - 1);
        ny_remaining = ny - (threads[0].y * blocks[0].y) * (kernels_y - 1);

        // (kernels_x-1, kernels_y-1)
        threads[(kernels_y-1)*kernels_x + kernels_x-1].x =
            (nx_remaining < int(sqrt(float(maxThreads)/nz))) ? nx_remaining : int(sqrt(float(maxThreads)/nz));
        threads[(kernels_y-1)*kernels_x + kernels_x-1].y =
            (ny_remaining < int(sqrt(float(maxThreads)/nz))) ? ny_remaining : int(sqrt(float(maxThreads)/nz));
        threads[(kernels_y-1)*kernels_x + kernels_x-1].z = nz;

        blocks[(kernels_y-1)*kernels_x + kernels_x-1].x =
            int(ceil(float(nx_remaining) /
            float(threads[(kernels_y-1)*kernels_x + kernels_x-1].x)));
        blocks[(kernels_y-1)*kernels_x + kernels_x-1].y =
            int(ceil(float(ny_remaining) /
            float(threads[(kernels_y-1)*kernels_x + kernels_x-1].y)));
        blocks[(kernels_y-1)*kernels_x + kernels_x-1].z = 1;

    } else {

        int total_threads = (total < maxThreads*2) ? nextPow2((total + 1)/ 2) : maxThreads;
        threads[0].x = int(floor(sqrt(float(total_threads)/nz)));
        threads[0].y = int(floor(sqrt(float(total_threads)/nz)));
        threads[0].z = nz;
        total_threads = threads[0].x * threads[0].y * threads[0].z;
        int total_blocks = int(ceil(float(total) / float(total_threads)));

        blocks[0].x = int(ceil(sqrt(float(total_blocks)/float(nx*ny))*nx));
        blocks[0].y = int(ceil(sqrt(float(total_blocks)/float(nx*ny))*ny));
        blocks[0].z = 1;

        total_blocks = blocks[0].x * blocks[0].y;

        if ((float)total_threads*total_blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
            printf("n is too large, please choose a smaller number!\n");
        }

        if (total_blocks > prop.maxGridSize[0]) {
            printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
                   total_blocks, prop.maxGridSize[0], total_threads*2, total_threads);

            blocks[0].x /= 2;
            blocks[0].y /= 2;
            threads[0].x *= 2;
            threads[0].y *= 2;
        }
    }
}

void bcs_fv(float * grid, int nx, int ny, int nz, int ng, int vec_dim) {
    /*
    Enforce boundary conditions on section of grid.

    Parameters
    ----------
    grid : float *
        grid of data
    nx, ny : int
        dimensions of grid
    ng : int
        number of ghost cells
    vec_dim : int
        dimension of state vector
    */
    // outflow
    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++){
            for (int g = 0; g < ng; g++) {
                for (int i = 0; i < vec_dim; i++) {
                    grid[((z * ny + y) * nx + g) * vec_dim+i] = grid[((z * ny + y) * nx + ng)*vec_dim+i];

                    grid[((z * ny + y) * nx + (nx-1-g))*vec_dim+i] = grid[((z * ny + y) * nx + (nx-1-ng))*vec_dim+i];
                }
            }
        }
        for (int g = 0; g < ng; g++) {
            for (int x = 0; x < nx; x++){
                for (int i = 0; i < vec_dim; i++) {
                    grid[((z * ny + g) * nx + x)*vec_dim+i] = grid[((z * ny + ng) * nx + x)*vec_dim+i];

                    grid[((z * ny + ny-1-g) * nx + x)*vec_dim+i] = grid[((z * ny + ny-1-ng) * nx + x)*vec_dim+i];
                }
            }
        }
    }
}

void bcs_mpi(float * grid, int nx, int ny, int nz, int vec_dim, int ng,
             MPI_Comm comm, MPI_Status status, int rank, int n_processes,
             int y_size) {
    /*
    Enforce boundary conditions across processes / at edges of grid.

    Loops have been ordered in a way so as to try and keep memory accesses as contiguous as possible.

    Need to do non-blocking send, blocking receive then wait.

    Parameters
    ----------
    grid : float *
        grid of data
    nx, ny : int
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
    */

    // x boundaries
    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int g = 0; g < ng; g++) {
                for (int i = 0; i < vec_dim; i++) {
                    grid[((z * ny + y) * nx + g) *vec_dim+i] = grid[((z * ny + y) * nx + ng) *vec_dim+i];

                    grid[((z * ny + y) * nx + (nx-1-g))*vec_dim+i] = grid[((z * ny + y) * nx + (nx-1-ng))*vec_dim+i];
                }
            }
        }
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
                        ysbuf[((z * ng + g) * nx + x)*vec_dim+i] = grid[((z * ny + y_size*rank+ng+g) * nx + x)*vec_dim+i];

                    }
                }
            }
        }
        mpi_err = MPI_Issend(ysbuf,nx*ng*nz*vec_dim, MPI_FLOAT, rank-1, tag, comm, &request);
        check_mpi_error(mpi_err);
        mpi_err = MPI_Recv(yrbuf, nx*ng*nz*vec_dim, MPI_FLOAT, rank+1, tag, comm, &status);
        check_mpi_error(mpi_err);
        MPI_Wait(&request, &status);

        // copy received data back to grid
        for (int z = 0; z < nz; z++) {
            for (int g = 0; g < ng; g++) {
                for (int x = 0; x < nx; x++) {
                    for (int i = 0; i < vec_dim; i++) {
                        grid[((z * ny + y_size*rank+ny-ng+g) * nx + x)*vec_dim+i] = yrbuf[((z * ng + g) * nx + x)*vec_dim+i];
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
                        ysbuf[((z * ng + g) * nx + x)*vec_dim+i] = grid[((z * ny + y_size*rank+ny-2*ng+g) * nx + x)*vec_dim+i];
                    }
                }
            }
        }
        MPI_Issend(ysbuf,nx*ng*nz*vec_dim, MPI_FLOAT, rank+1, tag, comm, &request);
        MPI_Recv(yrbuf, nx*ng*nz*vec_dim, MPI_FLOAT, rank-1, tag, comm, &status);
        MPI_Wait(&request, &status);

        // copy received data back to grid
        for (int z = 0; z < nz; z++) {
            for (int g = 0; g < ng; g++) {
                for (int x = 0; x < nx; x++) {
                    for (int i = 0; i < vec_dim; i++) {
                        grid[((z * ny + y_size*rank+g) * nx + x)*vec_dim+i] = yrbuf[((z * ng + g) * nx + x)*vec_dim+i];
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
                        ysbuf[((z * ng + g) * nx + x)*vec_dim+i] = grid[((z * ny + ny-2*ng+g) * nx + x)*vec_dim+i];
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
                        grid[((z * ny + ny-ng+g) * nx + x)*vec_dim+i] = yrbuf[((z * ng + g) * nx + x)*vec_dim+i];
                    }
                }
            }
        }

        // outflow stuff on top boundary
        for (int z = 0; z < nz; z++) {
            for (int g = 0; g < ng; g++) {
                for (int x = 0; x < nx; x++) {
                    for (int i = 0; i < vec_dim; i++) {
                        grid[((z * ny + g) * nx + x)*vec_dim+i] = grid[((z * ny + ng) * nx + x)*vec_dim+i];
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
                        ysbuf[((z * ng + g) * nx + x)*vec_dim+i] = grid[((z * ny + y_size*rank+ng+g) * nx + x)*vec_dim+i];
                    }
                }
            }
        }
        // bottom-most process
        MPI_Issend(ysbuf, nx*ng*nz*vec_dim, MPI_FLOAT, rank-1, tag, comm, &request);
        MPI_Recv(yrbuf, nx*ng*nz*vec_dim, MPI_FLOAT, rank-1, tag, comm, &status);
        MPI_Wait(&request, &status);

        // copy received data back to grid
        for (int z = 0; z < nz; z++) {
            for (int g = 0; g < ng; g++) {
                for (int x = 0; x < nx; x++) {
                    for (int i = 0; i < vec_dim; i++) {
                        grid[((z * ny + y_size*rank+g) * nx + x)*vec_dim+i] = yrbuf[((z * ng + g) * nx + x)*vec_dim+i];
                    }
                }
            }

            // outflow for bottom boundary
            for (int g = 0; g < ng; g++) {
                for (int x = 0; x < nx; x++) {
                    for (int i = 0; i < vec_dim; i++) {
                        grid[((z * ny + ny-1-g) * nx + x)*vec_dim+i] = grid[((z * ny + ny-1-ng) * nx + x)*vec_dim+i];
                    }
                }
            }
        }
    }

    delete[] ysbuf;
    delete[] yrbuf;
}

__host__ __device__ float W_swe(float * q, float * gamma_up) {
    // calculate Lorentz factor for conserved swe state vector
    return sqrt((q[1]*q[1] * gamma_up[0] +
            2.0 * q[1] * q[2] * gamma_up[1] +
            q[2] * q[2] * gamma_up[4]) / (q[0]*q[0]) + 1.0);
}

__host__ __device__ float phi(float r) {
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

__device__ float find_height(float ph) {
    /*
    Finds r given Phi.
    */
    const float M = 1.0; // set this for convenience
    return 2.0 * M / (1.0 - exp(-2.0 * ph));
}

__device__ float find_pot(float r) {
    /*
    Finds Phi given r.
    */
    const float M = 1.0; // set this for convenience
    return -0.5 * log(1.0 - 2.0 * M / r);
}

__device__ float rhoh_from_p(float p, float rho, float gamma) {
    // calculate rhoh using p for gamma law equation of state
    return rho + gamma * p / (gamma - 1.0);
}

__device__ float p_from_rhoh(float rhoh, float rho, float gamma) {
    // calculate p using rhoh for gamma law equation of state
    return (rhoh - rho) * (gamma - 1.0) / gamma;
}

__device__ float p_from_rho_eps_d(float rho, float eps, float gamma) {
    // calculate p using rho and epsilon for gamma law equation of state
    return (gamma - 1.0) * rho * eps;
}

__device__ __host__ float p_from_rho_eps(float rho, float eps, float gamma) {
    // calculate p using rho and epsilon for gamma law equation of state
    return (gamma - 1.0) * rho * eps;
}

__device__ __host__ float phi_from_p(float p, float rho, float gamma, float A) {
    // calculate the metric potential Phi given p for gamma law equation of
    // state
    return (gamma - 1.0) / gamma *
        log((rho + gamma * p / (gamma - 1.0)) / A);
}

__device__ __host__ float f_of_p(float p, float D, float Sx, float Sy,
                                 float Sz, float tau, float gamma,
                                 float * gamma_up) {
    // function of p whose root is to be found when doing conserved to
    // primitive variable conversion

    float sq = sqrt(pow(tau + p + D, 2) -
        Sx*Sx*gamma_up[0] - 2.0*Sx*Sy*gamma_up[1] - 2.0*Sz*Sz*gamma_up[2] -
        Sy*Sy*gamma_up[4] - 2.0*Sy*Sz*gamma_up[5] - Sz*Sz*gamma_up[8]);

    //if (nan_check(sq)) cout << "sq is nan :(\n";

    float rho = D * sq / (tau + p + D);
    float eps = (sq - p * (tau + p + D) / sq - D) / D;

    return (gamma - 1.0) * rho * eps - p;
}

__device__ float h_dot(float phi, float old_phi, float dt) {
    // Calculates the time derivative of the height given the shallow water
    // variable phi at current time and previous timestep
    // NOTE: this is an upwinded approximation of hdot - there may be a better
    // way to do this which will more accurately give hdot at current time.

    float h = find_height(phi);
    //float old_h = find_height(old_phi);

    return -2.0 * h * (phi - old_phi) / (dt * (exp(2.0 * phi) - 1.0));
}

__device__ void calc_As(float * rhos, float * phis, float * A,
                        float p_floor, int nlayers, float gamma) {
    // Calculates the As used to calculate the pressure given Phi, given
    // the pressure at the sea floor
    /*
    Parameters
    ----------
    rhos : float array
        densities of layers
    phis : float array
        Vector of Phi for different layers
    A : float array
        vector of As for layers
    p_floor : float
        pressure at sea floor
    nlayers : int
        number of layers
    gamma : float
        adiabatic index
    */

    // calculate A at sea floor
    A[nlayers-1] = (gamma * p_floor / (gamma - 1.0) +
        rhos[nlayers-1]) / exp(gamma * phis[nlayers-1] / (gamma - 1.0));
    for (int n = nlayers-2; n >= 0; n--) {
        // first calculate p at height of layer above previous layer
        float p = (gamma - 1.0) * (A[n+1] * exp(gamma * phis[n] /
            (gamma - 1.0)) - rhos[n+1]) / gamma;
        // now invert to calculate A
        A[n] = (gamma * p / (gamma - 1.0) +
            rhos[n]) / exp(gamma * phis[n] / (gamma - 1.0));
    }
}

__device__ void cons_to_prim_comp_d(float * q_cons, float * q_prim,
                       float gamma, float * gamma_up) {
    // convert compressible conserved variables to primitive variables

    const float TOL = 1.e-5;
    float D = q_cons[0];
    float Sx = q_cons[1];
    float Sy = q_cons[2];
    float Sz = q_cons[3];
    float tau = q_cons[4];

    // S^2
    float Ssq = Sx*Sx*gamma_up[0] + 2.0*Sx*Sy*gamma_up[1] +
        2.0*Sx*Sz*gamma_up[2] + Sy*Sy*gamma_up[4] + 2.0*Sy*Sz*gamma_up[5] +
        Sz*Sz*gamma_up[8];

    float pmin = (1.0 - Ssq) * (1.0 - Ssq) * tau * (gamma - 1.0);
    float pmax = (gamma - 1.0) * (tau + D) / (2.0 - gamma);

    if (pmin < 0.0) {
        pmin = 0.0;//1.0e-9;
    }
    if (pmax < 0.0 || pmax < pmin) {
        pmax = 1.0;
    }

    // check sign change
    if (f_of_p(pmin, D, Sx, Sy, Sz, tau, gamma, gamma_up) *
        f_of_p(pmax, D, Sx, Sy, Sz, tau, gamma, gamma_up) > 0.0) {
        pmin *= 0.1;
    }

    float p = zbrent((fptr)f_of_p, pmin, pmax, TOL, D, Sx, Sy, Sz,
                    tau, gamma, gamma_up);
    if (nan_check(p)){
        p = abs((gamma - 1.0) * (tau + D) / (2.0 - gamma)) > 1.0 ? 1.0 :
            abs((gamma - 1.0) * (tau + D) / (2.0 - gamma));
    }

    float sq = sqrt(pow(tau + p + D, 2) - Ssq);
    float eps = (sq - p * (tau + p + D)/sq - D) / D;
    float h = 1.0 + gamma * eps;
    float W = sqrt(1.0 + Ssq / (D*D*h*h));

    q_prim[0] = D * sq / (tau + p + D);//D / W;
    q_prim[1] = Sx / (W*W * h * q_prim[0]);
    q_prim[2] = Sy / (W*W * h * q_prim[0]);
    q_prim[3] = Sz / (W*W * h * q_prim[0]);
    q_prim[4] = eps;
}

void cons_to_prim_comp(float * q_cons, float * q_prim, int nxf, int nyf,
                       int nz,
                       float gamma, float * gamma_up) {
    /*
    Convert compressible conserved variables to primitive variables

    Parameters
    ----------
    q_cons : float *
        grid of conserved variables
    q_prim : float *
        grid where shall put the primitive variables
    nxf, nyf : int
        grid dimensions
    gamma : float
        adiabatic index
    gamma_up : float *
        spatial metric
    */

    const float TOL = 1.e-5;

    for (int i = 0; i < nxf*nyf*nz; i++) {
        float D = q_cons[i*5];
        float Sx = q_cons[i*5+1];
        float Sy = q_cons[i*5+2];
        float Sz = q_cons[i*5+3];
        float tau = q_cons[i*5+4];

        // S^2
        float Ssq = Sx*Sx*gamma_up[0] + 2.0*Sx*Sy*gamma_up[1] +
                2.0*Sx*Sz*gamma_up[2] + Sy*Sy*gamma_up[4] + 2.0*Sy*Sz*gamma_up[5] +
                Sz*Sz*gamma_up[8];

        float pmin = (1.0 - Ssq) * (1.0 - Ssq) * tau * (gamma - 1.0);
        float pmax = (gamma - 1.0) * (tau + D) / (2.0 - gamma);

        if (pmin < 0.0) {
            pmin = 0.0;//1.0e-9;
        }
        if (pmax < 0.0 || pmax < pmin) {
            pmax = 1.0;
        }

        // check sign change
        if (f_of_p(pmin, D, Sx, Sy, Sz, tau, gamma, gamma_up) *
            f_of_p(pmax, D, Sx, Sy, Sz, tau, gamma, gamma_up) > 0.0) {
            pmin = 0.0;
        }

        float p;
        try {
            p = zbrent((fptr)f_of_p, pmin, pmax, TOL, D, Sx, Sy, Sz,
                        tau, gamma, gamma_up);
        } catch (char const*){
            p = abs((gamma - 1.0) * (tau + D) / (2.0 - gamma)) > 1.0 ? 1.0 :
                abs((gamma - 1.0) * (tau + D) / (2.0 - gamma));
        }

        float sq = sqrt(pow(tau + p + D, 2) - Ssq);
        float eps = (sq - p * (tau + p + D)/sq - D) / D;
        float h = 1.0 + gamma * eps;
        float W = sqrt(1.0 + Ssq / (D*D*h*h));

        q_prim[i*5] = D * sq / (tau + p + D);//D / W;
        q_prim[i*5+1] = Sx / (W*W * h * q_prim[i*5]);
        q_prim[i*5+2] = Sy / (W*W * h * q_prim[i*5]);
        q_prim[i*5+3] = Sz / (W*W * h * q_prim[i*5]);
        q_prim[i*5+4] = eps;
    }
}

__device__ void shallow_water_fluxes(float * q, float * f, int dir,
                          float * gamma_up, float alpha, float * beta,
                          float gamma) {
    /*
    Calculate the flux vector of the shallow water equations

    Parameters
    ----------
    q : float *
        state vector
    f : float *
        grid where fluxes shall be stored
    dir : int
        0 if calculating flux in x-direction, 1 if in y-direction
    gamma_up : float *
        spatial metric
    alpha : float
        lapse function
    beta : float *
        shift vector
    gamma : float
        adiabatic index
    */
    if (nan_check(q[0])) q[0] = 1.0;
    if (nan_check(q[1])) q[1] = 0.0;
    if (nan_check(q[2])) q[2] = 0.0;

    float W = W_swe(q, gamma_up);
    if (nan_check(W)) {
        printf("W is nan! q0, q1, q2: %f, %f, %f\n", q[0], q[1], q[2]);
        W = 1.0;
    }

    float u = q[1] / (q[0] * W);
    float v = q[2] / (q[0] * W);

    if (dir == 0) {
        float qx = u * gamma_up[0] + v * gamma_up[1] -
            beta[0] / alpha;

        f[0] = q[0] * qx;
        f[1] = q[1] * qx + 0.5 * q[0] * q[0] / (W * W);
        f[2] = q[2] * qx;
    } else {
        float qy = v * gamma_up[4] + u * gamma_up[1] -
            beta[1] / alpha;

        f[0] = q[0] * qy;
        f[1] = q[1] * qy;
        f[2] = q[2] * qy + 0.5 * q[0] * q[0] / (W * W);
    }
}

__device__ void compressible_fluxes(float * q, float * f, int dir,
                         float * gamma_up, float alpha, float * beta,
                         float gamma) {
    /*
    Calculate the flux vector of the compressible GR hydrodynamics equations

    Parameters
    ----------
    q : float *
        state vector
    f : float *
        grid where fluxes shall be stored
    dir : int
        0 if calculating flux in x-direction, 1 if in y-direction,
        2 if in z-direction
    gamma_up : float *
        spatial metric
    alpha : float
        lapse function
    beta : float *
        shift vector
    gamma : float
        adiabatic index
    */

    // this is worked out on the fine grid
    float * q_prim;
    q_prim = (float *)malloc(5 * sizeof(float));

    cons_to_prim_comp_d(q, q_prim, gamma, gamma_up);

    float p = p_from_rho_eps_d(q_prim[0], q_prim[4], gamma);
    float u = q_prim[1];
    float v = q_prim[2];
    float w = q_prim[2];

    if (dir == 0) {
        float qx = u * gamma_up[0] + v * gamma_up[1] + w * gamma_up[2] - beta[0] / alpha;

        f[0] = q[0] * qx;
        f[1] = q[1] * qx + p;
        f[2] = q[2] * qx;
        f[3] = q[3] * qx;
        f[4] = q[4] * qx + p * u;
    } else if (dir == 1){
        float qy = v * gamma_up[4] + u * gamma_up[1] + w * gamma_up[5] - beta[1] / alpha;

        f[0] = q[0] * qy;
        f[1] = q[1] * qy;
        f[2] = q[2] * qy + p;
        f[3] = q[3] * qy;
        f[4] = q[4] * qy + p * v;
    } else {
        float qz = w * gamma_up[8] + u * gamma_up[2] + v * gamma_up[5] - beta[2] / alpha;

        f[0] = q[0] * qz;
        f[1] = q[1] * qz;
        f[2] = q[2] * qz;
        f[3] = q[3] * qz + p;
        f[4] = q[4] * qz + p * v;
    }

    free(q_prim);
}

void p_from_swe(float * q, float * p, int nx, int ny, int nz,
                 float * gamma_up, float rho, float gamma, float A) {
    /*
    Calculate p using SWE conserved variables

    Parameters
    ----------
    q : float *
        state vector
    p : float *
        grid where pressure shall be stored
    nx, ny : int
        grid dimensions
    gamma_up : float *
        spatial metric
    rho : float
        density
    gamma : float
        adiabatic index
    */

    for (int i = 0; i < nx*ny*nz; i++) {
        float W = W_swe(q, gamma_up);

        float ph = q[i*3] / W;

        p[i] = (gamma - 1.0) * (A * exp(gamma * ph /
            (gamma - 1.0)) - rho) / gamma;
    }
}

__device__ float p_from_swe(float * q, float * gamma_up, float rho,
                            float gamma, float W, float A) {
    /*
    Calculates p and returns using SWE conserved variables

    Parameters
    ----------
    q : float *
        state vector
    gamma_up : float *
        spatial metric
    rho : float
        density
    gamma : float
        adiabatic index
    W : float
        Lorentz factor
    */

    float ph = q[0] / W;

    return (gamma - 1.0) * (A * exp(gamma * ph /
        (gamma - 1.0)) - rho) / gamma;
}

__global__ void compressible_from_swe(float * q, float * q_comp,
                           int nx, int ny, int nz,
                           float * gamma_up, float * rho, float gamma,
                           int kx_offset, int ky_offset, float dt,
                           float * old_phi, float p_floor) {
    /*
    Calculates the compressible state vector from the SWE variables.

    Parameters
    ----------
    q : float *
        grid of SWE state vector
    q_comp : float *
        grid where compressible state vector to be stored
    nx, ny : int
        grid dimensions
    gamma_up : float *
        spatial metric
    rho, gamma : float
        density and adiabatic index
    kx_offset, ky_offset : int
        kernel offsets in the x and y directions
    */

    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;
    int offset = (z * ny + y) * nx + x;

    if ((x < nx) && (y < ny) && (z < nz)) {
        //printf("(%d, %d, %d): %f, %f, %f\n", x, y, z, q[offset*3], q[offset*3+1], q[offset*3+2]);

        float * q_swe;
        q_swe = (float *)malloc(3 * sizeof(float));

        for (int i = 0; i < 3; i++) {
            q_swe[i] = q[offset * 3 + i];
        }

        // calculate hdot = w (?)
        float hdot = h_dot(q[offset*3], old_phi[offset], dt);
        //printf("hdot(%d, %d, %d): %f, \n", x, y, z, hdot);

        float W = sqrt((q[offset*3+1] * q[offset*3+1] * gamma_up[0] +
                2.0 * q[offset*3+1] * q[offset*3+2] * gamma_up[1] +
                q[offset*3+2] * q[offset*3+2] * gamma_up[4]) /
                (q[offset*3] * q[offset*3]) +
                2.0 * hdot * (q[offset*3+1] * gamma_up[2] +
                q[offset*3+2] * gamma_up[5]) / q[offset*3] +
                hdot * hdot * gamma_up[8] + 1.0);
        //printf("%d\n",  gamma_up[8]);
        //printf("W(%d, %d, %d): %f, \n", x, y, z, W);
        // TODO: this is really inefficient as redoing the same calculation
        // on differnt layers
        float * A, * phis;
        A = (float *)malloc(nz * sizeof(float));
        phis = (float *)malloc(nz * sizeof(float));
        for (int i = 0; i < nz; i++) {
            phis[i] = q[((i * ny + y) * nx + x) * 3];
        }

        calc_As(rho, phis, A, p_floor, nz, gamma);

        float p = p_from_swe(q_swe, gamma_up, rho[z], gamma, W, A[z]);
        float rhoh = rhoh_from_p(p, rho[z], gamma);

        free(phis);
        free(A);

        q_comp[offset*5] = rho[z] * W;
        q_comp[offset*5+1] = rhoh * W * q[offset*3+1] / q[offset*3];
        q_comp[offset*5+2] = rhoh * W * q[offset*3+2] / q[offset*3];
        q_comp[offset*5+3] = rho[z] * W * hdot;
        q_comp[offset*5+4] = rhoh*W*W - p - rho[z] * W;

        //printf("s2c (%d, %d, %d): %f, %f\n", x, y, z, q_comp[offset*5+4], p);

        // NOTE: hack?
        if (q_comp[offset*5+4] < 0.0) {
            //printf("tau < 0, p: %f, tau: %f\n", p, q_comp[offset*5+4]);
            q_comp[offset*5+4] = 0.0;
        }

        free(q_swe);
    }
}

__global__ void prolong_reconstruct(float * q_comp, float * q_f, float * q_c,
                    int nx, int ny, int nlayers, int nxf, int nyf, int nz, float dx, float dy, float dz, float zmin,
                    int * matching_indices_d, float * gamma_up,
                    int kx_offset, int ky_offset) {
    /*
    Reconstruct fine grid variables from compressible variables on coarse grid

    Parameters
    ----------
    q_comp : float *
        compressible variables on coarse grid
    q_f : float *
        fine grid state vector
    q_c : float *
        coarse grid swe state vector
    nx, ny, nlayers : int
        coarse grid dimensions
    nxf, nyf, nz : int
        fine grid dimensions
    dx, dy, dz : float
        coarse grid spacings
    matching_indices_d : int *
        position of fine grid wrt coarse grid
    kx_offset, ky_offset : int
        kernel offsets in the x and y directions
    */

    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    if ((x>0) && (x < int(round(nxf*0.5)+1)) && (y > 0) && (y < int(round(nyf*0.5)+1)) && (z < nz)) {
        // corresponding x and y on the coarse grid
        int c_x = x + matching_indices_d[0];
        int c_y = y + matching_indices_d[2];

        // height of this layer
        float height = zmin + dz * (nz - z - 1.0);
        float * q_swe;
        q_swe = (float *)malloc(3 * sizeof(float));
        for (int i = 0; i < 3; i++) {
            q_swe[i] = q_c[(c_y*nx+c_x)*3+i];
        }
        float W = W_swe(q_swe, gamma_up);
        float r = find_height(q_c[(c_y * nx + c_x) * 3]/W);
        // Heights are sane here?
        //printf("z = %i, heights = %f, %f\n", z, height, r);
        float prev_r = r;

        int neighbour_layer = nlayers; // SWE layer just below compressible layer
        float layer_frac = 0.0; // fraction of distance between SWE layers that compressible is at

        if (height > r) { // compressible layer above top SWE layer
            neighbour_layer = 1;
            //if ((height - r) > dz) {
                //layer_frac = 0.0; // just copy across as another compressible layer between this and top SWE layer
            //} else {
            for (int i = 0; i < 3; i++) {
                q_swe[i] = q_c[((ny+c_y)*nx+c_x)*3+i];
            }
            W = W_swe(q_swe, gamma_up);
            r = find_height(q_c[((ny + c_y) * nx + c_x) * 3] / W);
            layer_frac = (height - prev_r) / (r - prev_r);
            //printf("Layer frac: %f  ", layer_frac);
            //}
        } else {

            // find heights of SWE layers - if height of SWE layer is above it, stop.
            for (int l = 1; l < nlayers-1; l++) {
                prev_r = r;
                for (int i = 0; i < 3; i++) {
                    q_swe[i] = q_c[((l*ny+c_y)*nx+c_x)*3+i];
                }
                W = W_swe(q_swe, gamma_up);
                r = find_height(q_c[((l * ny + c_y) * nx + c_x) * 3] / W);
                if (height > r) {
                    neighbour_layer = l;
                    layer_frac = (height - prev_r)/ (r - prev_r);
                    break;
                }
            }

            if (neighbour_layer == nlayers) {
                // lowest compressible beneath lowest SWE layer
                neighbour_layer = nlayers - 1;
                if (z == (nz-1)) {
                    layer_frac = 1.0;
                } else {
                    prev_r = r;
                    int l = neighbour_layer;
                    for (int i = 0; i < 3; i++) {
                        q_swe[i] = q_c[((l*ny+c_y)*nx+c_x)*3+i];
                    }
                    W = W_swe(q_swe, gamma_up);
                    r = find_height(q_c[((l * ny + c_y) * nx + c_x) * 3] / W);
                    layer_frac = (height - prev_r) / (r - prev_r);
                    //printf("Lower layer frac: %f  ", layer_frac);
                }
            }
        }

        free(q_swe);
        //printf("Layer frac: %f  ", layer_frac);

        //printf("z: %i, height: %f, neighbour_layer: %i, layer_frac: %f, \n", z, height, neighbour_layer, layer_frac);

        for (int n = 0; n < 5; n++) {
            // do some slope limiting
            // x-dir
            float S_upwind = (layer_frac *
                (q_comp[((neighbour_layer * ny + c_y) * nx + c_x+1) * 5 + n] -
                q_comp[((neighbour_layer * ny + c_y) * nx + c_x) * 5 + n]) +
                (1.0 - layer_frac) *
                (q_comp[(((neighbour_layer-1)*ny + c_y) * nx + c_x+1)*5 + n] -
                q_comp[(((neighbour_layer-1)*ny+c_y)*nx+c_x)*5 + n]))/ dx;
            float S_downwind = (layer_frac *
                (q_comp[((neighbour_layer * ny + c_y) * nx + c_x) * 5 + n] -
                q_comp[((neighbour_layer * ny + c_y) * nx + c_x-1) * 5 + n])
                + (1.0 - layer_frac) *
                (q_comp[(((neighbour_layer-1)*ny + c_y) * nx + c_x)*5 + n] -
                q_comp[(((neighbour_layer-1)*ny + c_y)*nx+c_x-1)*5+n]))/ dx;

            float Sx = 0.5 * (S_upwind + S_downwind);

            float r = 1.0e6;
            if (abs(S_downwind) > 1.0e-10) {
                r = S_upwind / S_downwind;
            }

            Sx *= phi(r);

            // y-dir
            S_upwind = (layer_frac *
                (q_comp[((neighbour_layer * ny + c_y+1) * nx + c_x) * 5 + n] -
                q_comp[((neighbour_layer * ny + c_y) * nx + c_x) * 5 + n]) +
                (1.0 - layer_frac) *
                (q_comp[(((neighbour_layer-1)*ny + c_y+1) * nx + c_x)*5 + n] -
                q_comp[(((neighbour_layer-1)*ny+c_y)*nx+c_x)*5 + n])) / dy;
            S_downwind = (layer_frac *
                (q_comp[((neighbour_layer * ny + c_y) * nx + c_x) * 5 + n] -
                q_comp[((neighbour_layer * ny + c_y-1) * nx + c_x) * 5 + n])
                + (1.0 - layer_frac) *
                (q_comp[(((neighbour_layer-1)*ny + c_y) * nx + c_x)*5 + n] -
                q_comp[(((neighbour_layer-1)*ny + c_y-1)*nx+c_x)*5+n])) / dy;

            float Sy = 0.5 * (S_upwind + S_downwind);

            r = 1.0e6;
            if (abs(S_downwind) > 1.0e-10) {
                r = S_upwind / S_downwind;
            }

            Sy *= phi(r);

            // vertically interpolated component of q_comp
            float interp_q_comp = layer_frac *
                q_comp[((neighbour_layer * ny + c_y) * nx + c_x) * 5 + n] +
                (1.0 - layer_frac) *
                q_comp[(((neighbour_layer-1) * ny + c_y) * nx + c_x) * 5 + n];

            q_f[((z * nyf + 2*y) * nxf + 2*x) * 5 + n] =
                interp_q_comp - 0.25 * (dx * Sx + dy * Sy);

            q_f[((z * nyf + 2*y) * nxf + 2*x+1) * 5 + n] =
                interp_q_comp + 0.25 * (dx * Sx - dy * Sy);

            q_f[((z * nyf + 2*y+1) * nxf + 2*x) * 5 + n] =
                interp_q_comp + 0.25 * (-dx * Sx + dy * Sy);

            q_f[((z * nyf + 2*y+1) * nxf + 2*x+1) * 5 + n] =
                interp_q_comp + 0.25 * (dx * Sx + dy * Sy);

        }
        //printf("(%d, %d, %d): %f, \n", 2*x, 2*y, z, q_f[((z * nyf + 2*y+1) * nxf + 2*x) * 5+4]);
        //printf("(%d, %d, %d): %f, \n", 2*x, 2*y+1, z, q_f[((z * nyf + 2*y+1) * nxf + 2*x) * 5]);
        //printf("(%d, %d, %d): %f, \n", 2*x, 2*y+1, z, q_f[((z * nyf + 2*y+1) * nxf + 2*x) * 5]);
        //printf("(%d, %d, %d): %f, \n", 2*x+1, 2*y+1, z, q_f[((z * nyf + 2*y+1) * nxf + 2*x+1) * 5]);
    }

    // going to get rid of tau < 0 (slightly hacky?)
    if ((x < nxf) && (y < nyf) && (z < nz)) {
        if (q_f[((z * nyf + y) * nxf + x) * 5 + 4] < 0.0) {
            //printf("tau < 0 (%d, %d, %d): %f, \n", x, y, z, q_f[((z * nyf + y) * nxf + x) * 5 + 4]);
            q_f[((z * nyf + y) * nxf + x) * 5 + 4] = 0.0;
        }

        //printf("(%d, %d, %d): %f, \n", x, y, z, q_f[((z * nyf + y) * nxf + x) * 5 + 4]);
    }
}

void prolong_grid(dim3 * kernels, dim3 * threads, dim3 * blocks,
                  int * cumulative_kernels, float * q_cd, float * q_fd,
                  int nx, int ny, int nlayers, int nxf, int nyf, int nz,
                  float dx, float dy, float dz, float dt, float zmin,
                  float * gamma_up_d, float * rho, float gamma,
                  int * matching_indices_d, int ng, int rank, float * qc_comp,
                  float * old_phi_d, float p_floor) {
    /*
    Prolong coarse grid data to fine grid

    Parameters
    ----------
    kernels, threads, blocks : dim3 *
        number of kernels, threads and blocks for each process/kernel
    cumulative_kernels : int *
        cumulative number of kernels in mpi processes of r < rank
    q_cd, q_fd : float *
        coarse and fine grids of state vectors
    nx, ny : int
        dimensions of coarse grid
    nxf, nyf : int
        dimensions of fine grid
    dx, dy : float
        coarse grid cell spacings
    gamma_up_d : float *
        spatial metric
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
            compressible_from_swe<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(q_cd, qc_comp, nx, ny, nlayers, gamma_up_d, rho, gamma, kx_offset, ky_offset, dt, old_phi_d, p_floor);
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
           prolong_reconstruct<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(qc_comp, q_fd, q_cd, nx, ny, nlayers, nxf, nyf, nz, dx, dy, dz, zmin, matching_indices_d, gamma_up_d, kx_offset, ky_offset);

           kx_offset += blocks[k_offset + j * kernels[rank].x + i].x *
                threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
       }
       ky_offset += blocks[k_offset + j * kernels[rank].x].y *
            threads[k_offset + j * kernels[rank].x].y - 2*ng;
    }
}

__global__ void swe_from_compressible(float * q, float * q_swe,
                                      int nxf, int nyf, int nz,
                                      float * gamma_up, float * rho,
                                      float gamma,
                                      int kx_offset, int ky_offset,
                                      float p_floor) {
    /*
    Calculates the SWE state vector from the compressible variables.

    Parameters
    ----------
    q : float *
        grid of compressible state vector
    q_swe : float *
        grid where SWE state vector to be stored
    nxf, nyf : int
        grid dimensions
    gamma_up : float *
        spatial metric
    rho, gamma : float
        density and adiabatic index
    kx_offset, ky_offset : int
        kernel offsets in the x and y directions
    */
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;
    int offset = (z * nyf + y) * nxf + x;

    /*if (x == 0 && y == 0 && z == 0) {
        for (int j = 0; j < 40; j++) {
            for (int i = 0; i < 40; i++) {
                printf("%d, ", q[(j*nxf+i)*5]);
            }
        }
        printf("\n\n");
    }*/

    if ((x < nxf) && (y < nyf) && (z < nz)) {
        float * q_prim, * q_con;
        q_con = (float *)malloc(5 * sizeof(float));
        q_prim = (float *)malloc(5 * sizeof(float));

        for (int i = 0; i < 5; i++) {
            q_con[i] = q[offset * 5 + i];
        }

        // find primitive variables
        cons_to_prim_comp_d(q_con, q_prim, gamma, gamma_up);

        float u = q_prim[1];
        float v = q_prim[2];
        float w = q_prim[3];

        float W = 1.0 / sqrt(1.0 -
                u*u*gamma_up[0] - 2.0 * u*v * gamma_up[1] -
                2.0 * u*w * gamma_up[2] - v*v*gamma_up[4] -
                2.0 * v*w*gamma_up[5] - w*w*gamma_up[8]);

        //rho = q_prim[0];

        // calculate SWE conserved variables on fine grid.
        float p = p_from_rho_eps(q_prim[0], q_prim[4], gamma);
        // TODO: calculate A
        float * A, * phis;
        A = (float *)malloc(nz * sizeof(float));
        phis = (float *)malloc(nz * sizeof(float));
        for (int i = 0; i < nz; i++) {
            phis[i] = q[((i * nyf + y) * nxf + x) * 3];
        }
        calc_As(rho, phis, A, p_floor, nz, gamma);

        float ph = phi_from_p(p, q_prim[0], gamma, A[z]);

        free(phis);
        free(A);

        //printf("W: %f, ph: %f, tau: %f, eps: %f\n", W, ph, q_con[4], q_prim[4]);

        q_swe[offset*3] = ph * W;
        q_swe[offset*3+1] = ph * W * W * u;
        q_swe[offset*3+2] = ph * W * W * v;

        free(q_con);
        free(q_prim);
    }
}

__device__ float height_err(float * q_c_new, float * qf_sw, float zmin,
                            int nxf, int nyf, int nz, float dz,
                            float * gamma_up, int x, int y,
                            float height_guess) {
    int z_index = nz;
    float z_frac = 0.0;

    if (height_guess > (zmin + (nz - 1.0) * dz)) { // SWE layer above top compressible layer
        //printf("hi :/\n");
        z_index = 1;
        float height = zmin + (nz - 1 - 1) * dz;
        z_frac = -(height_guess - (height+dz)) / dz;
    } else {

        for (int i = 1; i < (nz-1); i++) {
            float height = zmin + (nz - 1 - i) * dz;
            if (height_guess > height) {
                z_index = i;
                z_frac = -(height_guess - (height+dz)) / dz;
                break;
            }
        }

        if (z_index == nz) {
            //printf("oops..\n");
            z_index = nz - 1;
            z_frac = 1.0;
        }
    }

    //printf("height: %f, z_height: %f, z_index: %i, z_frac: %f\n", height_guess, zmin + (nz - 1 - z_index) * dz, z_index, z_frac);

    // interpolate between compressible cells nearest to SWE layer.
    for (int n = 0; n < 3; n++) {
        q_c_new[n] =
            0.25 * (z_frac *
            (qf_sw[((z_index * nyf + y*2) * nxf + x*2) * 3 + n] +
            qf_sw[((z_index * nyf + y*2) * nxf + x*2+1) * 3 + n] +
            qf_sw[((z_index * nyf + y*2+1) * nxf + x*2) * 3 + n] +
            qf_sw[((z_index * nyf + y*2+1) * nxf + x*2+1) * 3 + n]) +
            (1.0 - z_frac) *
            (qf_sw[(((z_index-1) * nyf + y*2) * nxf + x*2) * 3 + n] +
            qf_sw[(((z_index-1) * nyf + y*2) * nxf + x*2+1) * 3 + n] +
            qf_sw[(((z_index-1) * nyf + y*2+1) * nxf + x*2) * 3 + n] +
            qf_sw[(((z_index-1) * nyf + y*2+1) * nxf + x*2+1) * 3 + n]));
    }
    float W = W_swe(q_c_new, gamma_up);

    float actual_r = find_height(q_c_new[0] / W);
    return abs(height_guess - actual_r) / height_guess;
}

__global__ void restrict_interpolate(float * qf_sw, float * q_c,
                                     int nx, int ny, int nlayers,
                                     int nxf, int nyf, int nz,
                                     float dz, float zmin,
                                     int * matching_indices,
                                     float * gamma_up,
                                     int kx_offset, int ky_offset) {

    /*
    Interpolate SWE variables on fine grid to get them on coarse grid.

    Parameters
    ----------
    qf_swe : float *
        SWE variables on fine grid
    q_c : float *
        coarse grid state vector
    nx, ny : int
        coarse grid dimensions
    nxf, nyf : int
        fine grid dimensions
    matching_indices : int *
        position of fine grid wrt coarse grid
    kx_offset, ky_offset : int
        kernel offsets in the x and y directions
    */
    // interpolate fine grid to coarse grid
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    /*if (x == 0 && y == 0 && z == 0) {
        for (int j = 0; j < nyf; j++) {
            for (int i = 0; i < nxf; i++) {
                printf("%f, ", qf_sw[(j*nxf+i)*3]);
            }
        }
        printf("\n");
    }*/

    if ((x > 0) && (x < int(round(nxf*0.5))) && (y > 0) && (y < int(round(nyf*0.5))) && (z < nlayers-1)) {
        // first find position of layers relative to fine grid
        int coarse_index = ((z * ny + y+matching_indices[2]) * nx +
              x+matching_indices[0]) * 3;
        const float rel_tol = 1.0e-5;

        float * q_c_new;
        q_c_new = (float *)malloc(3 * sizeof(float));
        for (int i = 0; i < 3; i++) {
            q_c_new[i] = q_c[coarse_index+i];
        }

        float W = W_swe(q_c_new, gamma_up);
        float r = find_height(q_c[coarse_index] / W);
        float height_min = 0.9 * r;
        float height_max = 1.1 * r;

        float rel_err_min = height_err(q_c_new, qf_sw, zmin, nxf, nyf, nz, dz, gamma_up, x, y, height_min);
        float rel_err_max = height_err(q_c_new, qf_sw, zmin, nxf, nyf, nz, dz, gamma_up, x, y, height_max);

        int counter = 0;
        // TODO: This uses bisection - change to brentq?
        while ((rel_err_min > rel_tol) && (counter < 100)) {
            //if (x == 1 && y == 1 && z == 0) printf("\n\nCounter = %d\n\n", counter);

            if (rel_err_min > rel_err_max) {
                height_min = height_min + 0.5 * (height_max - height_min);
                rel_err_min = height_err(q_c_new, qf_sw, zmin, nxf, nyf, nz, dz, gamma_up, x, y, height_min);
            } else {
                height_max = height_min + 0.5 * (height_max - height_min);
                rel_err_max = height_err(q_c_new, qf_sw, zmin, nxf, nyf, nz, dz, gamma_up, x, y, height_max);
            }

            //printf("r: %f, W: %f, Phi: %f \n", r, W, q_c[coarse_index]);
            /*int z_index = nz;
            float z_frac = 0.0;

            if (r > (zmin + (nz - 1.0) * dz)) { // SWE layer above top compressible layer
                //printf("hi :/\n");
                z_index = 1;
            } else {

                for (int i = 1; i < nz; i++) {
                    float height = zmin + (nz - 1 - i) * dz;
                    if (r > height) {
                        z_index = i;
                        z_frac = 1.0 - (r - height) / dz;
                        break;
                    }
                }

                if (z_index == nz) {
                    //printf("oops..\n");
                    z_index = nz - 1;
                    z_frac = 1.0;
                }
            }

            //printf("z: %i, height: %f, z_index: %i, z_frac: %f\n", z, r, z_index, z_frac);

            // interpolate between compressible cells nearest to SWE layer.
            for (int n = 0; n < 3; n++) {
                q_c[coarse_index + n] =
                    0.25 * (z_frac *
                    (qf_sw[((z_index * nyf + y*2) * nxf + x*2) * 3 + n] +
                    qf_sw[((z_index * nyf + y*2) * nxf + x*2+1) * 3 + n] +
                    qf_sw[((z_index * nyf + y*2+1) * nxf + x*2) * 3 + n] +
                    qf_sw[((z_index * nyf + y*2+1) * nxf + x*2+1) * 3 + n]) +
                    (1.0 - z_frac) *
                    (qf_sw[(((z_index-1) * nyf + y*2) * nxf + x*2) * 3 + n] +
                    qf_sw[(((z_index-1) * nyf + y*2) * nxf + x*2+1) * 3 + n] +
                    qf_sw[(((z_index-1) * nyf + y*2+1) * nxf + x*2) * 3 + n] +
                    qf_sw[(((z_index-1) * nyf + y*2+1) * nxf + x*2+1) * 3 + n]));
            }
            W = sqrt((q_c[coarse_index+1] * q_c[coarse_index+1] *
                    gamma_up[0] +
                    2.0 * q_c[coarse_index+1] * q_c[coarse_index+2] *
                    gamma_up[1] +
                    q_c[coarse_index+2] * q_c[coarse_index+2] * gamma_up[4]) /
                    (q_c[coarse_index] * q_c[coarse_index]) + 1.0);

            float actual_r = find_height(q_c[coarse_index] / W);
            rel_err = abs(r - actual_r) / r;*/
            //printf("r - actual r: %f\n", rel_err_min);
            counter++;
        }
        // make sure calculate q_c_new using height_min
        rel_err_min = height_err(q_c_new, qf_sw, zmin, nxf, nyf, nz, dz, gamma_up, x, y, height_min);

        //printf("counter, err: %d, %f\n", counter, rel_err_min);

        for (int i = 0; i < 3; i++) {
            q_c[coarse_index + i] = q_c_new[i];
        }

        free(q_c_new);
    } else if ((x > 0) && (x < int(round(nxf*0.5))) && (y > 0) && (y < int(round(nyf*0.5))) && (z == nlayers-1)) {
        int coarse_index = ((z * ny + y+matching_indices[2]) * nx +
              x+matching_indices[0]) * 3;
        int z_index = nz-1;
        for (int n = 0; n < 3; n++) {
            q_c[coarse_index + n] = 0.25 *
                (qf_sw[((z_index * nyf + y*2) * nxf + x*2) * 3 + n] +
                qf_sw[((z_index * nyf + y*2) * nxf + x*2+1) * 3 + n] +
                qf_sw[((z_index * nyf + y*2+1) * nxf + x*2) * 3 + n] +
                qf_sw[((z_index * nyf + y*2+1) * nxf + x*2+1) * 3 + n]);
        }
    }
}

void restrict_grid(dim3 * kernels, dim3 * threads, dim3 * blocks,
                    int * cumulative_kernels, float * q_cd, float * q_fd,
                    int nx, int ny, int nlayers, int nxf, int nyf, int nz,
                    float dz, float zmin, int * matching_indices,
                    float * rho, float gamma, float * gamma_up,
                    int ng, int rank, float * qf_swe, float p_floor) {
    /*
    Restrict fine grid data to coarse grid

    Parameters
    ----------
    kernels, threads, blocks : dim3 *
        number of kernels, threads and blocks for each process/kernel
    cumulative_kernels : int *
        cumulative number of kernels in mpi processes of r < rank
    q_cd, q_fd : float *
        coarse and fine grids of state vectors
    nx, ny, nlayers : int
        dimensions of coarse grid
    nxf, nyf, nz : int
        dimensions of fine grid
    matching_indices : int *
        position of fine grid wrt coarse grid
    rho, gamma : float
        density and adiabatic index
    gamma_up : float *
        spatial metric
    ng : int
        number of ghost cells
    rank : int
        rank of MPI process
    qf_swe : float *
        grid of SWE variables on fine grid
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
            swe_from_compressible<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(q_fd, qf_swe, nxf, nyf, nz, gamma_up, rho, gamma, kx_offset, ky_offset, p_floor);
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
           restrict_interpolate<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(qf_swe, q_cd, nx, ny, nlayers, nxf, nyf, nz, dz, zmin, matching_indices, gamma_up, kx_offset, ky_offset);

           kx_offset += blocks[k_offset + j * kernels[rank].x + i].x *
                threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
       }
       ky_offset += blocks[k_offset + j * kernels[rank].x].y *
            threads[k_offset + j * kernels[rank].x].y - 2*ng;
    }
}

__global__ void evolve_fv(float * beta_d, float * gamma_up_d,
                     float * Un_d, flux_func_ptr flux_func,
                     float * qx_plus_half, float * qx_minus_half,
                     float * qy_plus_half, float * qy_minus_half,
                     float * fx_plus_half, float * fx_minus_half,
                     float * fy_plus_half, float * fy_minus_half,
                     int nx, int ny, int nz, int vec_dim, float alpha, float gamma,
                     float dx, float dy, float dt,
                     int kx_offset, int ky_offset) {
    /*
    First part of evolution through one timestep using finite volume methods.
    Reconstructs state vector to cell boundaries using slope limiter
    and calculates fluxes there.

    NOTE: we assume that beta is smooth so can get value at cell boundaries with simple averaging

    Parameters
    ----------
    beta_d : float *
        shift vector at each grid point.
    gamma_up_d : float *
        gamma matrix at each grid point
    Un_d : float *
        state vector at each grid point in each layer
    qx_plus_half, qx_minus_half : float *
        state vector reconstructed at right and left boundaries
    qy_plus_half, qy_minus_half : float *
        state vector reconstructed at top and bottom boundaries
    fx_plus_half, fx_minus_half : float *
        flux vector at right and left boundaries
    fy_plus_half, fy_minus_half : float *
        flux vector at top and bottom boundaries
    nx, ny, nz : int
        dimensions of grid
    alpha : float
        lapse function
    kx_offset, ky_offset : int
        x, y offset for current kernel
    */

    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    int offset = ((z * ny + y) * nx + x) * vec_dim;

    float * q_p, *q_m, * f;
    q_p = (float *)malloc(vec_dim * sizeof(float));
    q_m = (float *)malloc(vec_dim * sizeof(float));
    f = (float *)malloc(vec_dim * sizeof(float));

    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1)) && (z < nz)) {

        // x-direction
        for (int i = 0; i < vec_dim; i++) {
            float S_upwind = (Un_d[((z * ny + y) * nx + x+1) * vec_dim + i] -
                Un_d[((z * ny + y) * nx + x) * vec_dim + i]) / dx;
            float S_downwind = (Un_d[((z * ny + y) * nx + x) * vec_dim + i] -
                Un_d[((z * ny + y) * nx + x-1) * vec_dim + i]) / dx;
            float S = 0.5 * (S_upwind + S_downwind); // S_av

            float r = 1.0e6;

            // make sure don't divide by zero
            if (abs(S_downwind) > 1.0e-7) {
                r = S_upwind / S_downwind;
            }

            S *= phi(r);

            q_p[i] = Un_d[offset + i] + S * 0.5 * dx;
            q_m[i] = Un_d[offset + i] - S * 0.5 * dx;
        }

        // fluxes

        flux_func(q_p, f, 0, gamma_up_d, alpha, beta_d, gamma);

        for (int i = 0; i < vec_dim; i++) {
            qx_plus_half[offset + i] = q_p[i];
            fx_plus_half[offset + i] = f[i];
        }

        flux_func(q_m, f, 0, gamma_up_d, alpha, beta_d, gamma);

        for (int i = 0; i < vec_dim; i++) {
            qx_minus_half[offset + i] = q_m[i];
            fx_minus_half[offset + i] = f[i];
        }

        // y-direction
        for (int i = 0; i < vec_dim; i++) {
            float S_upwind = (Un_d[((z * ny + y+1) * nx + x) * vec_dim + i] -
                Un_d[((z * ny + y) * nx + x) * vec_dim + i]) / dy;
            float S_downwind = (Un_d[((z * ny + y) * nx + x) * vec_dim + i] -
                Un_d[((z * ny + y-1) * nx + x) * vec_dim + i]);
            float S = 0.5 * (S_upwind + S_downwind) / dy; // S_av

            float r = 1.0e6;

            // make sure don't divide by zero
            if (abs(S_downwind) > 1.0e-7) {
                r = S_upwind / S_downwind;
            }

            S *= phi(r);

            q_p[i] = Un_d[offset + i] + S * 0.5 * dy;
            q_m[i] = Un_d[offset + i] - S * 0.5 * dy;
        }

        // fluxes

        flux_func(q_p, f, 1, gamma_up_d, alpha, beta_d, gamma);

        for (int i = 0; i < vec_dim; i++) {
            qy_plus_half[offset + i] = q_p[i];
            fy_plus_half[offset + i] = f[i];
        }

        flux_func(q_m, f, 1, gamma_up_d, alpha, beta_d, gamma);

        for (int i = 0; i < vec_dim; i++) {
            qy_minus_half[offset + i] = q_m[i];
            fy_minus_half[offset + i] = f[i];
        }
    }

    free(q_p);
    free(q_m);
    free(f);
}

__global__ void evolve_z(float * beta_d, float * gamma_up_d,
                     float * Un_d, flux_func_ptr flux_func,
                     float * qz_plus_half, float * qz_minus_half,
                     float * fz_plus_half, float * fz_minus_half,
                     int nx, int ny, int nz, int vec_dim, float alpha, float gamma,
                     float dz, float dt,
                     int kx_offset, int ky_offset) {
    /*
    First part of evolution through one timestep using finite volume methods.
    Reconstructs state vector to cell boundaries using slope limiter
    and calculates fluxes there.

    NOTE: we assume that beta is smooth so can get value at cell boundaries with simple averaging

    Parameters
    ----------
    beta_d : float *
        shift vector at each grid point.
    gamma_up_d : float *
        gamma matrix at each grid point
    Un_d : float *
        state vector at each grid point in each layer
    qz_plus_half, qz_minus_half : float *
        state vector reconstructed at top and bottom boundaries
    fz_plus_half, fz_minus_half : float *
        flux vector at top and bottom boundaries
    nx, ny, nz : int
        dimensions of grid
    alpha : float
        lapse function
    kx_offset, ky_offset : int
        x, y offset for current kernel
    */

    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    int offset = ((z * ny + y) * nx + x) * vec_dim;

    float * q_p, *q_m, * f;
    q_p = (float *)malloc(vec_dim * sizeof(float));
    q_m = (float *)malloc(vec_dim * sizeof(float));
    f = (float *)malloc(vec_dim * sizeof(float));

    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1)) && (z > 0) && (z < (nz-1))) {

        // z-direction
        for (int i = 0; i < vec_dim; i++) {
            float S_upwind = (Un_d[(((z+1) * ny + y) * nx + x) * vec_dim + i] -
                Un_d[((z * ny + y) * nx + x) * vec_dim + i]) / dz;
            float S_downwind = (Un_d[((z * ny + y) * nx + x) * vec_dim + i] -
                Un_d[(((z-1) * ny + y) * nx + x) * vec_dim + i]) / dz;
            float S = 0.5 * (S_upwind + S_downwind); // S_av

            float r = 1.0e6;

            // make sure don't divide by zero
            if (abs(S_downwind) > 1.0e-7) {
                r = S_upwind / S_downwind;
            }

            S *= phi(r);

            q_p[i] = Un_d[offset + i] + S * 0.5 * dz;
            q_m[i] = Un_d[offset + i] - S * 0.5 * dz;
        }

        // fluxes

        flux_func(q_p, f, 2, gamma_up_d, alpha, beta_d, gamma);

        for (int i = 0; i < vec_dim; i++) {
            qz_plus_half[offset + i] = q_p[i];
            fz_plus_half[offset + i] = f[i];
        }

        flux_func(q_m, f, 2, gamma_up_d, alpha, beta_d, gamma);

        for (int i = 0; i < vec_dim; i++) {
            qz_minus_half[offset + i] = q_m[i];
            fz_minus_half[offset + i] = f[i];
        }
    }

    free(q_p);
    free(q_m);
    free(f);
}


__global__ void evolve_fv_fluxes(float * F,
                     float * qx_plus_half, float * qx_minus_half,
                     float * qy_plus_half, float * qy_minus_half,
                     float * fx_plus_half, float * fx_minus_half,
                     float * fy_plus_half, float * fy_minus_half,
                     int nx, int ny, int nz, int vec_dim, float alpha,
                     float dx, float dy, float dt,
                     int kx_offset, int ky_offset) {
    /*
    Calculates fluxes in finite volume evolution by solving the Riemann
    problem at the cell boundaries.

    Parameters
    ----------
    F : float *
        flux vector at each point in grid and each layer
    qx_plus_half, qx_minus_half : float *
        state vector reconstructed at right and left boundaries
    qy_plus_half, qy_minus_half : float *
        state vector reconstructed at top and bottom boundaries
    fx_plus_half, fx_minus_half : float *
        flux vector at right and left boundaries
    fy_plus_half, fy_minus_half : float *
        flux vector at top and bottom boundaries
    nx, ny, nlayers : int
        dimensions of grid
    alpha : float
        lapse function
    dx, dy, dt : float
        gridpoint spacing and timestep spacing
    kx_offset, ky_offset : int
        x, y offset for current kernel
    */
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    // do fluxes
    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1)) && (z < nz)) {
        for (int i = 0; i < vec_dim; i++) {
            // x-boundary
            // from i-1
            float fx_m = 0.5 * (
                fx_plus_half[((z * ny + y) * nx + x-1) * vec_dim + i] +
                fx_minus_half[((z * ny + y) * nx + x) * vec_dim + i] +
                qx_plus_half[((z * ny + y) * nx + x-1) * vec_dim + i] -
                qx_minus_half[((z * ny + y) * nx + x) * vec_dim + i]);
            // from i+1
            float fx_p = 0.5 * (
                fx_plus_half[((z * ny + y) * nx + x) * vec_dim + i] +
                fx_minus_half[((z * ny + y) * nx + x+1) * vec_dim + i] +
                qx_plus_half[((z * ny + y) * nx + x) * vec_dim + i] -
                qx_minus_half[((z * ny + y) * nx + x+1) * vec_dim + i]);

            // y-boundary
            // from j-1
            float fy_m = 0.5 * (
                fy_plus_half[((z * ny + y-1) * nx + x) * vec_dim + i] +
                fy_minus_half[((z * ny + y) * nx + x) * vec_dim + i] +
                qy_plus_half[((z * ny + y-1) * nx + x) * vec_dim + i] -
                qy_minus_half[((z * ny + y) * nx + x) * vec_dim + i]);
            // from j+1
            float fy_p = 0.5 * (
                fy_plus_half[((z * ny + y) * nx + x) * vec_dim + i] +
                fy_minus_half[((z * ny + y+1) * nx + x) * vec_dim + i] +
                qy_plus_half[((z * ny + y) * nx + x) * vec_dim + i] -
                qy_minus_half[((z * ny + y+1) * nx + x) * vec_dim + i]);

            F[((z * ny + y) * nx + x)*vec_dim + i] =
                -alpha * ((1.0/dx) * (fx_p - fx_m) +
                (1.0/dy) * (fy_p - fy_m));

            // hack?
            if (nan_check(F[((z * ny + y) * nx + x)*vec_dim + i])) F[((z * ny + y) * nx + x)*vec_dim + i] = 0.0;
        }
    }
}

__global__ void evolve_z_fluxes(float * F,
                     float * qz_plus_half, float * qz_minus_half,
                     float * fz_plus_half, float * fz_minus_half,
                     int nx, int ny, int nz, int vec_dim, float alpha,
                     float dz, float dt,
                     int kx_offset, int ky_offset) {
    /*
    Calculates fluxes in finite volume evolution by solving the Riemann
    problem at the cell boundaries in z direction.

    Parameters
    ----------
    F : float *
        flux vector at each point in grid and each layer
    qz_plus_half, qz_minus_half : float *
        state vector reconstructed at right and left boundaries
    fz_plus_half, fz_minus_half : float *
        flux vector at top and bottom boundaries
    nx, ny, nz : int
        dimensions of grid
    alpha : float
        lapse function
    dz, dt : float
        gridpoint spacing and timestep spacing
    kx_offset, ky_offset : int
        x, y offset for current kernel
    */
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    // do fluxes
    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1)) && (z > 0) && (z < (nz-1))) {
        for (int i = 0; i < vec_dim; i++) {
            // z-boundary
            // from i-1
            float fz_m = 0.5 * (
                fz_plus_half[(((z-1) * ny + y) * nx + x) * vec_dim + i] +
                fz_minus_half[((z * ny + y) * nx + x) * vec_dim + i] +
                qz_plus_half[(((z-1) * ny + y) * nx + x) * vec_dim + i] -
                qz_minus_half[((z * ny + y) * nx + x) * vec_dim + i]);
            // from i+1
            float fz_p = 0.5 * (
                fz_plus_half[((z * ny + y) * nx + x) * vec_dim + i] +
                fz_minus_half[(((z+1) * ny + y) * nx + x) * vec_dim + i] +
                qz_plus_half[((z * ny + y) * nx + x) * vec_dim + i] -
                qz_minus_half[(((z+1) * ny + y) * nx + x) * vec_dim + i]);

            F[((z * ny + y) * nx + x)*vec_dim + i] =
                F[((z * ny + y) * nx + x)*vec_dim + i]
                - alpha * (fz_p - fz_m) / dz;

            // hack?
            if (nan_check(F[((z * ny + y) * nx + x)*vec_dim + i])) F[((z * ny + y) * nx + x)*vec_dim + i] = 0.0;
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

    Parameters
    ----------
    gamma_up_d : float *
        gamma matrix at each grid point
    Un_d : float *
        state vector at each grid point in each layer at current timestep
    Up : float *
        state vector at next timestep
    U_half : float *
        state vector at half timestep
    qx_plus_half, qx_minus_half : float *
        state vector reconstructed at right and left boundaries
    qy_plus_half, qy_minus_half : float *
        state vector reconstructed at top and bottom boundaries
    fx_plus_half, fx_minus_half : float *
        flux vector at right and left boundaries
    fy_plus_half, fy_minus_half : float *
        flux vector at top and bottom boundaries
    sum_phs : float *
        sum of Phi in different layers
    rho_d : float *
        list of densities in different layers
    Q_d : float *
        heating rate at each grid point in each layer
    mu : float
        friction
    nx, ny, nlayers : int
        dimensions of grid
    alpha : float
        lapse function
    dx, dy, dt : float
        gridpoint spacing and timestep spacing
    burning : bool
        is burning present in this system?
    kx_offset, ky_offset : int
        x, y offset for current kernel
    */
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int l = threadIdx.z;
    int offset = (y * nx + x) * nlayers + l;

    // copy to U_half
    if ((x < nx) && (y < ny) && (l < nlayers)) {
        for (int i = 0; i < 3; i++) {
            U_half[offset*3+i] = Up[offset*3+i];
        }
    }

    // calculate Q
    //calc_Q(Up, rho_d, Q_d, nx, ny, nlayers, kx_offset, ky_offset, burning);

    float W = 1.0;

    // do source terms
    if ((x < nx) && (y < ny) && (l < nlayers)) {
        float * q_swe;
        q_swe = (float *)malloc(3 * sizeof(float));

        for (int i = 0; i < 3; i++) {
            q_swe[i] = U_half[offset * 3 + i];
        }
        W = W_swe(q_swe, gamma_up_d);
        free(q_swe);

        U_half[offset*3] /= W;
    }

    __syncthreads();

    if ((x < nx) && (y < ny) && (l < nlayers)) {

        sum_phs[offset] = 0.0;

        float sum_qs = 0.0;
        float deltaQx = 0.0;
        float deltaQy = 0.0;

        if (l < (nlayers - 1)) {
            sum_qs += (Q_d[offset+1] - Q_d[offset]);
            deltaQx = (Q_d[offset] + mu) *
                (U_half[offset*3+1] - U_half[offset*3+1]) /
                (W * U_half[offset*3]);
            deltaQy = (Q_d[offset] + mu) *
                (U_half[offset*3+2] - U_half[offset*3+2]) /
                (W * U_half[offset*3]);
        }
        if (l > 0) {
            sum_qs += -rho_d[l-1] / rho_d[l] * (Q_d[offset] - Q_d[offset-1]);
            deltaQx = rho_d[l-1] / rho_d[l] * (Q_d[offset] + mu) *
                (U_half[offset*3+1] - U_half[offset*3+1]) /
                 (W * U_half[offset*3]);
            deltaQy = rho_d[l-1] / rho_d[l] * (Q_d[offset] + mu) *
                (U_half[offset*3+2] - U_half[offset*3+2]) /
                 (W * U_half[offset*3]);
        }

        for (int j = 0; j < l; j++) {
            sum_phs[offset] += rho_d[j] / rho_d[l] *
                U_half[((y * nx + x) * nlayers + j)*3];
        }
        for (int j = l+1; j < nlayers; j++) {
            sum_phs[offset] = sum_phs[offset] +
                U_half[((y * nx + x) * nlayers + j)*3];
        }

        // D
        Up[offset*3] += dt * alpha * sum_qs;

        // Sx
        Up[offset*3+1] += dt * alpha * (-deltaQx);

        // Sy
        Up[offset*3+2] += dt * alpha * (-deltaQy);

        // zeta
        //Up[((y * nx + x) * nlayers + l)*4+3] += -dt * alpha * Q_d[(y * nx + x) * nlayers + l] * rho_d[l];
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

    Parameters
    ----------
    gamma_up_d : float *
        gamma matrix at each grid point
    Un_d : float *
        state vector at each grid point in each layer at current timestep
    Up : float *
        state vector at next timestep
    U_half : float *
        state vector at half timestep
    sum_phs : float *
        sum of Phi in different layers
    rho_d : float *
        list of densities in different layers
    Q_d : float *
        heating rate at each grid point in each layer
    mu : float
        friction
    nx, ny, nlayers : int
        dimensions of grid
    ng : int
        number of ghost cells
    alpha : float
        lapse function
    dx, dy, dt : float
        gridpoint spacing and timestep spacing
    kx_offset, ky_offset : int
        x, y offset for current kernel
    */
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int l = threadIdx.z;
    int offset = (y * nx + x) * nlayers + l;

    //printf("kx_offset: %i\n", kx_offset);

    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1)) && (l < nlayers)) {

        float a = dt * alpha * U_half[offset*3] * (0.5 / dx) *
            (sum_phs[(y * nx + x+1) * nlayers + l] -
            sum_phs[(y * nx + x-1) * nlayers + l]);

        if (abs(a) < 0.9 * dx / dt) {
            Up[offset*3+1] = Up[offset*3+1] - a;
        }

        a = dt * alpha * U_half[offset*3] * (0.5 / dy) *
            (sum_phs[((y+1) * nx + x) * nlayers + l] -
             sum_phs[((y-1) * nx + x) * nlayers + l]);

        if (abs(a) < 0.9 * dy / dt) {
            Up[offset*3+2] = Up[offset*3+2] - a;
        }
    }

    __syncthreads();

    //bcs_fv(Up, nx, ny, nlayers, ng, 3);

    // copy back to grid
    if ((x < nx) && (y < ny) && (l < nlayers)) {
        for (int i = 0; i < 3; i++) {
            Un_d[offset*3+i] = Up[offset*3+i];
        }
    }
}

void homogeneuous_fv(dim3 * kernels, dim3 * threads, dim3 * blocks,
       int * cumulative_kernels, float * beta_d, float * gamma_up_d,
       float * Un_d, float * F_d,
       float * qx_p_d, float * qx_m_d, float * qy_p_d, float * qy_m_d,
       float * qz_p_d, float * qz_m_d,
       float * fx_p_d, float * fx_m_d, float * fy_p_d, float * fy_m_d,
       float * fz_p_d, float * fz_m_d,
       int nx, int ny, int nz, int vec_dim, int ng, float alpha, float gamma,
       float dx, float dy, float dz, float dt, int rank,
       flux_func_ptr h_flux_func, bool do_z) {
    /*
    Solves the homogeneous part of the equation (ie the bit without source terms).

    Parameters
    ----------
    kernels, threads, blocks : dim3 *
        number of kernels, threads and blocks for each process/kernel
    beta_d : float *
        shift vector at each grid point
    gamma_up_d : float *
        gamma matrix at each grid point
    Un_d : float *
        state vector at each grid point in each layer at current timestep
    F_d : float *
        flux vector
    qx_p_d, qx_m_d : float *
        state vector reconstructed at right and left boundaries
    qy_p_d, qy_m_d : float *
        state vector reconstructed at top and bottom boundaries
    fx_p_d, fx_m_d : float *
        flux vector at right and left boundaries
    fy_p_d, fy_m_d : float *
        flux vector at top and bottom boundaries
    nx, ny : int
        dimensions of grid
    alpha : float
        lapse function
    dx, dy, dt : float
        gridpoint spacing and timestep spacing
    rank : int
        rank of MPI process
    do_z : bool
        should we evolve in the z direction?
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
           evolve_fv<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(beta_d, gamma_up_d, Un_d, h_flux_func,
                  qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                  fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                  nx, ny, nz, vec_dim, alpha, gamma,
                  dx, dy, dt, kx_offset, ky_offset);
           if (do_z) {
               evolve_z<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(beta_d, gamma_up_d, Un_d, h_flux_func,
                      qz_p_d, qz_m_d,
                      fz_p_d, fz_m_d,
                      nx, ny, nz, vec_dim, alpha, gamma,
                      dz, dt, kx_offset, ky_offset);
           }
          kx_offset += blocks[k_offset + j * kernels[rank].x + i].x * threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
       }
       ky_offset += blocks[k_offset + j * kernels[rank].x].y * threads[k_offset + j * kernels[rank].x].y - 2*ng;
    }

    ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

    for (int j = 0; j < kernels[rank].y; j++) {
       kx_offset = 0;
       for (int i = 0; i < kernels[rank].x; i++) {
           evolve_fv_fluxes<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(
                  F_d,
                  qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                  fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                  nx, ny, nz, vec_dim, alpha,
                  dx, dy, dt, kx_offset, ky_offset);

            if (do_z) {
                evolve_z_fluxes<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(
                       F_d,
                       qz_p_d, qz_m_d,
                       fz_p_d, fz_m_d,
                       nx, ny, nz, vec_dim, alpha,
                       dz, dt, kx_offset, ky_offset);
            }

           kx_offset += blocks[k_offset + j * kernels[rank].x + i].x * threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
       }
       ky_offset += blocks[k_offset + j * kernels[rank].x].y * threads[k_offset + j * kernels[rank].x].y - 2*ng;
    }
}

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
       flux_func_ptr h_flux_func, bool do_z) {
    /*
    Integrates the homogeneous part of the ODE in time using RK3.

    Parameters
    ----------
    kernels, threads, blocks : dim3 *
        number of kernels, threads and blocks for each process/kernel
    beta_d : float *
        shift vector at each grid point
    gamma_up_d : float *
        gamma matrix at each grid point
    Un_d : float *
        state vector at each grid point in each layer at current timestep on device
    F_d : float *
        flux vector on device
    Up_d : float *
        state vector at next timestep on device
    qx_p_d, qx_m_d : float *
        state vector reconstructed at right and left boundaries
    qy_p_d, qy_m_d : float *
        state vector reconstructed at top and bottom boundaries
    fx_p_d, fx_m_d : float *
        flux vector at right and left boundaries
    fy_p_d, fy_m_d : float *
        flux vector at top and bottom boundaries
    nx, ny : int
        dimensions of grid
    ng : int
        number of ghost cells
    alpha : float
        lapse function
    dx, dy, dt : float
        gridpoint spacing and timestep spacing
    Up_h, F_h, Un_h : float *
        state vector at next timestep, flux vector and state vector at current timestep on host
    comm : MPI_Comm
        MPI communicator
    status: MPI_Status
        status of MPI processes
    rank, n_processes : int
        rank of current MPI process and total number of MPI processes
    do_z
    */
    // u1 = un + dt * F(un)
    homogeneuous_fv(kernels, threads, blocks, cumulative_kernels,
          beta_d, gamma_up_d, Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
          nx, ny, nz, vec_dim, ng, alpha, gamma,
          dx, dy, dz, dt, rank, h_flux_func, do_z);

    // copy back flux
    cudaMemcpy(F_h, F_d, nx*ny*nz*vec_dim*sizeof(float), cudaMemcpyDeviceToHost);
    if (n_processes == 1) {
        bcs_fv(F_h, nx, ny, nz, ng, vec_dim);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(F_h, nx, ny, nz, vec_dim, ng, comm, status, rank, n_processes, y_size);
    }

    for (int n = 0; n < nx*ny*nz*vec_dim; n++) {
        Up_h[n] = Un_h[n] + dt * F_h[n];
    }
    // enforce boundaries and copy back
    if (n_processes == 1) {
        bcs_fv(Up_h, nx, ny, nz, ng, vec_dim);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(Up_h, nx, ny, nz, vec_dim, ng, comm, status, rank, n_processes, y_size);
    }

    cudaMemcpy(Un_d, Up_h, nx*ny*nz*vec_dim*sizeof(float), cudaMemcpyHostToDevice);

    // u2 = 0.25 * (3*un + u1 + dt*F(u1))
    homogeneuous_fv(kernels, threads, blocks, cumulative_kernels,
          beta_d, gamma_up_d, Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
          nx, ny, nz, vec_dim, ng, alpha, gamma,
          dx, dy, dz, dt, rank, h_flux_func, do_z);

    // copy back flux
    cudaMemcpy(F_h, F_d, nx*ny*nz*vec_dim*sizeof(float), cudaMemcpyDeviceToHost);

    if (n_processes == 1) {
        bcs_fv(F_h, nx, ny, nz, ng, vec_dim);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(F_h, nx, ny, nz, vec_dim, ng, comm, status, rank, n_processes, y_size);
    }

    for (int n = 0; n < nx*ny*nz*vec_dim; n++) {
        Up_h[n] = 0.25 * (3.0 * Un_h[n] + Up_h[n] + dt * F_h[n]);
    }

    // enforce boundaries and copy back
    if (n_processes == 1) {
        bcs_fv(Up_h, nx, ny, nz, ng, vec_dim);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(Up_h, nx, ny, nz, vec_dim, ng, comm, status, rank, n_processes, y_size);
    }
    cudaMemcpy(Un_d, Up_h, nx*ny*nz*vec_dim*sizeof(float), cudaMemcpyHostToDevice);

    // un+1 = (1/3) * (un + 2*u2 + 2*dt*F(u2))
    homogeneuous_fv(kernels, threads, blocks, cumulative_kernels,
          beta_d, gamma_up_d, Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
          nx, ny, nz, vec_dim, ng, alpha, gamma,
          dx, dy, dz, dt, rank, h_flux_func, do_z);

    // copy back flux
    cudaMemcpy(F_h, F_d, nx*ny*nz*vec_dim*sizeof(float), cudaMemcpyDeviceToHost);

    if (n_processes == 1) {
        bcs_fv(F_h, nx, ny, nz, ng, vec_dim);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(F_h, nx, ny, nz, vec_dim, ng, comm, status, rank, n_processes, y_size);
    }

    for (int n = 0; n < nx*ny*nz*vec_dim; n++) {
        Up_h[n] = (1/3.0) * (Un_h[n] + 2.0*Up_h[n] + 2.0*dt * F_h[n]);
    }

    // enforce boundaries
    if (n_processes == 1) {
        bcs_fv(Up_h, nx, ny, nz, ng, vec_dim);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(Up_h, nx, ny, nz, vec_dim, ng, comm, status, rank, n_processes, y_size);
    }

    for (int j = 0; j < nx*ny*nz*vec_dim; j++) {
        Un_h[j] = Up_h[j];
    }

}

// device-side function pointers to __device__ functions
__device__ flux_func_ptr d_compressible_fluxes = compressible_fluxes;
__device__ flux_func_ptr d_shallow_water_fluxes = shallow_water_fluxes;

void cuda_run(float * beta, float * gamma_up, float * Uc_h, float * Uf_h,
         float * rho, float p_floor, float mu,
         int nx, int ny, int nlayers,
         int nxf, int nyf, int nz, int ng,
         int nt, float alpha, float gamma, float zmin,
         float dx, float dy, float dz, float dt, bool burning,
         int dprint, char * filename,
         MPI_Comm comm, MPI_Status status, int rank, int n_processes,
         int * matching_indices) {
    /*
    Evolve system through nt timesteps, saving data to filename every dprint timesteps.

    Parameters
    ----------
    beta : float *
        shift vector at each grid point
    gamma_up : float *
        gamma matrix at each grid point
    Un_h : float *
        state vector at each grid point in each layer at current timestep on host
    rho : float *
        densities in each layer
    Q : float *
        heating rate at each point and in each layer
    mu : float
        friction
    nx, ny, nlayers : int
        dimensions of coarse grid
    nxf, nyf, nz : int
        dimensions of fine grid
    ng : int
        number of ghost cells
    nt : int
        total number of timesteps
    alpha : float
        lapse function
    dx, dy, dz, dt : float
        gridpoint spacing and timestep spacing
    burning : bool
        is burning included in this system?
    dprint : int
        number of timesteps between each printout
    filename : char *
        name of file to which output is printed
    comm : MPI_Comm
        MPI communicator
    status: MPI_Status
        status of MPI processes
    rank, n_processes : int
        rank of current MPI process and total number of MPI processes
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
    if (n_processes > count) {
        n_processes = count;
    }

    if (rank == 0) {
        printf("Running on %i processor(s)\n", n_processes);
    }

    int maxThreads = 256;
    int maxBlocks = 256; //64;

    dim3 *kernels = new dim3[n_processes];
    int *cumulative_kernels = new int[n_processes];

    //getNumKernels(max(nx, nxf), max(ny, nyf), max(nlayers, nz+2*ng), ng, n_processes, &maxBlocks, &maxThreads, kernels, cumulative_kernels);
    getNumKernels(max(nx, nxf), max(ny, nyf), max(nlayers, nz), ng, n_processes, &maxBlocks, &maxThreads, kernels, cumulative_kernels);

    int total_kernels = cumulative_kernels[n_processes-1];

    dim3 *blocks = new dim3[total_kernels];
    dim3 *threads = new dim3[total_kernels];

    //getNumBlocksAndThreads(max(nx, nxf), max(ny, nyf), max(nlayers, nz+2*ng), ng, maxBlocks, maxThreads, n_processes, kernels, blocks, threads);
    getNumBlocksAndThreads(max(nx, nxf), max(ny, nyf), max(nlayers, nz), ng, maxBlocks, maxThreads, n_processes, kernels, blocks, threads);

    printf("rank: %i\n", rank);
    printf("kernels: (%i, %i)\n", kernels[rank].x, kernels[rank].y);
    printf("cumulative kernels: %i\n", cumulative_kernels[rank]);

    int k_offset = 0;
    if (rank > 0) {
      k_offset = cumulative_kernels[rank-1];
    }

    for (int i = k_offset; i < cumulative_kernels[rank]; i++) {
        printf("blocks: (%i, %i) , threads: (%i, %i)\n",
               blocks[i].x, blocks[i].y,
               threads[i].x, threads[i].y);
    }

    // gpu variables
    float * beta_d;
    float * gamma_up_d;
    float * Uc_d;
    float * Uf_d;
    float * rho_d;
    //float * Q_d;

    // initialise Uf_h
    for (int i = 0; i < nxf*nyf*nz*5; i++) {
        Uf_h[i] = 0.0;
    }

    // set device
    cudaSetDevice(rank);

    // allocate memory on device
    cudaMalloc((void**)&beta_d, 3*sizeof(float));
    cudaMalloc((void**)&gamma_up_d, 9*sizeof(float));
    cudaMalloc((void**)&Uc_d, nx*ny*nlayers*3*sizeof(float));
    cudaMalloc((void**)&Uf_d, nxf*nyf*nz*5*sizeof(float));
    cudaMalloc((void**)&rho_d, nlayers*sizeof(float));
    //cudaMalloc((void**)&Q_d, nlayers*nx*ny*sizeof(float));

    // copy stuff to GPU
    cudaMemcpy(beta_d, beta, 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma_up_d, gamma_up, 9*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Uc_d, Uc_h, nx*ny*nlayers*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Uf_d, Uf_h, nxf*nyf*nz*5*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rho_d, rho, nlayers*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(Q_d, Q, nlayers*nx*ny*sizeof(float), cudaMemcpyHostToDevice);

    float *Upc_d, *Uc_half_d, *Upf_d, *Uf_half_d, *old_phi_d;//*sum_phs_d;
    cudaMalloc((void**)&Upc_d, nx*ny*nlayers*3*sizeof(float));
    cudaMalloc((void**)&Uc_half_d, nx*ny*nlayers*3*sizeof(float));
    cudaMalloc((void**)&Upf_d, nxf*nyf*nz*5*sizeof(float));
    cudaMalloc((void**)&Uf_half_d, nxf*nyf*nz*5*sizeof(float));
    cudaMalloc((void**)&old_phi_d, nlayers*nx*ny*sizeof(float));
    //cudaMalloc((void**)&sum_phs_d, nlayers*nx*ny*sizeof(float));

    // need to fill old_phi with current phi to initialise
    float *pphi = new float[nlayers*nx*ny];
    for (int i = 0; i < nlayers*nx*ny; i++) {
        pphi[i] = Uc_h[i*3];
    }
    cudaMemcpy(old_phi_d, pphi, nx*ny*nlayers*sizeof(float), cudaMemcpyHostToDevice);


    float *qx_p_d, *qx_m_d, *qy_p_d, *qy_m_d, *qz_p_d, *qz_m_d, *fx_p_d, *fx_m_d, *fy_p_d, *fy_m_d, *fz_p_d, *fz_m_d;
    float *Upc_h = new float[nx*ny*nlayers*3];
    float *Fc_h = new float[nx*ny*nlayers*3];

    float *Upf_h = new float[nxf*nyf*nz*5];
    float *Ff_h = new float[nxf*nyf*nz*5];

    // initialise
    for (int j = 0; j < nxf*nyf*nz*5; j++) {
        Upf_h[j] = 0.0;
    }

    int grid_size = max(nx*ny*nlayers*3, nxf*nyf*nz*5);

    cudaMalloc((void**)&qx_p_d, grid_size*sizeof(float));
    cudaMalloc((void**)&qx_m_d, grid_size*sizeof(float));
    cudaMalloc((void**)&qy_p_d, grid_size*sizeof(float));
    cudaMalloc((void**)&qy_m_d, grid_size*sizeof(float));
    cudaMalloc((void**)&qz_p_d, grid_size*sizeof(float));
    cudaMalloc((void**)&qz_m_d, grid_size*sizeof(float));
    cudaMalloc((void**)&fx_p_d, grid_size*sizeof(float));
    cudaMalloc((void**)&fx_m_d, grid_size*sizeof(float));
    cudaMalloc((void**)&fy_p_d, grid_size*sizeof(float));
    cudaMalloc((void**)&fy_m_d, grid_size*sizeof(float));
    cudaMalloc((void**)&fz_p_d, grid_size*sizeof(float));
    cudaMalloc((void**)&fz_m_d, grid_size*sizeof(float));

    float * q_comp_d;
    cudaMalloc((void**)&q_comp_d, nx*ny*nlayers*5*sizeof(float));
    float * qf_swe;
    cudaMalloc((void**)&qf_swe, nxf*nyf*nz*3*sizeof(float));

    int * matching_indices_d;
    cudaMalloc((void**)&matching_indices_d, 4*sizeof(int));
    cudaMemcpy(matching_indices_d, matching_indices, 4*sizeof(int), cudaMemcpyHostToDevice);

    // make host-side function pointers to __device__ functions
    flux_func_ptr h_compressible_fluxes;
    flux_func_ptr h_shallow_water_fluxes;

    // copy function pointers to host equivalent
    cudaMemcpyFromSymbol(&h_compressible_fluxes, d_compressible_fluxes, sizeof(flux_func_ptr));
    cudaMemcpyFromSymbol(&h_shallow_water_fluxes, d_shallow_water_fluxes, sizeof(flux_func_ptr));

    if (strcmp(filename, "na") != 0) {
        hid_t outFile, dset, mem_space, file_space;

        if (rank == 0) {
            // create file
            outFile = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

            // create dataspace
            int ndims = 5;
            hsize_t dims[] = {hsize_t((nt+1)/dprint+1), hsize_t(nlayers), (ny), hsize_t(nx), 3};
            file_space = H5Screate_simple(ndims, dims, NULL);

            hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_layout(plist, H5D_CHUNKED);
            hsize_t chunk_dims[] = {1, hsize_t(nlayers), hsize_t(ny), hsize_t(nx), 3};
            H5Pset_chunk(plist, ndims, chunk_dims);

            // create dataset
            dset = H5Dcreate(outFile, "SwerveOutput", H5T_NATIVE_FLOAT, file_space, H5P_DEFAULT, plist, H5P_DEFAULT);

            H5Pclose(plist);

            // make a memory dataspace
            mem_space = H5Screate_simple(ndims, chunk_dims, NULL);

            // select a hyperslab
            file_space = H5Dget_space(dset);
            hsize_t start[] = {0, 0, 0, 0, 0};
            hsize_t hcount[] = {1, hsize_t(nlayers), hsize_t(ny), hsize_t(nx), 3};
            H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, hcount, NULL);
            // write to dataset
            printf("Printing t = %i\n", 0);
            H5Dwrite(dset, H5T_NATIVE_FLOAT, mem_space, file_space, H5P_DEFAULT, Uc_h);
            // close file dataspace
            H5Sclose(file_space);
        }

        cudaError_t err;
        err = cudaGetLastError();
        if (err != cudaSuccess){
            cout << "Before evolution\n";
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        // main loop
        for (int t = 0; t < nt; t++) {

            cout << "Evolving t = " << t << '\n';

            int ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

            // good here
            /*cout << "\nCoarse grid before prolonging\n\n";
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                        cout << '(' << x << ',' << y << "): " << Uc_h[((y*nx)+x)*3] << ',' <<  Uc_h[(((ny+y)*nx)+x)*3] << '\n';
                }
            }*/

            //cout << "\n\nProlonging\n\n";

            // prolong to fine grid
            prolong_grid(kernels, threads, blocks, cumulative_kernels,
                         Uc_d, Uf_d, nx, ny, nlayers, nxf, nyf, nz, dx, dy, dz, dt, zmin, gamma_up_d,
                         rho_d, gamma, matching_indices_d, ng, rank, q_comp_d, old_phi_d, p_floor);


            cudaMemcpy(Uf_h, Uf_d, nxf*nyf*nz*5*sizeof(float), cudaMemcpyDeviceToHost);
            err = cudaGetLastError();
            if (err != cudaSuccess){
                cout << "After prolonging\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }
            // EVERYTHING HAS NAN'd
            cout << "\nFine grid after prolonging\n\n";
            for (int y = 0; y < nyf; y++) {
                for (int x = 0; x < nxf; x++) {
                        cout << '(' << x << ',' << y << "): ";
                        for (int z = 0; z < nz; z++) {
                            cout << Uf_h[(((z*nyf + y)*nxf)+x)*5+4] << ',';
                        }
                        cout << '\n';
                }
            }


            // enforce boundaries
            if (n_processes == 1) {
                bcs_fv(Uf_h, nxf, nyf, nz, ng, 5);
            } else {
                int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
                bcs_mpi(Uf_h, nxf, nyf, nz, 5, ng, comm, status, rank, n_processes, y_size);
            }

            /*cout << "\nFine grid after prolonging\n\n";
            for (int y = 0; y < nyf; y++) {
                for (int x = 0; x < nxf; x++) {
                        cout << '(' << x << ',' << y << "): ";
                        for (int z = 0; z < nz; z++) {
                            cout << Uf_h[(((z*nyf + y)*nxf)+x)*5+4] << ',';
                        }
                        cout << '\n';
                }
            }*/

            cudaMemcpy(Uf_d, Uf_h, nxf*nyf*nz*5*sizeof(float), cudaMemcpyHostToDevice);

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                cout << "Before fine rk3\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }

            // evolve fine grid through two subcycles
            for (int i = 0; i < 2; i++) {

                rk3(kernels, threads, blocks, cumulative_kernels,
                        beta_d, gamma_up_d, Uf_d, Uf_half_d, Upf_d,
                        qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
                        fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
                        nxf, nyf, nz, 5, ng, alpha, gamma,
                        dx*0.5, dy*0.5, dz, dt*0.5, Upf_h, Ff_h, Uf_h,
                        comm, status, rank, n_processes,
                        h_compressible_fluxes, true);

                // enforce boundaries is done within rk3
                /*if (n_processes == 1) {
                    bcs_fv(Uf_h, nxf, nyf, nz, ng, 5);
                } else {
                    int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
                    bcs_mpi(Uf_h, nxf, nyf, nz, 5, ng, comm, status, rank, n_processes, y_size);
                }*/

                /*cout << "\nFine grid\n\n";
                for (int y = 0; y < ny; y++) {
                    for (int x = 0; x < nx; x++) {
                            cout << '(' << x << ',' << y << "): ";
                            for (int z = 0; z < nz; z++) {
                                cout << Uf_h[(((z*nyf + y)*nxf)+x)*5] << ',';
                            }
                            cout << '\n';
                    }
                }*/

                cudaDeviceSynchronize();

                // copy to device
                cudaMemcpy(Uf_d, Uf_h, nxf*nyf*nz*5*sizeof(float), cudaMemcpyHostToDevice);

            }
            err = cudaGetLastError();
            if (err != cudaSuccess){
                cout << "Before restricting\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }

            //cout << "\n\nRestricting\n\n";
            // probably good here
            /*cout << "\nFine grid before restricting\n\n";
            for (int y = 0; y < nyf; y++) {
                for (int x = 0; x < nxf; x++) {
                        cout << '(' << x << ',' << y << "): ";
                        for (int z = 0; z < nz; z++) {
                            cout << Uf_h[(((z*nyf + y)*nxf)+x)*5+4] << ',';
                        }
                        cout << '\n';
                }
            }*/

            /*cout << "\nCoarse grid before restricting\n\n";
            for (int z = 0; z < nlayers; z++) {
                for (int y = 0; y < ny; y++) {
                    for (int x = 0; x < nx; x++) {
                            cout << '(' << x << ',' << y << ',' << z << "): " << Uc_h[(((z*ny+y)*nx)+x)*3] << ',' <<  Uc_h[(((z*ny+y)*nx)+x)*3+1] << '\n';
                    }
                }
            }*/

            // restrict to coarse grid
            restrict_grid(kernels, threads, blocks, cumulative_kernels,
                          Uc_d, Uf_d, nx, ny, nlayers, nxf, nyf, nz,
                          dz, zmin, matching_indices_d,
                          rho_d, gamma, gamma_up_d, ng, rank, qf_swe,
                          p_floor);
            err = cudaGetLastError();
            if (err != cudaSuccess){
                cout << "After restricting\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }

            cudaMemcpy(Uc_h, Uc_d, nx*ny*nlayers*3*sizeof(float), cudaMemcpyDeviceToHost);

            /*cout << "\nCoarse grid after restricting\n\n";
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                        cout << '(' << x << ',' << y << "): " << Uc_h[((y*nx)+x)*3] << ',' <<  Uc_h[(((ny+y)*nx)+x)*3] << '\n';
                }
            }*/

            err = cudaGetLastError();
            if (err != cudaSuccess){
                cout << "After copying\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }


            // enforce boundaries
            if (n_processes == 1) {
                bcs_fv(Uc_h, nx, ny, nlayers, ng, 3);
            } else {
                int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
                bcs_mpi(Uc_h, nx, ny, nlayers, 3, ng, comm, status, rank, n_processes, y_size);
            }

            cudaMemcpy(Uc_d, Uc_h, nx*ny*nlayers*3*sizeof(float), cudaMemcpyHostToDevice);

            err = cudaGetLastError();
            if (err != cudaSuccess){
                cout << "Coarse rk3\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }

            rk3(kernels, threads, blocks, cumulative_kernels,
                beta_d, gamma_up_d, Uc_d, Uc_half_d, Upc_d,
                qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
                fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
                nx, ny, nlayers, 3, ng, alpha, gamma,
                dx, dy, dz, dt, Upc_h, Fc_h, Uc_h,
                comm, status, rank, n_processes,
                h_shallow_water_fluxes, false);

            err = cudaGetLastError();
            if (err != cudaSuccess){
                cout << "Done coarse rk3\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }

            cudaMemcpy(Uc_d, Uc_h, nx*ny*nlayers*3*sizeof(float), cudaMemcpyHostToDevice);

            // update old_phi
            for (int i = 0; i < nlayers*nx*ny; i++) {
                pphi[i] = Uc_h[i*3];
            }
            cudaMemcpy(old_phi_d, pphi, nx*ny*nlayers*sizeof(float), cudaMemcpyHostToDevice);

            /*cout << "\nCoarse grid after rk3\n\n";
            for (int z = 0; z < nlayers; z++) {
                for (int y = 0; y < ny; y++) {
                    for (int x = 0; x < nx; x++) {
                            cout << '(' << x << ',' << y << ',' << z << "): " << Uc_h[(((z*ny+y)*nx)+x)*3] << ',' <<  Uc_h[(((z*ny+y)*nx)+x)*3+1] << '\n';
                    }
                }
            }*/

            /*for (int j = 0; j < kernels[rank].y; j++) {
                kx_offset = 0;
                for (int i = 0; i < kernels[rank].x; i++) {
                    evolve_fv_heating<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(
                           gamma_up_d, Un_d,
                           Up_d, U_half_d,
                           qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                           fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                           sum_phs_d, rho_d, Q_d, mu,
                           nx, ny, nlayers, alpha,
                           dx, dy, dt, burning, kx_offset, ky_offset);
                    kx_offset += blocks[k_offset + j * kernels[rank].x + i].x * threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
                }
                ky_offset += blocks[k_offset + j * kernels[rank].x].y * threads[k_offset + j * kernels[rank].x].y - 2*ng;
            }

            kx_offset = 0;
            ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

            for (int j = 0; j < kernels[rank].y; j++) {
                kx_offset = 0;
                for (int i = 0; i < kernels[rank].x; i++) {
                    evolve2<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(gamma_up_d, Un_d,
                           Up_d, U_half_d, sum_phs_d, rho_d, Q_d, mu,
                           nx, ny, nlayers, ng, alpha,
                           dx, dy, dt, kx_offset, ky_offset);
                    kx_offset += blocks[k_offset + j * kernels[rank].x + i].x * threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
                }
                ky_offset += blocks[k_offset + j * kernels[rank].x].y * threads[k_offset + j * kernels[rank].x].y - 2*ng;
            }*/

            cudaDeviceSynchronize();

            err = cudaGetLastError();

            if (err != cudaSuccess)
                printf("Error: %s\n", cudaGetErrorString(err));

            // boundaries
            /*cudaMemcpy(Uc_h, Uc_d, nx*ny*3*sizeof(float), cudaMemcpyDeviceToHost);
            if (n_processes == 1) {
                bcs_fv(Uc_h, nx, ny, ng, 3);
            } else {
                int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
                bcs_mpi(Uc_h, nx, ny, 3, ng, comm, status, rank, n_processes, y_size);
            }
            cudaMemcpy(Uc_d, Uc_h, nx*ny*3*sizeof(float), cudaMemcpyHostToDevice);*/

            int mpi_err;

            if ((t+1) % dprint == 0) {
                if (rank == 0) {
                    printf("Printing t = %i\n", t+1);

                    if (n_processes > 1) { // only do MPI stuff if needed
                        float * buf = new float[nx*ny*nlayers*3];
                        int tag = 0;
                        for (int source = 1; source < n_processes; source++) {
                            mpi_err = MPI_Recv(buf, nx*ny*nlayers*3, MPI_FLOAT, source, tag, comm, &status);

                            check_mpi_error(mpi_err);

                            // copy data back to grid
                            ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;
                            // cheating slightly and using the fact that are moving from bottom to top to make calculations a bit easier.
                            for (int z = 0; z < nlayers; z++) {
                                for (int y = ky_offset; y < ny; y++) {
                                    for (int x = 0; x < nx; x++) {
                                        for (int i = 0; i < 3; i++) {
                                            Uc_h[((z * ny + y) * nx + x) * 3 + i] = buf[((z * ny + y) * nx + x) * 3 + i];
                                        }
                                    }
                                }
                            }
                        }
                        delete[] buf;
                    }

                    // receive data from other processes and copy to grid

                    // select a hyperslab
                    file_space = H5Dget_space(dset);
                    hsize_t start[] = {hsize_t((t+1)/dprint), 0, 0, 0, 0};
                    hsize_t hcount[] = {1, hsize_t(nlayers), hsize_t(ny), hsize_t(nx), 3};
                    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, hcount, NULL);
                    // write to dataset
                    H5Dwrite(dset, H5T_NATIVE_FLOAT, mem_space, file_space, H5P_DEFAULT, Uc_h);
                    // close file dataspae
                    H5Sclose(file_space);
                } else { // send data to rank 0
                    int tag = 0;
                    mpi_err = MPI_Ssend(Uc_h, ny*nx*nlayers*3, MPI_FLOAT, 0, tag, comm);
                    check_mpi_error(mpi_err);
                }
            }
        }

        if (rank == 0) {
            H5Sclose(mem_space);
            H5Fclose(outFile);
        }

    } else { // don't print

        for (int t = 0; t < nt; t++) {

            // prolong to fine grid
            prolong_grid(kernels, threads, blocks, cumulative_kernels, Uc_d,
                         Uf_d, nx, ny, nlayers, nxf, nyf, nz, dx, dy, dz,
                         dt, zmin, gamma_up,
                         rho_d, gamma, matching_indices_d, ng, rank, q_comp_d, old_phi_d, p_floor);

            cudaMemcpy(Uf_h, Uf_d, nxf*nyf*nz*5*sizeof(float), cudaMemcpyDeviceToHost);

            // evolve fine grid through two subcycles
            for (int i = 0; i < 2; i++) {
                rk3(kernels, threads, blocks, cumulative_kernels,
                        beta_d, gamma_up_d, Uf_d, Uf_half_d, Upf_d,
                        qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
                        fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
                        nxf, nyf, nz, 5, ng, alpha, gamma,
                        dx*0.5, dy*0.5, dz, dt*0.5, Upf_h, Ff_h, Uf_h,
                        comm, status, rank, n_processes,
                        h_compressible_fluxes, true);

                // if not last step, copy output array to input array
                if (i < 1) {
                    for (int j = 0; j < nxf*nyf*nz*5; j++) {
                        Uf_h[j] = Upf_h[j];
                    }
                }
            }

            // restrict to coarse grid
            restrict_grid(kernels, threads, blocks, cumulative_kernels,
                          Uc_d, Uf_d, nx, ny, nlayers, nxf, nyf, nz,
                          dz, zmin, matching_indices_d,
                          rho_d, gamma, gamma_up_d, ng, rank, qf_swe,
                          p_floor);

            rk3(kernels, threads, blocks, cumulative_kernels,
                beta_d, gamma_up_d, Uc_d, Uc_half_d, Upc_d,
                qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
                fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
                nx, ny, nlayers, 3, ng, alpha, gamma,
                dx, dy, dz, dt, Upc_h, Fc_h, Uc_h,
                comm, status, rank, n_processes,
                h_shallow_water_fluxes, false);

            /*int k_offset = 0;
            if (rank > 0) {
                k_offset = cumulative_kernels[rank-1];
            }
            for (int j = 0; j < kernels[rank].y; j++) {
                kx_offset = 0;
                for (int i = 0; i < kernels[rank].x; i++) {
                    evolve_fv_heating<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(
                           gamma_up_d, Un_d,
                           Up_d, U_half_d,
                           qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                           fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                           sum_phs_d, rho_d, Q_d, mu,
                           nx, ny, nlayers, alpha,
                           dx, dy, dt, burning, kx_offset, ky_offset);
                    kx_offset += blocks[k_offset + j * kernels[rank].x + i].x * threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
                }
                ky_offset += blocks[k_offset + j * kernels[rank].x].y * threads[j * kernels[rank].x].y - 2*ng;
            }


            kx_offset = 0;
            ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

            for (int j = 0; j < kernels[rank].y; j++) {
                kx_offset = 0;
                for (int i = 0; i < kernels[rank].x; i++) {
                    evolve2<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(gamma_up_d, Un_d,
                           Up_d, U_half_d, sum_phs_d, rho_d, Q_d, mu,
                           nx, ny, nlayers, ng, alpha,
                           dx, dy, dt, kx_offset, ky_offset);
                    kx_offset += blocks[k_offset + j * kernels[rank].x + i].x * threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
                }
                ky_offset += blocks[k_offset + j * kernels[rank].x].y * threads[k_offset + j * kernels[rank].x].y - 2*ng;
            }*/

            cudaDeviceSynchronize();

            // boundaries
            cudaMemcpy(Uc_h, Uc_d, nx*ny*nlayers*3*sizeof(float), cudaMemcpyDeviceToHost);
            if (n_processes == 1) {
                bcs_fv(Uc_h, nx, ny, nlayers, ng, 3);
            } else {
                int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
                bcs_mpi(Uc_h, nx, ny, nlayers, 3, ng, comm, status, rank, n_processes, y_size);
            }
            cudaMemcpy(Uc_d, Uc_h, nx*ny*nlayers*3*sizeof(float), cudaMemcpyHostToDevice);

            cudaError_t err = cudaGetLastError();

            if (err != cudaSuccess)
                printf("Error: %s\n", cudaGetErrorString(err));
        }
    }

    // delete some stuff
    cudaFree(beta_d);
    cudaFree(gamma_up_d);
    cudaFree(Uc_d);
    cudaFree(Uf_d);
    cudaFree(rho_d);
    //cudaFree(Q_d);
    cudaFree(Upc_d);
    cudaFree(Uc_half_d);
    cudaFree(Upf_d);
    cudaFree(Uf_half_d);
    cudaFree(old_phi_d);
    //cudaFree(sum_phs_d);

    cudaFree(qx_p_d);
    cudaFree(qx_m_d);
    cudaFree(qy_p_d);
    cudaFree(qy_m_d);
    cudaFree(qz_p_d);
    cudaFree(qz_m_d);
    cudaFree(fx_p_d);
    cudaFree(fx_m_d);
    cudaFree(fy_p_d);
    cudaFree(fy_m_d);
    cudaFree(fz_p_d);
    cudaFree(fz_m_d);
    cudaFree(q_comp_d);
    cudaFree(qf_swe);
    cudaFree(matching_indices_d);

    delete[] kernels;
    delete[] cumulative_kernels;
    delete[] threads;
    delete[] blocks;
    delete[] Upc_h;
    delete[] Fc_h;
    delete[] Upf_h;
    delete[] Ff_h;
    delete[] pphi;
}

#endif
