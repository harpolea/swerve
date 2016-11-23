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

__device__ bool nan_check_d(float a) {
    // check to see whether float a is a nan
    if (a != a) {
        return true;
    } else {
        return false;
    }
}

float zbrent(fptr func, const float x1, const float x2, const float tol,
             float D, float Sx, float Sy, float tau, float gamma,
             float * gamma_up) {
    /*
    Using Brent's method, return the root of a function or functor func known
    to lie between x1 and x2. The root will be regined until its accuracy is
    tol.
    */

    const int ITMAX = 300;

    float a = x1, b = x2;
    float c, d=0.0, e=0.0;
    float fa = func(a, D, Sx, Sy, tau, gamma, gamma_up);
    float fb = func(b, D, Sx, Sy, tau, gamma, gamma_up);
    float fc=0.0, fs, s;

    if (fa * fb >= 0.0) {
        //cout << "Root must be bracketed in zbrent.\n";
        throw("Root must be bracketed in zbrent.");
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

        fs = func(s, D, Sx, Sy, tau, gamma, gamma_up);

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
    throw("Maximum number of iterations exceeded in zbrent.");
    //return 0;
}


__device__ float zbrent_d(fptr func, const float x1, const float x2, const float tol,
             float D, float Sx, float Sy, float tau, float gamma,
             float * gamma_up) {
    /*
    Using Brent's method, return the root of a function or functor func known
    to lie between x1 and x2. The root will be regined until its accuracy is
    tol.
    */

    const int ITMAX = 300;

    float a = x1, b = x2;
    float c, d=0.0, e=0.0;
    float fa = func(a, D, Sx, Sy, tau, gamma, gamma_up);
    float fb = func(b, D, Sx, Sy, tau, gamma, gamma_up);
    float fc=0.0, fs, s;

    if (fa * fb >= 0.0) {
        //cout << "Root must be bracketed in zbrent.\n";
        printf("Root must be bracketed in zbrent.");
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

        fs = func(s, D, Sx, Sy, tau, gamma, gamma_up);

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
    printf("Maximum number of iterations exceeded in zbrent.");
    return 0;
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
            fprintf(stderr,"Invalid rank used in MPI send call\n");
            MPI_Error_string(mpi_err,err_buffer,&resultlen);
            fprintf(stderr,err_buffer);
            MPI_Finalize();
        } else {
            fprintf(stderr, "Other MPI error\n");
            MPI_Error_string(mpi_err,err_buffer,&resultlen);
            fprintf(stderr,err_buffer);
            MPI_Finalize();
        }
    }
}

void getNumKernels(int nx, int ny, int ng, int n_processes, int *maxBlocks, int *maxThreads, dim3 * kernels, int * cumulative_kernels) {
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
    *maxThreads = int(sqrt(float(*maxThreads))) * int(sqrt(*maxThreads));
    *maxBlocks = int(sqrt(float(*maxBlocks))) * int(sqrt(float(*maxBlocks)));

    // calculate number of kernels needed

    if (nx*ny > *maxBlocks * *maxThreads) {
        int kernels_x = int(ceil(float(nx-2*ng) / (sqrt(float(*maxThreads * *maxBlocks)) - 2.0*ng)));
        int kernels_y = int(ceil(float(ny-2*ng) / (sqrt(float(*maxThreads * *maxBlocks)) - 2.0*ng)));

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

void getNumBlocksAndThreads(int nx, int ny, int ng, int maxBlocks, int maxThreads, int n_processes, dim3 *kernels, dim3 *blocks, dim3 *threads)
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

    int total = (nx - 2*ng) * (ny - 2*ng);

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
                threads[j*kernels_x + i].x = int(sqrt(float(maxThreads)));
                threads[j*kernels_x + i].y = int(sqrt(float(maxThreads)));

                blocks[j*kernels_x + i].x = int(sqrt(float(maxBlocks)));
                blocks[j*kernels_x + i].y = int(sqrt(float(maxBlocks)));
            }
        }
        // kernels_x-1
        int nx_remaining = nx - (threads[0].x * blocks[0].x - 2*ng) * (kernels_x - 1);

        //printf("nx_remaining: %i\n", nx_remaining);

        for (int j = 0; j < (kernels_y-1); j++) {

            threads[j*kernels_x + kernels_x-1].y =
                int(sqrt(float(maxThreads)));

            threads[j*kernels_x + kernels_x-1].x =
                (nx_remaining < threads[j*kernels_x + kernels_x-1].y) ? nx_remaining : threads[j*kernels_x + kernels_x-1].y;

            blocks[j*kernels_x + kernels_x-1].x = int(ceil(float(nx_remaining) /
                float(threads[j*kernels_x + kernels_x-1].x)));
            blocks[j*kernels_x + kernels_x-1].y = int(sqrt(float(maxBlocks)));
        }

        // kernels_y-1
        int ny_remaining = ny - (threads[0].y * blocks[0].y - 2*ng) * (kernels_y - 1);
        //printf("ny_remaining: %i\n", ny_remaining);
        for (int i = 0; i < (kernels_x-1); i++) {

            threads[(kernels_y-1)*kernels_x + i].x =
                int(sqrt(float(maxThreads)));
            threads[(kernels_y-1)*kernels_x + i].y =
                (ny_remaining < threads[(kernels_y-1)*kernels_x + i].x) ? ny_remaining : threads[(kernels_y-1)*kernels_x + i].x;

            blocks[(kernels_y-1)*kernels_x + i].x = int(sqrt(float(maxBlocks)));
            blocks[(kernels_y-1)*kernels_x + i].y = int(ceil(float(ny_remaining) /
                float(threads[(kernels_y-1)*kernels_x + i].y)));
        }

        // recalculate
        nx_remaining = nx - (threads[0].x * blocks[0].x - 2*ng) * (kernels_x - 1);
        ny_remaining = ny - (threads[0].y * blocks[0].y - 2*ng) * (kernels_y - 1);
        //printf("nx_remaining: %i\n", nx_remaining);
        //printf("ny_remaining: %i\n", ny_remaining);

        // (kernels_x-1, kernels_y-1)
        threads[(kernels_y-1)*kernels_x + kernels_x-1].x =
            (nx_remaining < int(sqrt(float(maxThreads)))) ? nx_remaining : int(sqrt(float(maxThreads)));
        threads[(kernels_y-1)*kernels_x + kernels_x-1].y =
            (ny_remaining < int(sqrt(float(maxThreads)))) ? ny_remaining : int(sqrt(float(maxThreads)));

        blocks[(kernels_y-1)*kernels_x + kernels_x-1].x =
            int(ceil(float(nx_remaining) /
            float(threads[(kernels_y-1)*kernels_x + kernels_x-1].x)));
        blocks[(kernels_y-1)*kernels_x + kernels_x-1].y =
            int(ceil(float(ny_remaining) /
            float(threads[(kernels_y-1)*kernels_x + kernels_x-1].y)));

    } else {

        int total_threads = (total < maxThreads*2) ? nextPow2((total + 1)/ 2) : maxThreads;
        threads[0].x = int(floor(sqrt(float(total_threads))));
        threads[0].y = int(floor(sqrt(float(total_threads))));
        total_threads = threads[0].x * threads[0].y;
        int total_blocks = int(ceil(float(total) / float(total_threads)));

        blocks[0].x = int(ceil(sqrt(float(total_blocks)/float(nx*ny))*nx));
        blocks[0].y = int(ceil(sqrt(float(total_blocks)/float(nx*ny))*ny));

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

void bcs_fv(float * grid, int nx, int ny, int ng, int vec_dim) {
    /*
    Enforce boundary conditions on section of grid.
    */
    // outflow

    for (int y = 0; y < ny; y++){
        for (int i = 0; i < vec_dim; i++) {
            for (int g = 0; g < ng; g++) {
                grid[(y * nx + g) * vec_dim+i] = grid[(y * nx + ng)*vec_dim+i];

                grid[(y * nx + (nx-1-g))*vec_dim+i] = grid[(y * nx + (nx-1-ng))*vec_dim+i];
            }
        }
    }
    for (int x = 0; x < nx; x++){
        for (int i = 0; i < vec_dim; i++) {
            for (int g = 0; g < ng; g++) {
                grid[(g * nx + x)*vec_dim+i] = grid[(ng * nx + x)*vec_dim+i];

                grid[((ny-1-g) * nx + x)*vec_dim+i] = grid[((ny-1-ng) * nx + x)*vec_dim+i];
            }
        }
    }
}


void bcs_mpi(float * grid, int nx, int ny, int vec_dim, int ng, MPI_Comm comm, MPI_Status status, int rank, int n_processes, int y_size) {
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
    for (int y = 0; y < ny; y++) {
        for (int g = 0; g < ng; g++) {
            for (int i = 0; i < vec_dim; i++) {
                grid[(y * nx + g) *vec_dim+i] = grid[(y * nx + ng) *vec_dim+i];

                grid[(y * nx + (nx-1-g))*vec_dim+i] = grid[(y * nx + (nx-1-ng))*vec_dim+i];
            }
        }
    }

    // interior cells between processes

    // make some buffers for sending and receiving
    float * ysbuf = new float[nx*ng*vec_dim];
    float * yrbuf = new float[nx*ng*vec_dim];

    int tag = 1;
    int mpi_err;
    MPI_Request request;

    // if there are process above and below, send/receive
    if ((rank > 0) && (rank < n_processes-1)) {
        // send to below, receive from above
        // copy stuff to buffer
        for (int g = 0; g < ng; g++) {
            for (int x = 0; x < nx; x++) {
                for (int i = 0; i < vec_dim; i++) {
                    ysbuf[(g * nx + x)*vec_dim+i] = grid[((y_size*rank+ng+g) * nx + x)*vec_dim+i];

                }
            }
        }
        mpi_err = MPI_Issend(ysbuf,nx*ng*vec_dim, MPI_FLOAT, rank-1, tag, comm, &request);
        check_mpi_error(mpi_err);
        mpi_err = MPI_Recv(yrbuf, nx*ng*vec_dim, MPI_FLOAT, rank+1, tag, comm, &status);
        check_mpi_error(mpi_err);
        MPI_Wait(&request, &status);

        // copy received data back to grid
        for (int g = 0; g < ng; g++) {
            for (int x = 0; x < nx; x++) {
                for (int i = 0; i < vec_dim; i++) {
                    grid[((y_size*rank+ny-ng+g) * nx + x)*vec_dim+i] = yrbuf[(g * nx + x)*vec_dim+i];
                }
            }
        }
        // send to above, receive from below
        // copy stuff to buffer
        for (int g = 0; g < ng; g++) {
            for (int x = 0; x < nx; x++) {
                for (int i = 0; i < vec_dim; i++) {
                    ysbuf[(g * nx + x)*vec_dim+i] = grid[((y_size*rank+ny-2*ng+g) * nx + x)*vec_dim+i];
                }
            }
        }
        MPI_Issend(ysbuf,nx*ng*vec_dim, MPI_FLOAT, rank+1, tag, comm, &request);
        MPI_Recv(yrbuf, nx*ng*vec_dim, MPI_FLOAT, rank-1, tag, comm, &status);
        MPI_Wait(&request, &status);

        // copy received data back to grid
        for (int g = 0; g < ng; g++) {
            for (int x = 0; x < nx; x++) {
                for (int i = 0; i < vec_dim; i++) {
                    grid[((y_size*rank+g) * nx + x)*vec_dim+i] = yrbuf[(g * nx + x)*vec_dim+i];
                }
            }
        }

    } else if (rank == 0) {
        // do outflow for top boundary
        // copy stuff to buffer
        for (int g = 0; g < ng; g++) {
            for (int x = 0; x < nx; x++) {
                for (int i = 0; i < vec_dim; i++) {
                    ysbuf[(g * nx + x)*vec_dim+i] = grid[((ny-2*ng+g) * nx + x)*vec_dim+i];
                }
            }
        }

        MPI_Issend(ysbuf, nx*ng*vec_dim, MPI_FLOAT, 1, tag, comm, &request);
        MPI_Recv(yrbuf, nx*ng*vec_dim, MPI_FLOAT, 1, tag, comm, &status);
        MPI_Wait(&request, &status);

        // copy received data back to grid
        for (int g = 0; g < ng; g++) {
            for (int x = 0; x < nx; x++) {
                for (int i = 0; i < vec_dim; i++) {
                    grid[((ny-ng+g) * nx + x)*vec_dim+i] = yrbuf[(g * nx + x)*vec_dim+i];
                }
            }
        }

        // outflow stuff on top boundary
        for (int g = 0; g < ng; g++) {
            for (int x = 0; x < nx; x++) {
                for (int i = 0; i < vec_dim; i++) {
                    grid[(g * nx + x)*vec_dim+i] = grid[(ng * nx + x)*vec_dim+i];
                }
            }
        }

    } else {
        // copy stuff to buffer
        for (int g = 0; g < ng; g++) {
            for (int x = 0; x < nx; x++) {
                for (int i = 0; i < vec_dim; i++) {
                    ysbuf[(g * nx + x)*vec_dim+i] = grid[((y_size*rank+ng+g) * nx + x)*vec_dim+i];
                }
            }
        }
        // bottom-most process
        MPI_Issend(ysbuf, nx*ng*vec_dim, MPI_FLOAT, rank-1, tag, comm, &request);
        MPI_Recv(yrbuf, nx*ng*vec_dim, MPI_FLOAT, rank-1, tag, comm, &status);
        MPI_Wait(&request, &status);

        // copy received data back to grid
        for (int g = 0; g < ng; g++) {
            for (int x = 0; x < nx; x++) {
                for (int i = 0; i < vec_dim; i++) {
                    grid[((y_size*rank+g) * nx + x)*vec_dim+i] = yrbuf[(g * nx + x)*vec_dim+i];
                }
            }
        }

        // outflow for bottom boundary
        for (int g = 0; g < ng; g++) {
            for (int x = 0; x < nx; x++) {
                for (int i = 0; i < vec_dim; i++) {
                    grid[((ny-1-g) * nx + x)*vec_dim+i] = grid[((ny-1-ng) * nx + x)*vec_dim+i];
                }
            }
        }
    }

    delete[] ysbuf;
    delete[] yrbuf;
}

float phi(float r) {
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

__device__ float phi_d(float r) {
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

float rhoh_from_p(float p, float rho, float gamma) {
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

float p_from_rho_eps(float rho, float eps, float gamma) {
    // calculate p using rho and epsilon for gamma law equation of state
    return (gamma - 1.0) * rho * eps;
}

float phi_from_p(float p, float rho, float gamma) {
    // calculate the metric potential Phi given p for gamma law equation of
    // state
    return 1.0 + (gamma - 1.0) / gamma *
        log(1.0 + gamma * p / ((gamma - 1.0) * rho));
}



__device__ float f_of_p_d(float p, float D, float Sx, float Sy, float tau,
            float gamma, float * gamma_up) {
    // function of p whose root is to be found when doing conserved to
    // primitive variable conversion

    float sq = sqrt(pow(tau + p + D, 2) -
        Sx*Sx*gamma_up[0] - 2.0*Sx*Sy*gamma_up[1] - Sy*Sy*gamma_up[3]);

    //if (nan_check(sq)) cout << "sq is nan :(\n";

    float rho = D * sq / (tau + p + D);
    float eps = (sq - p * (tau + p + D) / sq - D) / D;

    return (gamma - 1.0) * rho * eps - p;
}

float f_of_p(float p, float D, float Sx, float Sy, float tau,
            float gamma, float * gamma_up) {
    // function of p whose root is to be found when doing conserved to
    // primitive variable conversion

    float sq = sqrt(pow(tau + p + D, 2) -
        Sx*Sx*gamma_up[0] - 2.0*Sx*Sy*gamma_up[1] - Sy*Sy*gamma_up[3]);

    //if (nan_check(sq)) cout << "sq is nan :(\n";

    float rho = D * sq / (tau + p + D);
    float eps = (sq - p * (tau + p + D) / sq - D) / D;

    return (gamma - 1.0) * rho * eps - p;
}

__device__ void cons_to_prim_comp_d(float * q_cons, float * q_prim,
                       float gamma, float * gamma_up) {
    // convert compressible conserved variables to primitive variables

    const float TOL = 1.e-5;
    float D = q_cons[0];
    float Sx = q_cons[1];
    float Sy = q_cons[2];
    float tau = q_cons[3];

    // S^2
    float Ssq = Sx*Sx*gamma_up[0] + 2.0*Sx*Sy*gamma_up[1] +
        Sy*Sy*gamma_up[3];

    float pmin = (1.0 - Ssq) * (1.0 - Ssq) * tau * (gamma - 1.0);
    float pmax = (gamma - 1.0) * (tau + D) / (2.0 - gamma);

    if (pmin < 0.0) {
        pmin = 0.0;//1.0e-9;
    }
    if (pmax < 0.0 || pmax < pmin) {
        pmax = 1.0;
    }

    // check sign change
    if (f_of_p_d(pmin, D, Sx, Sy, tau, gamma, gamma_up) *
        f_of_p_d(pmax, D, Sx, Sy, tau, gamma, gamma_up) > 0.0) {
        pmin = 0.0;
    }

    float p = zbrent_d((fptr)f_of_p_d, pmin, pmax, TOL, D, Sx, Sy,
                    tau, gamma, gamma_up);
    if (nan_check_d(p)){
        //printf("NAN ALERT\n");
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
    q_prim[3] = eps;
}

void cons_to_prim_comp(float * q_cons, float * q_prim, int nxf, int nyf,
                       float gamma, float * gamma_up) {
    // convert compressible conserved variables to primitive variables

    const float TOL = 1.e-5;
    //printf("size of q_cons: %i\n", sizeof(q_cons)/sizeof(q_cons[0]));
    //printf("size of nxf*nyf*4: %i\n", nxf*nyf*4);
    /*cout << "Cons to prim\n";
    for (int y = 0; y < nyf; y++) {
        for (int x = 0; x < nxf; x++) {
            cout << '(' << x << ',' << y << "): " << q_cons[(y*nxf+x)*4+3] << '\n';
        }
    }*/

    for (int i = 0; i < nxf*nyf; i++) {
        float D = q_cons[i*4];
        float Sx = q_cons[i*4+1];
        float Sy = q_cons[i*4+2];
        float tau = q_cons[i*4+3];

        //printf("D = %f\n", D);

        // S^2
        float Ssq = Sx*Sx*gamma_up[0] + 2.0*Sx*Sy*gamma_up[1] +
            Sy*Sy*gamma_up[3];

        float pmin = (1.0 - Ssq) * (1.0 - Ssq) * tau * (gamma - 1.0);
        float pmax = (gamma - 1.0) * (tau + D) / (2.0 - gamma);

        if (pmin < 0.0) {
            pmin = 0.0;//1.0e-9;
        }
        if (pmax < 0.0 || pmax < pmin) {
            pmax = 1.0;
        }

        // check sign change
        if (f_of_p(pmin, D, Sx, Sy, tau, gamma, gamma_up) *
            f_of_p(pmax, D, Sx, Sy, tau, gamma, gamma_up) > 0.0) {
            pmin = 0.0;
        }

        // nan check inputs
        //if (nan_check(pmin)) cout << "pmin is nan!\n";
        //if (nan_check(pmax)) cout << "pmax is nan!\n";
        //if (nan_check(D)) cout << "D is nan!\n";
        //if (nan_check(Sx)) cout << "Sx is nan!\n";
        //if (nan_check(Sy)) cout << "Sy is nan!\n";
        //if (nan_check(tau)) cout << "tau is nan!\n";

        float p;
        try {
            p = zbrent((fptr)f_of_p, pmin, pmax, TOL, D, Sx, Sy,
                        tau, gamma, gamma_up);
        } catch (char const*){
            p = abs((gamma - 1.0) * (tau + D) / (2.0 - gamma)) > 1.0 ? 1.0 :
                abs((gamma - 1.0) * (tau + D) / (2.0 - gamma));
        }

        float sq = sqrt(pow(tau + p + D, 2) - Ssq);
        float eps = (sq - p * (tau + p + D)/sq - D) / D;
        float h = 1.0 + gamma * eps;
        float W = sqrt(1.0 + Ssq / (D*D*h*h));

        q_prim[i*4] = D * sq / (tau + p + D);//D / W;
        q_prim[i*4+1] = Sx / (W*W * h * q_prim[i*4]);
        q_prim[i*4+2] = Sy / (W*W * h * q_prim[i*4]);
        q_prim[i*4+3] = eps;
    }
}

__device__ void shallow_water_fluxes(float * q, float * f, bool x_dir,
                          float * gamma_up, float alpha, float * beta,
                          float gamma) {
    // calculate the flux vector of the shallow water equations

    float W = sqrt((q[1] * q[1] * gamma_up[0] +
                2.0 * q[1] * q[2] * gamma_up[1] +
                q[2] * q[2] * gamma_up[3]) / (q[0] * q[0]) + 1.0);

    float u = q[1] / (q[0] * W);
    float v = q[2] / (q[0] * W);

    if (x_dir) {
        float qx = u * gamma_up[0] + v * gamma_up[1] -
            beta[0] / alpha;

        f[0] = q[0] * qx;
        f[1] = q[1] * qx + 0.5 * q[0] * q[0] / (W * W);
        f[2] = q[2] * qx;
    } else {
        float qy = v * gamma_up[3] + u * gamma_up[1] -
            beta[1] / alpha;

        f[0] = q[0] * qy;
        f[1] = q[1] * qy;
        f[2] = q[2] * qy + 0.5 * q[0] * q[0] / (W * W);
    }
}

__device__ void compressible_fluxes(float * q, float * f, bool x_dir,
                         float * gamma_up, float alpha, float * beta,
                         float gamma) {
    // calculate the flux vector of the compressible GR hydrodynamics equations

    // this is worked out on the fine grid
    float * q_prim;
    q_prim = (float *)malloc(4 * sizeof(float));

    cons_to_prim_comp_d(q, q_prim, gamma, gamma_up);

    float p = p_from_rho_eps_d(q_prim[0], q_prim[3], gamma);
    float u = q_prim[1];
    float v = q_prim[2];

    if (x_dir) {
        float qx = u * gamma_up[0] + v * gamma_up[1] - beta[0] / alpha;

        f[0] = q[0] * qx;
        f[1] = q[1] * qx + p;
        f[2] = q[2] * qx;
        f[3] = q[3] * qx + p * u;
    } else {
        float qy = v * gamma_up[3] + u * gamma_up[1] - beta[1] / alpha;

        f[0] = q[0] * qy;
        f[1] = q[1] * qy;
        f[2] = q[2] * qy + p;
        f[3] = q[3] * qy + p * v;
    }

    free(q_prim);
}


void p_from_swe(float * q, float * p, int nx, int ny,
                 float * gamma_up, float rho, float gamma) {
    // calculate p using SWE conserved variables

    for (int i = 0; i < nx*ny; i++) {
        float W = sqrt((q[i*3+1]*q[i*3+1] * gamma_up[0] +
                2.0 * q[i*3+1] * q[i*3+2] * gamma_up[1] +
                q[i*3+2] * q[i*3+2] * gamma_up[3]) / (q[i*3]*q[i*3]) + 1.0);

        float ph = q[i*3] / W;

        p[i] = rho * (gamma - 1.0) * (exp(gamma * (ph - 1.0) /
            (gamma - 1.0)) - 1.0) / gamma;
    }
}

void prolong_grid(float * q_c, float * q_f,
                       int nx, int ny, int nxf, int nyf, float dx, float dy,
                       float * gamma_up, float rho, float gamma,
                       int * matching_indices) {
    // prolong coarse grid to fine one
    float * qc_comp = new float[int(nx*ny*4)];
    float * Sx = new float[int(nx*ny*4)];
    float * Sy = new float[int(nx*ny*4)];
    float * p = new float[int(nx*ny)];

    p_from_swe(q_c, p, nx, ny, gamma_up, rho, gamma);

    // first calculate the compressible conserved variables on the coarse grid
    for (int i = 0; i < nx*ny; i++) {
        float rhoh = rhoh_from_p(p[i], rho, gamma);
        float W = sqrt((q_c[i*3+1] * q_c[i*3+1] * gamma_up[0] +
                2.0 * q_c[i*3+1] * q_c[i*3+2] * gamma_up[1] +
                q_c[i*3+2] * q_c[i*3+2] * gamma_up[3]) /
                (q_c[i*3] * q_c[i*3]) + 1.0);

        qc_comp[i*4] = rho * W;
        qc_comp[i*4+1] = rhoh * W * q_c[i*3+1] / q_c[i*3];
        qc_comp[i*4+2] = rhoh * W * q_c[i*3+2] / q_c[i*3];
        qc_comp[i*4+3] = rhoh*W*W - p[i] - qc_comp[i*4];

        // NOTE: hack?
        if (qc_comp[i*4+3] < 0.0) qc_comp[i*4+3] = 0.0;
    }

    // do some slope limiting
    for (int j = matching_indices[2]; j < matching_indices[3]+1; j++) {
        for (int i = matching_indices[0]; i < matching_indices[1]+1; i++) {
            for (int n = 0; n < 4; n++) {

                // x-dir
                float S_upwind = (qc_comp[(j * nx + i+1) * 4 + n] -
                    qc_comp[(j * nx + i) * 4 + n]) / dx;
                float S_downwind = (qc_comp[(j * nx + i) * 4 + n] -
                    qc_comp[(j * nx + i-1) * 4 + n]) / dx;

                Sx[(j * nx + i) * 4 + n] = 0.5 * (S_upwind + S_downwind);

                float r = 1.0e6;
                if (abs(S_downwind) > 1.0e-10) {
                    r = S_upwind / S_downwind;
                }

                Sx[(j * nx + i) * 4 + n] *= phi(r);

                // y-dir
                S_upwind = (qc_comp[((j+1) * nx + i) * 4 + n] -
                    qc_comp[(j * nx + i) * 4 + n]) / dy;
                S_downwind = (qc_comp[(j * nx + i) * 4 + n] -
                    qc_comp[((j-1) * nx + i) * 4 + n]) / dy;

                Sy[(j * nx + i) * 4 + n] = 0.5 * (S_upwind + S_downwind);

                r = 1.0e6;
                if (abs(S_downwind) > 1.0e-10) {
                    r = S_upwind / S_downwind;
                }

                Sy[(j * nx + i) * 4 + n] *= phi(r);
            }
        }
    }

    // reconstruct values at fine grid cell centres
    for (int j = 0; j < matching_indices[3] - matching_indices[2]+1; j++) {
        for (int i = 0; i < matching_indices[1] - matching_indices[0]+1; i++) {
            for (int n = 0; n < 4; n++) {
                int coarse_index = ((j + matching_indices[2]) * nx + i +
                    matching_indices[0]) * 4 + n;

                q_f[(2*j * nxf + 2*i) * 4 + n] = qc_comp[coarse_index] -
                    0.25 * (dx * Sx[coarse_index] + dy * Sy[coarse_index]);

                q_f[(2*j * nxf + 2*i+1) * 4 + n] = qc_comp[coarse_index] +
                    0.25 * (dx * Sx[coarse_index] - dy * Sy[coarse_index]);

                q_f[((2*j+1) * nxf + 2*i) * 4 + n] = qc_comp[coarse_index] +
                    0.25 * (-dx * Sx[coarse_index] + dy * Sy[coarse_index]);

                q_f[((2*j+1) * nxf + 2*i+1) * 4 + n] = qc_comp[coarse_index] +
                    0.25 * (dx * Sx[coarse_index] + dy * Sy[coarse_index]);
            }
        }
    }

    delete[] qc_comp;
    delete[] Sx;
    delete[] Sy;
    delete[] p;
}

void restrict_grid(float * q_c, float * q_f,
                        int nx, int ny, int nxf, int nyf,
                        int * matching_indices,
                        float rho, float gamma, float * gamma_up) {
    // restrict fine grid to coarse grid

    float * q_prim = new float[nxf*nyf*4];
    float * qf_sw = new float[nxf*nyf*3];

    // initialise q_prim
    for (int i = 0; i < nxf*nyf*4; i++) {
        q_prim[i] = 0.0;
    }

    /*cout << "Restricting grid\n";
    for (int y = 0; y < nyf; y++) {
        for (int x = 0; x < nxf; x++) {
            cout << '(' << x << ',' << y << "): " << q_f[((y*nxf)+x)*4+3] << '\n';
        }
    }*/

    // find primitive variables
    cons_to_prim_comp(q_f, q_prim, nxf, nyf, gamma, gamma_up);

    // calculate SWE conserved variables on fine grid
    for (int i = 0; i < nxf*nyf; i++) {
        float p = p_from_rho_eps(q_prim[i*4], q_prim[i*4+3], gamma);
        float phi = phi_from_p(p, rho, gamma);

        float u = q_prim[i*4+1];
        float v = q_prim[i*4+2];

        float W = 1.0 / sqrt(1.0 -
                u*u*gamma_up[0] - 2.0 * u*v * gamma_up[1] - v*v*gamma_up[3]);

        qf_sw[i*3] = phi * W;
        qf_sw[i*3+1] = phi * W * W * u;
        qf_sw[i*3+2] = phi * W * W * v;
    }

    // interpolate fine grid to coarse grid
    for (int j = 1; j < matching_indices[3] - matching_indices[2]; j++) {
        for (int i = 1; i < matching_indices[1] - matching_indices[0]; i++) {
            for (int n = 0; n < 3; n++) {
                q_c[((j+matching_indices[2]) * nx +
                      i+matching_indices[0]) * 3+n] =
                      0.25 * (qf_sw[(j*2 * nxf + i*2) * 3 + n] +
                              qf_sw[(j*2 * nxf + i*2+1) * 3 + n] +
                              qf_sw[((j*2+1) * nxf + i*2) * 3 + n] +
                              qf_sw[((j*2+1) * nxf + i*2+1) * 3 + n]);
            }
        }
    }

    delete[] q_prim;
    delete[] qf_sw;
}


__global__ void evolve_fv(float * beta_d, float * gamma_up_d,
                     float * Un_d, flux_func_ptr flux_func,
                     float * qx_plus_half, float * qx_minus_half,
                     float * qy_plus_half, float * qy_minus_half,
                     float * fx_plus_half, float * fx_minus_half,
                     float * fy_plus_half, float * fy_minus_half,
                     int nx, int ny, int vec_dim, float alpha, float gamma,
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
    nx, ny : int
        dimensions of grid
    alpha : float
        lapse function
    kx_offset, ky_offset : int
        x, y offset for current kernel
    */

    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;

    int offset = (y * nx + x) * vec_dim;

    float * q_p, *q_m, * f;
    q_p = (float *)malloc(vec_dim * sizeof(float));
    q_m = (float *)malloc(vec_dim * sizeof(float));
    f = (float *)malloc(vec_dim * sizeof(float));

    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1))) {

        // x-direction
        for (int i = 0; i < vec_dim; i++) {
            float S_upwind = (Un_d[(y * nx + x+1) * vec_dim + i] -
                Un_d[(y * nx + x) * vec_dim + i]) / dx;
            float S_downwind = (Un_d[(y * nx + x) * vec_dim + i] -
                Un_d[(y * nx + x-1) * vec_dim + i]) / dx;
            float S = 0.5 * (S_upwind + S_downwind); // S_av

            float r = 1.0e6;

            // make sure don't divide by zero
            if (abs(S_downwind) > 1.0e-7) {
                r = S_upwind / S_downwind;
            }

            S *= phi_d(r);

            q_p[i] = Un_d[offset + i] + S * 0.5 * dx;
            q_m[i] = Un_d[offset + i] - S * 0.5 * dx;

        }

        // fluxes

        //printf("x, y: %i, %i\n", x, y);

        flux_func(q_p, f, true, gamma_up_d, alpha, beta_d, gamma);

        //printf("Called flux func?\n");

        for (int i = 0; i < vec_dim; i++) {
            fx_plus_half[offset + i] = f[i];
        }

        flux_func(q_m, f, true, gamma_up_d, alpha, beta_d, gamma);

        for (int i = 0; i < vec_dim; i++) {
            fx_minus_half[offset + i] = f[i];
        }

        // y-direction
        for (int i = 0; i < vec_dim; i++) {
            float S_upwind = (Un_d[((y+1) * nx + x) * vec_dim + i] -
                Un_d[(y * nx + x) * vec_dim + i]) / dy;
            float S_downwind = (Un_d[(y * nx + x) * vec_dim + i] -
                Un_d[((y-1) * nx + x) * vec_dim + i]);
            float S = 0.5 * (S_upwind + S_downwind) / dy; // S_av

            float r = 1.0e6;

            // make sure don't divide by zero
            if (abs(S_downwind) > 1.0e-7) {
                r = S_upwind / S_downwind;
            }

            S *= phi_d(r);

            q_p[i] = Un_d[offset + i] + S * 0.5 * dy;
            q_m[i] = Un_d[offset + i] - S * 0.5 * dy;
        }

        // fluxes

        flux_func(q_p, f, false, gamma_up_d, alpha, beta_d, gamma);

        for (int i = 0; i < vec_dim; i++) {
            fy_plus_half[offset + i] = f[i];
        }

        flux_func(q_m, f, false, gamma_up_d, alpha, beta_d, gamma);

        for (int i = 0; i < vec_dim; i++) {
            fy_minus_half[offset + i] = f[i];
        }

        //printf("evolve_fv (%i, %i): %f\n", x, y, fx_plus_half[offset + vec_dim-1]);
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
                     int nx, int ny, int vec_dim, float alpha,
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

    // do fluxes
    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1))) {
        for (int i = 0; i < vec_dim; i++) {
            // x-boundary
            // from i-1
            float fx_m = 0.5 * (
                fx_plus_half[(y * nx + x-1) * vec_dim + i] +
                fx_minus_half[(y * nx + x) * vec_dim + i] +
                qx_plus_half[(y * nx + x-1) * vec_dim + i] -
                qx_minus_half[(y * nx + x) * vec_dim + i]);
            // from i+1
            float fx_p = 0.5 * (
                fx_plus_half[(y * nx + x) * vec_dim + i] +
                fx_minus_half[(y * nx + x+1) * vec_dim + i] +
                qx_plus_half[(y * nx + x) * vec_dim + i] -
                qx_minus_half[(y * nx + x+1) * vec_dim + i]);

            // y-boundary
            // from j-1
            float fy_m = 0.5 * (
                fy_plus_half[((y-1) * nx + x) * vec_dim + i] +
                fy_minus_half[(y * nx + x) * vec_dim + i] +
                qy_plus_half[((y-1) * nx + x) * vec_dim + i] -
                qy_minus_half[(y * nx + x) * vec_dim + i]);
            // from j+1
            float fy_p = 0.5 * (
                fy_plus_half[(y * nx + x) * vec_dim + i] +
                fy_minus_half[((y+1) * nx + x) * vec_dim + i] +
                qy_plus_half[(y * nx + x) * vec_dim + i] -
                qy_minus_half[((y+1) * nx + x) * vec_dim + i]);

            F[(y * nx + x)*vec_dim + i] =
                -alpha * ((1.0/dx) * (fx_p - fx_m) +
                (1.0/dy) * (fy_p - fy_m));
        }
    }
}

void homogeneuous_fv(dim3 * kernels, dim3 * threads, dim3 * blocks,
       int * cumulative_kernels, float * beta_d, float * gamma_up_d,
       float * Un_d, float * F_d,
       float * qx_p_d, float * qx_m_d, float * qy_p_d, float * qy_m_d,
       float * fx_p_d, float * fx_m_d, float * fy_p_d, float * fy_m_d,
       int nx, int ny, int vec_dim, int ng, float alpha, float gamma,
       float dx, float dy, float dt, int rank,
       flux_func_ptr h_flux_func) {
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
                  nx, ny, vec_dim, alpha, gamma,
                  dx, dy, dt, kx_offset, ky_offset);
          kx_offset += blocks[k_offset + j * kernels[rank].x + i].x * threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
       }
       ky_offset += blocks[k_offset + j * kernels[rank].x].y * threads[k_offset + j * kernels[rank].x].y - 2*ng;
    }

    ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

    for (int j = 0; j < kernels[rank].y; j++) {
       kx_offset = 0;
       for (int i = 0; i < kernels[rank].x; i++) {
           evolve_fv_fluxes<<<blocks[k_offset + j * kernels[rank].x + i],
                              threads[k_offset + j * kernels[rank].x + i]>>>(
                  F_d,
                  qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                  fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                  nx, ny, vec_dim, alpha,
                  dx, dy, dt, kx_offset, ky_offset);

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
       float * fx_p_d, float * fx_m_d, float * fy_p_d, float * fy_m_d,
       int nx, int ny, int vec_dim, int ng, float alpha, float gamma,
       float dx, float dy, float dt,
       float * Up_h, float * F_h, float * Un_h,
       MPI_Comm comm, MPI_Status status, int rank, int n_processes,
       flux_func_ptr h_flux_func) {
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
    */

    // u1 = un + dt * F(un)
    homogeneuous_fv(kernels, threads, blocks, cumulative_kernels,
          beta_d, gamma_up_d, Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d,
          nx, ny, vec_dim, ng, alpha, gamma,
          dx, dy, dt, rank, h_flux_func);

    // copy back flux
    cudaMemcpy(F_h, F_d, nx*ny*vec_dim*sizeof(float), cudaMemcpyDeviceToHost);
    if (n_processes == 1) {
        bcs_fv(F_h, nx, ny, ng, vec_dim);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(F_h, nx, ny, vec_dim, ng, comm, status, rank, n_processes, y_size);
    }

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            for (int i = 0; i < vec_dim; i++) {
                Up_h[(y * nx + x) * vec_dim + i] = Un_h[(y * nx + x) * vec_dim + i] + dt * F_h[(y * nx + x) * vec_dim + i];
            }
        }
    }

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            cout << '(' << x << ',' << y << "): " << F_h[((y*nx)+x)*vec_dim+vec_dim-1] << '\n';
        }
    }

    // enforce boundaries and copy back
    if (n_processes == 1) {
        bcs_fv(Up_h, nx, ny, ng, vec_dim);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(Up_h, nx, ny, vec_dim, ng, comm, status, rank, n_processes, y_size);
    }
    cudaMemcpy(Un_d, Up_h, nx*ny*vec_dim*sizeof(float), cudaMemcpyHostToDevice);

    // u2 = 0.25 * (3*un + u1 + dt*F(u1))
    homogeneuous_fv(kernels, threads, blocks, cumulative_kernels,
          beta_d, gamma_up_d, Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d,
          nx, ny, vec_dim, ng, alpha, gamma,
          dx, dy, dt, rank, h_flux_func);

    // copy back flux
    cudaMemcpy(F_h, F_d, nx*ny*vec_dim*sizeof(float), cudaMemcpyDeviceToHost);
    //bcs_fv(F_h, nx, ny, nlayers, ng);
    if (n_processes == 1) {
        bcs_fv(F_h, nx, ny, ng, vec_dim);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(F_h, nx, ny, vec_dim, ng, comm, status, rank, n_processes, y_size);
    }

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            for (int i = 0; i < vec_dim; i++) {
                Up_h[(y * nx + x) * vec_dim + i] = 0.25 * (
                    3.0 * Un_h[(y * nx + x) * vec_dim + i] +
                    Up_h[(y * nx + x) * vec_dim + i] +
                    dt * F_h[(y * nx + x) * vec_dim + i]);
            }
        }
    }

    // enforce boundaries and copy back
    //bcs_fv(Up_h, nx, ny, nlayers, ng);
    if (n_processes == 1) {
        bcs_fv(Up_h, nx, ny, ng, vec_dim);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(Up_h, nx, ny, vec_dim, ng, comm, status, rank, n_processes, y_size);
    }
    cudaMemcpy(Un_d, Up_h, nx*ny*vec_dim*sizeof(float), cudaMemcpyHostToDevice);

    // un+1 = (1/3) * (un + 2*u2 + 2*dt*F(u2))
    homogeneuous_fv(kernels, threads, blocks, cumulative_kernels,
          beta_d, gamma_up_d, Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d,
          nx, ny, vec_dim, ng, alpha, gamma,
          dx, dy, dt, rank, h_flux_func);

    // copy back flux
    cudaMemcpy(F_h, F_d, nx*ny*vec_dim*sizeof(float), cudaMemcpyDeviceToHost);
    //bcs_fv(F_h, nx, ny, nlayers, ng);
    if (n_processes == 1) {
        bcs_fv(F_h, nx, ny, ng, vec_dim);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(F_h, nx, ny, vec_dim, ng, comm, status, rank, n_processes, y_size);
    }

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
                for (int i = 0; i < vec_dim; i++) {
                    Up_h[(y * nx + x) * vec_dim + i] = (1/3.0) * (
                        Un_h[(y * nx + x) * vec_dim + i] +
                        2.0*Up_h[(y * nx + x) * vec_dim + i] +
                        2.0*dt * F_h[(y * nx + x) * vec_dim + i]);
            }
        }
    }

    // enforce boundaries
    //bcs_fv(Up_h, nx, ny, nlayers, ng);
    if (n_processes == 1) {
        bcs_fv(Up_h, nx, ny, ng, vec_dim);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(Up_h, nx, ny, vec_dim, ng, comm, status, rank, n_processes, y_size);
    }

    for (int j = 0; j < nx*ny*vec_dim; j++) {
        Un_h[j] = Up_h[j];
    }

    //cudaMemcpy(Up_d, Up_h, nx*ny*vec_dim*sizeof(float), cudaMemcpyHostToDevice);

}

// device-side function pointers to __device__ functions
__device__ flux_func_ptr d_compressible_fluxes = compressible_fluxes;
__device__ flux_func_ptr d_shallow_water_fluxes = shallow_water_fluxes;

void cuda_run(float * beta, float * gamma_up, float * Uc_h, float * Uf_h,
         float rho, float mu, int nx, int ny,
         int nxf, int nyf, int ng,
         int nt, float alpha, float gamma, float dx, float dy, float dt, bool burning,
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
    nx, ny : int
        dimensions of coarse grid
    ng : int
        number of ghost cells
    nt : int
        total number of timesteps
    alpha : float
        lapse function
    dx, dy, dt : float
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

    getNumKernels(nx, ny, ng, n_processes, &maxBlocks, &maxThreads, kernels, cumulative_kernels);

    int total_kernels = cumulative_kernels[n_processes-1];

    //int kernels_y = kernels[0].y;

    dim3 *blocks = new dim3[total_kernels];
    dim3 *threads = new dim3[total_kernels];

    getNumBlocksAndThreads(nx, ny, ng, maxBlocks, maxThreads, n_processes, kernels, blocks, threads);

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
    //float * rho_d;
    //float * Q_d;

    // initialise Uf_h
    for (int i = 0; i < nxf*nyf*4; i++) {
        Uf_h[i] = 0.0;
    }

    // set device
    cudaSetDevice(rank);

    // allocate memory on device
    cudaMalloc((void**)&beta_d, 2*sizeof(float));
    cudaMalloc((void**)&gamma_up_d, 4*sizeof(float));
    cudaMalloc((void**)&Uc_d, nx*ny*3*sizeof(float));
    cudaMalloc((void**)&Uf_d, nxf*nyf*4*sizeof(float));
    //cudaMalloc((void**)&Q_d, nlayers*nx*ny*sizeof(float));

    // copy stuff to GPU
    cudaMemcpy(beta_d, beta, 2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma_up_d, gamma_up, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Uc_d, Uc_h, nx*ny*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Uf_d, Uf_h, nxf*nyf*4*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(rho_d, rho, nlayers*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(Q_d, Q, nlayers*nx*ny*sizeof(float), cudaMemcpyHostToDevice);

    float *Upc_d, *Uc_half_d, *Upf_d, *Uf_half_d;//*sum_phs_d;
    cudaMalloc((void**)&Upc_d, nx*ny*3*sizeof(float));
    cudaMalloc((void**)&Uc_half_d, nx*ny*3*sizeof(float));
    cudaMalloc((void**)&Upf_d, nxf*nyf*4*sizeof(float));
    cudaMalloc((void**)&Uf_half_d, nxf*nyf*4*sizeof(float));
    //cudaMalloc((void**)&sum_phs_d, nlayers*nx*ny*sizeof(float));

    float *qx_p_d, *qx_m_d, *qy_p_d, *qy_m_d, *fx_p_d, *fx_m_d, *fy_p_d, *fy_m_d;
    float *Upc_h = new float[nx*ny*3];
    float *Fc_h = new float[nx*ny*3];

    float *Upf_h = new float[nxf*nyf*4];
    float *Ff_h = new float[nxf*nyf*4];

    // initialise
    for (int j = 0; j < nxf*nyf*4; j++) {
        Upf_h[j] = 0.0;
    }

    int grid_size = max(nx*ny*3, nxf*nyf*4);

    cudaMalloc((void**)&qx_p_d, grid_size*sizeof(float));
    cudaMalloc((void**)&qx_m_d, grid_size*sizeof(float));
    cudaMalloc((void**)&qy_p_d, grid_size*sizeof(float));
    cudaMalloc((void**)&qy_m_d, grid_size*sizeof(float));
    cudaMalloc((void**)&fx_p_d, grid_size*sizeof(float));
    cudaMalloc((void**)&fx_m_d, grid_size*sizeof(float));
    cudaMalloc((void**)&fy_p_d, grid_size*sizeof(float));
    cudaMalloc((void**)&fy_m_d, grid_size*sizeof(float));

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
            int ndims = 4;
            hsize_t dims[] = {hsize_t((nt+1)/dprint+1), hsize_t(ny), hsize_t(nx), 3};
            file_space = H5Screate_simple(ndims, dims, NULL);

            hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_layout(plist, H5D_CHUNKED);
            hsize_t chunk_dims[] = {1, hsize_t(ny), hsize_t(nx), 3};
            H5Pset_chunk(plist, ndims, chunk_dims);

            // create dataset
            dset = H5Dcreate(outFile, "SwerveOutput", H5T_NATIVE_FLOAT, file_space, H5P_DEFAULT, plist, H5P_DEFAULT);

            H5Pclose(plist);

            // make a memory dataspace
            mem_space = H5Screate_simple(ndims, chunk_dims, NULL);

            // select a hyperslab
            file_space = H5Dget_space(dset);
            hsize_t start[] = {0, 0, 0, 0};
            hsize_t hcount[] = {1, hsize_t(ny), hsize_t(nx), 3};
            H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, hcount, NULL);
            // write to dataset
            printf("Printing t = %i\n", 0);
            H5Dwrite(dset, H5T_NATIVE_FLOAT, mem_space, file_space, H5P_DEFAULT, Uc_h);
            // close file dataspace
            H5Sclose(file_space);
        }

        // prolong to fine grid
        //prolong_grid(Uc_h, Uf_h, nx, ny, nxf, nyf, dx, dy, gamma_up,
                     //rho, gamma, matching_indices);

        // main loop
        for (int t = 0; t < nt; t++) {
            cout << "Evolving t = " << t << '\n';
            //printf("t = %i\n", t);
            // offset by kernels in previous
            //int kx_offset = 0;
            int ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;
            //int k_offset = 0;
            //if (rank > 0) {
                //k_offset = cumulative_kernels[rank - 1];
            //}

            /*cout << "Coarse grid: \n";
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    cout << '(' << x << ',' << y << "): " << Uc_h[((y*nx)+x)*3] << '\n';
                }
            }*/

            // prolong to fine grid
            prolong_grid(Uc_h, Uf_h, nx, ny, nxf, nyf, dx, dy, gamma_up,
                         rho, gamma, matching_indices);


            // enforce boundaries
            //bcs_fv(Up_h, nx, ny, nlayers, ng);
            if (n_processes == 1) {
                bcs_fv(Uf_h, nxf, nyf, ng, 4);
            } else {
                int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
                bcs_mpi(Uf_h, nxf, nyf, 4, ng, comm, status, rank, n_processes, y_size);
            }

            /*for (int y = 0; y < nyf; y++) {
                for (int x = 0; x < nxf; x++) {
                    cout << '(' << x << ',' << y << "): " << Uf_h[((y*nxf)+x)*4+3] << '\n';
                }
            }*/

            cudaMemcpy(Uf_d, Uf_h, nxf*nyf*4*sizeof(float), cudaMemcpyHostToDevice);

            // evolve fine grid through two subcycles
            for (int i = 0; i < 2; i++) {

                /*for (int y = 0; y < nyf; y++) {
                    for (int x = 0; x < nxf; x++) {
                        cout << '(' << x << ',' << y << "): " << Uf_h[((y*nxf)+x)*4+3] << '\n';
                    }
                }*/
                rk3(kernels, threads, blocks, cumulative_kernels,
                        beta_d, gamma_up_d, Uf_d, Uf_half_d, Upf_d,
                        qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                        fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                        nxf, nyf, 4, ng, alpha, gamma,
                        dx*0.5, dy*0.5, dt*0.5, Upf_h, Ff_h, Uf_h,
                        comm, status, rank, n_processes,
                        h_compressible_fluxes);

                cudaDeviceSynchronize();

                //for (int j = 0; j < nxf*nyf*4; j++) {
                    //Uf_h[j] = Upf_h[j];
                //}

                // if not last step, copy to device
                if (i < 1) {
                    cudaMemcpy(Uf_d, Uf_h, nxf*nyf*4*sizeof(float), cudaMemcpyHostToDevice);
                }
            }

            /*for (int y = 0; y < nyf; y++) {
                for (int x = 0; x < nxf; x++) {
                    cout << '(' << x << ',' << y << "): " << Upf_h[((y*nxf)+x)*4+3] << '\n';
                }
            }*/

            // restrict to coarse grid
            restrict_grid(Uc_h, Uf_h, nx, ny, nxf, nyf, matching_indices,
                          rho, gamma, gamma_up);

            cudaMemcpy(Uc_d, Uc_h, nx*ny*3*sizeof(float), cudaMemcpyHostToDevice);

            rk3(kernels, threads, blocks, cumulative_kernels,
                beta_d, gamma_up_d, Uc_d, Uc_half_d, Upc_d,
                qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                nx, ny, 3, ng, alpha, gamma,
                dx, dy, dt, Upc_h, Fc_h, Uc_h,
                comm, status, rank, n_processes,
                h_shallow_water_fluxes);

            //for (int j = 0; j < nx*ny*3; j++) {
                //Uc_h[j] = Upc_h[j];
            //}

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

            cudaError_t err = cudaGetLastError();

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
                        float * buf = new float[nx*ny*3];
                        int tag = 0;
                        for (int source = 1; source < n_processes; source++) {
                            //printf("Receiving from rank %i\n", source);
                            mpi_err = MPI_Recv(buf, nx*ny*3, MPI_FLOAT, source, tag, comm, &status);

                            check_mpi_error(mpi_err);

                            // copy data back to grid
                            ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;
                            // cheating slightly and using the fact that are moving from bottom to top to make calculations a bit easier.
                            for (int y = ky_offset; y < ny; y++) {
                                for (int x = 0; x < nx; x++) {
                                    for (int i = 0; i < 3; i++) {
                                        Uc_h[(y * nx + x) * 3 + i] = buf[(y * nx + x) * 3 + i];
                                    }
                                }
                            }
                        }

                        delete[] buf;
                    }

                    // receive data from other processes and copy to grid

                    // select a hyperslab
                    file_space = H5Dget_space(dset);
                    hsize_t start[] = {hsize_t((t+1)/dprint), 0, 0, 0};
                    hsize_t hcount[] = {1, hsize_t(ny), hsize_t(nx), 3};
                    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, hcount, NULL);
                    // write to dataset
                    H5Dwrite(dset, H5T_NATIVE_FLOAT, mem_space, file_space, H5P_DEFAULT, Uc_h);
                    // close file dataspae
                    H5Sclose(file_space);
                } else { // send data to rank 0
                    //printf("Rank %i sending\n", rank);
                    int tag = 0;
                    mpi_err = MPI_Ssend(Uc_h, ny*nx*3, MPI_FLOAT, 0, tag, comm);
                    check_mpi_error(mpi_err);
                }

            }
        }

        if (rank == 0) {
            H5Sclose(mem_space);
            H5Fclose(outFile);
        }

    } else { // don't print

        // prolong to fine grid
        prolong_grid(Uc_h, Uf_h, nx, ny, nxf, nyf, dx, dy, gamma_up,
                     rho, gamma, matching_indices);

        for (int t = 0; t < nt; t++) {

            //int kx_offset = 0;
            //int ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

            // prolong to fine grid
            prolong_grid(Uc_h, Uf_h, nx, ny, nxf, nyf, dx, dy, gamma_up,
                         rho, gamma, matching_indices);

            // evolve fine grid through two subcycles
            for (int i = 0; i < 2; i++) {
                rk3(kernels, threads, blocks, cumulative_kernels,
                        beta_d, gamma_up_d, Uf_d, Uf_half_d, Upf_d,
                        qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                        fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                        nxf, nyf, 4, ng, alpha, gamma,
                        dx*0.5, dy*0.5, dt*0.5, Upf_h, Ff_h, Uf_h,
                        comm, status, rank, n_processes,
                        h_compressible_fluxes);

                // if not last step, copy output array to input array
                if (i < 1) {
                    for (int j = 0; j < nxf*nyf*4; j++) {
                        Uf_h[j] = Upf_h[j];
                    }
                }
            }

            // restrict to coarse grid
            restrict_grid(Uc_h, Uf_h, nx, ny, nxf, nyf, matching_indices,
                          rho, gamma, gamma_up);

            rk3(kernels, threads, blocks, cumulative_kernels,
                beta_d, gamma_up_d, Uc_d, Uc_half_d, Upc_d,
                qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                nx, ny, 3, ng, alpha, gamma,
                dx, dy, dt, Upc_h, Fc_h, Uc_h,
                comm, status, rank, n_processes,
                h_shallow_water_fluxes);

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
            cudaMemcpy(Uc_h, Uc_d, nx*ny*3*sizeof(float), cudaMemcpyDeviceToHost);
            if (n_processes == 1) {
                bcs_fv(Uc_h, nx, ny, ng, 3);
            } else {
                int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
                bcs_mpi(Uc_h, nx, ny, 3, ng, comm, status, rank, n_processes, y_size);
            }
            cudaMemcpy(Uc_d, Uc_h, nx*ny*3*sizeof(float), cudaMemcpyHostToDevice);

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
    //cudaFree(rho_d);
    //cudaFree(Q_d);
    cudaFree(Upc_d);
    cudaFree(Uc_half_d);
    cudaFree(Upf_d);
    cudaFree(Uf_half_d);
    //cudaFree(sum_phs_d);

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
    delete[] Upc_h;
    delete[] Fc_h;
    delete[] Upf_h;
    delete[] Ff_h;
}


#endif
