#ifndef _GR_CUDA_KERNEL_H_
#define _GR_CUDA_KERNEL_H_

#include <stdio.h>
#include "H5Cpp.h"

using namespace std;

// prototypes

dim3 getNumKernels(int nx, int ny, int nlayers, int ng, int *maxBlocks, int *maxThreads);

void getNumBlocksAndThreads(int nx, int ny, int nlayers, int ng, int maxBlocks, int maxThreads, dim3 kernels, dim3 *blocks, dim3 *threads);

unsigned int nextPow2(unsigned int x);

void bcs_fv(float * grid, int nx, int ny, int nlayers, int ng);

void homogeneuous_fv(dim3 kernels, dim3 * threads, dim3 * blocks, float * beta_d, float * gamma_up_d,
       float * Un_d, float * F_d,
       float * qx_p_d, float * qx_m_d, float * qy_p_d, float * qy_m_d,
       float * fx_p_d, float * fx_m_d, float * fy_p_d, float * fy_m_d,
       int nx, int ny, int nlayers, float alpha,
       float dx, float dy, float dt);

void rk3_fv(dim3 kernels, dim3 * threads, dim3 * blocks,
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

dim3 getNumKernels(int nx, int ny, int nlayers, int ng, int *maxBlocks, int *maxThreads) {
    /*
    Return the number of kernels needed to run the problem given its size and the constraints of the GPU.
    */
    // won't actually use maxThreads - fix to account for the fact we want something square
    *maxThreads = nlayers * int(sqrt(float(*maxThreads)/nlayers)) * int(sqrt(*maxThreads/nlayers));
    *maxBlocks = int(sqrt(float(*maxBlocks))) * int(sqrt(float(*maxBlocks)));

    //int numBlocks = 0;
    //int numThreads = 0;

    dim3 kernels;

    // calculate number of kernels needed

    if (nx*ny*nlayers > *maxBlocks * *maxThreads) {
        kernels.x = int(ceil(float(nx-2*ng) / (sqrt(float(*maxThreads * *maxBlocks)/nlayers) - 2.0*ng)));
        kernels.y = int(ceil(float(ny-2*ng) / (sqrt(float(*maxThreads * *maxBlocks)/nlayers) - 2.0*ng)));

    } else {

        kernels.x = 1;
        kernels.y = 1;
    }

    return kernels;
}

void getNumBlocksAndThreads(int nx, int ny, int nlayers, int ng, int maxBlocks, int maxThreads, dim3 kernels, dim3 *blocks, dim3 *threads)
{
    /*
    Returns the number of blocks and threads required for each kernel given the size of the problem and the constraints of the device.
    */

    //get device capability, to avoid block/grid size exceed the upper bound
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    int total = nx*ny*nlayers;

    if ((kernels.x > 1) || (kernels.y > 1)) {
        // initialise
        threads[0].x = 0;
        threads[0].y = 0;
        blocks[0].x = 0;
        blocks[0].y = 0;


        for (int j = 0; j < (kernels.y-1); j++) {
            for (int i = 0; i < (kernels.x-1); i++) {
                threads[j*kernels.x + i].x = int(sqrt(float(maxThreads)/nlayers));
                threads[j*kernels.x + i].y = int(sqrt(float(maxThreads)/nlayers));
                threads[j*kernels.x + i].z = nlayers;

                blocks[j*kernels.x + i].x = int(sqrt(float(maxBlocks)));
                blocks[j*kernels.x + i].y = int(sqrt(float(maxBlocks)));
                blocks[j*kernels.x + i].z = 1;
            }


        }
        // kernels.x-1
        int nx_remaining = nx - threads[0].x * blocks[0].x * (kernels.x - 1);
        for (int j = 0; j < (kernels.y-1); j++) {


            threads[j*kernels.x + kernels.x-1].y =
                int(sqrt(float(maxThreads)/nlayers));
            threads[j*kernels.x + kernels.x-1].z = nlayers;

            threads[j*kernels.x + kernels.x-1].x =
                (nx_remaining < threads[j*kernels.x + kernels.x-1].y) ? nx_remaining : threads[j*kernels.x + kernels.x-1].y;

            blocks[j*kernels.x + kernels.x-1].x = int(ceil(float(nx_remaining) /
                float(threads[j*kernels.x + kernels.x-1].x)));
            blocks[j*kernels.x + kernels.x-1].y = int(sqrt(float(maxBlocks)));
            blocks[j*kernels.x + kernels.x-1].z = 1;
        }

        // kernels.y-1
        int ny_remaining = ny - threads[0].y * blocks[0].y * (kernels.y - 1);
        for (int i = 0; i < (kernels.x-1); i++) {

            threads[(kernels.y-1)*kernels.x + i].x =
                int(sqrt(float(maxThreads)/nlayers));
            threads[(kernels.y-1)*kernels.x + i].y =
                (ny_remaining < threads[(kernels.y-1)*kernels.x + i].x) ? ny_remaining : threads[(kernels.y-1)*kernels.x + i].x;
            threads[(kernels.y-1)*kernels.x + i].z = nlayers;

            blocks[(kernels.y-1)*kernels.x + i].x = int(sqrt(float(maxBlocks)));
            blocks[(kernels.y-1)*kernels.x + i].y = int(ceil(float(ny_remaining) /
                float(threads[(kernels.y-1)*kernels.x + i].y)));
            blocks[(kernels.y-1)*kernels.x + i].z = 1;
        }

        // (kernels.x-1, kernels.y-1)
        threads[(kernels.y-1)*kernels.x + kernels.x-1].x =
            (nx_remaining < int(sqrt(float(maxThreads)/nlayers))) ? nx_remaining : int(sqrt(float(maxThreads/nlayers)));
        threads[(kernels.y-1)*kernels.x + kernels.x-1].y =
            (ny_remaining < int(sqrt(float(maxThreads)/nlayers))) ? ny_remaining : int(sqrt(float(maxThreads)/nlayers));
        threads[(kernels.y-1)*kernels.x + kernels.x-1].z = nlayers;

        blocks[(kernels.y-1)*kernels.x + kernels.x-1].x =
            int(ceil(float(nx_remaining) /
            float(threads[(kernels.y-1)*kernels.x + kernels.x-1].x)));
        blocks[(kernels.y-1)*kernels.x + kernels.x-1].y =
            int(ceil(float(ny_remaining) /
            float(threads[(kernels.y-1)*kernels.x + kernels.x-1].y)));
        blocks[(kernels.y-1)*kernels.x + kernels.x-1].z = 1;

    } else {

        int total_threads = (total < maxThreads*2) ? nextPow2((total + 1)/ 2) : maxThreads;
        threads[0].x = int(floor(sqrt(float(total_threads)/float(nlayers))));
        threads[0].y = int(floor(sqrt(float(total_threads)/float(nlayers))));
        threads[0].z = nlayers;
        total_threads = threads[0].x * threads[0].y * threads[0].z;
        int total_blocks = int(ceil(float(total) / float(total_threads)));

        //printf("total blocks: %i\n", total_blocks);

        blocks[0].x = int(ceil(sqrt(float(total_blocks)/float(nx*ny))*nx));
        blocks[0].y = int(ceil(sqrt(float(total_blocks)/float(nx*ny))*ny));
        blocks[0].z = 1;

        total_blocks = blocks[0].x * blocks[0].y;

        //printf("total blocks: %i\n", total_blocks);

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

__device__ void Jx(float * u, float * beta_d, float * gamma_up_d, float * jx, float alpha) {
    /*
    Calculate Jacobian in the x-direction.
    */

    float W = sqrt((u[1]*u[1] * gamma_up_d[0] +
                2.0 * u[1]* u[2] * gamma_up_d[1] +
                u[2]*u[2] * gamma_up_d[3]) / (u[0]*u[0]) + 1.0);
    //cout << "W = " << W << '\n';
    //cout << "u = " << u[0] << ' ' << u[1] << ' ' << u[2] << '\n';

    float ph = u[0] / W;
    float vx = u[1] / (u[0] * W); // u_down
    float vy = u[2] / (u[0] * W); // v_down

    float qx = vx * gamma_up_d[0] + vy * gamma_up_d[1] - beta_d[0]/alpha;

    float chi = 1.0 / (1.0 - vx*vx * W*W - vy*vy * W*W);

    jx[0*3+0] = qx/chi - vx;
    jx[0*3+1] = (1.0 + vy*vy*W*W)/W;
    jx[0*3+2] = -W * vx * vy;

    jx[1*3+0] = -2.0*pow(W,3)*vx*qx*(vx*vx + vy*vy) + ph*(1.0/W - W*vx*vx);
    jx[1*3+1] = qx * (1.0+W*W*vx*vx + W*W*vy*vy) + 0.5*ph*vx*(vy*vy*W*W-1.0);
    jx[1*3+2] = -vy*ph*(1.0 + 0.5*W*W*vx*vx);

    jx[2*3+0] = -W*vy*(2.0*W*W*qx*(vx*vx+vy*vy) + 0.5*ph*vx);
    jx[2*3+1] = 0.5*ph*vy*(1.0+vy*vy*W*W);
    jx[2*3+2] = qx*(1.0+W*W*vx*vx+W*W*vy*vy) - 0.5*ph*W*W*vx*vy*vy;

    for (int i = 0; i < 9; i++) {
        jx[i] *= chi;
    }
}

__device__ void Jy(float * u, float * beta_d, float * gamma_up_d, float * jy, float alpha) {
    /*
    Calculate Jacobian in the y-direction.
    */

    float W = sqrt((u[1]*u[1] * gamma_up_d[0] +
                2.0 * u[1]* u[2] * gamma_up_d[1] +
                u[2]*u[2] * gamma_up_d[3]) / (u[0]*u[0]) + 1.0);

    float ph = u[0] / W;
    float vx = u[1] / (u[0] * W); // u_down
    float vy = u[2] / (u[0] * W); // v_down

    float qy = vy * gamma_up_d[3] + vx * gamma_up_d[1] - beta_d[1]/alpha;

    float chi = 1.0 / (1.0 - vx*vx * W*W - vy*vy * W*W);

    jy[0] = qy/chi - vx;
    jy[1] = -W * vx * vy;
    jy[2] = (1.0 + vx*vx*W*W)/W;

    jy[1*3] = -W*vx*(2.0*W*W*qy*(vx*vx+vy*vy) + 0.5*ph*vy);
    jy[1*3+1] = qy*(1.0+W*W*vx*vx+W*W*vy*vy) - 0.5*ph*W*W*vx*vx*vy;
    jy[1*3+2] = 0.5*ph*vx*(1.0+vx*vx*W*W);

    jy[2*3+0] = -2.0*pow(W,3)*vy*qy*(vx*vx + vy*vy) + ph*(1.0/W - W*vy*vy);
    jy[2*3+1] = -vx*ph*(1.0 + 0.5*W*W*vy*vy);
    jy[2*3+2] = qy * (1.0+W*W*vx*vx + W*W*vy*vy) + 0.5*ph*vy*(vx*vx*W*W-1.0);

    for (int i = 0; i < 9; i++) {
        jy[i] *= chi;
    }

}

__device__ void calc_Q(float * U, float * rho_d, float * Q_d,
                       int nx, int ny, int nlayers,
                       int kx_offset, int ky_offset) {
    /*
    Calculate heating rate using equation 64 of Spitkovsky 2002.
    */
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

__global__ void evolve(float * beta_d, float * gamma_up_d,
                     float * Un_d, float * Up, float * U_half,
                     float * sum_phs, float * rho_d, float * Q_d,
                     float mu,
                     int nx, int ny, int nlayers, float alpha,
                     float dx, float dy, float dt,
                     int kx_offset, int ky_offset) {
    /*
    First part of evolution through one timestep.
    */
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int l = threadIdx.z;

    //if (x*y*l == 0) {
    //    printf("evolving\n");
    //}

    float *u, *A, *B, *A2, *B2, *AB;

    u = (float *) malloc(3*sizeof(float));
    A = (float *) malloc(9*sizeof(float));
    B = (float *) malloc(9*sizeof(float));
    A2 = (float *) malloc(9*sizeof(float));
    B2 = (float *) malloc(9*sizeof(float));
    AB = (float *) malloc(9*sizeof(float));

    //if (x*y*l == 0) {
        //printf("evolving\n");
    //}

    float d, e, f, g, h;
    float * beta;
    beta = (float *) malloc(2*sizeof(float));

    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1)) && (l < nlayers)) {

        for (int i = 0; i < 4; i++) {
            u[i] = Un_d[((y * nx + x) * nlayers + l)*4+i];
        }
        beta[0] = beta_d[(y * nx + x) * 2];
        beta[1] = beta_d[(y * nx + x) * 2 + 1];

        Jx(u, beta, gamma_up_d, A, alpha);
        Jy(u, beta, gamma_up_d, B, alpha);

        // matrix multiplication
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                A2[i*3+j] = 0;
                B2[i*3+j] = 0;
                AB[i*3+j] = 0;
                for (int k = 0; k < 3; k++) {
                    A2[i*3+j] += A[i*3+k] * A[k*3+j];
                    B2[i*3+j] += B[i*3+k] * B[k*3+j];
                    AB[i*3+j] += A[i*3+k] * B[k*3+j];
                }
            }
        }

        // going to do matrix calculations to calculate different terms
        for (int i = 0; i < 3; i ++) {
            d = 0;
            e = 0;
            f = 0;
            g = 0;
            h = 0;
            for (int j = 0; j < 4; j++) {
                d += A[i*3+j] *
                    (Un_d[((y * nx + x+1) * nlayers + l)*4+j] -
                    Un_d[((y * nx + x-1) * nlayers + l)*4+j]);

                e += B[i*3+j] *
                    (Un_d[(((y+1) * nx + x) * nlayers + l)*4+j] -
                    Un_d[(((y-1) * nx + x) * nlayers + l)*4+j]);

                f += A2[i*3+j] *
                    (Un_d[((y * nx + x+1) * nlayers + l)*4+j] - 2.0 *
                    Un_d[((y * nx + x) * nlayers + l)*4+j] +
                    Un_d[((y * nx + x-1) * nlayers + l)*4+j]);

                g += B2[i*3+j] *
                    (Un_d[(((y+1) * nx + x) * nlayers + l)*4+j] - 2.0 *
                    Un_d[((y * nx + x) * nlayers + l)*4+j] +
                    Un_d[(((y-1) * nx + x) * nlayers + l)*4+j]);

                h += AB[i*3+j] *
                    (Un_d[(((y+1) * nx + x+1) * nlayers + l)*4+j] -
                    Un_d[(((y-1) * nx + x+1) * nlayers + l)*4+j] -
                    Un_d[(((y+1) * nx + x-1) * nlayers + l)*4+j] +
                    Un_d[(((y-1) * nx + x-1) * nlayers + l)*4+j]);
            }

            Up[((y * nx + x) * nlayers + l) * 4 + i] = u[i] + alpha * (
                    -0.5 * dt/dx * d -
                    0.5 * dt/dy * e +
                    0.5 * dt*dt/(dx*dx) * f +
                    0.5 * dt*dt/(dy*dy) * g -
                    0.25 * dt*dt/(dx*dy) * h);

        }

        //if (isnan(Up[((y * nx + x) * nlayers + l)*4])) {
            //printf("Up is %f! ", Up[((y * nx + x) * nlayers + l)*4]);
        //}


    }

    free(u);
    free(A);
    free(B);
    free(A2);
    free(B2);
    free(AB);
    free(beta);

    __syncthreads();

    // enforce boundary conditions
    bcs(Up, nx, ny, nlayers, kx_offset, ky_offset);

    // copy to U_half
    if ((x < nx) && (y < ny) && (l < nlayers)) {
        for (int i = 0; i < 4; i++) {
            U_half[((y * nx + x) * nlayers + l)*4+i] =
                Up[((y * nx + x) * nlayers + l)*4+i];
        }
    }

    float W = 1.0;

    // do source terms
    if ((x < nx) && (y < ny) && (l < nlayers)) {

        //ph[l] = U_half[((y * nx + x) * nlayers + l)*4];
        //Sx[l] = U_half[((y * nx + x) * nlayers + l)*4+1];
        //Sy[l] = U_half[((y * nx + x) * nlayers + l)*4+2];
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

        //if (isnan(U_half[((y * nx + x) * nlayers + l)*4])) {
            //printf("ph is %f! ", U_half[((y * nx + x) * nlayers + l)*4]);
        //}
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
            float fx_m = 0.5 * (
                fx_plus_half[((y * nx + x-1) * nlayers + l) * 4 + i] +
                fx_minus_half[((y * nx + x) * nlayers + l) * 4 + i] +
                qx_plus_half[((y * nx + x-1) * nlayers + l) * 4 + i] -
                qx_minus_half[((y * nx + x) * nlayers + l) * 4 + i]);

            float fx_p = 0.5 * (
                fx_plus_half[((y * nx + x) * nlayers + l) * 4 + i] +
                fx_minus_half[((y * nx + x+1) * nlayers + l) * 4 + i] +
                qx_plus_half[((y * nx + x) * nlayers + l) * 4 + i] -
                qx_minus_half[((y * nx + x+1) * nlayers + l) * 4 + i]);

            float fy_m = 0.5 * (
                fy_plus_half[(((y-1) * nx + x) * nlayers + l) * 4 + i] +
                fy_minus_half[((y * nx + x) * nlayers + l) * 4 + i] +
                qy_plus_half[(((y-1) * nx + x) * nlayers + l) * 4 + i] -
                qy_minus_half[((y * nx + x) * nlayers + l) * 4 + i]);


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
    calc_Q(Up, rho_d, Q_d, nx, ny, nlayers, kx_offset, ky_offset);

    float W = 1.0;


    // do source terms
    if ((x < nx) && (y < ny) && (l < nlayers)) {

        //ph[l] = U_half[((y * nx + x) * nlayers + l)*4];
        //Sx[l] = U_half[((y * nx + x) * nlayers + l)*4+1];
        //Sy[l] = U_half[((y * nx + x) * nlayers + l)*4+2];
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

        //if (isnan(U_half[((y * nx + x) * nlayers + l)*4])) {
            //printf("ph is %f! ", U_half[((y * nx + x) * nlayers + l)*4]);
        //}
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
                     int nx, int ny, int nlayers, float alpha,
                     float dx, float dy, float dt,
                     int kx_offset, int ky_offset) {
    /*
    Adds buoyancy terms.
    */
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int l = threadIdx.z;

    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1)) && (l < nlayers)) {

        float a = dt * alpha *
            U_half[((y * nx + x) * nlayers + l)*4] * (0.5 / dx) * (sum_phs[(y * nx + x+1) * nlayers + l] -
            sum_phs[(y * nx + x-1) * nlayers + l]);

        if (abs(a) < 0.9 * dx / dt) {
            //printf("a is %f! ", a);
            Up[((y * nx + x) * nlayers + l)*4+1] = Up[((y * nx + x) * nlayers + l)*4+1] - a;
        }

        a = dt * alpha *
            U_half[((y * nx + x) * nlayers + l)*4] * (0.5 / dy) *
            (sum_phs[((y+1) * nx + x) * nlayers + l] -
             sum_phs[((y-1) * nx + x) * nlayers + l]);

        if (abs(a) < 0.9 * dy / dt) {
            //printf("a is %f! ", a);
            Up[((y * nx + x) * nlayers + l)*4+2] = Up[((y * nx + x) * nlayers + l)*4+2] - a;
        }


    }

    __syncthreads();

    bcs(Up, nx, ny, nlayers, kx_offset, ky_offset);

    // copy back to grid
    if ((x < nx) && (y < ny) && (l < nlayers)) {
        for (int i = 0; i < 4; i++) {
            Un_d[((y * nx + x) * nlayers + l)*4+i] =
                Up[((y * nx + x) * nlayers + l)*4+i];
        }
    }


}

void homogeneuous_fv(dim3 kernels, dim3 * threads, dim3 * blocks, float * beta_d, float * gamma_up_d,
       float * Un_d, float * F_d,
       float * qx_p_d, float * qx_m_d, float * qy_p_d, float * qy_m_d,
       float * fx_p_d, float * fx_m_d, float * fy_p_d, float * fy_m_d,
       int nx, int ny, int nlayers, float alpha,
       float dx, float dy, float dt) {
    /*
    Solves the homogeneous part of the equation (ie the bit without source terms).
    */

    int kx_offset = 0;
    int ky_offset = 0;

    for (int j = 0; j < kernels.y; j++) {
       kx_offset = 0;
       for (int i = 0; i < kernels.x; i++) {
           evolve_fv<<<blocks[j * kernels.x + i], threads[j * kernels.x + i]>>>(beta_d, gamma_up_d, Un_d,
                  qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                  fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                  nx, ny, nlayers, alpha,
                  dx, dy, dt, kx_offset, ky_offset);
          kx_offset += blocks[j * kernels.x + i].x * threads[j * kernels.x + i].x;
       }
       ky_offset += blocks[j * kernels.x].y * threads[j * kernels.x].y;
    }

    ky_offset = 0;

    for (int j = 0; j < kernels.y; j++) {
       kx_offset = 0;
       for (int i = 0; i < kernels.x; i++) {
           evolve_fv_fluxes<<<blocks[j * kernels.x + i],
                              threads[j * kernels.x + i]>>>(
                  F_d,
                  qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                  fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                  nx, ny, nlayers, alpha,
                  dx, dy, dt, kx_offset, ky_offset);

           kx_offset += blocks[j * kernels.x + i].x * threads[j * kernels.x + i].x;
       }
       ky_offset += blocks[j * kernels.x].y * threads[j * kernels.x].y;
    }


}

void rk3_fv(dim3 kernels, dim3 * threads, dim3 * blocks,
       float * beta_d, float * gamma_up_d, float * Un_d,
       float * F_d, float * Up_d,
       float * qx_p_d, float * qx_m_d, float * qy_p_d, float * qy_m_d,
       float * fx_p_d, float * fx_m_d, float * fy_p_d, float * fy_m_d,
       int nx, int ny, int nlayers, int ng, float alpha,
       float dx, float dy, float dt,
       float * Up_h, float * F_h, float * Un_h) {
    /*
    Integrates the homogeneous part of the ODE in time using RK3.
    */

    // u1 = un + dt * F(un)
    homogeneuous_fv(kernels, threads, blocks,
          beta_d, gamma_up_d, Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d,
          nx, ny, nlayers, alpha,
          dx, dy, dt);

    // copy back flux
    cudaMemcpy(F_h, F_d, nx*ny*nlayers*4*sizeof(float), cudaMemcpyDeviceToHost);
    bcs_fv(F_h, nx, ny, nlayers, ng);

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
    bcs_fv(Up_h, nx, ny, nlayers, ng);
    cudaMemcpy(Un_d, Up_h, nx*ny*nlayers*4*sizeof(float), cudaMemcpyHostToDevice);

    // u2 = 0.25 * (3*un + u1 + dt*F(u1))
    homogeneuous_fv(kernels, threads, blocks,
          beta_d, gamma_up_d, Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d,
          nx, ny, nlayers, alpha,
          dx, dy, dt);

    // copy back flux
    cudaMemcpy(F_h, F_d, nx*ny*nlayers*4*sizeof(float), cudaMemcpyDeviceToHost);
    bcs_fv(F_h, nx, ny, nlayers, ng);

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
    bcs_fv(Up_h, nx, ny, nlayers, ng);
    cudaMemcpy(Un_d, Up_h, nx*ny*nlayers*4*sizeof(float), cudaMemcpyHostToDevice);

    // un+1 = (1/3) * (un + 2*u2 + 2*dt*F(u2))
    homogeneuous_fv(kernels, threads, blocks,
          beta_d, gamma_up_d, Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d,
          nx, ny, nlayers, alpha,
          dx, dy, dt);

    // copy back flux
    cudaMemcpy(F_h, F_d, nx*ny*nlayers*4*sizeof(float), cudaMemcpyDeviceToHost);
    bcs_fv(F_h, nx, ny, nlayers, ng);

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
    bcs_fv(Up_h, nx, ny, nlayers, ng);

    cudaMemcpy(Up_d, Up_h, nx*ny*nlayers*4*sizeof(float), cudaMemcpyHostToDevice);

}


void cuda_run(float * beta, float * gamma_up, float * Un_h,
         float * rho, float * Q, float mu, int nx, int ny, int nlayers, int ng,
         int nt, float alpha, float dx, float dy, float dt, int dprint, char * filename) {
    /*
    Evolve system through nt timesteps, saving data to filename every dprint timesteps.
    */

    bool finite_volume = true;

    // set up GPU stuff
    int count;
    cudaGetDeviceCount(&count);

    //int size = 3 * nx * ny * nlayers;
    int maxThreads = 256;
    int maxBlocks = 256; //64;

    dim3 kernels = getNumKernels(nx, ny, nlayers, ng, &maxBlocks, &maxThreads);

    dim3 *blocks = new dim3[kernels.x*kernels.y];
    dim3 *threads = new dim3[kernels.x*kernels.y];

    getNumBlocksAndThreads(nx, ny, nlayers, ng, maxBlocks, maxThreads, kernels, blocks, threads);
    //int numBlocks = blocks.x * blocks.y * blocks.z;

    printf("kernels: (%i, %i)\n", kernels.x, kernels.y);

    for (int i = 0; i < kernels.x*kernels.y; i++) {
        printf("blocks: (%i, %i, %i) , threads: (%i, %i, %i)\n",
               blocks[i].x, blocks[i].y, blocks[i].z,
               threads[i].x, threads[i].y, threads[i].z);
    }

    // copy
    float * beta_d;
    float * gamma_up_d;
    float * Un_d;
    float * rho_d;
    float * Q_d;

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

    if (finite_volume) {
        cudaMalloc((void**)&qx_p_d, nlayers*nx*ny*4*sizeof(float));
        cudaMalloc((void**)&qx_m_d, nlayers*nx*ny*4*sizeof(float));
        cudaMalloc((void**)&qy_p_d, nlayers*nx*ny*4*sizeof(float));
        cudaMalloc((void**)&qy_m_d, nlayers*nx*ny*4*sizeof(float));
        cudaMalloc((void**)&fx_p_d, nlayers*nx*ny*4*sizeof(float));
        cudaMalloc((void**)&fx_m_d, nlayers*nx*ny*4*sizeof(float));
        cudaMalloc((void**)&fy_p_d, nlayers*nx*ny*4*sizeof(float));
        cudaMalloc((void**)&fy_m_d, nlayers*nx*ny*4*sizeof(float));
    }

    if (strcmp(filename, "na") != 0) {

        // create file
        hid_t outFile = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        // create dataspace
        int ndims = 5;
        hsize_t dims[] = {hsize_t((nt+1)/dprint+1), hsize_t(ny), hsize_t(nx), hsize_t(nlayers), 4};
        hid_t file_space = H5Screate_simple(ndims, dims, NULL);

        hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_layout(plist, H5D_CHUNKED);
        hsize_t chunk_dims[] = {1, hsize_t(ny), hsize_t(nx), hsize_t(nlayers), 4};
        H5Pset_chunk(plist, ndims, chunk_dims);

        // create dataset
        hid_t dset = H5Dcreate(outFile, "SwerveOutput", H5T_NATIVE_FLOAT, file_space, H5P_DEFAULT, plist, H5P_DEFAULT);

        H5Pclose(plist);

        // make a memory dataspace
        hid_t mem_space = H5Screate_simple(ndims, chunk_dims, NULL);

        // select a hyperslab
        //printf("hyperslab selection\n");
        file_space = H5Dget_space(dset);
        hsize_t start[] = {0, 0, 0, 0, 0};
        hsize_t hcount[] = {1, hsize_t(ny), hsize_t(nx), hsize_t(nlayers), 4};
        H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, hcount, NULL);
        //printf("writing\n");
        // write to dataset
        printf("Printing t = %i\n", 0);
        H5Dwrite(dset, H5T_NATIVE_FLOAT, mem_space, file_space, H5P_DEFAULT, Un_h);
        // close file dataspace
        //printf("wrote\n");
        H5Sclose(file_space);

        for (int t = 0; t < nt; t++) {

            int kx_offset = 0;
            int ky_offset = 0;

            if (finite_volume) {

                rk3_fv(kernels, threads, blocks,
                    beta_d, gamma_up_d, Un_d, U_half_d, Up_d,
                    qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                    fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                    nx, ny, nlayers, ng, alpha,
                    dx, dy, dt, Up_h, F_h, Un_h);

                for (int j = 0; j < kernels.y; j++) {
                    kx_offset = 0;
                    for (int i = 0; i < kernels.x; i++) {
                        evolve_fv_heating<<<blocks[j * kernels.x + i], threads[j * kernels.x + i]>>>(
                               gamma_up_d, Un_d,
                               Up_d, U_half_d,
                               qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                               fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                               sum_phs_d, rho_d, Q_d, mu,
                               nx, ny, nlayers, alpha,
                               dx, dy, dt, kx_offset, ky_offset);
                        kx_offset += blocks[j * kernels.x + i].x * threads[j * kernels.x + i].x;
                    }
                    ky_offset += blocks[j * kernels.x].y * threads[j * kernels.x].y;
                }


            } else {
                for (int j = 0; j < kernels.y; j++) {
                    kx_offset = 0;
                    for (int i = 0; i < kernels.x; i++) {
                        evolve<<<blocks[j * kernels.x + i], threads[j * kernels.x + i]>>>(beta_d, gamma_up_d, Un_d,
                               Up_d, U_half_d, sum_phs_d, rho_d, Q_d, mu,
                               nx, ny, nlayers, alpha,
                               dx, dy, dt, kx_offset, ky_offset);
                        kx_offset += blocks[j * kernels.x + i].x * threads[j * kernels.x + i].x;
                    }
                    ky_offset += blocks[j * kernels.x].y * threads[j * kernels.x].y;
                }
            }

            kx_offset = 0;
            ky_offset = 0;

            for (int j = 0; j < kernels.y; j++) {
                kx_offset = 0;
                for (int i = 0; i < kernels.x; i++) {
                    evolve2<<<blocks[j * kernels.x + i], threads[j * kernels.x + i]>>>(gamma_up_d, Un_d,
                           Up_d, U_half_d, sum_phs_d, rho_d, Q_d, mu,
                           nx, ny, nlayers, alpha,
                           dx, dy, dt, kx_offset, ky_offset);
                    kx_offset += blocks[j * kernels.x + i].x * threads[j * kernels.x + i].x;
                }
                ky_offset += blocks[j * kernels.x].y * threads[j * kernels.x].y;
            }

            cudaDeviceSynchronize();

            cudaError_t err = cudaGetLastError();

            if (err != cudaSuccess)
                printf("Error: %s\n", cudaGetErrorString(err));

            if (finite_volume) {
                // boundaries
                cudaMemcpy(Un_h, Un_d, nx*ny*nlayers*4*sizeof(float), cudaMemcpyDeviceToHost);
                bcs_fv(Un_h, nx, ny, nlayers, ng);
                cudaMemcpy(Un_d, Un_h, nx*ny*nlayers*4*sizeof(float), cudaMemcpyHostToDevice);
            }


            if ((t+1) % dprint == 0) {
                printf("Printing t = %i\n", t+1);

                if (finite_volume == false) {
                    // copy stuff back
                    cudaMemcpy(Un_h, Un_d, nx*ny*nlayers*4*sizeof(float), cudaMemcpyDeviceToHost);
                }

                // select a hyperslab
                file_space = H5Dget_space(dset);
                hsize_t start[] = {hsize_t((t+1)/dprint), 0, 0, 0, 0};
                hsize_t hcount[] = {1, hsize_t(ny), hsize_t(nx), hsize_t(nlayers), 4};
                H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, hcount, NULL);
                // write to dataset
                H5Dwrite(dset, H5T_NATIVE_FLOAT, mem_space, file_space, H5P_DEFAULT, Un_h);
                // close file dataspae
                H5Sclose(file_space);
            }
        }
        H5Sclose(mem_space);
        H5Fclose(outFile);

    } else { // don't print
        for (int t = 0; t < nt; t++) {



            //if (t % 50 == 0) {
                //printf("t =  %i\n", t);
            //}
            int kx_offset = 0;
            int ky_offset = 0;

            if (finite_volume) {

                rk3_fv(kernels, threads, blocks,
                    beta_d, gamma_up_d, Un_d, U_half_d, Up_d,
                    qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                    fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                    nx, ny, nlayers, ng, alpha,
                    dx, dy, dt, Up_h, F_h, Un_h);

                for (int j = 0; j < kernels.y; j++) {
                    kx_offset = 0;
                    for (int i = 0; i < kernels.x; i++) {
                        evolve_fv_heating<<<blocks[j * kernels.x + i], threads[j * kernels.x + i]>>>(
                               gamma_up_d, Un_d,
                               Up_d, U_half_d,
                               qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                               fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                               sum_phs_d, rho_d, Q_d, mu,
                               nx, ny, nlayers, alpha,
                               dx, dy, dt, kx_offset, ky_offset);
                        kx_offset += blocks[j * kernels.x + i].x * threads[j * kernels.x + i].x;
                    }
                    ky_offset += blocks[j * kernels.x].y * threads[j * kernels.x].y;
                }


            } else {
                for (int j = 0; j < kernels.y; j++) {
                    kx_offset = 0;
                    for (int i = 0; i < kernels.x; i++) {
                        evolve<<<blocks[j * kernels.x + i], threads[j * kernels.x + i]>>>(beta_d, gamma_up_d, Un_d,
                               Up_d, U_half_d, sum_phs_d, rho_d, Q_d, mu,
                               nx, ny, nlayers, alpha,
                               dx, dy, dt, kx_offset, ky_offset);
                        kx_offset += blocks[j * kernels.x + i].x * threads[j * kernels.x + i].x;
                    }
                    ky_offset += blocks[j * kernels.x].y * threads[j * kernels.x].y;
                }
            }

            kx_offset = 0;
            ky_offset = 0;

            for (int j = 0; j < kernels.y; j++) {
                kx_offset = 0;
                for (int i = 0; i < kernels.x; i++) {
                    evolve2<<<blocks[j * kernels.x + i], threads[j * kernels.x + i]>>>(gamma_up_d, Un_d,
                           Up_d, U_half_d, sum_phs_d, rho_d, Q_d, mu,
                           nx, ny, nlayers, alpha,
                           dx, dy, dt, kx_offset, ky_offset);
                    kx_offset += blocks[j * kernels.x + i].x * threads[j * kernels.x + i].x;
                }
                ky_offset += blocks[j * kernels.x].y * threads[j * kernels.x].y;
            }

            cudaDeviceSynchronize();

            cudaError_t err = cudaGetLastError();

            if (err != cudaSuccess)
                printf("Error: %s\n", cudaGetErrorString(err));

            if ((t+1) % dprint == 0) {
                printf("Printing t = %i\n", t+1);
                // copy stuff back
                cudaMemcpy(Un_h, Un_d, nx*ny*nlayers*4*sizeof(float), cudaMemcpyDeviceToHost);

            }
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

    if (finite_volume) {
        cudaFree(qx_p_d);
        cudaFree(qx_m_d);
        cudaFree(qy_p_d);
        cudaFree(qy_m_d);
        cudaFree(fx_p_d);
        cudaFree(fx_m_d);
        cudaFree(fy_p_d);
        cudaFree(fy_m_d);
    }

    //delete[] Un_h;
    delete[] threads;
    delete[] blocks;

    delete[] Up_h;
    delete[] F_h;
}


#endif
