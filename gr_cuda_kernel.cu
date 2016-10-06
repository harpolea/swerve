#ifndef _GR_CUDA_KERNEL_H_
#define _GR_CUDA_KERNEL_H_

#include <stdio.h>
#include "H5Cpp.h"

using namespace std;

dim3 getNumKernels(int nx, int ny, int nlayers, int *maxBlocks, int *maxThreads);

void getNumBlocksAndThreads(int nx, int ny, int nlayers, int maxBlocks, int maxThreads, dim3 kernels, dim3 *blocks, dim3 *threads);

unsigned int nextPow2(unsigned int x);

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

dim3 getNumKernels(int nx, int ny, int nlayers, int *maxBlocks, int *maxThreads) {
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
        kernels.x = int(ceil(float(nx-2) / (sqrt(float(*maxThreads * *maxBlocks)/nlayers) - 2.0)));
        kernels.y = int(ceil(float(ny-2) / (sqrt(float(*maxThreads * *maxBlocks)/nlayers) - 2.0)));

    } else {

        kernels.x = 1;
        kernels.y = 1;
    }

    return kernels;
}

void getNumBlocksAndThreads(int nx, int ny, int nlayers, int maxBlocks, int maxThreads, dim3 kernels, dim3 *blocks, dim3 *threads)
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
        for (int i = 0; i < 3; i++) {
            if (x == 0) {
                grid[((y * nx) * nlayers + l)*3+i] = grid[((y * nx + 1) * nlayers + l)*3+i];
            } else if (x == (nx-1)) {
                grid[((y * nx + (nx-1)) * nlayers + l)*3+i] = grid[((y * nx + (nx-2)) * nlayers + l)*3+i];
            } else if (y == 0) {
                grid[(x * nlayers + l)*3+i] = grid[((nx + x) * nlayers + l)*3+i];
            } else if (y == (ny-1)) {
                grid[(((ny-1) * nx + x) * nlayers + l)*3+i] = grid[(((ny-2) * nx + x) * nlayers + l)*3+i];
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

    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1)) && (l < nlayers)) {

        for (int i = 0; i < 3; i++) {
            u[i] = Un_d[((y * nx + x) * nlayers + l)*3+i];
        }

        Jx(u, beta_d, gamma_up_d, A, alpha);
        Jy(u, beta_d, gamma_up_d, B, alpha);

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
            for (int j = 0; j < 3; j++) {
                d += A[i*3+j] *
                    (Un_d[((y * nx + x+1) * nlayers + l)*3+j] -
                    Un_d[((y * nx + x-1) * nlayers + l)*3+j]);

                e += B[i*3+j] *
                    (Un_d[(((y+1) * nx + x) * nlayers + l)*3+j] -
                    Un_d[(((y-1) * nx + x) * nlayers + l)*3+j]);

                f += A2[i*3+j] *
                    (Un_d[((y * nx + x+1) * nlayers + l)*3+j] - 2.0 *
                    Un_d[((y * nx + x) * nlayers + l)*3+j] +
                    Un_d[((y * nx + x-1) * nlayers + l)*3+j]);

                g += B2[i*3+j] *
                    (Un_d[(((y+1) * nx + x) * nlayers + l)*3+j] - 2.0 *
                    Un_d[((y * nx + x) * nlayers + l)*3+j] +
                    Un_d[(((y-1) * nx + x) * nlayers + l)*3+j]);

                h += AB[i*3+j] *
                    (Un_d[(((y+1) * nx + x+1) * nlayers + l)*3+j] -
                    Un_d[(((y-1) * nx + x+1) * nlayers + l)*3+j] -
                    Un_d[(((y+1) * nx + x-1) * nlayers + l)*3+j] +
                    Un_d[(((y-1) * nx + x-1) * nlayers + l)*3+j]);
            }

            Up[((y * nx + x) * nlayers + l) * 3 + i] = u[i] + alpha * (
                    -0.5 * dt/dx * d -
                    0.5 * dt/dy * e +
                    0.5 * dt*dt/(dx*dx) * f +
                    0.5 * dt*dt/(dy*dy) * g -
                    0.25 * dt*dt/(dx*dy) * h);

        }

        //if (isnan(Up[((y * nx + x) * nlayers + l)*3])) {
            //printf("Up is %f! ", Up[((y * nx + x) * nlayers + l)*3]);
        //}


    }

    free(u);
    free(A);
    free(B);
    free(A2);
    free(B2);
    free(AB);

    __syncthreads();

    // enforce boundary conditions
    bcs(Up, nx, ny, nlayers, kx_offset, ky_offset);

    // copy to U_half
    if ((x < nx) && (y < ny) && (l < nlayers)) {
        for (int i = 0; i < 3; i++) {
            U_half[((y * nx + x) * nlayers + l)*3+i] =
                Up[((y * nx + x) * nlayers + l)*3+i];
        }
    }

    float W = 1.0;

    // do source terms
    if ((x < nx) && (y < ny) && (l < nlayers)) {

        //ph[l] = U_half[((y * nx + x) * nlayers + l)*3];
        //Sx[l] = U_half[((y * nx + x) * nlayers + l)*3+1];
        //Sy[l] = U_half[((y * nx + x) * nlayers + l)*3+2];
        W = sqrt(float((U_half[((y * nx + x) * nlayers + l)*3+1] *
            U_half[((y * nx + x) * nlayers + l)*3+1] * gamma_up_d[0] +
            2.0 * U_half[((y * nx + x) * nlayers + l)*3+1] *
            U_half[((y * nx + x) * nlayers + l)*3+2] *
            gamma_up_d[1] +
            U_half[((y * nx + x) * nlayers + l)*3+2] *
            U_half[((y * nx + x) * nlayers + l)*3+2] *
            gamma_up_d[3]) /
            (U_half[((y * nx + x) * nlayers + l)*3] *
            U_half[((y * nx + x) * nlayers + l)*3]) + 1.0));

        //if (isnan(U_half[((y * nx + x) * nlayers + l)*3])) {
            //printf("ph is %f! ", U_half[((y * nx + x) * nlayers + l)*3]);
        //}
        U_half[((y * nx + x) * nlayers + l)*3] /= W;

    }

    __syncthreads();

    if ((x < nx) && (y < ny) && (l < nlayers)) {

        sum_phs[(y * nx + x) * nlayers + l] = 0.0;


        float sum_qs = 0.0;
        float deltaQx = 0.0;
        float deltaQy = 0.0;

        if (l < (nlayers - 1)) {
            sum_qs += -rho_d[l+1] / rho_d[l] * abs(Q_d[(y * nx + x) * nlayers + l+1] - Q_d[(y * nx + x) * nlayers + l]);
            deltaQx = rho_d[l+1] / rho_d[l] *
                (max(float(0.0), Q_d[(y * nx + x) * nlayers + l] - Q_d[(y * nx + x) * nlayers + l+1]) + mu) *
                (U_half[((y * nx + x) * nlayers + l)*3+1] -
                 U_half[((y * nx + x) * nlayers + (l+1))*3+1]) /
                 U_half[((y * nx + x) * nlayers + l)*3];
            deltaQy = rho_d[l+1] / rho_d[l] *
                (max(float(0.0), Q_d[(y * nx + x) * nlayers + l] - Q_d[(y * nx + x) * nlayers + l+1]) + mu) *
                (U_half[((y * nx + x) * nlayers + l)*3+2] -
                 U_half[((y * nx + x) * nlayers + (l+1))*3+2]) /
                 U_half[((y * nx + x) * nlayers + l)*3];
        }
        if (l > 0) {
            sum_qs += abs(Q_d[(y * nx + x) * nlayers + l] - Q_d[(y * nx + x) * nlayers + l-1]);
            deltaQx = (max(float(0.0), Q_d[(y * nx + x) * nlayers + l] - Q_d[(y * nx + x) * nlayers + l-1]) + mu) *
                (U_half[((y * nx + x) * nlayers + l)*3+1] -
                 U_half[((y * nx + x) * nlayers + l-1)*3+1]) /
                 U_half[((y * nx + x) * nlayers + l)*3];
            deltaQy = (max(float(0.0), Q_d[(y * nx + x) * nlayers + l] - Q_d[(y * nx + x) * nlayers + l-1]) + mu) *
                (U_half[((y * nx + x) * nlayers + l)*3+2] -
                 U_half[((y * nx + x) * nlayers + l-1)*3+2]) /
                 U_half[((y * nx + x) * nlayers + l)*3];
        }

        for (int j = 0; j < l; j++) {
            sum_phs[(y * nx + x) * nlayers + l] += rho_d[j] / rho_d[l] *
                U_half[((y * nx + x) * nlayers + j)*3];
        }
        for (int j = l+1; j < nlayers; j++) {
            sum_phs[(y * nx + x) * nlayers + l] = sum_phs[(y * nx + x) * nlayers + l] +
                U_half[((y * nx + x) * nlayers + j)*3];
        }

        // D
        Up[((y * nx + x) * nlayers + l)*3] += dt * alpha * sum_qs;

        // Sx
        Up[((y * nx + x) * nlayers + l)*3+1] += dt * alpha *
            U_half[((y * nx + x) * nlayers + l)*3] * (-deltaQx);

        // Sy
        Up[((y * nx + x) * nlayers + l)*3+2] += dt * alpha *
            U_half[((y * nx + x) * nlayers + l)*3] * (-deltaQy);

    }

}


__global__ void evolve2(float * beta_d, float * gamma_up_d,
                     float * Un_d, float * Up, float * U_half,
                     float * sum_phs, float * rho_d, float * Q_d,
                     float mu,
                     int nx, int ny, int nlayers, float alpha,
                     float dx, float dy, float dt,
                     int kx_offset, int ky_offset) {
    /*
    Second part of evolution through one timestep.
    */
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int l = threadIdx.z;


    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1)) && (l < nlayers)) {

        float a = dt * alpha *
            U_half[((y * nx + x) * nlayers + l)*3] * (0.5 / dx) * (sum_phs[(y * nx + x+1) * nlayers + l] -
            sum_phs[(y * nx + x-1) * nlayers + l]);

        if (abs(a) < 0.9 * dx / dt) {
            //printf("a is %f! ", a);
            Up[((y * nx + x) * nlayers + l)*3+1] = Up[((y * nx + x) * nlayers + l)*3+1] - a;
        }

        a = dt * alpha *
            U_half[((y * nx + x) * nlayers + l)*3] * (0.5 / dy) *
            (sum_phs[((y+1) * nx + x) * nlayers + l] -
             sum_phs[((y-1) * nx + x) * nlayers + l]);

        if (abs(a) < 0.9 * dy / dt) {
            //printf("a is %f! ", a);
            Up[((y * nx + x) * nlayers + l)*3+2] = Up[((y * nx + x) * nlayers + l)*3+2] - a;
        }


    }

    __syncthreads();

    bcs(Up, nx, ny, nlayers, kx_offset, ky_offset);

    // copy back to grid
    if ((x < nx) && (y < ny) && (l < nlayers)) {
        for (int i = 0; i < 3; i++) {
            Un_d[((y * nx + x) * nlayers + l)*3+i] =
                Up[((y * nx + x) * nlayers + l)*3+i];
        }
    }


}


void cuda_run(float * beta, float * gamma_up, float * Un_h,
         float * rho, float * Q, float mu, int nx, int ny, int nlayers,
         int nt, float alpha, float dx, float dy, float dt, int dprint, char * filename) {
    /*
    Evolve system through nt timesteps, saving data to filename every dprint timesteps.
    */


    // set up GPU stuff
    int count;
    cudaGetDeviceCount(&count);

    //int size = 3 * nx * ny * nlayers;
    int maxThreads = 256;
    int maxBlocks = 256; //64;

    dim3 kernels = getNumKernels(nx, ny, nlayers, &maxBlocks, &maxThreads);

    dim3 *blocks = new dim3[kernels.x*kernels.y];
    dim3 *threads = new dim3[kernels.x*kernels.y];

    getNumBlocksAndThreads(nx, ny, nlayers, maxBlocks, maxThreads, kernels, blocks, threads);
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
    cudaMalloc((void**)&beta_d, 2*sizeof(float));
    cudaMalloc((void**)&gamma_up_d, 4*sizeof(float));
    cudaMalloc((void**)&Un_d, nx*ny*nlayers*3*sizeof(float));
    cudaMalloc((void**)&rho_d, nlayers*sizeof(float));
    cudaMalloc((void**)&Q_d, nlayers*nx*ny*sizeof(float));

    // copy stuff to GPU
    cudaMemcpy(beta_d, beta, 2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma_up_d, gamma_up, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Un_d, Un_h, nx*ny*nlayers*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rho_d, rho, nlayers*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Q_d, Q, nlayers*nx*ny*sizeof(float), cudaMemcpyHostToDevice);

    float *Up_d, *U_half_d, *sum_phs_d;
    cudaMalloc((void**)&Up_d, nlayers*nx*ny*3*sizeof(float));
    cudaMalloc((void**)&U_half_d, nlayers*nx*ny*3*sizeof(float));
    cudaMalloc((void**)&sum_phs_d, nlayers*nx*ny*sizeof(float));

    if (strcmp(filename, "na") != 0) {

        // create file
        hid_t outFile = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        // create dataspace
        int ndims = 5;
        hsize_t dims[] = {hsize_t((nt+1)/dprint+1), hsize_t(ny), hsize_t(nx), hsize_t(nlayers), 3};
        hid_t file_space = H5Screate_simple(ndims, dims, NULL);

        hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_layout(plist, H5D_CHUNKED);
        hsize_t chunk_dims[] = {1, hsize_t(ny), hsize_t(nx), hsize_t(nlayers), 3};
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
        hsize_t hcount[] = {1, hsize_t(ny), hsize_t(nx), hsize_t(nlayers), 3};
        H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, hcount, NULL);
        //printf("writing\n");
        // write to dataset
        printf("Printing t = %i\n", 0);
        H5Dwrite(dset, H5T_NATIVE_FLOAT, mem_space, file_space, H5P_DEFAULT, Un_h);
        // close file dataspace
        //printf("wrote\n");
        H5Sclose(file_space);

        for (int t = 0; t < nt; t++) {

            //if (t % 50 == 0) {
                //printf("t =  %i\n", t);
            //}
            int kx_offset = 0;
            int ky_offset = 0;

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

            kx_offset = 0;
            ky_offset = 0;

            for (int j = 0; j < kernels.y; j++) {
                kx_offset = 0;
                for (int i = 0; i < kernels.x; i++) {
                    evolve2<<<blocks[j * kernels.x + i], threads[j * kernels.x + i]>>>(beta_d, gamma_up_d, Un_d,
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
                cudaMemcpy(Un_h, Un_d, nx*ny*nlayers*3*sizeof(float), cudaMemcpyDeviceToHost);

                // select a hyperslab
                file_space = H5Dget_space(dset);
                hsize_t start[] = {hsize_t((t+1)/dprint), 0, 0, 0, 0};
                hsize_t hcount[] = {1, hsize_t(ny), hsize_t(nx), hsize_t(nlayers), 3};
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

            kx_offset = 0;
            ky_offset = 0;

            for (int j = 0; j < kernels.y; j++) {
                kx_offset = 0;
                for (int i = 0; i < kernels.x; i++) {
                    evolve2<<<blocks[j * kernels.x + i], threads[j * kernels.x + i]>>>(beta_d, gamma_up_d, Un_d,
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
                cudaMemcpy(Un_h, Un_d, nx*ny*nlayers*3*sizeof(float), cudaMemcpyDeviceToHost);

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

    //delete[] Un_h;
    delete[] threads;
    delete[] blocks;
}


#endif
