#include <iostream>
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <algorithm>
#include "SeaCuda.h"
#include "reduction.h"

using namespace std;


// TODO: GET RID OF THIS
void __syncthreads() {}

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

void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

    //get device capability, to avoid block/grid size exceed the upper bound
    cudaDeviceProp prop;
    int device;
    checkCudaErrors(cudaGetDevice(&device));
    checkCudaErrors(cudaGetDeviceProperties(&prop, device));

    threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);

    if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0])
    {
        printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, prop.maxGridSize[0], threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }
}


SeaCuda::SeaCuda(int n_layers, int _nx, int _ny, int _nt,
        float xmin, float xmax,
        float ymin, float ymax, float * _rho,
        float * _Q,
        float _alpha, float * _beta, float ** _gamma,
        bool _periodic)
        : nlayers(n_layers), nx(_nx), ny(_ny), nt(_nt), alpha(_alpha), periodic(_periodic)
{
    xs = new float[nx-2];
    for (int i = 0; i < (nx - 2); i++) {
        xs[i] = xmin + i * (xmax - xmin) / (nx-2);
    }

    ys = new float[ny-2];
    for (int i = 0; i < (ny - 2); i++) {
        ys[i] = ymin + i * (ymax - ymin) / (ny-2);
    }

    dx = xs[1] - xs[0];
    dy = ys[1] - ys[0];
    dt = 0.1 * min(dx, dy);

    rho = new float[nlayers];
    Q = new float[nlayers];

    for (int i = 0; i < nlayers; i++) {
        rho[i] = _rho[i];
        Q[i] = _Q[i];
    }

    for (int i = 0; i < 2; i++) {
        beta[i] = _beta[i];
        for (int j = 0; j < 2; j++) {
            gamma[i][j] = _gamma[i][j];
        }
    }

    // find inverse of gamma
    float det = gamma[0][0] * gamma[1][1] - gamma[0][1] * gamma[1][0];
    //cout << "det = " << det << '\n';
    gamma_up[0][0] = gamma[1][1] / det;
    gamma[0][1] = -gamma[0][1]/det;
    gamma[1][0] = -gamma[1][0]/det;
    gamma_up[1][1] = gamma[0][0]/det;

    U_grid = new float*[nlayers*nx*ny*(nt+1)];
    for (int i=0; i < nlayers*nx*ny*(nt+1); i++){
        U_grid[i] = new float[3];
    }

    cout << "Made a Sea.\n";
    //cout << "dt = " << dt << "\tdx = " << dx << "\tdy = " << dy << '\n';

}

SeaCuda::SeaCuda(const SeaCuda &seaToCopy)
    : nlayers(seaToCopy.nlayers), nx(seaToCopy.nx), ny(seaToCopy.ny), nt(seaToCopy.nt), dx(seaToCopy.dx), dy(seaToCopy.dy), dt(seaToCopy.dt), alpha(seaToCopy.alpha), periodic(seaToCopy.periodic)
{

    xs = new float[nx-2];
    for (int i = 0; i < (nx - 2); i++) {
        xs[i] = seaToCopy.xs[i];
    }

    ys = new float[ny-2];
    for (int i = 0; i < (ny - 2); i++) {
        ys[i] = seaToCopy.ys[i];
    }

    rho = new float[nlayers];
    Q = new float[nlayers];

    for (int i = 0; i < nlayers; i++) {
        rho[i] = seaToCopy.rho[i];
        Q[i] = seaToCopy.Q[i];
    }

    U_grid = new float*[nlayers*nx*ny*(nt+1)];
    for (int i=0; i < nlayers*nx*ny*(nt+1); i++){
        U_grid[i] = new float[3];
    }

    for (int i = 0; i < 2; i++) {
        beta[i] = seaToCopy.beta[i];
        for (int j = 0; j < 2; j++) {
            gamma[i][j] = seaToCopy.gamma[i][j];
            gamma_up[i][j] = seaToCopy.gamma_up[i][j];
        }
    }

}

// deconstructor
SeaCuda::~SeaCuda() {
    delete[] xs;
    delete[] ys;
    delete[] rho;
    delete[] Q;

    for (int i = 0; i < nlayers*nx*ny*(nt+1); i++) {
        delete[] U_grid[i];
    }
}

// return U state vector
void SeaCuda::U(float * grid, int l, int x, int y, int t, float * u) {
    for (int i=0; i < 3; i++) {
        u[i] = grid[(((t * ny + y) * nx + x) * nlayers + l)*3+i];
    }
}

// set the initial data
void SeaCuda::initial_data(float * D0, float * Sx0, float * Sy0) {
    for (int i = 0; i < nlayers*nx*ny; i++) {
        U_grid[i][0] = D0[i];
        U_grid[i][1] = Sx0[i];
        U_grid[i][2] = Sy0[i];
    }

    bcs(0);

    cout << "Set initial data.\n";
}

void SeaCuda::bcs(int t) {
    if (periodic) {

        for (int l = 0; l < nlayers; l++) {
            for (int y = 0; y < ny; y++){
                for (int i = 0; i < 3; i++) {
                    U_grid[((t * ny + y) * nx) * nlayers + l][i] = U_grid[((t * ny + y) * nx + (nx-2)) * nlayers + l][i];

                    U_grid[((t * ny + y) * nx + (nx-1)) * nlayers + l][i] = U_grid[((t * ny + y) * nx + 1) * nlayers + l][i];
                }
            }
            for (int x = 0; x < nx; x++){
                for (int i = 0; i < 3; i++) {
                    U_grid[((t * ny) * nx + x) * nlayers + l][i] = U_grid[((t * ny + (ny-2)) * nx + x) * nlayers + l][i];

                    U_grid[((t * ny + (ny-1)) * nx + x) * nlayers + l][i] = U_grid[((t * ny + 1) * nx + x) * nlayers + l][i];
                }
            }
        }
    } else { // outflow
        for (int l = 0; l < nlayers; l++) {
            for (int y = 0; y < ny; y++){
                for (int i = 0; i < 3; i++) {
                    U_grid[((t * ny + y) * nx) * nlayers + l][i] = U_grid[((t * ny + y) * nx + 1) * nlayers + l][i];

                    U_grid[((t * ny + y) * nx + (nx-1)) * nlayers + l][i] = U_grid[((t * ny + y) * nx + (nx-2)) * nlayers + l][i];
                }
            }
            for (int x = 0; x < nx; x++){
                for (int i = 0; i < 3; i++) {
                    U_grid[((t * ny) * nx + x) * nlayers + l][i] = U_grid[((t * ny + 1) * nx + x) * nlayers + l][i];

                    U_grid[((t * ny + (ny-1)) * nx + x) * nlayers + l][i] = U_grid[((t * ny + (ny-2)) * nx + x) * nlayers + l][i];
                }

            }
        }
    }
}

void SeaCuda::bcs(float * grid) {
    if (periodic) {

        for (int l = 0; l < nlayers; l++) {
            for (int y = 0; y < ny; y++){
                for (int i = 0; i < 3; i++) {
                    grid[((y * nx) * nlayers + l)*3+i] = grid[((y * nx + (nx-2)) * nlayers + l)*3+i];

                    grid[((y * nx + (nx-1)) * nlayers + l)*3+i] = grid[((y * nx + 1) * nlayers + l)*3+i];
                }
            }
            for (int x = 0; x < nx; x++){
                for (int i = 0; i < 3; i++) {
                    grid[(x * nlayers + l)*3+i] = grid[(((ny-2) * nx + x) * nlayers + l)*3+i];

                    grid[(((ny-1) * nx + x) * nlayers + l)*3+i] = grid[((nx + x) * nlayers + l)*3+i];
                }
            }
        }
    } else { // outflow
        for (int l = 0; l < nlayers; l++) {
            for (int y = 0; y < ny; y++){
                for (int i = 0; i < 3; i++) {
                    grid[((y * nx) * nlayers + l)*3+i] = grid[((y * nx + 1) * nlayers + l)*3+i];

                    grid[((y * nx + (nx-1)) * nlayers + l)*3+i] = grid[((y * nx + (nx-2)) * nlayers + l)*3+i];
                }
            }
            for (int x = 0; x < nx; x++){
                for (int i = 0; i < 3; i++) {
                    grid[(x * nlayers + l)*3+i] = grid[((nx + x) * nlayers + l)*3+i];

                    grid[(((ny-1) * nx + x) * nlayers + l)*3+i] = grid[(((ny-2) * nx + x) * nlayers + l)*3+i];
                }

            }
        }
    }
}

__device__ void Jx(float * u, float * beta_d, float * gamma_up_d, float * jx, float alpha) {

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

__global__ void evolve(int t, float * beta_d, float * gamma_up_d,
                     float * U_grid_d, float * rho_d, float * Q_d,
                     int nx, int ny, int nlayers, float alpha,
                     float dx, float dy, float dt) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int l = threadIdx.z;

    if (t % 50 == 0) {
        cout << "t = " << t << "\n";
    }

    float *u, *u_ip, *u_im, *u_jp, *u_jm, *u_pp, *u_mm, *u_imjp, *u_ipjm;
    float *A, *B, *A2, *B2, *AB;

    u = (float *) malloc(3*sizeof(float));
    u_ip = (float *) malloc(3*sizeof(float));
    u_im = (float *) malloc(3*sizeof(float));
    u_jp = (float *) malloc(3*sizeof(float));
    u_jm = (float *) malloc(3*sizeof(float));
    u_pp = (float *) malloc(3*sizeof(float));
    u_mm = (float *) malloc(3*sizeof(float));
    u_imjp = (float *) malloc(3*sizeof(float));
    u_ipjm = (float *) malloc(3*sizeof(float));

    A = (float *) malloc(9*sizeof(float));
    B = (float *) malloc(9*sizeof(float));
    A2 = (float *) malloc(9*sizeof(float));
    B2 = (float *) malloc(9*sizeof(float));
    AB = (float *) malloc(9*sizeof(float));

    float d, e, f, g, h;

    float *Up, *U_half;
    Up = (float *) malloc(nlayers*nx*ny*3*sizeof(float));
    U_half = (float *) malloc(nlayers*nx*ny*3*sizeof(float));
    for (int i=0; i < nlayers*nx*ny*3; i++){
        // initialise
        Up[i] = 0.0;
        U_half[i] = 0.0;
    }

    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1)) && (l < nlayers)) {

        for (int i = 0; i < 3; i++) {
            u[i] = U_grid_d[(((t * ny + y) * nx + x) * nlayers + l)*3+i];
            u_ip[i] = U_grid_d[(((t * ny + y) * nx + x+1) * nlayers + l)*3+i];
            u_im[i] = U_grid_d[(((t * ny + y) * nx + x-1) * nlayers + l)*3+i];
            u_jp[i] = U_grid_d[(((t * ny + y+1) * nx + x) * nlayers + l)*3+i];
            u_jm[i] = U_grid_d[(((t * ny + y-1) * nx + x) * nlayers + l)*3+i];
            u_pp[i] = U_grid_d[(((t * ny + y+1) * nx + x+1) * nlayers + l)*3+i];
            u_mm[i] = U_grid_d[(((t * ny + y-1) * nx + x-1) * nlayers + l)*3+i];
            u_ipjm[i] = U_grid_d[(((t * ny + y-1) * nx + x+1) * nlayers + l)*3+i];
            u_imjp[i] = U_grid_d[(((t * ny + y+1) * nx + x-1) * nlayers + l)*3+i];
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
                d += A[i*3+j] * (u_ip[j] - u_im[j]);
                e += B[i*3+j] * (u_jp[j] - u_jm[j]);
                f += A2[i*3+j] * (u_ip[j] - 2.0 * u[j] + u_im[j]);
                g += B2[i*3+j] * (u_jp[j] - 2.0 * u[j] + u_jm[j]);
                h += AB[i*3+j] * (u_pp[j] - u_ipjm[j] - u_imjp[j] + u_mm[j]);
            }

            Up[((y * nx + x) * nlayers + l) * 3 + i] = u[i] -
                    0.5 * dt/dx * d -
                    0.5 * dt/dy * e +
                    0.5 * dt*dt/(dx*dx) * f +
                    0.5 * dt*dt/(dy*dy) * g -
                    0.25 * dt*dt/(dx*dy) * h;

        }

    }

    free(u);
    free(u_ip);
    free(u_im);
    free(u_jp);
    free(u_pp);
    free(u_mm);
    free(u_imjp);
    free(u_ipjm);
    free(A);
    free(B);
    free(A2);
    free(B2);
    free(AB);

    __syncthreads();

    // enforce boundary conditions
    bcs(Up);

    // copy to U_half
    if ((x < nx) && (y < ny) && (l < nlayers)) {
        for (int i = 0; i < 3; i++) {
            U_half[((y * nx + x) * nlayers + l)*3+i] = Up[((y * nx + x) * nlayers + l)*3+i];
        }
    }

    __syncthreads();


    float *ph, *Sx, *Sy, *W, *sum_phs;
    ph = (float *) malloc(nlayers*sizeof(float));
    Sx = (float *) malloc(nlayers*sizeof(float));
    Sy = (float *) malloc(nlayers*sizeof(float));
    W = (float *) malloc(nlayers*sizeof(float));

    sum_phs = (float *) malloc(nlayers*nx*ny*sizeof(float));

    // do source terms
    if ((x < nx) && (y < ny) && (l < nlayers)) {
        ph[l] = U_half[((y * nx + x) * nlayers + l)*3];
        Sx[l] = U_half[((y * nx + x) * nlayers + l)*3+1];
        Sy[l] = U_half[((y * nx + x) * nlayers + l)*3+2];
        W[l] = sqrt((Sx[l] * Sx[l] * gamma_up_d[0] +
                       2.0 * Sx[l] * Sy[l] * gamma_up_d[1] +
                       Sy[l] * Sy[l] * gamma_up_d[3]) /
                       (ph[l] * ph[l]) + 1.0);
        ph[l] /= W[l];


        __syncthreads();

        float sum_qs = 0.0;
        float deltaQx = 0.0;
        float deltaQy = 0.0;
        sum_phs[(y * nx + x) * nlayers + l] = 0.0;

        if (l < (nlayers - 1)) {
            sum_qs += -rho_d[l+1] / rho_d[l] * abs(Q_d[l+1] - Q_d[l]);
            deltaQx = rho_d[l+1] / rho_d[l] * max(float(0.0), Q_d[l] - Q_d[l+1]) * (Sx[l] - Sx[l+1]) / ph[l];
            deltaQy = rho_d[l+1] / rho_d[l] * max(float(0.0), Q_d[l] - Q_d[l+1]) * (Sy[l] - Sy[l+1]) / ph[l];
        }
        if (l > 0) {
            sum_qs += abs(Q_d[l] - Q_d[l-1]);
            deltaQx = max(float(0.0), Q_d[l] - Q_d[l-1]) * (Sx[l] - Sx[l-1]) / ph[l];
            deltaQy = max(float(0.0), Q_d[l] - Q_d[l-1]) * (Sy[l] - Sy[l-1]) / ph[l];
        }

        for (int j = 0; j < l; j++) {
            sum_phs[(y * nx + x) * nlayers + l] += rho_d[j] / rho_d[l] * ph[j];
        }
        for (int j = l+1; j < nlayers; j++) {
            sum_phs[(y * nx + x) * nlayers + l] += ph[j];
        }

        // D
        Up[((y * nx + x) * nlayers + l)*3] += dt * sum_qs;

        // Sx
        Up[((y * nx + x) * nlayers + l)*3+1] += dt * ph[l] * (-deltaQx);

        // Sy
        Up[((y * nx + x) * nlayers + l)*3+2] += dt * ph[l] * (-deltaQy);

    }

    __syncthreads();

    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1)) && (l < nlayers)) {

    // Sx
    Up[((y * nx + x) * nlayers + l)*3+1] -= dt * U_half[((y * nx + x) * nlayers + l)*3] * 0.5 / dx * (sum_phs[(y * nx + (x+1)) * nlayers + l] - sum_phs[(y * nx + (x-1)) * nlayers + l]);

    // Sy
    Up[((y * nx + x) * nlayers + l)*3+2] -= dt * U_half[((y * nx + x) * nlayers + l)*3] * 0.5 / dy * (sum_phs[((y+1) * nx + x) * nlayers + l] - sum_phs[((y-1) * nx + x) * nlayers + l]);


    }

    __syncthreads();


    // copy back to grid
    if ((x < nx) && (y < ny) && (l < nlayers)) {
        for (int i = 0; i < 3; i++) {
            U_grid_d[((y * nx + x) * nlayers + l)*3+i] = Up[((y * nx + x) * nlayers + l)*3+i];
        }
    }

    __syncthreads();

    bcs(t+1);

    free(ph);
    free(Sx);
    free(Sy);
    free(W);
    free(sum_phs);

    free(Up);
    free(U_half);

}

void SeaCuda::run() {
    cout << "Beginning evolution.\n";

    // set up GPU stuff
    int count;
    cudaGetDeviceCount(&count);
    //dim3 threadsPerBlock(20,20,nlayers);
    //dim3 numBlocks(nx/threadsPerBlock.x,ny/threadsPerBlock.y,1);

    int size = 3 * nx * ny * nlayers * (nt+1);
    int maxThreads = 1024;
    int maxBlocks = 64;

    int numBlocks = 0;
    int numThreads = 0;
    getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);

    dim3 threadsPerBlock(int(ceil(sqrt(numThreads/nlayers))), int(ceil(sqrt(numThreads/nlayers))), nlayers);
    dim3 grid(numBlocks, 1, 1);

    float * beta_d;
    float * gamma_up_d;
    float * U_grid_d;
    float * rho_d;
    float * Q_d;

    // allocate memory on device
    cudaMalloc((void**)&beta_d, 2*sizeof(float));
    cudaMalloc((void**)&gamma_up_d, 4*sizeof(float));
    cudaMalloc((void**)&U_grid_d, numBlocks*sizeof(float));
    cudaMalloc((void**)&rho_d, nlayers*sizeof(float));
    cudaMalloc((void**)&Q_d, nlayers*sizeof(float));

    // copy stuff to GPU
    cudaMemcpy(beta_d, beta, 2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma_up_d, gamma_up, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(U_grid_d, U_grid, numBlocks*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rho_d, rho, nlayers*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Q_d, Q, nlayers*sizeof(float), cudaMemcpyHostToDevice);

    for (int t = 0; t < nt; t++) {
        evolve(t, beta_d, gamma_up_d, U_grid_d, rho_d, Q_d,
               nx, ny, nlayers, alpha,
               dx, dy, dt);
    }

    // copy stuff back
    cudaMemcpy(U_grid, U_grid_d, numBlocks*sizeof(float), cudaMemcpyDeviceToHost);

    // delete some stuff
    cudaFree(beta_d);
    cudaFree(gamma_up_d);
    cudaFree(U_grid_d);
    cudaFree(rho_d);
    cudaFree(Q_d);
}

void SeaCuda::output(char * filename) {
    // open file
    ofstream outFile(filename);

    // only going to output every 10 because file size is ridiculous
    for (int t = 0; t < (nt+1); t+=10) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                for (int l = 0; l < nlayers; l++) {
                    outFile << t << ", " << x << ", " << y << ", " << l;
                    for (int i = 0; i < 3; i++ ) {
                        outFile << ", " << U_grid[((t * ny + y) * nx + x) * nlayers + l][i];
                    }
                    outFile << '\n';
                }
            }
        }
    }
}

int main() {

    // initialise parameters
    static const int nlayers = 2;
    int nx = 200;
    int ny = 200;
    int nt = 600;
    float xmin = 0.0;
    float xmax = 10.0;
    float ymin = 0.0;
    float ymax = 10.0;
    float rho[nlayers];
    float Q[nlayers];
    float alpha = 0.9;
    float beta[2];
    float *gamma[2];
    bool periodic = false;

    float D0[nlayers*nx*ny];
    float Sx0[nlayers*nx*ny];
    float Sy0[nlayers*nx*ny];

    for (int i =0; i < nlayers; i++) {
        rho[i] = 1.0;
        Q[i] = 0.0;
    }

    for (int i =0; i < 2; i++) {
        beta[i] = 0.0;
        gamma[i] = new float[2];
        gamma[i][i] = 1.0 / (alpha*alpha);
    }

    // make a sea
    SeaCuda sea(nlayers, nx, ny, nt, xmin, xmax, ymin, ymax, rho, Q, alpha, beta, gamma, periodic);

    // set initial data
    for (int x = 1; x < (nx - 1); x++) {
        for (int y = 1; y < (ny - 1); y++) {
            D0[(y * nx + x) * nlayers] = 1.0 + 0.4 * exp(-(pow(sea.xs[x-1]-2.0, 2) + pow(sea.ys[y-1]-2.0, 2)) * 2.0);
            D0[(y * nx + x) * nlayers + 1] = 0.8 + 0.2 * exp(-(pow(sea.xs[x-1]-7.0, 2) + pow(sea.ys[y-1]-7.0, 2)) * 2.0);
            for (int l = 0; l < nlayers; l++) {
                Sx0[(y * nx + x) * nlayers + l] = 0.0;
                Sy0[(y * nx + x) * nlayers + l] = 0.0;
            }
        }
    }

    sea.initial_data(D0, Sx0, Sy0);

    // run simulation
    sea.run();

    char filename[] = "out.dat";
    sea.output(filename);

    cout << "Output data to file.\n";

    // clean up
    for (int i =0; i < 2; i++) {
        delete[] gamma[i];
    }

}
