#include <iostream>
#include <cmath>
//#include <cuda.h>
#include "Sea.h"
using namespace std;

SquareMatrix operator*(SquareMatrix a, SquareMatrix b) {
    SquareMatrix c;
    for (int i = 0; i < a.dim; i++) {
        for (int j = 0; j < a.dim; j++) {
            c.mat[i][j] = 0;
            for (int k = 0; k < a.dim; k++) {
                c.mat[i][j] += a.mat[i][k] * b.mat[k][j];
            }
        }
    }
    return c;
}

Vec SquareMatrix::operator*(Vec v) {
    Vec av;
    for (int i = 0; i < dim; i++) {
        av.vec[i] = 0;
        for (int j = 0; j < dim; j++) {
            av.vec[i] += mat[i][j] * v.vec[j];
        }
    }

    return av;
}

SquareMatrix SquareMatrix::operator*(float a) {
    SquareMatrix c;
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            c.mat[i][j] = mat[i][j] * a;
        }
    }

    return c;
}

SquareMatrix operator*(float a, SquareMatrix b) {
    SquareMatrix c;
    for (int i = 0; i < b.dim; i++) {
        for (int j = 0; j < b.dim; j++) {
            c.mat[i][j] = b.mat[i][j] * a;
        }
    }

    return c;
}

Vec Vec::operator*(float a) {
    Vec av;
    for (int i = 0; i < dim; i++) {
        av.vec[i] = vec[i] * a;
    }

    return av;
}

Vec operator*(float a, Vec b) {
    Vec av;
    for (int i = 0; i < b.dim; i++) {
        av.vec[i] = b.vec[i] * a;
    }

    return av;
}

Vec operator-(Vec a, Vec b) {
    Vec c;
    for (int i=0; i<a.dim; i++) {
        c.vec[i] = a.vec[i] - b.vec[i];
    }
    return c;
}

Vec operator+(Vec a, Vec b) {
    Vec c;
    for (int i=0; i<a.dim; i++) {
        c.vec[i] = a.vec[i] + b.vec[i];
    }
    return c;
}

float dot(Vec a, Vec b) {
    float c = 0;
    for (int i = 0; i<a.dim; i++) {
        c += a.vec[i] * b.vec[i];
    }
}

Sea::Sea(int n_layers, int _nx, int _ny, int _nt,
        float xmin, float xmax,
        float ymin, float ymax, float * _rho,
        float * _Q,
        float _alpha, float * _beta, float ** _gamma,
        bool _periodic)
        : nlayers(n_layers), nx(_nx), ny(_ny), nt(_nt), alpha(_alpha), periodic(_periodic)
{
    x = new float[nx-2];
    for (int i = 0; i < nx - 2; i++) {
        x[i] = xmin + i * (xmax - xmin) / (nx-2);
    }

    y = new float[ny-2];
    for (int i = 0; i < ny - 2; i++) {
        y[i] = ymin + i * (ymax - ymin) / (ny-2);
    }

    dx = x[1] - x[0];
    dy = y[1] - y[0];
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
    gamma_up[0][0] = gamma[1][1] / det;
    gamma[0][1] = -gamma[0][1]/det;
    gamma[1][0] = -gamma[1][0]/det;
    gamma_up[1][1] = gamma[0][0]/det;

    U_grid = new float*[3];
    for (int i=0; i < 3; i++){
        U_grid[i] = new float[nlayers*nx*ny*nt+1];
    }

}

// deconstructor
Sea::~Sea() {
    delete[] x;
    delete[] y;
    delete[] rho;
    delete[] Q;

    for (int i = 0; i < 3; i++) {
        delete[] U_grid[i];
    }
}


Vec Sea::U(int l, int x, int y, int t, Vec u) {
    for (int i=0; i < 3; i++) {
        u.vec[i] = U_grid[i][((t * ny + y) * nx + x) * nlayers + l];
    }

    return u;
}


void Sea::initial_data(float * D0, float * Sx0, float * Sy0) {
    for (int i = 0; i < nlayers*nx*ny; i++) {
        U_grid[0][i] = D0[i];
        U_grid[1][i] = Sx0[i];
        U_grid[2][i] = Sy0[i];
    }
}

void Sea::bcs(int t) {
    if (periodic) {
        for (int i = 0; i < 3; i++) {
            for (int l = 0; l < nlayers; l++) {
                for (int y = 0; y < ny; y++){
                    U_grid[i][((t * ny + y) * nx) * nlayers + l] = U_grid[i][((t * ny + y) * nx + (nx-2)) * nlayers + l];

                    U_grid[i][((t * ny + y) * nx + (nx-1)) * nlayers + l] = U_grid[i][((t * ny + y) * nx + 1) * nlayers + l];
                }
                for (int x = 0; x < nx; x++){
                    U_grid[i][((t * ny) * nx + x) * nlayers + l] = U_grid[i][((t * ny + (ny-2)) * nx + x) * nlayers + l];

                    U_grid[i][((t * ny + (ny-1)) * nx + x) * nlayers + l] = U_grid[i][((t * ny + 1) * nx + x) * nlayers + l];
                }

            }
        }
    } else { // outflow
        for (int i = 0; i < 3; i++) {
            for (int l = 0; l < nlayers; l++) {
                for (int y = 0; y < ny; y++){
                    U_grid[i][((t * ny + y) * nx) * nlayers + l] = U_grid[i][((t * ny + y) * nx + 1) * nlayers + l];

                    U_grid[i][((t * ny + y) * nx + (nx-1)) * nlayers + l] = U_grid[i][((t * ny + y) * nx + (nx-2)) * nlayers + l];
                }
                for (int x = 0; x < nx; x++){
                    U_grid[i][((t * ny) * nx + x) * nlayers + l] = U_grid[i][((t * ny + 1) * nx + x) * nlayers + l];

                    U_grid[i][((t * ny + (ny-1)) * nx + x) * nlayers + l] = U_grid[i][((t * ny + (ny-2)) * nx + x) * nlayers + l];
                }

            }
        }
    }
}

SquareMatrix Sea::Jx(Vec u) {

    float W = sqrt((u.vec[1]*u.vec[1] * gamma_up[0][0] +
                2.0 * u.vec[1]* u.vec[2] * gamma_up[0][1] +
                u.vec[2]*u.vec[2] * gamma_up[1][1]) / u.vec[0]*u.vec[0] + 1.0);

    float ph = u.vec[0] / W;
    float vx = u.vec[1] / (u.vec[0] * W); // u_down
    float vy = u.vec[2] / (u.vec[0] * W); // v_down

    float qx = vx * gamma_up[0][0] + vy * gamma_up[0][1] - beta[0]/alpha;

    float chi = 1.0 / (1.0 - vx*vx * W*W - vy*vy * W*W);

    SquareMatrix jx;

    jx.mat[0][0] = qx/chi - vx;
    jx.mat[0][1] = (1.0 + vy*vy*W*W)/W;
    jx.mat[0][2] = -W * vx * vy;

    jx.mat[1][0] = -2.0*W*W*W*vx*qx*(vx*vx + vy*vy) + ph*(1.0/W - W*vx*vx);
    jx.mat[1][1] = qx * (1.0+W*W*vx*vx + W*W*vy*vy) + 0.5*ph*vx*(vy*vy*W*W-1.0);
    jx.mat[1][2] = -vy*ph*(1.0 + 0.5*W*W*vx*vx);

    jx.mat[2][0] = -W*vy*(2.0*W*W*qx*(vx*vx+vy*vy) + 0.5*ph*vx);
    jx.mat[2][1] = 0.5*ph*vy*(1.0+vy*vy*W*W);
    jx.mat[2][2] = qx*(1.0+W*W*vx*vx+W*W*vy*vy) - 0.5*ph*W*W*vx*vy*vy;

    for (int i=0; i< 3; i++){
        for (int j=0; j < 3; j++) {
            jx.mat[i][j] *= chi;
        }
    }

    return jx;
}

SquareMatrix Sea::Jy(Vec u) {

    float W = sqrt((u.vec[1]*u.vec[1] * gamma_up[0][0] +
                2.0 * u.vec[1]* u.vec[2] * gamma_up[0][1] +
                u.vec[2]*u.vec[2] * gamma_up[1][1]) / u.vec[0]*u.vec[0] + 1.0);

    float ph = u.vec[0] / W;
    float vx = u.vec[1] / (u.vec[0] * W); // u_down
    float vy = u.vec[2] / (u.vec[0] * W); // v_down

    float qy = vy * gamma_up[1][1] + vx * gamma_up[0][1] - beta[1]/alpha;

    float chi = 1.0 / (1.0 - vx*vx * W*W - vy*vy * W*W);

    SquareMatrix jy;

    jy.mat[0][0] = qy/chi - vx;
    jy.mat[0][1] = -W * vx * vy;
    jy.mat[0][2] = (1.0 + vx*vx*W*W)/W;

    jy.mat[1][0] = -W*vx*(2.0*W*W*qy*(vx*vx+vy*vy) + 0.5*ph*vy);
    jy.mat[1][1] = qy*(1.0+W*W*vx*vx+W*W*vy*vy) - 0.5*ph*W*W*vx*vx*vy;
    jy.mat[1][2] = 0.5*ph*vx*(1.0+vx*vx*W*W);

    jy.mat[2][0] = -2.0*W*W*W*vy*qy*(vx*vx + vy*vy) + ph*(1.0/W - W*vy*vy);
    jy.mat[2][1] = -vx*ph*(1.0 + 0.5*W*W*vy*vy);
    jy.mat[2][2] = qy * (1.0+W*W*vx*vx + W*W*vy*vy) + 0.5*ph*vy*(vx*vx*W*W-1.0);

    for (int i=0; i< 3; i++){
        for (int j=0; j < 3; j++) {
            jy.mat[i][j] *= chi;
        }
    }

    return jy;
}

void Sea::evolve(int t) {
    Vec u, u_ip, u_im, u_jp, u_jm, u_pp, u_mm, up;
    for (int l = 0; l < nlayers; l++) {
        for (int x = 1; x < (nx-1); x++) {
            for (int y = 1; y < (ny-1); y++) {
                U(l, x, y, t, u);
                U(l, x, y, t, u_ip);
                U(l, x, y, t, u_im);
                U(l, x, y, t, u_jp);
                U(l, x, y, t, u_jm);

                SquareMatrix A = Jx(u);
                SquareMatrix B = Jy(u);


                up = up -
                    0.5 * (dt/dx) * A * (u_ip - u_im) -
                    0.5 * (dt/dy) * B * (u_jp - u_jm) +
                    0.5 * dt*dt/(dx*dx) * A * A * (u_ip - 2.0 * u + u_im) +
                    0.5 * dt*dt/(dy*dy) * B * B * (u_jp - 2.0 * u + u_jm) -
                    0.25 * dt*dt/(dx*dy) * A * B * (u_ip - 2.0 * u + u_im)


            }
        }
    }
}

__global__ void func() {

}

int main() {

}
