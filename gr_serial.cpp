#include <iostream>
#include <cmath>
//#include <cuda.h>
#include "Sea.h"
#include <fstream>
using namespace std;

Vec::Vec() {
    for (int i = 0; i < dim; i++) {
        vec[i] = 0;
    }
}

// overload some operators to make vector/matrix calculations easier

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

    return c;
}

Sea::Sea(int n_layers, int _nx, int _ny, int _nt,
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

Sea::Sea(const Sea &seaToCopy)
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
Sea::~Sea() {
    delete[] xs;
    delete[] ys;
    delete[] rho;
    delete[] Q;

    for (int i = 0; i < nlayers*nx*ny*(nt+1); i++) {
        delete[] U_grid[i];
    }
}

// return U state vector
Vec Sea::U(int l, int x, int y, int t) {
    Vec u;
    for (int i=0; i < 3; i++) {
        u.vec[i] = U_grid[((t * ny + y) * nx + x) * nlayers + l][i];
    }
    return u;
}

// set the initial data
void Sea::initial_data(float * D0, float * Sx0, float * Sy0) {
    for (int i = 0; i < nlayers*nx*ny; i++) {
        U_grid[i][0] = D0[i];
        U_grid[i][1] = Sx0[i];
        U_grid[i][2] = Sy0[i];
    }

    bcs(0);

    cout << "Set initial data.\n";
}

void Sea::bcs(int t) {
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

void Sea::bcs(float ** grid) {
    if (periodic) {

        for (int l = 0; l < nlayers; l++) {
            for (int y = 0; y < ny; y++){
                for (int i = 0; i < 3; i++) {
                    grid[(y * nx) * nlayers + l][i] = grid[(y * nx + (nx-2)) * nlayers + l][i];

                    grid[(y * nx + (nx-1)) * nlayers + l][i] = grid[(y * nx + 1) * nlayers + l][i];
                }
            }
            for (int x = 0; x < nx; x++){
                for (int i = 0; i < 3; i++) {
                    grid[x * nlayers + l][i] = grid[((ny-2) * nx + x) * nlayers + l][i];

                    grid[((ny-1) * nx + x) * nlayers + l][i] = grid[(nx + x) * nlayers + l][i];
                }
            }
        }
    } else { // outflow
        for (int l = 0; l < nlayers; l++) {
            for (int y = 0; y < ny; y++){
                for (int i = 0; i < 3; i++) {
                    grid[(y * nx) * nlayers + l][i] = grid[(y * nx + 1) * nlayers + l][i];

                    grid[(y * nx + (nx-1)) * nlayers + l][i] = grid[(y * nx + (nx-2)) * nlayers + l][i];
                }
            }
            for (int x = 0; x < nx; x++){
                for (int i = 0; i < 3; i++) {
                    grid[x * nlayers + l][i] = grid[(nx + x) * nlayers + l][i];

                    grid[((ny-1) * nx + x) * nlayers + l][i] = grid[((ny-2) * nx + x) * nlayers + l][i];
                }

            }
        }
    }
}

SquareMatrix Sea::Jx(Vec u) {

    float W = sqrt((u.vec[1]*u.vec[1] * gamma_up[0][0] +
                2.0 * u.vec[1]* u.vec[2] * gamma_up[0][1] +
                u.vec[2]*u.vec[2] * gamma_up[1][1]) / (u.vec[0]*u.vec[0]) + 1.0);
    //cout << "W = " << W << '\n';
    //cout << "u = " << u.vec[0] << ' ' << u.vec[1] << ' ' << u.vec[2] << '\n';

    float ph = u.vec[0] / W;
    float vx = u.vec[1] / (u.vec[0] * W); // u_down
    float vy = u.vec[2] / (u.vec[0] * W); // v_down

    float qx = vx * gamma_up[0][0] + vy * gamma_up[0][1] - beta[0]/alpha;

    float chi = 1.0 / (1.0 - vx*vx * W*W - vy*vy * W*W);

    SquareMatrix jx;

    jx.mat[0][0] = qx/chi - vx;
    jx.mat[0][1] = (1.0 + vy*vy*W*W)/W;
    jx.mat[0][2] = -W * vx * vy;

    jx.mat[1][0] = -2.0*pow(W,3)*vx*qx*(vx*vx + vy*vy) + ph*(1.0/W - W*vx*vx);
    jx.mat[1][1] = qx * (1.0+W*W*vx*vx + W*W*vy*vy) + 0.5*ph*vx*(vy*vy*W*W-1.0);
    jx.mat[1][2] = -vy*ph*(1.0 + 0.5*W*W*vx*vx);

    jx.mat[2][0] = -W*vy*(2.0*W*W*qx*(vx*vx+vy*vy) + 0.5*ph*vx);
    jx.mat[2][1] = 0.5*ph*vy*(1.0+vy*vy*W*W);
    jx.mat[2][2] = qx*(1.0+W*W*vx*vx+W*W*vy*vy) - 0.5*ph*W*W*vx*vy*vy;

    jx = jx * chi;

    return jx;
}

SquareMatrix Sea::Jy(Vec u) {

    float W = sqrt((u.vec[1]*u.vec[1] * gamma_up[0][0] +
                2.0 * u.vec[1]* u.vec[2] * gamma_up[0][1] +
                u.vec[2]*u.vec[2] * gamma_up[1][1]) / (u.vec[0]*u.vec[0]) + 1.0);

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

    jy.mat[2][0] = -2.0*pow(W,3)*vy*qy*(vx*vx + vy*vy) + ph*(1.0/W - W*vy*vy);
    jy.mat[2][1] = -vx*ph*(1.0 + 0.5*W*W*vy*vy);
    jy.mat[2][2] = qy * (1.0+W*W*vx*vx + W*W*vy*vy) + 0.5*ph*vy*(vx*vx*W*W-1.0);

    jy  = jy * chi;

    return jy;
}

void Sea::evolve(int t) {

    if (t % 50 == 0) {
        cout << "t = " << t << "\n";
    }

    Vec u, u_ip, u_im, u_jp, u_jm, u_pp, u_mm, u_imjp, u_ipjm, up;

    float ** Up = new float*[nlayers*nx*ny];
    float ** U_half = new float*[nlayers*nx*ny];
    for (int i=0; i < nlayers*nx*ny; i++){
        Up[i] = new float[3];
        U_half[i] = new float[3];
        for (int j = 0; j < 3; j++) {
            // initialise
            Up[i][j] = 0.0;
            U_half[i][j] = 0.0;
        }
    }

    for (int l = 0; l < nlayers; l++) {
        for (int x = 1; x < (nx-1); x++) {
            for (int y = 1; y < (ny-1); y++) {
                u = U(l, x, y, t);
                u_ip = U(l, x+1, y, t);
                u_im = U(l, x-1, y, t);
                u_jp = U(l, x, y+1, t);
                u_jm = U(l, x, y-1, t);
                u_pp = U(l, x+1, y+1, t);
                u_mm = U(l, x-1, y-1, t);
                u_ipjm = U(l, x+1, y-1, t);
                u_imjp = U(l, x-1, y+1, t);

                SquareMatrix A = Jx(u);
                SquareMatrix B = Jy(u);

                up = u -
                    0.5 * (dt/dx) * A * (u_ip - u_im) -
                    0.5 * (dt/dy) * B * (u_jp - u_jm) +
                    0.5 * dt*dt/(dx*dx) * A * A * (u_ip - 2.0 * u + u_im) +
                    0.5 * dt*dt/(dy*dy) * B * B * (u_jp - 2.0 * u + u_jm) -
                    0.25 * dt*dt/(dx*dy) * A * B * (u_pp - u_ipjm - u_imjp + u_mm);

                // copy to array
                for (int i = 0; i < 3; i++) {
                    Up[(y * nx + x) * nlayers + l][i] = up.vec[i];
                }

            }
        }
    }

    // enforce boundary conditions
    bcs(Up);

    // copy to U_half
    for (int n = 0; n < nlayers*nx*ny; n++) {
        for (int i = 0; i < 3; i++) {
            U_half[n][i] = Up[n][i];
        }
    }

    float * ph = new float[nlayers];
    float * Sx = new float[nlayers];
    float * Sy = new float[nlayers];
    float * W = new float[nlayers];

    float * sum_phs = new float[nlayers*nx*ny];

    // do source terms
    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {

            for (int l = 0; l < nlayers; l++) {
                ph[l] = U_half[(y * nx + x) * nlayers + l][0];
                Sx[l] = U_half[(y * nx + x) * nlayers + l][1];
                Sy[l] = U_half[(y * nx + x) * nlayers + l][2];
                W[l] = sqrt((Sx[l] * Sx[l] * gamma_up[0][0] +
                               2.0 * Sx[l] * Sy[l] * gamma_up[0][1] +
                               Sy[l] * Sy[l] * gamma_up[1][1]) /
                               (ph[l] * ph[l]) + 1.0);
                ph[l] /= W[l];
            }

            for (int l = 0; l < nlayers; l++) {
                float sum_qs = 0.0;
                float deltaQx = 0.0;
                float deltaQy = 0.0;
                sum_phs[(y * nx + x) * nlayers + l] = 0.0;

                if (l < (nlayers - 1)) {
                    sum_qs += -rho[l+1] / rho[l] * abs(Q[l+1] - Q[l]);
                    deltaQx = rho[l+1] / rho[l] * max(float(0.0), Q[l] - Q[l+1]) * (Sx[l] - Sx[l+1]) / ph[l];
                    deltaQy = rho[l+1] / rho[l] * max(float(0.0), Q[l] - Q[l+1]) * (Sy[l] - Sy[l+1]) / ph[l];
                }
                if (l > 0) {
                    sum_qs += abs(Q[l] - Q[l-1]);
                    deltaQx = max(float(0.0), Q[l] - Q[l-1]) * (Sx[l] - Sx[l-1]) / ph[l];
                    deltaQy = max(float(0.0), Q[l] - Q[l-1]) * (Sy[l] - Sy[l-1]) / ph[l];
                }

                for (int j = 0; j < l; j++) {
                    sum_phs[(y * nx + x) * nlayers + l] += rho[j] / rho[l] * ph[j];
                }
                for (int j = l+1; j < nlayers; j++) {
                    sum_phs[(y * nx + x) * nlayers + l] += ph[j];
                }

                // D
                Up[(y * nx + x) * nlayers + l][0] += dt * sum_qs;

                // Sx
                Up[(y * nx + x) * nlayers + l][1] += dt * ph[l] * (-deltaQx);

                // Sy
                Up[(y * nx + x) * nlayers + l][2] += dt * ph[l] * (-deltaQy);


            }
        }
    }

    for (int x = 1; x < (nx-1); x++) {
        for (int y = 1; y < (ny-1); y++) {
            for (int l = 0; l < nlayers; l++) {
                // Sx
                Up[(y * nx + x) * nlayers + l][1] -= dt * U_half[(y * nx + x) * nlayers + l][0] * 0.5 / dx * (sum_phs[(y * nx + (x+1)) * nlayers + l] - sum_phs[(y * nx + (x-1)) * nlayers + l]);

                // Sy
                Up[(y * nx + x) * nlayers + l][2] -= dt * U_half[(y * nx + x) * nlayers + l][0] * 0.5 / dy * (sum_phs[((y+1) * nx + x) * nlayers + l] - sum_phs[((y-1) * nx + x) * nlayers + l]);


            }
        }
    }


    // copy back to grid
    for (int n = 0; n < nlayers*nx*ny; n++) {
        for (int i = 0; i < 3; i++) {
            U_grid[(t+1) * nlayers*nx*ny + n][i] = Up[n][i];
        }
    }

    bcs(t+1);

    delete[] ph;
    delete[] Sx;
    delete[] Sy;
    delete[] W;
    delete[] sum_phs;

    for (int i=0; i < nlayers*nx*ny; i++){
        delete[] Up[i];
        delete[] U_half[i];
    }

}

void Sea::run() {
    cout << "Beginning evolution.\n";

    for (int t = 0; t < nt; t++) {
        evolve(t);
    }
}

void Sea::output(char * filename) {
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
    Sea sea(nlayers, nx, ny, nt, xmin, xmax, ymin, ymax, rho, Q, alpha, beta, gamma, periodic);

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
