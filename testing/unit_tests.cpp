#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <algorithm>
#include "mpi.h"
#include "../Mesh_cuda.h"
#include "../mesh_cuda_kernel.h"
#include <cstdlib>
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "unit_tests.h"

using namespace std;

/*
This does some unit tests on some of the functions in swerve.
*/

bool test_invert_matrix() {
    /*
    Tests the static function invert_mat in the Sea class.
    */
    // define some test matrices
    float A[] = {1,2,3,4};
    float B[] = {2,0,0,0,2,0,0,0,2};
    float C[] = {4,7,1.4,67,1,2,3,7,2.66,1.2,5.0,2.0,5,1,2.5,1};

    Sea::invert_mat(A, 2, 2);
    Sea::invert_mat(B, 3, 3);
    Sea::invert_mat(C, 4, 4);

    // inverses calculated using numpy.linalg.inv
    float a_inv[] = {-2, 1, 1.5, -0.5};
    float b_inv[] = {0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5};
    float c_inv[] = {0.01198038, -0.13723728, -0.04079525,  0.23956623, -0.10991996,
        1.25915208, -0.87570359,  0.30197986,  0.00981152, -0.18135812,
        0.39762458, -0.18311387,  0.02548928, -0.11957036,  0.08561838,
       -0.04202634};

    const float tol = 1.0e-5; // tolerance
    for (int i = 0; i < 4; i++) {
        if (abs((A[i] - a_inv[i])/A[i]) > tol) {
            cout << A[i] << ',' << a_inv[i] << '\n';
            return false;
        }
    }
    for (int i = 0; i < 9; i++) {
        if (abs((B[i] - b_inv[i])/B[i]) > tol) {
            cout << B[i] << ',' << b_inv[i] << '\n';
            return false;
        }
    }
    for (int i = 0; i < 16; i++) {
        if (abs((C[i] - c_inv[i])/C[i]) > tol) {
            cout << C[i] << ',' << c_inv[i] << '\n';
            return false;
        }
    }

    return true;
}

float r() {
    // generate random number between 0 and 1
    return (float) rand() / RAND_MAX;
}

bool test_cons_to_prim_comp() {
    /*
    Tests the function cons_to_prim_comp defined in mesh_cuda_kernel.cu,
    which converts conserved compressible variables to primitive compressible
    variables.
    */

    float gamma = 5.0 / 3.0;
    float gamma_up[] = {0.80999862,  0.0 ,  0.0,  0.0,  0.80999862,
        0.0,  0.0,  0.0,  0.80999862};
    int nx = 1;
    int ny = 1;
    int nz = 1;

    float * q_new_prim = new float[5];

    for (int i = 0; i < 10000; i++) {

        // Define some primitive variables (rho, u, v, w, eps)
        float q_prim[] = {10*r(), r()-0.5f, 1.2f*r()-0.6f, 1.2f*r()-0.6f, 15*r()};

        // Define corresponding conserved variables
        float W = 1.0 / sqrt(1.0 - (q_prim[1]*q_prim[1]*gamma_up[0] + 2.0*q_prim[1]*q_prim[2]*gamma_up[1] + 2.0*q_prim[1]*q_prim[3]*gamma_up[2] + q_prim[2]*q_prim[2]*gamma_up[4] + 2.0*q_prim[2]*q_prim[3]*gamma_up[5] + q_prim[3]*q_prim[3]*gamma_up[8]));

        float h = 1.0 + gamma * q_prim[4];
        float p = (gamma - 1.0) * q_prim[0] * q_prim[4];

        float q_cons[] = {q_prim[0]*W, q_prim[0]*h*W*W*q_prim[1], q_prim[0]*h*W*W*q_prim[2], q_prim[0]*h*W*W*q_prim[3], q_prim[0]*W*(h*W-1) - p};

        cons_to_prim_comp(q_cons, q_new_prim, nx, ny, nz, gamma, gamma_up);

        const float tol = 1.0e-4;
        for (int i = 0; i < 5; i++) {
            if ((abs((q_prim[i] - q_new_prim[i]) / q_prim[i]) > tol) && (abs(q_prim[i] - q_new_prim[i]) > 0.01*tol)) {
                cout << i << ' ' << W << ' '<< q_prim[i] << ',' << q_new_prim[i] << '\n';
                delete[] q_new_prim;
                return false;
            }
        }
    }

    delete[] q_new_prim;

    return true;
}

bool test_nan_check() {
    float a[] = {1.0e-6, 4.0e10, 978324, 0.00000284765893}; //should pass

    float b = sqrt(-1.0); // should not pass

    for (int i = 0; i < 4; i++) {
        if (nan_check(a[i])) {
            return false;
        }
    }

    if (!(nan_check(b))) {
        return false;
    }

    return true;
}

bool test_zbrent() {
    return true;
}

bool test_W_swe() {
    float gamma_up[] = {0.80999862,  0.0 ,  0.0,  0.0,  0.80999862,
        0.0,  0.0,  0.0,  0.80999862};

    for (int i = 0; i < 100; i++) {
        // generate primitive variables
        float q_prim[] = {0.5f*r()+1.1f, 1.2f*r()-0.6f, 1.2f*r()-0.6f};

        // calculate W
        float W = 1.0 / sqrt(1.0 - (q_prim[1] * q_prim[1]*gamma_up[0] + 2.0 * q_prim[1] * q_prim[2] * gamma_up[1] + q_prim[2] * q_prim[2] * gamma_up[4]));

        // turn into conserved variables
        float q_cons[] = {q_prim[0] * W, q_prim[0]*q_prim[1]*W*W, q_prim[0]*q_prim[2]*W*W};

        const float tol = 1.0e-4;
        if ((abs((W - W_swe(q_cons, gamma_up)) / W) > tol) && (abs(W - W_swe(q_cons, gamma_up)) > 0.01*tol)) {
            return false;
        }
    }

    return true;
}

bool test_phi() {
    // test superbee slope limiter

    float r[] = {-2.0e6, -1.0f, 0.0f, 0.1f, 0.498374983475f, 0.7f, 0.9999f, 1.5f, 742.362f};
    float phis[] = {0.0f, 0.0f, 0.0f, 0.2f, 0.996749967f, 1.0f, 1.0f, 0.8f, 2.690479201e-3f};

    const float tol = 1.0e-4;
    for (int i = 0; i < 9; i++) {
        if ((abs((phis[i] - phi(r[i])) / phis[i]) > tol) && (abs(phis[i] - phi(r[i])) > 0.01*tol)) {

            cout << phis[i] << ',' << phi(r[i]) << '\n';
            return false;
        }
    }
    return true;
}

bool test_p_from_rho_eps() {

    float gamma = 5.0/3.0;

    float rho[] = {1.0e-3f, 1.0e-3f, 0.1f, 123.812f, 1.0e6};
    float eps[] = {1.0e-3f, 1.0e3f, 123, 0.1f, 2345.234};
    float p[] = {6.6666666667e-7, 0.6666666667f, 8.2f, 8.2541333333f, 1563489333.0f};

    const float tol = 1.0e-5;
    for (int i = 0; i < 5; i++) {
        if ((abs((p[i] - p_from_rho_eps(rho[i], eps[i], gamma)) / p[i]) > tol) && (abs(p[i] - p_from_rho_eps(rho[i], eps[i], gamma)) > 0.01*tol)) {

            cout << p[i] << ',' << p_from_rho_eps(rho[i], eps[i], gamma) << '\n';
            return false;
        }
    }

    return true;
}

bool test_phi_from_p() {

    float gamma = 5.0/3.0;

    float rho[] = {1.0e-3, 1.0e-3, 0.5, 1.5, 1000.0};
    float p[] = {1.0e-3, 1.0e3, 2.5, 12987.23, 0.1};
    float A[] = {0.00001, 1.0, 3.0, 1000.0, 1.123987};
    float phi[] = {2.343173262, 3.129618564, 0.3243720865, 1.392121399, 2.716449225};

    const float tol = 1.0e-5;
    for (int i = 0; i < 5; i++) {
        if ((abs((phi[i] - phi_from_p(p[i], rho[i], gamma, A[i])) / phi[i]) > tol) && (abs(phi[i] - phi_from_p(p[i], rho[i], gamma, A[i])) > 0.01*tol)) {

            cout << phi[i] << ',' << phi_from_p(p[i], rho[i], gamma, A[i]) << '\n';
            return false;
        }
    }

    return true;
}

bool test_f_of_p() {
    // Calculated the test data using python

    float gamma = 5.0 / 3.0;
    float gamma_up[] = {0.80999862,  0.0 ,  0.0,  0.0,  0.80999862,
        0.0,  0.0,  0.0,  0.80999862};

    float q_cons[][5] = {{1.0e-3, 0.0, 0.0, 0.0, 3.0},
                         {1.0e-3, 0.4, -0.4, 0.4, 1.0e3},
                         {1.0e3, 0.0, 0.0, 0.0, 1.0e-3},
                         {5.0, 0.3, 0.1, 0.4, 1.0}};
    float p[] = {2.0, 50.0, 20.0, 1.0};
    float f[] = {0.0, 616.66641981029716, -19.99933333333335, -0.34621947550289045};

    for (int i = 0; i < 4; i++) {

        // Define some primitive variables (rho, u, v, w, eps)
        float new_f = f_of_p(p[i], q_cons[i][0], q_cons[i][1], q_cons[i][2], q_cons[i][3], q_cons[i][4], gamma, gamma_up);

        const float tol = 1.0e-5;
        if ((abs((f[i] - new_f) / f[i]) > tol) && (abs(f[i] - new_f) > 0.1*tol)) {
                cout << f[i] << ',' << new_f << '\n';
                return false;
        }
    }

    return true;
}

bool test_p_from_swe() {

    float gamma_up[] = {0.80999862,  0.0 ,  0.0,  0.0,  0.80999862,
        0.0,  0.0,  0.0,  0.80999862};
    float gamma = 5.0/3.0;

    float q_cons[][3] = {{1.0e-3, 0.0, 0.0},
                         {1.0, 0.0, 0.0},
                         {1.0, 0.4, 0.4},
                         {2.0e-2, 0.4, 0.4}};

    float A[] = {1.0e-3, 0.23, 5.0, 1.0e3};
    float rho[] = {1.e-3,1.0, 15.0, 500.0};
    float p[] = {6.67222531e-7, 0.7207894444, 12.56043128, 200.7858403};

    int nx = 1;
    int ny = 1;
    int nz = 1;

    const float tol = 1.0e-4;

    for (int i = 0; i < 4; i++) {
        float p_new;
        p_from_swe(q_cons[i], &p_new, nx, ny, nz, gamma_up, rho[i], gamma, A[i]);
        if ((abs((p[i] - p_new) / p[i]) > tol) && (abs(p[i] - p_new) > 0.01*tol)) {
            cout << p[i] << ',' <<  p_new << '\n';
            return false;
        }
    }

    return true;
}

int main(int argc, char *argv[]) {

    srand(time(0));

    bool passed = test_invert_matrix();
    if (passed) {
        cout << "Invert matrix passed!\n";
    } else {
        cout << "Invert matrix did not pass :(\n";
    }

    passed = test_cons_to_prim_comp();
    if (passed) {
        cout << "Cons to prim passed!\n";
    } else {
        cout << "Cons to prim did not pass :(\n";
    }

    passed = test_nan_check();
    if (passed) {
        cout << "nan_check passed!\n";
    } else {
        cout << "nan_check did not pass :(\n";
    }

    passed = test_W_swe();
    if (passed) {
        cout << "W_swe passed!\n";
    } else {
        cout << "W_swe did not pass :(\n";
    }

    passed = test_phi();
    if (passed) {
        cout << "phi passed!\n";
    } else {
        cout << "phi did not pass :(\n";
    }

    passed = test_p_from_rho_eps();
    if (passed) {
        cout << "p_from_rho_eps passed!\n";
    } else {
        cout << "p_from_rho_eps did not pass :(\n";
    }

    passed = test_phi_from_p();
    if (passed) {
        cout << "phi_from_p passed!\n";
    } else {
        cout << "phi_from_p did not pass :(\n";
    }

    passed = test_f_of_p();
    if (passed) {
        cout << "f_of_p passed!\n";
    } else {
        cout << "f_of_p did not pass :(\n";
    }

    passed = test_p_from_swe();
    if (passed) {
        cout << "p_from_swe passed!\n";
    } else {
        cout << "p_from_swe did not pass :(\n";
    }

    run_cuda_tests();
}
