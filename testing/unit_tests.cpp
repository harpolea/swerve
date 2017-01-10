#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <algorithm>
#include "mpi.h"
#include "../Mesh_cuda.h"

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

    const float tol = 1.0e-6; // tolerance
    for (int i = 0; i < 4; i++) {
        if (abs(A[i] - a_inv[i]) > tol) {
            return false;
        }
    }
    for (int i = 0; i < 9; i++) {
        if (abs(B[i] - b_inv[i]) > tol) {
            return false;
        }
    }
    for (int i = 0; i < 16; i++) {
        if (abs(C[i] - c_inv[i]) > tol) {
            return false;
        }
    }

    return true;
}

bool test_cons_to_prim_comp() {
    /*
    Tests the function cons_to_prim_comp defined in mesh_cuda_kernel.cu,
    which converts conserved compressible variables to primitive compressible
    variables.
    */
    //(float * q_cons, float * q_prim, int nxf, int nyf,int nz, float gamma, float * gamma_up)

    float gamma = 5.0 / 3.0;
    float gamma_up[] = {0.80999862,  0.0 ,  0.0,  0.0,  0.80999862,
        0.0,  0.0,  0.0,  0.80999862};
    int nx = 1;
    int ny = 1;
    int nz = 1;

    // Define some primitive variables (rho, u, v, w, eps)
    float q_prim[] = {1.0, 0.1, 0.2, 0.3, 1.5};

    // Define corresponding conserved variables
    float W = 1./sqrt(1.0 - (q_prim[1]*q_prim[1]*gamma_up[0] + 2.0*q_prim[1]*q_prim[2]*gamma_up[1] + 2.0*q_prim[1]*q_prim[3]*gamma_up[2] + q_prim[2]*q_prim[2]*gamma_up[4] + 2.0*q_prim[2]*q_prim[3]*gamma_up[5] + q_prim[3]*q_prim[3]*gamma_up[8]));

    float h = 1.0 + gamma * q_prim[4];
    float p = (gamma - 1.0) * q_prim[0] * q_prim[4];

    float q_cons[] = {q_prim[0]*W, q_prim[0]*h*W*W*q_prim[1], q_prim[0]*h*W*W*q_prim[2], q_prim[0]*h*W*W*q_prim[3], q_prim[0]*W*(h*W-1) - p};

    float * q_new_prim = new float[5];

    cons_to_prim_comp(q_cons, q_new_prim, nx, ny, nz, gamma, gamma_up);

    const float tol = 1.0e-6;
    for (int i = 0; i < 5; i++) {
        if (abs(q_prim[i] - q_new_prim[i]) > tol) {
            return false;
        }
    }

    delete[] q_new_prim;

    return true;
}

int main(int argc, char *argv[]) {

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
}
