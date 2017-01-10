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
    // define some test matrices
    float A[] = {1,2,3,4};
    float B[] = {2,0,0,0,2,0,0,0,2};
    float C[] = {4,7,1.4,67,1,2,3,7,2.66,1.2,5.0,2.0,5,1,2.5,1};

    Sea::invert_mat(A, 2, 2);
    Sea::invert_mat(B, 3, 3);
    Sea::invert_mat(C, 4, 4);

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

int main(int argc, char *argv[]) {

    bool passed = test_invert_matrix();

    if (passed) {
        cout << "Invert matrix passed!\n";
    } else {
        cout << "Invert matrix did not pass :(\n";
    }
}
