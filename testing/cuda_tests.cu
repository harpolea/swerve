#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_functions.h>
#include "../Mesh_cuda.h"
#include "../mesh_cuda_kernel.h"
#include "unit_tests.h"

using namespace std;

/*
This does some unit tests on some of the functions in swerve.
*/

__global__ void test_find_height(bool * passed) {

    float ph[] = {};
    float h[] = {};

    const float tol = 1.0e-5;

    for (int i = 0; i < 4; i++) {
        if ((abs((h[i] - find_height(ph[i])) / h[i]) > tol) && (abs(h[i] - find_height(ph[i])) > 0.01*tol)) {
            cout << h[i] << ',' <<  find_height(ph[i]) << '\n';
            *passed = false;
        }
    }

    *passed = true;
}

__global__ void test_find_pot(bool * passed) {
    *passed = true;
}

__global__ void test_rhoh_from_p(bool * passed) {
    *passed = true;
}

__global__ void test_p_from_rhoh(bool * passed) {
    *passed = true;
}

__global__ void test_h_dot(bool * passed) {
    *passed = true;
}

__global__ void test_calc_As(bool * passed) {
    *passed = true;
}

__global__ void test_shallow_water_fluxes(bool * passed) {
    *passed = true;
}

__global__ void test_compressible_fluxes(bool * passed) {
    *passed = true;
}

__global__ void test_p_from_swe_d(bool * passed) {
    *passed = true;
}

__global__ void test_height_err(bool * passed) {
    *passed = true;
}

void run() {

    bool passed = true;
    bool *passed_d;
    cudaMalloc((void**)&passed_d, sizeof(bool));

    test_find_height<<<1,1>>>(passed_d);
    cudaMemcpy(&passed, passed_d, sizeof(bool), cudaMemcpyDeviceToHost);
    if (passed) {
        cout << "find_height passed!\n";
    } else {
        cout << "find_height did not pass :(\n";
    }

    cudaFree(passed_d);
}
