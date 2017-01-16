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


/*__global__ void test_find_height(bool * passed) {

    float ph[] = {1.0e-3, 1.0, 1.0e3};
    float h[] = {10001.000333, 2.313035285, 1.0};

    const float tol = 1.0e-5;

    for (int i = 0; i < 3; i++) {
        if ((abs((h[i] - find_height(ph[i])) / h[i]) > tol) && (abs(h[i] - find_height(ph[i])) > 0.01*tol)) {
            printf("%f, %f\n", h[i], find_height(ph[i]));
            *passed = false;
        }
    }

    *passed = true;
}*/

bool test_cons_to_prim_comp_d_wrapper() {
    int ntests = 100;
    bool *passed_vec_d, *passed_vec;
    passed_vec = (bool *)malloc(ntests*sizeof(bool));
    cudaMalloc((void**)&passed_vec_d, ntests*sizeof(bool));

    float * q_prim, *q_prim_d;
    q_prim = (float *)malloc(5*ntests*sizeof(float));
    cudaMalloc((void**)&q_prim_d, 5*ntests*sizeof(float));

    for (int i = 0; i < ntests; i++) {
        q_prim[i*5+0] = 10*r();
        q_prim[i*5+1] = 0.8*r()-0.4;
        q_prim[i*5+2] = r()-0.5;
        q_prim[i*5+3] = r()-0.5;
        q_prim[i*5+4] = 15*r();
    }
    cudaMemcpy(q_prim_d, q_prim, ntests*5*sizeof(float), cudaMemcpyHostToDevice);
    test_cons_to_prim_comp_d<<<1,ntests>>>(passed_vec_d, q_prim_d);
    cudaMemcpy(passed_vec, passed_vec_d, ntests*sizeof(bool), cudaMemcpyDeviceToHost);

    bool passed = true;
    for (int i = 0; i < ntests; i++) {
        if (!(passed_vec[i])) {
            passed = false;
        }
    }

    free(passed_vec);
    cudaFree(passed_vec_d);
    cudaFree(q_prim_d);
    free(q_prim);

    return passed;
}

bool test_swe_from_compressible() {
    bool passed = true;
    float gamma_up[] = {0.80999862,  0.0 ,  0.0,  0.0,  0.80999862,
        0.0,  0.0,  0.0,  0.80999862};
    float gamma = 5.0/3.0;
    int kx_offset = 0;
    int ky_offset = 0;
    float p_floor = 1.0;
    const int nxf = 5;
    int nyf = 1;
    int nz = 1;

    float * gamma_up_d;
    cudaMalloc((void**)&gamma_up_d, 9*sizeof(float));
    cudaMemcpy(gamma_up_d, gamma_up, 9*sizeof(float), cudaMemcpyHostToDevice);

    float q[] = {};
    float q_swe[] = {};
    float rho[] = {};

    float * q_d, *q_swe_new, * q_swe_d, *rho_d;
    cudaMalloc((void**)&q_d, 5*nxf*sizeof(float));
    q_swe_new = (float *)malloc(3*nxf*sizeof(float));
    cudaMalloc((void**)&q_swe_d, 3*nxf*sizeof(float));
    cudaMalloc((void**)&rho_d, nxf*sizeof(float));
    cudaMemcpy(q_d, q, 5*nxf*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rho_d, rho, nxf*sizeof(float), cudaMemcpyHostToDevice);

    swe_from_compressible<<<1,nxf>>>(q_d, q_swe_d, nxf, nyf, nz, gamma_up_d,
                                   rho_d, gamma, kx_offset, ky_offset,
                                   p_floor);

    cudaMemcpy(q_swe_new, q_swe_d, 3*nxf*sizeof(float), cudaMemcpyDeviceToHost);

    const float tol = 1.0e-5;
    for (int i = 0; i < nxf; i++) {
        for (int n = 0; n < 3; n++) {
            if ((abs((q_swe[i*3+n] - q_swe_new[i*3+n]) / q_swe[i*3+n]) > tol) && (abs(q_swe[i*3+n] - q_swe_new[i*3+n]) > 0.1*tol)) {
                printf("%f, %f\n", q_swe[i*3+n], q_swe_new[i*3+n]);
                passed = false;
            }
        }
    }

    cudaFree(gamma_up_d);
    cudaFree(q_d);
    free(q_swe_new);
    cudaFree(q_swe_d);
    cudaFree(rho_d);

    return passed;
}

void run_cuda_tests() {

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

    test_find_pot<<<1,1>>>(passed_d);
    cudaMemcpy(&passed, passed_d, sizeof(bool), cudaMemcpyDeviceToHost);
    if (passed) {
        cout << "find_pot passed!\n";
    } else {
        cout << "find_pot did not pass :(\n";
    }

    test_rhoh_from_p<<<1,1>>>(passed_d);
    cudaMemcpy(&passed, passed_d, sizeof(bool), cudaMemcpyDeviceToHost);
    if (passed) {
        cout << "rhoh_from_p passed!\n";
    } else {
        cout << "rhoh_from_p did not pass :(\n";
    }

    test_p_from_rhoh<<<1,1>>>(passed_d);
    cudaMemcpy(&passed, passed_d, sizeof(bool), cudaMemcpyDeviceToHost);
    if (passed) {
        cout << "p_from_rhoh passed!\n";
    } else {
        cout << "p_from_rhoh did not pass :(\n";
    }

    test_p_from_rho_eps<<<1,1>>>(passed_d);
    cudaMemcpy(&passed, passed_d, sizeof(bool), cudaMemcpyDeviceToHost);
    if (passed) {
        cout << "p_from_rho_eps passed!\n";
    } else {
        cout << "p_from_rho_eps did not pass :(\n";
    }

    test_hdot<<<1,1>>>(passed_d);
    cudaMemcpy(&passed, passed_d, sizeof(bool), cudaMemcpyDeviceToHost);
    if (passed) {
        cout << "h_dot passed!\n";
    } else {
        cout << "h_dot did not pass :(\n";
    }


    test_calc_As<<<1,1>>>(passed_d);
    cudaMemcpy(&passed, passed_d, sizeof(bool), cudaMemcpyDeviceToHost);
    if (passed) {
        cout << "calc_As passed!\n";
    } else {
        cout << "calc_As did not pass :(\n";
    }

    passed = test_cons_to_prim_comp_d_wrapper();
    if (passed) {
        cout << "cons_to_prim_comp_d passed!\n";
    } else {
        cout << "cons_to_prim_comp_d did not pass :(\n";
    }

    test_shallow_water_fluxes<<<1,1>>>(passed_d);
    cudaMemcpy(&passed, passed_d, sizeof(bool), cudaMemcpyDeviceToHost);
    if (passed) {
        cout << "shallow_water_fluxes passed!\n";
    } else {
        cout << "shallow_water_fluxes did not pass :(\n";
    }

    test_compressible_fluxes<<<1,1>>>(passed_d);
    cudaMemcpy(&passed, passed_d, sizeof(bool), cudaMemcpyDeviceToHost);
    if (passed) {
        cout << "compressible_fluxes passed!\n";
    } else {
        cout << "compressible_fluxes did not pass :(\n";
    }

    test_p_from_swe<<<1,1>>>(passed_d);
    cudaMemcpy(&passed, passed_d, sizeof(bool), cudaMemcpyDeviceToHost);
    if (passed) {
        cout << "p_from_swe passed!\n";
    } else {
        cout << "p_from_swe did not pass :(\n";
    }

    cudaFree(passed_d);
}
