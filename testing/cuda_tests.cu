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

bool test_cons_to_prim_comp_d_wrapper() {
    int ntests = 100;
    bool *passed_vec_d, *passed_vec;
    passed_vec = (bool *)malloc(ntests*sizeof(bool));
    cudaMalloc((void**)&passed_vec_d, ntests*sizeof(bool));

    float * q_prim, *q_prim_d;
    q_prim = (float *)malloc(6*ntests*sizeof(float));
    cudaMalloc((void**)&q_prim_d, 6*ntests*sizeof(float));

    for (int i = 0; i < ntests; i++) {
        q_prim[i*6+0] = 10*r();
        q_prim[i*6+1] = 0.8*r()-0.4;
        q_prim[i*6+2] = r()-0.5;
        q_prim[i*6+3] = r()-0.5;
        q_prim[i*6+4] = 15*r();
        q_prim[i*6+5] = r();
    }
    cudaMemcpy(q_prim_d, q_prim, ntests*6*sizeof(float), cudaMemcpyHostToDevice);
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
    /*
    TODO: FIX THIS TEST
    */
    bool passed = true;
    float gamma_up[] = {0.80999862,  0.0 ,  0.0,  0.0,  0.80999862,
        0.0,  0.0,  0.0,  0.80999862};
    float gamma = 5.0/3.0;
    int kx_offset = 0;
    int ky_offset = 0;
    int nxs[] = {10, 5};
    int nys[] = {1,1};
    int nzs[] = {1, 2};
    const int coarse_level = 0;

    float * gamma_up_d;
    cudaMalloc((void**)&gamma_up_d, 9*sizeof(float));
    cudaMemcpy(gamma_up_d, gamma_up, 9*sizeof(float), cudaMemcpyHostToDevice);

    int matching_indices[] = {1, 10, 0, 0};

    float q[] = {1.0,0.0,0.0,0.0,1.0, 0.0,
                 0.001, 0.0, 0.0, 0.0, 0.001, 0.0,
                 1000., 0.0, 0.0, 0.0, 1000., 0.0,
                 1.05245657, 0.59075458, 0.59075458, 0.59075458, 1.23464966, 0.0,
                 1.03406473, 0.2142144, 0.2142144, 0.0, 0.03634062, 0.0, 1.0,0.0,0.0,0.0,1.0, 0.0,
                  0.001, 0.0, 0.0, 0.0, 0.001, 0.0,
                  1000., 0.0, 0.0, 0.0, 1000., 0.0,
                  1.05245657, 0.59075458, 0.59075458, 0.59075458, 1.23464966, 0.0,
                  1.03406473, 0.2142144, 0.2142144, 0.0, 0.03634062, 0.0};
    float qc[] = {0.39233170120469052, 0.0, -0.0, 0.0,
                     -2.3707704103881642, -0.0, 0.0, 0.0,
                     3.1554338127975452, 0.0, -0.0, 0.0,
                     0.41291207788984285, 0.086914406123075383, 0.086914406123075383, 0.0,
                     0.00068880264357815526, 0.000142453303890117, 0.000142453303890117, 0.0};
    float q_swe[] = {0.39233170120469052, 0.0, -0.0, 0.0,
                     -2.3707704103881642, -0.0, 0.0, 0.0,
                     3.1554338127975452, 0.0, -0.0, 0.0,
                     0.41291207788984285, 0.086914406123075383, 0.086914406123075383, 0.0,
                     0.00068880264357815526, 0.000142453303890117, 0.000142453303890117, 0.0};
    float rho[] = {1.0, 1.0e-3, 1.0e3, 1.0, 1.0,1.0, 1.0e-3, 1.0e3, 1.0, 1.0};

    float * q_d, *q_swe_new, * q_swe_d, *rho_d, *qc_d;
    cudaMalloc((void**)&q_d, 6*nxs[1]*nzs[1]*sizeof(float));
    q_swe_new = (float *)malloc(4*nxs[1]*nzs[1]*sizeof(float));
    cudaMalloc((void**)&q_swe_d, 4*nxs[1]*nzs[1]*sizeof(float));

    cudaMalloc((void**)&rho_d, nxs[1]*nzs[1]*sizeof(float));
    cudaMemcpy(q_d, q, 6*nxs[1]*nzs[1]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rho_d, rho, nxs[1]*nzs[1]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&qc_d, 4*nxs[0]*nys[0]*nzs[0]*sizeof(float));
    cudaMemcpy(qc_d, qc, 4*nxs[0]*nys[0]*nzs[0]*sizeof(float), cudaMemcpyHostToDevice);

    int *nxs_d, * nys_d, * nzs_d;
    cudaMalloc((void**)&nxs_d, 2*sizeof(int));
    cudaMemcpy(nxs_d, nxs, 2*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&nys_d, 2*sizeof(int));
    cudaMemcpy(nys_d, nys, 2*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&nzs_d, 2*sizeof(int));
    cudaMemcpy(nzs_d, nzs, 2*sizeof(int), cudaMemcpyHostToDevice);

    int * matching_indices_d;
    cudaMalloc((void**)&matching_indices_d, 4*sizeof(int));
    cudaMemcpy(matching_indices_d, matching_indices, 4*sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(nxs[1], nys[1], nzs[1]);

    swe_from_compressible<<<1,block>>>(q_d, q_swe_d,
                                   nxs_d, nys_d, nzs_d, gamma_up_d,
                                   rho_d, gamma, kx_offset, ky_offset,
                                   qc_d, matching_indices_d, coarse_level);

    cudaMemcpy(q_swe_new, q_swe_d, 4*nxs[1]*nzs[1]*sizeof(float), cudaMemcpyDeviceToHost);

    const float tol = 1.0e-5;
    for (int i = 0; i < nxs[1]; i++) {
        for (int n = 0; n < 4; n++) {
            if ((abs((q_swe[i*4+n] - q_swe_new[i*4+n]) / q_swe[i*4+n]) > tol) && (abs(q_swe[i*4+n] - q_swe_new[i*4+n]) > 0.1*tol)) {
                printf("component %d: %f, %f\n", n, q_swe[i*4+n], q_swe_new[i*4+n]);
                passed = false;
            }
        }
    }

    cudaFree(gamma_up_d);
    cudaFree(q_d);
    free(q_swe_new);
    cudaFree(q_swe_d);
    cudaFree(rho_d);

    cudaFree(nxs_d);
    cudaFree(nys_d);
    cudaFree(nzs_d);
    cudaFree(matching_indices_d);

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

    passed = test_swe_from_compressible();
    if (passed) {
        cout << "swe_from_compressible passed!\n";
    } else {
        cout << "swe_from_compressible did not pass :(\n";
    }

    cudaFree(passed_d);
}
