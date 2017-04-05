/**
File containing test routines.
**/

__global__ void test_find_height(bool * passed) {
    *passed = true;

    float ph[] = {1.0e-3, 1.0, 1.0e3};
    float h[] = {1001.000333, 2.313035285, 2.0};

    const float tol = 1.0e-5;

    for (int i = 0; i < 4; i++) {
        if ((abs((h[i] - find_height(ph[i])) / h[i]) > tol) && (abs(h[i] - find_height(ph[i])) > 0.01*tol)) {
            printf("%f, %f\n", h[i], find_height(ph[i]));
            *passed = false;
        }
    }
}

__global__ void test_find_pot(bool * passed) {
    *passed = true;

    float r[] = {2.001, 25.0, 1.0e3};
    float ph[] = {3.800701167, 0.04169080447, 1.001001335e-3};

    const float tol = 1.0e-5;

    for (int i = 0; i < 4; i++) {
        if ((abs((ph[i] - find_pot(r[i])) / ph[i]) > tol) && (abs(ph[i] - find_pot(r[i])) > 0.01*tol)) {
            printf("%f, %f\n", ph[i], find_pot(r[i]));
            *passed = false;
        }
    }
}

__global__ void test_rhoh_from_p(bool * passed) {
    *passed = true;

    float gamma = 5.0/3.0;

    float rho[] = {1.0e-3, 1.0e-3, 1.0e3, 1.0e3, 1.124};
    float p[] = {1.0e-3, 1.0e3, 1.0e-3, 1.0e3, 13.12};
    float rhoh[] = {3.5e-3, 2500.001, 1000.0025, 3500, 33.924};

    const float tol = 1.0e-5;

    for (int i = 0; i < 6; i++) {
        float new_rhoh = rhoh_from_p(p[i], rho[i], gamma);
        if ((abs((rhoh[i] - new_rhoh) / rhoh[i]) > tol) && (abs(rhoh[i] - new_rhoh) > 0.01*tol)) {
            printf("%f, %f\n", rhoh[i], new_rhoh);
            *passed = false;
        }
    }
}

__global__ void test_p_from_rhoh(bool * passed) {
    *passed = true;

    float gamma = 5.0/3.0;

    float rho[] = {1.0e-3, 1.0e-3, 1.0e3, 1.0e3, 1.124};
    float rhoh[] = {3.5e-3, 2500.001, 1000.0025, 3500, 33.924};
    float p[] = {1.0e-3, 1.0e3, 1.0e-3, 1.0e3, 13.12};

    const float tol = 1.0e-5;

    for (int i = 0; i < 6; i++) {
        float new_p = p_from_rhoh(rhoh[i], rho[i], gamma);
        if ((abs((p[i] - new_p) / p[i]) > tol) && (abs(p[i] - new_p) > 0.1*tol)) {
            printf("%f, %f\n", p[i], new_p);
            *passed = false;
        }
    }
}

__global__ void test_p_from_rho_eps(bool * passed) {
    *passed = true;

    float gamma = 5.0/3.0;

    float rho[] = {1.0e-3, 0.1, 1.0e3, 1.0e3, 1.124};
    float eps[] = {0.1, 1.e-3, 1.0e3, 1.0, 33.924};
    float p[] = {6.6666666667e-5, 6.6666666667e-5, 666666.667, 666.6666667, 25.420384};

    const float tol = 1.0e-5;

    for (int i = 0; i < 6; i++) {
        float new_p = p_from_rho_eps(rho[i], eps[i], gamma);
        if ((abs((p[i] - new_p) / p[i]) > tol) && (abs(p[i] - new_p) > 0.1*tol)) {
            printf("%f, %f\n", p[i], new_p);
            *passed = false;
        }
    }
}

__global__ void test_hdot(bool * passed) {
    *passed = true;

    float phi[] = {1.0e-3, 1.e-3, 1.0e-3, 1.0, 100.0};
    float old_phi[] = {1.e-3, 2.e-3, 2.0e-3, 1.1, 101.0};
    float dt[] = {1.e-3, 1.e-3, 1.0, 1.0e-3, 0.1};
    float hdot[] = {0.0, 999999.66666672146, 999.9996666667214, 72.406166096631111, 0.0};

    const float tol = 1.0e-5;

    for (int i = 0; i < 6; i++) {
        float new_hdot = h_dot(phi[i], old_phi[i], dt[i]);
        if ((abs((hdot[i] - new_hdot) / hdot[i]) > tol) && (abs(hdot[i] - new_hdot) > 0.1*tol)) {
            printf("%f, %f\n", hdot[i], new_hdot);
            *passed = false;
        }
    }
}

__global__ void test_calc_As(bool * passed) {
    *passed = true;

    int nlayers = 2;
    float gamma = 5.0/3.0;

    float rhos[] = {1.0, 1.0,1.0,1.0, 1.0, 1.0e-3, 1.0, 1.0e-3, 1.0e-3, 1.0e3};
    float phis[] = {1.0, 1.0, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1.0,1.0, 1.0, 1.0};
    float As[] = {0.082085,  0.99750312, 9.97503122e-01, 8.20849986e-02, 8.20849986e-05};

    float * A, *rho, *phi;

    A = (float *)malloc(nlayers * sizeof(float));
    rho = (float *)malloc(nlayers * sizeof(float));
    phi = (float *)malloc(nlayers * sizeof(float));

    const float tol = 1.0e-5;

    for (int i = 0; i < 6; i++) {
        for (int l = 0; l < nlayers; l++) {
            rho[l] = rhos[i*nlayers+l];
            phi[l] = phis[i*nlayers+l];
        }
        calc_As(rho, phi, A, nlayers, gamma, phi[0], rho[0]);
        if ((abs((As[i] - A[0]) / As[i]) > tol) && (abs(As[i] - A[0]) > 0.1*tol)) {
            printf("%f, %f\n", As[i], A[0]);
            *passed = false;
        }
    }
    free(As);
    free(rho);
    free(phi);
}

__global__ void test_cons_to_prim_comp_d(bool * passed, float * q_prims) {

    int i = blockIdx.x * blockDim.y * blockDim.x + threadIdx.x + threadIdx.y * blockDim.x;
    passed[i] = true;

    float gamma = 5.0 / 3.0;
    //gamma_up_d = {0.80999862,  0.0 ,  0.0,  0.0,  0.80999862,
        //0.0,  0.0,  0.0,  0.80999862};

    float * q_new_prim, *q_prim;
    q_new_prim = (float *)malloc(6*sizeof(float));
    q_prim = (float *)malloc(6*sizeof(float));

    // Define some primitive variables (rho, u, v, w, eps)
    for (int j = 0; j < 6; j++) {
        q_prim[j] = q_prims[i*6+j];
    }

    // Define corresponding conserved variables
    float W = 1.0 / sqrt(1.0 - (q_prim[1]*q_prim[1]*gamma_up_d[0] + 2.0*q_prim[1]*q_prim[2]*gamma_up_d[1] + 2.0*q_prim[1]*q_prim[3]*gamma_up_d[2] + q_prim[2]*q_prim[2]*gamma_up_d[4] + 2.0*q_prim[2]*q_prim[3]*gamma_up_d[5] + q_prim[3]*q_prim[3]*gamma_up_d[8]));

    float h = 1.0 + gamma * q_prim[4];
    float p = (gamma - 1.0) * q_prim[0] * q_prim[4];

    float q_cons[] = {q_prim[0]*W, q_prim[0]*h*W*W*q_prim[1], q_prim[0]*h*W*W*q_prim[2], q_prim[0]*h*W*W*q_prim[3], q_prim[0]*W*(h*W-1) - p, q_prim[0]*q_prim[5]*W};

    cons_to_prim_comp_d(q_cons, q_new_prim, gamma);

    const float tol = 1.0e-4;
    for (int j = 0; j < 6; j++) {
        if ((abs((q_prim[j] - q_new_prim[j]) / q_prim[j]) > tol) && (abs(q_prim[j] - q_new_prim[j]) > 0.01*tol)) {
            printf("%f, %f\n", q_prim[j], q_new_prim[j]);
            passed[i] = false;
        }
    }
    free(q_new_prim);
    free(q_prim);
}

__global__ void test_shallow_water_fluxes(bool * passed) {
    *passed = true;

    float gamma = 5.0 / 3.0;
    //gamma_up_d = {0.80999862,  0.0 ,  0.0,  0.0,  0.80999862,
        //0.0,  0.0,  0.0,  0.80999862};
    float alpha = 0.9;
    //beta_d = {0.1, -0.2, 0.0};

    float qs[] = {0.1,0.0,0.0,0.0,
                  0.1,0.0,0.0,0.0,
                  1.0e-3,0.5,0.0,0.0,
                  1.0e-3,0.5,0.0,0.0,
                  1.e3,0.5,0.5,0.0,
                  1.e3,0.5,0.5,0.0};
    int dirs[] = {0,1,0,1,0,1};
    float fs[] = {-0.01111111,0.005, -0.0, 0.0,
                  0.02222222,  0.0        ,  0.005, 0.0,
                  0.00078889,  0.39444295,  0.0, 0.0,
                  2.22222222e-04,   1.11111111e-01,   0.0, 0.0,
                  -1.10706112e+02,   4.99999742e+05,  -5.53530559e-02, 0.0,
                  2.22627221e+02,   1.11313611e-01,   4.99999909e+05, 0.0};
    float * f, * q;
    f = (float *)malloc(4*sizeof(float));
    q = (float *)malloc(4*sizeof(float));

    const float tol = 1.0e-5;
    for (int i = 0; i < 6; i++) {
        for (int n = 0; n < 4; n++) {
            q[n] = qs[4*i+n];
        }
        shallow_water_fluxes(q, f, dirs[i], alpha, gamma);
        for (int n = 0; n < 4; n++) {
            if ((abs((fs[4*i+n] - f[n]) / fs[4*i+n]) > tol) && (abs(fs[4*i+n] - f[n]) > 0.1*tol)) {
                printf("%f, %f\n", fs[4*i+n], f[n]);
                *passed = false;
            }
        }
    }
    free(f);
    free(q);
}

__global__ void test_compressible_fluxes(bool * passed) {
    *passed = true;

    float gamma = 5.0 / 3.0;
    //gamma_up_d = {0.80999862,  0.0 ,  0.0,  0.0,  0.80999862,
        //0.0,  0.0,  0.0,  0.80999862};
    float alpha = 0.9;
    //beta_d = {0.1, -0.2, 0.3};

    float qs[] = {1.,  0.,  0.,  0.,  1., 0.0,
                  1.,  0.,  0.,  0.,  1., 0.0,
                  1.,  0.,  0.,  0.,  1., 0.0,
                  1.13133438,  1.02393398,  1.02393398,  1.02393398,  1.61511222, 0.0,
                  1.13133438,  1.02393398,  1.02393398,  1.02393398,  1.61511222, 0.0,
                  0.00113133,  0.00102393, -0.00102393,  0.00102393,  0.00161511, 0.0,
                  0.00113133,  0.00102393, -0.00102393,  0.00102393,  0.00161511, 0.0,
                  0.01012376,  0.00104199, -0.00104199,  0.00104199,  0.00022944, 0.0};
    int dirs[] = {0,1,2,0,1,0,1,2};
    float fs[] = {-0.11111111,  0.66666667, -0.0, -0.0, -0.11111111, 0.0,
                  0.22222222,  0.0,  0.66666667,  0.0,  0.22222222, 0.0,
                  -0.33333333, -0.0,  0.0,  0.66666667, -0.33333333, 0.0,
                  0.14920997,  0.80171176,  0.13504509,  0.13504509,  0.41301469, 0.0,
                  0.52632143,  0.47635642,  1.14302308,  0.47635642,  0.95138543, 0.0,
                  0.00014921,  0.00080171, -0.00013505,  0.00013505,  0.00041301, 0.0,
                  -2.35061459e-05,  -2.12746488e-05,   6.87941315e-04, -2.12746488e-05,  -2.33557774e-04, 0.0,
                  -2.55456349e-03,  -2.62928173e-04,   2.62928173e-04, -1.96261506e-04,  -5.12293429e-05, 0.0};
    float * f, * q;
    f = (float *)malloc(6*sizeof(float));
    q = (float *)malloc(6*sizeof(float));

    const float tol = 1.0e-4;
    for (int i = 0; i < 8; i++) {
        for (int n = 0; n < 6; n++) {
            q[n] = qs[6*i+n];
        }
        compressible_fluxes(q, f, dirs[i], alpha, gamma);
        for (int n = 0; n < 6; n++) {
            if ((abs((fs[6*i+n] - f[n]) / fs[6*i+n]) > tol) && (abs(fs[6*i+n] - f[n]) > 0.1*tol)) {
                printf("%f, %f\n", fs[6*i+n], f[n]);
                *passed = false;
            }
        }
    }
    free(f);
    free(q);
}

__global__ void test_p_from_swe(bool * passed) {
    *passed = true;
    float gamma = 5.0 / 3.0;
    //gamma_up_d = {0.80999862,  0.0 ,  0.0,
                        //0.0,  0.80999862, 0.0,
                        //0.0,  0.0,  0.80999862};

    float qs[] = {1.0, 0.0, 0.0, 0.0,
                  1.0, 0.5, 0.5, 0.0,
                  1.0, 0.0, 0.0, 0.0,
                  1.0, 0.0, 0.0, 0.0,
                  5.0, 0.3, -0.3, 0.0};
    float rhos[] = {1.0, 1.0, 1.e-3, 1.0, 1.0e3};
    float Ws[] = {1.0, 1.1853266680540011, 1.0, 1.0, 1.0029117558708742};
    float As[] = {1.0, 1.0, 1.0, 1.e-3, 1.0};
    float ps[] = {4.472997584, 2.896404957, 4.872597584, -0.3951270024, 103109.4291};

    float * q;
    q = (float *)malloc(4*sizeof(float));

    const float tol = 1.0e-5;
    for (int i = 0; i < 5; i++) {
        for (int n = 0; n < 4; n++) {
            q[n] = qs[i*4+n];
        }
        float p = p_from_swe(q, rhos[i], gamma, Ws[i], As[i]);

        if ((abs((ps[i] - p) / ps[i]) > tol) && (abs(ps[i] - p) > 0.1*tol)) {
            printf("%f, %f\n", ps[i], p);
            *passed = false;
        }
    }
    free(q);
}
