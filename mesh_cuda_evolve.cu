/**
File containing routines which model the evolution.
**/

__global__ void evolve_fv(float * beta_d, float * gamma_up_d,
                     float * Un_d, flux_func_ptr flux_func,
                     float * qx_plus_half, float * qx_minus_half,
                     float * qy_plus_half, float * qy_minus_half,
                     float * fx_plus_half, float * fx_minus_half,
                     float * fy_plus_half, float * fy_minus_half,
                     int nx, int ny, int nz, int vec_dim, float alpha, float gamma,
                     float dx, float dy, float dt,
                     int kx_offset, int ky_offset) {
    /**
    First part of evolution through one timestep using finite volume methods.
    Reconstructs state vector to cell boundaries using slope limiter
    and calculates fluxes there.

    NOTE: we assume that beta is smooth so can get value at cell boundaries with simple averaging

    Parameters
    ----------
    beta_d : float *
        shift vector at each grid point.
    gamma_up_d : float *
        gamma matrix at each grid point
    Un_d : float *
        state vector at each grid point in each layer
    flux_func : flux_func_ptr
        pointer to function to be used to calulate fluxes
    qx_plus_half, qx_minus_half : float *
        state vector reconstructed at right and left boundaries
    qy_plus_half, qy_minus_half : float *
        state vector reconstructed at top and bottom boundaries
    fx_plus_half, fx_minus_half : float *
        flux vector at right and left boundaries
    fy_plus_half, fy_minus_half : float *
        flux vector at top and bottom boundaries
    nx, ny, nz : int
        dimensions of grid
    alpha, gamma : float
        lapse function and adiabatic index
    dx, dy, dt : float
        grid dimensions and timestep
    kx_offset, ky_offset : int
        x, y offset for current kernel
    */

    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    int offset = ((z * ny + y) * nx + x) * vec_dim;

    float * q_p, *q_m, * f;
    q_p = (float *)malloc(vec_dim * sizeof(float));
    q_m = (float *)malloc(vec_dim * sizeof(float));
    f = (float *)malloc(vec_dim * sizeof(float));

    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1)) && (z < nz)) {

        // x-direction
        for (int i = 0; i < vec_dim; i++) {
            float S_upwind = (Un_d[((z * ny + y) * nx + x+1) * vec_dim + i] -
                Un_d[((z * ny + y) * nx + x) * vec_dim + i]);
            float S_downwind = (Un_d[((z * ny + y) * nx + x) * vec_dim + i] -
                Un_d[((z * ny + y) * nx + x-1) * vec_dim + i]);
            float S = 0.5 * (S_upwind + S_downwind); // S_av

            float r = 1.0e6;

            // make sure don't divide by zero
            if (abs(S_downwind) > 1.0e-7) {
                r = S_upwind / S_downwind;
            }

            S *= phi(r);

            q_p[i] = Un_d[offset + i] + S * 0.5;
            q_m[i] = Un_d[offset + i] - S * 0.5;
        }

        // fluxes
        flux_func(q_p, f, 0, gamma_up_d, alpha, beta_d, gamma);

        for (int i = 0; i < vec_dim; i++) {
            qx_plus_half[offset + i] = q_p[i];
            fx_plus_half[offset + i] = f[i];
        }

        flux_func(q_m, f, 0, gamma_up_d, alpha, beta_d, gamma);

        for (int i = 0; i < vec_dim; i++) {
            qx_minus_half[offset + i] = q_m[i];
            fx_minus_half[offset + i] = f[i];
            //if (nan_check(q_p[i]) || nan_check(q_m[i]) || nan_check(fx_plus_half[offset + i]) || nan_check(fx_minus_half[offset + i])) printf("(%d, %d, %d) i: %d, qx_p: %f, qx_m: %f, fx_p: %f, fx_m: %f\n", x, y, z, i, q_p[i], q_m[i], fx_plus_half[offset + i], fx_minus_half[offset + i]);
        }

        // y-direction
        for (int i = 0; i < vec_dim; i++) {
            float S_upwind = (Un_d[((z * ny + y+1) * nx + x) * vec_dim + i] -
                Un_d[((z * ny + y) * nx + x) * vec_dim + i]);
            float S_downwind = (Un_d[((z * ny + y) * nx + x) * vec_dim + i] -
                Un_d[((z * ny + y-1) * nx + x) * vec_dim + i]);
            float S = 0.5 * (S_upwind + S_downwind); // S_av

            float r = 1.0e6;

            // make sure don't divide by zero
            if (abs(S_downwind) > 1.0e-7) {
                r = S_upwind / S_downwind;
            }

            S *= phi(r);

            q_p[i] = Un_d[offset + i] + S * 0.5;
            q_m[i] = Un_d[offset + i] - S * 0.5;
        }

        // fluxes

        flux_func(q_p, f, 1, gamma_up_d, alpha, beta_d, gamma);

        for (int i = 0; i < vec_dim; i++) {
            qy_plus_half[offset + i] = q_p[i];
            fy_plus_half[offset + i] = f[i];
        }

        flux_func(q_m, f, 1, gamma_up_d, alpha, beta_d, gamma);

        for (int i = 0; i < vec_dim; i++) {
            qy_minus_half[offset + i] = q_m[i];
            fy_minus_half[offset + i] = f[i];
            //if (nan_check(q_p[i]) || nan_check(q_m[i])) printf("(%d, %d, %d) i: %d, qy_p: %f, qy_m: %f\n", x, y, z, i, q_p[i], q_m[i]);
        }
    }

    free(q_p);
    free(q_m);
    free(f);
}

__global__ void evolve_z(float * beta_d, float * gamma_up_d,
                     float * Un_d, flux_func_ptr flux_func,
                     float * qz_plus_half, float * qz_minus_half,
                     float * fz_plus_half, float * fz_minus_half,
                     int nx, int ny, int nz, int vec_dim, float alpha, float gamma,
                     float dz, float dt,
                     int kx_offset, int ky_offset) {
    /**
    First part of evolution through one timestep using finite volume methods.
    Reconstructs state vector to cell boundaries using slope limiter
    and calculates fluxes there.

    NOTE: we assume that beta is smooth so can get value at cell boundaries with simple averaging

    Parameters
    ----------
    beta_d : float *
        shift vector at each grid point.
    gamma_up_d : float *
        gamma matrix at each grid point
    Un_d : float *
        state vector at each grid point in each layer
    flux_func : flux_func_ptr
        pointer to function to be used to calculate fluxes
    qz_plus_half, qz_minus_half : float *
        state vector reconstructed at top and bottom boundaries
    fz_plus_half, fz_minus_half : float *
        flux vector at top and bottom boundaries
    nx, ny, nz : int
        dimensions of grid
    vec_dim : int
        dimension of state vector
    alpha, gamma : float
        lapse function and adiabatic index
    dz, dt : float
        vertical grid spacing and timestep
    kx_offset, ky_offset : int
        x, y offset for current kernel
    */

    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    int offset = ((z * ny + y) * nx + x) * vec_dim;

    float * q_p, *q_m, * f;
    q_p = (float *)malloc(vec_dim * sizeof(float));
    q_m = (float *)malloc(vec_dim * sizeof(float));
    f = (float *)malloc(vec_dim * sizeof(float));

    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1)) && (z > 0) && (z < (nz-1))) {

        // z-direction
        for (int i = 0; i < vec_dim; i++) {
            float S_upwind = (Un_d[(((z+1) * ny + y) * nx + x) * vec_dim + i] -
                Un_d[((z * ny + y) * nx + x) * vec_dim + i]);
            float S_downwind = (Un_d[((z * ny + y) * nx + x) * vec_dim + i] -
                Un_d[(((z-1) * ny + y) * nx + x) * vec_dim + i]);
            float S = 0.5 * (S_upwind + S_downwind); // S_av

            float r = 1.0e6;

            // make sure don't divide by zero
            if (abs(S_downwind) > 1.0e-7) {
                r = S_upwind / S_downwind;
            }

            S *= phi(r);

            q_p[i] = Un_d[offset + i] + S * 0.5;
            q_m[i] = Un_d[offset + i] - S * 0.5;
        }

        // fluxes
        flux_func(q_p, f, 2, gamma_up_d, alpha, beta_d, gamma);

        for (int i = 0; i < vec_dim; i++) {
            qz_plus_half[offset + i] = q_p[i];
            fz_plus_half[offset + i] = f[i];
        }

        flux_func(q_m, f, 2, gamma_up_d, alpha, beta_d, gamma);

        for (int i = 0; i < vec_dim; i++) {
            qz_minus_half[offset + i] = q_m[i];
            fz_minus_half[offset + i] = f[i];
        }
    }
    free(q_p);
    free(q_m);
    free(f);
}


__global__ void evolve_fv_fluxes(float * F,
                     float * qx_plus_half, float * qx_minus_half,
                     float * qy_plus_half, float * qy_minus_half,
                     float * fx_plus_half, float * fx_minus_half,
                     float * fy_plus_half, float * fy_minus_half,
                     int nx, int ny, int nz, int vec_dim, float alpha,
                     float dx, float dy, float dt,
                     int kx_offset, int ky_offset) {
    /**
    Calculates fluxes in finite volume evolution by solving the Riemann
    problem at the cell boundaries.

    Parameters
    ----------
    F : float *
        flux vector at each point in grid and each layer
    qx_plus_half, qx_minus_half : float *
        state vector reconstructed at right and left boundaries
    qy_plus_half, qy_minus_half : float *
        state vector reconstructed at top and bottom boundaries
    fx_plus_half, fx_minus_half : float *
        flux vector at right and left boundaries
    fy_plus_half, fy_minus_half : float *
        flux vector at top and bottom boundaries
    nx, ny, nz : int
        dimensions of grid
    vec_dim : int
        dimension of state vector
    alpha : float
        lapse function
    dx, dy, dt : float
        gridpoint spacing and timestep spacing
    kx_offset, ky_offset : int
        x, y offset for current kernel
    */
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    float fx_m, fx_p, fy_m, fy_p;

    // do fluxes
    if ((x > 1) && (x < (nx-2)) && (y > 1) && (y < (ny-2)) && (z < nz)) {
        for (int i = 0; i < vec_dim; i++) {
            // x-boundary
            // from i-1
            fx_m = 0.5 * (
                fx_plus_half[((z * ny + y) * nx + x-1) * vec_dim + i] +
                fx_minus_half[((z * ny + y) * nx + x) * vec_dim + i] +
                qx_plus_half[((z * ny + y) * nx + x-1) * vec_dim + i] -
                qx_minus_half[((z * ny + y) * nx + x) * vec_dim + i]);
            // from i+1
            fx_p = 0.5 * (
                fx_plus_half[((z * ny + y) * nx + x) * vec_dim + i] +
                fx_minus_half[((z * ny + y) * nx + x+1) * vec_dim + i] +
                qx_plus_half[((z * ny + y) * nx + x) * vec_dim + i] -
                qx_minus_half[((z * ny + y) * nx + x+1) * vec_dim + i]);

            // y-boundary
            // from j-1
            fy_m = 0.5 * (
                fy_plus_half[((z * ny + y-1) * nx + x) * vec_dim + i] +
                fy_minus_half[((z * ny + y) * nx + x) * vec_dim + i] +
                qy_plus_half[((z * ny + y-1) * nx + x) * vec_dim + i] -
                qy_minus_half[((z * ny + y) * nx + x) * vec_dim + i]);
            // from j+1
            fy_p = 0.5 * (
                fy_plus_half[((z * ny + y) * nx + x) * vec_dim + i] +
                fy_minus_half[((z * ny + y+1) * nx + x) * vec_dim + i] +
                qy_plus_half[((z * ny + y) * nx + x) * vec_dim + i] -
                qy_minus_half[((z * ny + y+1) * nx + x) * vec_dim + i]);

            float old_F = F[((z * ny + y) * nx + x)*vec_dim + i];
            F[((z * ny + y) * nx + x)*vec_dim + i] =
                -alpha * ((fx_p - fx_m)/dx + (fy_p - fy_m)/dy);

            // hack?
            if (nan_check(F[((z * ny + y) * nx + x)*vec_dim + i])) {
                //printf("nan :( (%d, %d, %d) i: %d, fx_p: %f, fx_m: %f, fy_p: %f, fy_m: %f\n", x, y, z, i, fx_p, fx_m, fy_p, fy_m);
                F[((z * ny + y) * nx + x)*vec_dim + i] = old_F;
            }
        }
        //printf("fxm, fxp: %f, %f fym, fyp: %f, %f F(tau): %f\n", fx_m, fx_p, fy_m, fy_p, F[((z * ny + y) * nx + x)*vec_dim +4]);
    }
}

__global__ void evolve_z_fluxes(float * F,
                     float * qz_plus_half, float * qz_minus_half,
                     float * fz_plus_half, float * fz_minus_half,
                     int nx, int ny, int nz, int vec_dim, float alpha,
                     float dz, float dt,
                     int kx_offset, int ky_offset) {
    /**
    Calculates fluxes in finite volume evolution by solving the Riemann
    problem at the cell boundaries in z direction.

    Parameters
    ----------
    F : float *
        flux vector at each point in grid and each layer
    qz_plus_half, qz_minus_half : float *
        state vector reconstructed at right and left boundaries
    fz_plus_half, fz_minus_half : float *
        flux vector at top and bottom boundaries
    nx, ny, nz : int
        dimensions of grid
    vec_dim : int
        dimension of state vector
    alpha : float
        lapse function
    dz, dt : float
        gridpoint spacing and timestep spacing
    kx_offset, ky_offset : int
        x, y offset for current kernel
    */
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    // do fluxes
    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1)) && (z > 0) && (z < (nz-1))) {
        for (int i = 0; i < vec_dim; i++) {
            // z-boundary
            // from i-1
            float fz_m = 0.5 * (
                fz_plus_half[(((z-1) * ny + y) * nx + x) * vec_dim + i] +
                fz_minus_half[((z * ny + y) * nx + x) * vec_dim + i] +
                qz_plus_half[(((z-1) * ny + y) * nx + x) * vec_dim + i] -
                qz_minus_half[((z * ny + y) * nx + x) * vec_dim + i]);
            // from i+1
            float fz_p = 0.5 * (
                fz_plus_half[((z * ny + y) * nx + x) * vec_dim + i] +
                fz_minus_half[(((z+1) * ny + y) * nx + x) * vec_dim + i] +
                qz_plus_half[((z * ny + y) * nx + x) * vec_dim + i] -
                qz_minus_half[(((z+1) * ny + y) * nx + x) * vec_dim + i]);

            float old_F = F[((z * ny + y) * nx + x)*vec_dim + i];

            F[((z * ny + y) * nx + x)*vec_dim + i] =
                F[((z * ny + y) * nx + x)*vec_dim + i]
                - alpha * (fz_p - fz_m) / dz;

            // hack?
            if (nan_check(F[((z * ny + y) * nx + x)*vec_dim + i])) F[((z * ny + y) * nx + x)*vec_dim + i] = old_F;
        }
    }
}

__global__ void evolve_fv_heating(float * gamma_up,
                     float * Up, float * U_half,
                     float * qx_plus_half, float * qx_minus_half,
                     float * qy_plus_half, float * qy_minus_half,
                     float * fx_plus_half, float * fx_minus_half,
                     float * fy_plus_half, float * fy_minus_half,
                     float * sum_phs, float * rho, float * Q_d,
                     int nx, int ny, int nlayers, float alpha, float gamma,
                     float dx, float dy, float dt,
                     bool burning, float Cv, float E_He,
                     int kx_offset, int ky_offset) {
    /**
    Does the heating part of the evolution.

    Parameters
    ----------
    gamma_up : float *
        gamma matrix at each grid point
    Up : float *
        state vector at next timestep
    U_half : float *
        state vector at half timestep
    qx_plus_half, qx_minus_half : float *
        state vector reconstructed at right and left boundaries
    qy_plus_half, qy_minus_half : float *
        state vector reconstructed at top and bottom boundaries
    fx_plus_half, fx_minus_half : float *
        flux vector at right and left boundaries
    fy_plus_half, fy_minus_half : float *
        flux vector at top and bottom boundaries
    sum_phs : float *
        sum of Phi in different layers
    rho : float *
        list of densities in different layers
    Q_d : float *
        heating rate in each layer
    nx, ny, nlayers : int
        dimensions of grid
    alpha, gamma : float
        lapse function and adiabatic index
    dx, dy, dt : float
        gridpoint spacing and timestep spacing
    burning : bool
        is burning present in this system?
    Cv, E_He : float
        specific heat in constant volume and energy release per unit mass of He
    kx_offset, ky_offset : int
        x, y offset for current kernel
    */
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;
    int offset = (z * ny + y) * nx + x;

    // calculate Q
    //calc_Q(Up, rho_d, Q_d, nx, ny, nlayers, kx_offset, ky_offset, burning);

    float W = 1.0;

    float X_dot = 0.0;

    // do source terms
    if ((x < nx) && (y < ny) && (z < nlayers)) {
        float * q_swe;
        q_swe = (float *)malloc(4 * sizeof(float));

        for (int i = 0; i < 4; i++) {
            q_swe[i] = U_half[offset * 4 + i];
        }
        W = W_swe(q_swe, gamma_up);

        float * A, * phis;
        A = (float *)malloc(nlayers * sizeof(float));
        phis = (float *)malloc(nlayers * sizeof(float));
        for (int i = 0; i < nlayers; i++) {
            phis[i] = U_half[((i * ny + y) * nx + x)* 4];
        }

        calc_As(rho, phis, A, nlayers, gamma, phis[0], rho[0]);

        float p = p_from_swe(q_swe, gamma_up, rho[z], gamma, W, A[z]);
        float Y = q_swe[3] / q_swe[0];

        X_dot = calc_Q_swe(rho[z], p, gamma, Y, Cv) / E_He;

        //printf("p: %f, A: %f, X_dot : %f\n", p, A[z], X_dot);

        free(phis);
        free(A);
        free(q_swe);

        U_half[offset*4] /= W;
    }

    __syncthreads();

    if ((x < nx) && (y < ny) && (z < nlayers)) {

        sum_phs[offset] = 0.0;

        float sum_qs = 0.0;
        float deltaQx = 0.0;
        float deltaQy = 0.0;

        if (z < (nlayers - 1)) {
            sum_qs += (Q_d[z + 1] - Q_d[z]);
            deltaQx = Q_d[z] *
                (U_half[offset*4+1] - U_half[(((z + 1) * ny + y) * nx + x)*4+1]) /
                (W * U_half[offset*4]);
            deltaQy = (Q_d[z]) *
                (U_half[offset*4+2] - U_half[(((z + 1) * ny + y) * nx + x)*4+2]) /
                (W * U_half[offset*4]);
        }
        if (z > 0) {
            sum_qs += -rho[z-1] / rho[z] * (Q_d[z] - Q_d[z - 1]);
            deltaQx = rho[z-1] / rho[z] * Q_d[z] *
                (U_half[offset*4+1] - U_half[(((z - 1) * ny + y) * nx + x)*4+1]) /
                 (W * U_half[offset*4]);
            deltaQy = rho[z-1] / rho[z] * Q_d[z] *
                (U_half[offset*4+2] - U_half[(((z - 1) * ny + y) * nx + x)*4+2]) /
                 (W * U_half[offset*4]);
        }

        for (int j = 0; j < z; j++) {
            sum_phs[offset] += rho[j] / rho[z] *
                U_half[((j * ny + y) * nx + x)*4];
        }
        for (int j = z+1; j < nlayers; j++) {
            sum_phs[offset] += U_half[((j * ny + y) * nx + x)*4];
        }

        // NOTE: for now going to make Xdot a constant
        //const float X_dot = 0.01;

        // D
        Up[offset*4] += dt * alpha * sum_qs;

        //if (x < 10 && y < 10) printf("(%d, %d, %d) Q: %f, sum_qs: %f, deltaQx: %f, deltaQy: %f\n", x, y, z, Q_d[z], sum_qs, deltaQx, deltaQy);

        // Sx
        Up[offset*4+1] += dt * alpha * (-deltaQx);

        // Sy
        Up[offset*4+2] += dt * alpha * (-deltaQy);

        // DX
        Up[offset*4+3] += dt * alpha * X_dot;
    }
}

__global__ void evolve2(float * Un_d, float * Up, float * U_half,
                     float * sum_phs, int nx, int ny, int nlayers, int ng,
                     float alpha, float dx, float dy, float dt,
                     int kx_offset, int ky_offset) {
    /**
    Adds buoyancy terms.

    Parameters
    ----------
    Un_d : float *
        state vector at each grid point in each layer at current timestep
    Up : float *
        state vector at next timestep
    U_half : float *
        state vector at half timestep
    sum_phs : float *
        sum of Phi in different layers
    nx, ny, nlayers : int
        dimensions of grid
    ng : int
        number of ghost cells
    alpha : float
        lapse function
    dx, dy, dt : float
        gridpoint spacing and timestep spacing
    kx_offset, ky_offset : int
        x, y offset for current kernel
    */
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;
    int offset =  (z * ny + y) * nx + x;

    //printf("kx_offset: %i\n", kx_offset);

    if ((x > 1) && (x < (nx-2)) && (y > 1) && (y < (ny-2)) && (z < nlayers)) {

        float a_upwind = sum_phs[(z * ny + y) * nx + x+1] - sum_phs[offset];
        float a_downwind = sum_phs[offset] - sum_phs[(z * ny + y) * nx + x-1];

        float a = 0.5 * (a_upwind + a_downwind);

        float r = 1.0e6;
        if (abs(a_downwind) > 1.0e-10) {
            r = a_upwind / a_downwind;
        }

        a *= dt * alpha * U_half[offset*4] * 0.5 * phi(r);
        if (abs(a) < 0.9 * dx / dt) {
            Up[offset*4+1] -= a;
        }

        a_upwind = sum_phs[(z * ny + y+1) * nx + x] - sum_phs[offset];
        a_downwind = sum_phs[offset] - sum_phs[(z * ny + y-1) * nx + x];

        a = 0.5 * (a_upwind + a_downwind);

        r = 1.0e6;
        if (abs(a_downwind) > 1.0e-10) {
            r = a_upwind / a_downwind;
        }

        a *= dt * alpha * U_half[offset*4] * 0.5 * phi(r);

        if (abs(a) < 0.9 * dy / dt) {
            Up[offset*4+2] -= a;
        }

        // copy back to grid
        for (int i = 0; i < 4; i++) {
            Un_d[offset*4+i] = Up[offset*4+i];
        }
    }
}

void homogeneuous_fv(dim3 * kernels, dim3 * threads, dim3 * blocks,
       int * cumulative_kernels, float * beta_d, float * gamma_up_d,
       float * Un_d, float * F_d,
       float * qx_p_d, float * qx_m_d, float * qy_p_d, float * qy_m_d,
       float * qz_p_d, float * qz_m_d,
       float * fx_p_d, float * fx_m_d, float * fy_p_d, float * fy_m_d,
       float * fz_p_d, float * fz_m_d,
       int nx, int ny, int nz, int vec_dim, int ng, float alpha, float gamma,
       float dx, float dy, float dz, float dt, int rank,
       flux_func_ptr h_flux_func, bool do_z) {
    /**
    Solves the homogeneous part of the equation (ie the bit without source terms).

    Parameters
    ----------
    kernels, threads, blocks : dim3 *
        number of kernels, threads and blocks for each process/kernel
    cumulative_kernels : int *
        Cumulative total of kernels in ranks < rank of current MPI process
    beta_d : float *
        shift vector at each grid point
    gamma_up_d : float *
        gamma matrix at each grid point
    Un_d : float *
        state vector at each grid point in each layer at current timestep
    F_d : float *
        flux vector
    qx_p_d, qx_m_d : float *
        state vector reconstructed at right and left boundaries
    qy_p_d, qy_m_d : float *
        state vector reconstructed at top and bottom boundaries
    fx_p_d, fx_m_d : float *
        flux vector at right and left boundaries
    fy_p_d, fy_m_d : float *
        flux vector at top and bottom boundaries
    nx, ny, nz : int
        dimensions of grid
    alpha, gamma : float
        lapse function and adiabatic index
    dx, dy, dz, dt : float
        gridpoint spacing and timestep spacing
    rank : int
        rank of MPI process
    flux_func : flux_func_ptr
        pointer to function to be used to calculate fluxes
    do_z : bool
        should we evolve in the z direction?
    */

    int kx_offset = 0;
    int ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

    int k_offset = 0;
    if (rank > 0) {
        k_offset = cumulative_kernels[rank - 1];
    }

    for (int j = 0; j < kernels[rank].y; j++) {
       kx_offset = 0;
       for (int i = 0; i < kernels[rank].x; i++) {
           evolve_fv<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(beta_d, gamma_up_d, Un_d, h_flux_func,
                  qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                  fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                  nx, ny, nz, vec_dim, alpha, gamma,
                  dx, dy, dt, kx_offset, ky_offset);
           if (do_z) {
               evolve_z<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(beta_d, gamma_up_d, Un_d, h_flux_func,
                      qz_p_d, qz_m_d,
                      fz_p_d, fz_m_d,
                      nx, ny, nz, vec_dim, alpha, gamma,
                      dz, dt, kx_offset, ky_offset);
           }
           kx_offset += blocks[k_offset + j * kernels[rank].x + i].x * threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
       }
       ky_offset += blocks[k_offset + j * kernels[rank].x].y * threads[k_offset + j * kernels[rank].x].y - 2*ng;
    }

    ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

    for (int j = 0; j < kernels[rank].y; j++) {
       kx_offset = 0;
       for (int i = 0; i < kernels[rank].x; i++) {
           evolve_fv_fluxes<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(
                  F_d,
                  qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                  fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                  nx, ny, nz, vec_dim, alpha,
                  dx, dy, dt, kx_offset, ky_offset);

            if (do_z) {
                evolve_z_fluxes<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(
                       F_d,
                       qz_p_d, qz_m_d,
                       fz_p_d, fz_m_d,
                       nx, ny, nz, vec_dim, alpha,
                       dz, dt, kx_offset, ky_offset);
            }

            kx_offset += blocks[k_offset + j * kernels[rank].x + i].x * threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
       }
       ky_offset += blocks[k_offset + j * kernels[rank].x].y * threads[k_offset + j * kernels[rank].x].y - 2*ng;
    }
}

void rk3(dim3 * kernels, dim3 * threads, dim3 * blocks,
       int * cumulative_kernels,
       float * beta_d, float * gamma_up_d, float * Un_d,
       float * F_d, float * Up_d,
       float * qx_p_d, float * qx_m_d, float * qy_p_d, float * qy_m_d,
       float * qz_p_d, float * qz_m_d,
       float * fx_p_d, float * fx_m_d, float * fy_p_d, float * fy_m_d,
       float * fz_p_d, float * fz_m_d,
       int nx, int ny, int nz, int vec_dim, int ng, float alpha, float gamma,
       float dx, float dy, float dz, float dt,
       float * Up_h, float * F_h, float * Un_h,
       MPI_Comm comm, MPI_Status status, int rank, int n_processes,
       flux_func_ptr flux_func, bool do_z) {
    /**
    Integrates the homogeneous part of the ODE in time using RK3.

    Parameters
    ----------
    kernels, threads, blocks : dim3 *
        number of kernels, threads and blocks for each process/kernel
    cumulative_kernels : int *
        Cumulative total of kernels in ranks < rank of current MPI process
    beta_d : float *
        shift vector at each grid point
    gamma_up_d : float *
        gamma matrix at each grid point
    Un_d : float *
        state vector at each grid point in each layer at current timestep on device
    F_d : float *
        flux vector on device
    Up_d : float *
        state vector at next timestep on device
    qx_p_d, qx_m_d : float *
        state vector reconstructed at right and left boundaries
    qy_p_d, qy_m_d : float *
        state vector reconstructed at top and bottom boundaries
    fx_p_d, fx_m_d : float *
        flux vector at right and left boundaries
    fy_p_d, fy_m_d : float *
        flux vector at top and bottom boundaries
    nx, ny, nz : int
        dimensions of grid
    vec_dim : int
        dimension of state vector
    ng : int
        number of ghost cells
    alpha, gamma : float
        lapse function and adiabatic index
    dx, dy, dz, dt : float
        gridpoint spacing and timestep spacing
    Up_h, F_h, Un_h : float *
        state vector at next timestep, flux vector and state vector at current timestep on host
    comm : MPI_Comm
        MPI communicator
    status: MPI_Status
        status of MPI processes
    rank, n_processes : int
        rank of current MPI process and total number of MPI processes
    flux_func : flux_func_ptr
        pointer to function to be used to calculate fluxes
    do_z : bool
        should we evolve in the z direction?
    */
    //cout << "\nu1\n\n\n";
    // u1 = un + dt * F(un)
    homogeneuous_fv(kernels, threads, blocks, cumulative_kernels,
          beta_d, gamma_up_d, Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
          nx, ny, nz, vec_dim, ng, alpha, gamma,
          dx, dy, dz, dt, rank, flux_func, do_z);

    // copy back flux
    cudaMemcpy(F_h, F_d, nx*ny*nz*vec_dim*sizeof(float), cudaMemcpyDeviceToHost);
    if (n_processes == 1) {
        bcs_fv(F_h, nx, ny, nz, ng, vec_dim);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(F_h, nx, ny, nz, vec_dim, ng, comm, status, rank, n_processes, y_size, do_z);
    }

    for (int n = 0; n < nx*ny*nz*vec_dim; n++) {
        Up_h[n] = Un_h[n] + dt * F_h[n];
    }
    // enforce boundaries and copy back
    if (n_processes == 1) {
        bcs_fv(Up_h, nx, ny, nz, ng, vec_dim);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(Up_h, nx, ny, nz, vec_dim, ng, comm, status, rank, n_processes, y_size, do_z);
    }

    if (do_z) {
        // HACK:
        // going to do some hacky data sanitisation here
        // NOTE: could argue that this is actually a form of artificial
        // dissipation to ensure stability (as it is just smoothing out
        // spikes in the data after all)
        for (int x = 0; x < nx * ny * nz; x++) {
            if (abs(Up_h[x*6]) > 1.0e2) {
                Up_h[x*6] = 0.5;
            }
            if (abs(Up_h[x*6+4]) > 1.0e3 || Up_h[x*6+4] < 0.0) {
                Up_h[x*6+4] = Up_h[x*6];
            }
            if (Up_h[x*6+5] > 1.0) Up_h[x*6+5] = 1.0;
            if (Up_h[x*6+5] < 0.0) Up_h[x*6+5] = 0.0;
            for (int i = 1; i < 4; i++) {
                if (abs(Up_h[x*6+i]) > Up_h[x*6]) {
                    Up_h[x*6+i] = 0.0;
                }
            }
        }
    }

    cudaMemcpy(Un_d, Up_h, nx*ny*nz*vec_dim*sizeof(float), cudaMemcpyHostToDevice);
    //cout << "\nu2\n\n\n";
    // u2 = 0.25 * (3*un + u1 + dt*F(u1))
    homogeneuous_fv(kernels, threads, blocks, cumulative_kernels,
          beta_d, gamma_up_d, Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
          nx, ny, nz, vec_dim, ng, alpha, gamma,
          dx, dy, dz, dt, rank, flux_func, do_z);

    // copy back flux
    cudaMemcpy(F_h, F_d, nx*ny*nz*vec_dim*sizeof(float), cudaMemcpyDeviceToHost);

    if (n_processes == 1) {
        bcs_fv(F_h, nx, ny, nz, ng, vec_dim);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(F_h, nx, ny, nz, vec_dim, ng, comm, status, rank, n_processes, y_size, do_z);
    }

    for (int n = 0; n < nx*ny*nz*vec_dim; n++) {
        Up_h[n] = 0.25 * (3.0 * Un_h[n] + Up_h[n] + dt * F_h[n]);
    }

    // enforce boundaries and copy back
    if (n_processes == 1) {
        bcs_fv(Up_h, nx, ny, nz, ng, vec_dim);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(Up_h, nx, ny, nz, vec_dim, ng, comm, status, rank, n_processes, y_size, do_z);
    }

    if (do_z) {
        // HACK:
        // going to do some hacky data sanitisation here
        for (int x = 0; x < nx * ny * nz; x++) {
            if (abs(Up_h[x*6]) > 1.0e2) {
                Up_h[x*6] = 0.5;
            }
            if (abs(Up_h[x*6+4]) > 1.0e3 || Up_h[x*6+4] < 0.0) {
                Up_h[x*6+4] = Up_h[x*6];
            }
            if (Up_h[x*6+5] > 1.0) Up_h[x*6+5] = 1.0;
            if (Up_h[x*6+5] < 0.0) Up_h[x*6+5] = 0.0;
            for (int i = 1; i < 4; i++) {
                if (abs(Up_h[x*6+i]) > Up_h[x*6]) {
                    Up_h[x*6+i] = 0.0;
                }
            }
        }
    }

    cudaMemcpy(Un_d, Up_h, nx*ny*nz*vec_dim*sizeof(float), cudaMemcpyHostToDevice);
    //cout << "\nun+1\n\n\n";
    // un+1 = (1/3) * (un + 2*u2 + 2*dt*F(u2))
    homogeneuous_fv(kernels, threads, blocks, cumulative_kernels,
          beta_d, gamma_up_d, Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
          nx, ny, nz, vec_dim, ng, alpha, gamma,
          dx, dy, dz, dt, rank, flux_func, do_z);

    // copy back flux
    cudaMemcpy(F_h, F_d, nx*ny*nz*vec_dim*sizeof(float), cudaMemcpyDeviceToHost);

    if (n_processes == 1) {
        bcs_fv(F_h, nx, ny, nz, ng, vec_dim);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(F_h, nx, ny, nz, vec_dim, ng, comm, status, rank, n_processes, y_size, do_z);
    }

    for (int n = 0; n < nx*ny*nz*vec_dim; n++) {
        Up_h[n] = (1/3.0) * (Un_h[n] + 2.0*Up_h[n] + 2.0*dt * F_h[n]);
    }

    // enforce boundaries
    if (n_processes == 1) {
        bcs_fv(Up_h, nx, ny, nz, ng, vec_dim);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(Up_h, nx, ny, nz, vec_dim, ng, comm, status, rank, n_processes, y_size, do_z);
    }

    if (do_z) {
        // HACK: going to do some hacky data sanitisation here
        for (int x = 0; x < nx * ny * nz; x++) {
            if (abs(Up_h[x*6]) > 1.0e2) {
                Up_h[x*6] = 0.5;
            }
            if (abs(Up_h[x*6+4]) > 1.0e3 || Up_h[x*6+4] < 0.0) {
                Up_h[x*6+4] = Up_h[x*6];
            }
            if (Up_h[x*6+5] > 1.0) Up_h[x*6+5] = 1.0;
            if (Up_h[x*6+5] < 0.0) Up_h[x*6+5] = 0.0;
            for (int i = 1; i < 4; i++) {
                if (abs(Up_h[x*6+i]) > Up_h[x*6]) {
                    Up_h[x*6+i] = 0.0;
                }
            }
        }
    }

    for (int j = 0; j < nx*ny*nz*vec_dim; j++) {
        Un_h[j] = Up_h[j];
    }
}

template<typename T>
T array_max(T * a, int length) {
    // Returns the maximum value of array a
    T max = a[0];
    for (int i = 1; i < length; i++) {
        if (a[i] > max) max = a[i];
    }
    return max;
}

void cuda_run(float * beta, float * gamma_up, float * Uc_h, float * Uf_h,
         float ** Us_h, float * rho, float * Q, int nx, int ny, int nlayers,
         int nxf, int nyf, int nz,
         int * nxs, int * nys, int * nzs, int nlevels, char * models,
         int * vec_dims, int ng,
         int nt, float alpha, float gamma, float E_He, float Cv,
         float zmin,
         float dx, float dy, float dz, float dt, bool burning,
         int dprint, char * filename,
         MPI_Comm comm, MPI_Status status, int rank, int n_processes,
         int * matching_indices) {
    /**
    Evolve system through nt timesteps, saving data to filename every dprint timesteps.

    Parameters
    ----------
    beta : float *
        shift vector at each grid point
    gamma_up : float *
        gamma matrix at each grid point
    Uc_h : float *
        state vector at each grid point in each layer at current timestep on host in coarse grid
    Uf_h : float *
        state vector at each grid point in each layer at current timestep on host in fine grid
    rho : float *
        densities in each layer
    Q : float *
        heating rate at each point and in each layer
    nx, ny, nlayers : int
        dimensions of coarse grid
    nxf, nyf, nz : int
        dimensions of fine grid
    ng : int
        number of ghost cells
    nt : int
        total number of timesteps
    alpha : float
        lapse function
    gamma : float
        adiabatic index
    E_He : float
        energy release per unit mass of helium
    Cv : float
        specific heat in constant volume
    zmin : float
        height of sea floor
    dx, dy, dz, dt : float
        gridpoint spacing and timestep spacing
    burning : bool
        is burning included in this system?
    dprint : int
        number of timesteps between each printout
    filename : char *
        name of file to which output is printed
    comm : MPI_Comm
        MPI communicator
    status: MPI_Status
        status of MPI processes
    rank, n_processes : int
        rank of current MPI process and total number of MPI processes
    matching_indices : int *
        position of fine grid wrt coarse grid
    */

    // set up GPU stuff
    int count;
    cudaGetDeviceCount(&count);

    if (rank == 0) {
        cudaError_t err = cudaGetLastError();
        // check that we actually have some GPUS
        if (err != cudaSuccess) {
            printf("Error: %s\n", cudaGetErrorString(err));
            printf("Aborting program.\n");
            return;
        }
        printf("Found %i CUDA devices\n", count);
    }

    // if rank > number of GPUs, exit now
    if (rank >= count) {
        return;
    }

    // redefine - we only want to run on as many cores as we have GPUs
    if (n_processes > count) {
        n_processes = count;
    }

    if (rank == 0) {
        printf("Running on %i processor(s)\n", n_processes);
    }

    int maxThreads = 256;
    int maxBlocks = 256; //64;

    dim3 *kernels = new dim3[n_processes];
    int *cumulative_kernels = new int[n_processes];

    //getNumKernels(max(nx, nxf), max(ny, nyf), max(nlayers, nz), ng, n_processes, &maxBlocks, &maxThreads, kernels, cumulative_kernels);

    getNumKernels(array_max(nxs, nlevels), array_max(nys, nlevels), array_max(nzs, nlevels), ng, n_processes, &maxBlocks, &maxThreads, kernels, cumulative_kernels);

    int total_kernels = cumulative_kernels[n_processes-1];

    dim3 *blocks = new dim3[total_kernels];
    dim3 *threads = new dim3[total_kernels];

    //getNumBlocksAndThreads(max(nx, nxf), max(ny, nyf), max(nlayers, nz), ng, maxBlocks, maxThreads, n_processes, kernels, blocks, threads);

    getNumBlocksAndThreads(array_max(nxs, nlevels), array_max(nys, nlevels), array_max(nzs, nlevels), ng, maxBlocks, maxThreads, n_processes, kernels, blocks, threads);

    printf("rank: %i\n", rank);
    printf("kernels: (%i, %i)\n", kernels[rank].x, kernels[rank].y);
    printf("cumulative kernels: %i\n", cumulative_kernels[rank]);

    int k_offset = 0;
    if (rank > 0) {
      k_offset = cumulative_kernels[rank-1];
    }

    for (int i = k_offset; i < cumulative_kernels[rank]; i++) {
        printf("blocks: (%i, %i, %i) , threads: (%i, %i, %i)\n",
               blocks[i].x, blocks[i].y, blocks[i].z,
               threads[i].x, threads[i].y, threads[i].z);
    }

    // gpu variables
    float * beta_d, * gamma_up_d, * Uc_d, * Uf_d, * rho_d, * Q_d;

    // initialise Uf_h
    for (int i = 0; i < nxf*nyf*nz*6; i++) {
        Uf_h[i] = 0.0;
    }

    // set device
    cudaSetDevice(rank);

    // allocate memory on device
    cudaMalloc((void**)&beta_d, 3*sizeof(float));
    cudaMalloc((void**)&gamma_up_d, 9*sizeof(float));
    cudaMalloc((void**)&Uc_d, nx*ny*nlayers*4*sizeof(float));
    cudaMalloc((void**)&Uf_d, nxf*nyf*nz*6*sizeof(float));
    cudaMalloc((void**)&rho_d, nlayers*sizeof(float));
    cudaMalloc((void**)&Q_d, nlayers*sizeof(float));

    // copy stuff to GPU
    cudaMemcpy(beta_d, beta, 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma_up_d, gamma_up, 9*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Uc_d, Uc_h, nx*ny*nlayers*4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Uf_d, Uf_h, nxf*nyf*nz*6*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rho_d, rho, nlayers*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Q_d, Q, nlayers*sizeof(float), cudaMemcpyHostToDevice);

    float *Upc_d, *Uc_half_d, *Upf_d, *Uf_half_d, *old_phi_d, *sum_phs_d;
    cudaMalloc((void**)&Upc_d, nx*ny*nlayers*4*sizeof(float));
    cudaMalloc((void**)&Uc_half_d, nx*ny*nlayers*4*sizeof(float));
    cudaMalloc((void**)&Upf_d, nxf*nyf*nz*6*sizeof(float));
    cudaMalloc((void**)&Uf_half_d, nxf*nyf*nz*6*sizeof(float));
    cudaMalloc((void**)&old_phi_d, nlayers*nx*ny*sizeof(float));
    cudaMalloc((void**)&sum_phs_d, nlayers*nx*ny*sizeof(float));

    // need to fill old_phi with current phi to initialise
    float *pphi = new float[nlayers*nx*ny];
    for (int i = 0; i < nlayers*nx*ny; i++) {
        pphi[i] = Uc_h[i*4];
    }
    cudaMemcpy(old_phi_d, pphi, nx*ny*nlayers*sizeof(float), cudaMemcpyHostToDevice);

    float *qx_p_d, *qx_m_d, *qy_p_d, *qy_m_d, *qz_p_d, *qz_m_d, *fx_p_d, *fx_m_d, *fy_p_d, *fy_m_d, *fz_p_d, *fz_m_d;
    float *Upc_h = new float[nx*ny*nlayers*4];
    float *Fc_h = new float[nx*ny*nlayers*4];

    float *Upf_h = new float[nxf*nyf*nz*6];
    float *Ff_h = new float[nxf*nyf*nz*6];

    float * sum_phs_h = new float[nx*ny*nlayers];

    // initialise
    for (int j = 0; j < nxf*nyf*nz*6; j++) {
        Upf_h[j] = 0.0;
    }

    //int grid_size = max(nx*ny*nlayers*4, nxf*nyf*nz*6);
    int grid_size = nxs[0]*nys[0]*nzs[0]*vec_dims[0];
    for (int i = 1; i < nlevels; i++) {
        grid_size = max(nxs[i]*nys[i]*nzs[i]*vec_dims[i], grid_size);
    }

    cudaMalloc((void**)&qx_p_d, grid_size*sizeof(float));
    cudaMalloc((void**)&qx_m_d, grid_size*sizeof(float));
    cudaMalloc((void**)&qy_p_d, grid_size*sizeof(float));
    cudaMalloc((void**)&qy_m_d, grid_size*sizeof(float));
    cudaMalloc((void**)&qz_p_d, grid_size*sizeof(float));
    cudaMalloc((void**)&qz_m_d, grid_size*sizeof(float));
    cudaMalloc((void**)&fx_p_d, grid_size*sizeof(float));
    cudaMalloc((void**)&fx_m_d, grid_size*sizeof(float));
    cudaMalloc((void**)&fy_p_d, grid_size*sizeof(float));
    cudaMalloc((void**)&fy_m_d, grid_size*sizeof(float));
    cudaMalloc((void**)&fz_p_d, grid_size*sizeof(float));
    cudaMalloc((void**)&fz_m_d, grid_size*sizeof(float));

    float * q_comp_d;
    cudaMalloc((void**)&q_comp_d, nx*ny*nlayers*6*sizeof(float));
    float * qf_swe;
    cudaMalloc((void**)&qf_swe, nxf*nyf*nz*4*sizeof(float));

    int * matching_indices_d;
    cudaMalloc((void**)&matching_indices_d, 4*sizeof(int));
    cudaMemcpy(matching_indices_d, matching_indices, 4*sizeof(int), cudaMemcpyHostToDevice);

    // make host-side function pointers to __device__ functions
    flux_func_ptr h_compressible_fluxes;
    flux_func_ptr h_shallow_water_fluxes;

    // copy function pointers to host equivalent
    cudaMemcpyFromSymbol(&h_compressible_fluxes, d_compressible_fluxes, sizeof(flux_func_ptr));
    cudaMemcpyFromSymbol(&h_shallow_water_fluxes, d_shallow_water_fluxes, sizeof(flux_func_ptr));

    if (strcmp(filename, "na") != 0) {
        hid_t outFile, dset, mem_space, file_space;

        if (rank == 0) {
            // create file
            outFile = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

            // create dataspace
            int ndims = 5;
            hsize_t dims[] = {hsize_t((nt+1)/dprint+1), hsize_t(nlayers), (ny), hsize_t(nx), 4};
            file_space = H5Screate_simple(ndims, dims, NULL);

            hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_layout(plist, H5D_CHUNKED);
            hsize_t chunk_dims[] = {1, hsize_t(nlayers), hsize_t(ny), hsize_t(nx), 4};
            H5Pset_chunk(plist, ndims, chunk_dims);

            // create dataset
            dset = H5Dcreate(outFile, "SwerveOutput", H5T_NATIVE_FLOAT, file_space, H5P_DEFAULT, plist, H5P_DEFAULT);

            H5Pclose(plist);

            // make a memory dataspace
            mem_space = H5Screate_simple(ndims, chunk_dims, NULL);

            // select a hyperslab
            file_space = H5Dget_space(dset);
            hsize_t start[] = {0, 0, 0, 0, 0};
            hsize_t hcount[] = {1, hsize_t(nlayers), hsize_t(ny), hsize_t(nx), 4};
            H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, hcount, NULL);
            // write to dataset
            printf("Printing t = %i\n", 0);
            H5Dwrite(dset, H5T_NATIVE_FLOAT, mem_space, file_space, H5P_DEFAULT, Uc_h);
            // close file dataspace
            H5Sclose(file_space);
        }

        cudaError_t err;
        err = cudaGetLastError();
        if (err != cudaSuccess){
            cout << "Before evolution\n";
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        // main loop
        for (int t = 0; t < nt; t++) {

            cout << "Evolving t = " << t << '\n';

            int ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

            // good here
            /*cout << "\nCoarse grid before prolonging\n\n";
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    cout << '(' << x << ',' << y << "): ";
                    for (int z = 0; z < nlayers; z++) {
                        cout << Uc_h[(((z*ny + y)*nx)+x)*4] << ',';
                    }
                    cout << '\n';
                }
            }*/

            //cout << "\n\nProlonging\n\n";

            // prolong to fine grid
            prolong_grid(kernels, threads, blocks, cumulative_kernels,
                         Uc_d, Uf_d, nx, ny, nlayers, nxf, nyf, nz, dx, dy, dz, dt, zmin, gamma_up_d,
                         rho_d, gamma, matching_indices_d, ng, rank, q_comp_d, old_phi_d);


            cudaMemcpy(Uf_h, Uf_d, nxf*nyf*nz*6*sizeof(float), cudaMemcpyDeviceToHost);
            err = cudaGetLastError();
            if (err != cudaSuccess){
                cout << "After prolonging\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }

            /*cout << "\nFine grid after prolonging\n\n";
            for (int y = 0; y < nyf; y++) {
                for (int x = 0; x < nxf; x++) {
                        cout << '(' << x << ',' << y << "): ";
                        for (int z = 0; z < nz; z++) {
                            cout << Uf_h[(((z*nyf + y)*nxf)+x)*6+4] << ',';
                        }
                        cout << '\n';
                }
            }*/

            // enforce boundaries
            if (n_processes == 1) {
                bcs_fv(Uf_h, nxf, nyf, nz, ng, 6);
            } else {
                int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
                bcs_mpi(Uf_h, nxf, nyf, nz, 6, ng, comm, status, rank, n_processes, y_size, true);
            }

            /*cout << "\nFine grid after prolonging\n\n";
            for (int y = 0; y < nyf; y++) {
                for (int x = 0; x < nxf; x++) {
                        cout << '(' << x << ',' << y << "): ";
                        for (int z = 0; z < nz; z++) {
                            cout << Uf_h[(((z*nyf + y)*nxf)+x)*6+4] << ',';
                        }
                        cout << '\n';
                }
            }*/

            cudaMemcpy(Uf_d, Uf_h, nxf*nyf*nz*6*sizeof(float), cudaMemcpyHostToDevice);

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                cout << "Before fine rk3\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }

            // evolve fine grid through two subcycles
            for (int i = 0; i < 2; i++) {

                rk3(kernels, threads, blocks, cumulative_kernels,
                        beta_d, gamma_up_d, Uf_d, Uf_half_d, Upf_d,
                        qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
                        fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
                        nxf, nyf, nz, 6, ng, alpha, gamma,
                        dx*0.5, dy*0.5, dz, dt*0.5, Upf_h, Ff_h, Uf_h,
                        comm, status, rank, n_processes,
                        h_compressible_fluxes, true);

                // enforce boundaries is done within rk3
                /*if (n_processes == 1) {
                    bcs_fv(Uf_h, nxf, nyf, nz, ng, 6);
                } else {
                    int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
                    bcs_mpi(Uf_h, nxf, nyf, nz, 6, ng, comm, status, rank, n_processes, y_size);
                }*/

                /*cout << "\nFine grid\n\n";
                for (int y = 0; y < nyf; y++) {
                    for (int x = 0; x < nxf; x++) {
                            cout << '(' << x << ',' << y << "): ";
                            for (int z = 0; z < nz; z++) {
                                if (abs(Uf_h[(((z*nyf + y)*nxf)+x)*6+4]) > 30.0)
                                cout << Uf_h[(((z*nyf + y)*nxf)+x)*6+4] << ',';
                            }
                            cout << '\n';
                    }
                }*/

                cudaDeviceSynchronize();

                // hack on the burning
                float * H = new float[nxf*nyf*nz];
                calc_Q(rho, Uf_h, nxf, nyf, nz, gamma, gamma_up, H, Cv);
                for (int z = 0; z < nz; z++) {
                    for (int y = ng; y < nyf-ng; y++) {
                        for (int x = ng; x < nxf - ng; x++) {
                            // tau
                            Uf_h[((z * nyf + y) * nxf + x) * 6 + 4] += dt * 0.5 * alpha * Uf_h[((z * nyf + y) * nxf + x) * 6] * H[(z * nyf + y) * nxf + x];
                            float X_dot = H[(z * nyf + y) * nxf + x] / E_He;
                            // DX
                            Uf_h[((z * nyf + y) * nxf + x) * 6 + 5] += dt * 0.5 * alpha * rho[0] * X_dot;
                        }
                    }
                }
                delete[] H;

                if (n_processes == 1) {
                    bcs_fv(Uf_h, nxf, nyf, nz, ng, 6);
                } else {
                    int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
                    bcs_mpi(Uf_h, nxf, nyf, nz, 6, ng, comm, status, rank, n_processes, y_size, false);
                }

                // copy to device
                cudaMemcpy(Uf_d, Uf_h, nxf*nyf*nz*6*sizeof(float), cudaMemcpyHostToDevice);

            }
            err = cudaGetLastError();
            if (err != cudaSuccess){
                cout << "Before restricting\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }

            //cout << "\n\nRestricting\n\n";
            // probably good here
            /*cout << "\nFine grid before restricting\n\n";
            for (int y = 0; y < nyf; y++) {
                for (int x = 0; x < nxf; x++) {
                        cout << '(' << x << ',' << y << "): ";
                        for (int z = 0; z < nz; z++) {
                            cout << Uf_h[(((z*nyf + y)*nxf)+x)*6+4] << ',';
                        }
                        cout << '\n';
                }
            }*/

            /*cout << "\nCoarse grid before restricting\n\n";
            for (int z = 0; z < nlayers; z++) {
                for (int y = 0; y < ny; y++) {
                    for (int x = 0; x < nx; x++) {
                            cout << '(' << x << ',' << y << ',' << z << "): " << Uc_h[(((z*ny+y)*nx)+x)*4+1] << ',' <<  Uc_h[(((z*ny+y)*nx)+x)*4+2] << ',' <<  Uc_h[(((z*ny+y)*nx)+x)*4+3] << '\n';
                    }
                }
            }*/

            // restrict to coarse grid
            restrict_grid(kernels, threads, blocks, cumulative_kernels,
                          Uc_d, Uf_d, nx, ny, nlayers, nxf, nyf, nz,
                          dz, zmin, matching_indices_d,
                          rho_d, gamma, gamma_up_d, ng, rank, qf_swe);
            err = cudaGetLastError();
            if (err != cudaSuccess){
                cout << "After restricting\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }

            cudaMemcpy(Uc_h, Uc_d, nx*ny*nlayers*4*sizeof(float), cudaMemcpyDeviceToHost);

            err = cudaGetLastError();
            if (err != cudaSuccess){
                cout << "After copying\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }

            // enforce boundaries
            if (n_processes == 1) {
                bcs_fv(Uc_h, nx, ny, nlayers, ng, 4);
            } else {
                int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
                bcs_mpi(Uc_h, nx, ny, nlayers, 4, ng, comm, status, rank, n_processes, y_size, false);
            }

            cudaMemcpy(Uc_d, Uc_h, nx*ny*nlayers*4*sizeof(float), cudaMemcpyHostToDevice);

            /*cout << "\nCoarse grid after restricting\n\n";
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    cout << '(' << x << ',' << y << "): ";
                    for (int z = 0; z < nlayers; z++) {
                        cout << Uc_h[(((z*ny + y)*nx)+x)*4] << ',';
                    }
                    cout << '\n';
                }
            }*/

            err = cudaGetLastError();
            if (err != cudaSuccess){
                cout << "Coarse rk3\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }

            rk3(kernels, threads, blocks, cumulative_kernels,
                beta_d, gamma_up_d, Uc_d, Uc_half_d, Upc_d,
                qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
                fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
                nx, ny, nlayers, 4, ng, alpha, gamma,
                dx, dy, dz, dt, Upc_h, Fc_h, Uc_h,
                comm, status, rank, n_processes,
                h_shallow_water_fluxes, false);

            err = cudaGetLastError();
            if (err != cudaSuccess){
                cout << "Done coarse rk3\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }

            cudaMemcpy(Uc_d, Uc_h, nx*ny*nlayers*4*sizeof(float), cudaMemcpyHostToDevice);

            // update old_phi
            for (int i = 0; i < nlayers*nx*ny; i++) {
                pphi[i] = Uc_h[i*4];
            }
            cudaMemcpy(old_phi_d, pphi, nx*ny*nlayers*sizeof(float), cudaMemcpyHostToDevice);

            /*cout << "\nCoarse grid after rk3\n\n";
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    cout << '(' << x << ',' << y << "): ";
                    for (int z = 0; z < nlayers; z++) {
                        cout << Uc_h[(((z*ny + y)*nx)+x)*4] << ',';
                    }
                    cout << '\n';
                }
            }*/

            cudaMemcpy(Upc_d, Uc_h, nx*ny*nlayers*4*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(Uc_half_d, Uc_h, nx*ny*nlayers*4*sizeof(float), cudaMemcpyHostToDevice);

            float kx_offset = 0;
            ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

            for (int j = 0; j < kernels[rank].y; j++) {
                kx_offset = 0;
                for (int i = 0; i < kernels[rank].x; i++) {
                    evolve_fv_heating<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(
                           gamma_up_d,
                           Upc_d, Uc_half_d,
                           qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                           fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                           sum_phs_d, rho_d, Q_d,
                           nx, ny, nlayers, alpha, gamma,
                           dx, dy, dt, burning, Cv, E_He,
                           kx_offset, ky_offset);
                    kx_offset += blocks[k_offset + j * kernels[rank].x + i].x * threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
                }
                ky_offset += blocks[k_offset + j * kernels[rank].x].y * threads[k_offset + j * kernels[rank].x].y - 2*ng;
            }

            cudaMemcpy(Upc_h, Upc_d, nx*ny*nlayers*4*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(sum_phs_h, sum_phs_d, nx*ny*nlayers*sizeof(float), cudaMemcpyDeviceToHost);

            // enforce boundaries
            if (n_processes == 1) {
                bcs_fv(Upc_h, nx, ny, nlayers, ng, 4);
                bcs_fv(sum_phs_h, nx, ny, nlayers, ng, 1);
            } else {
                int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
                bcs_mpi(Upc_h, nx, ny, nlayers, 4, ng, comm, status, rank, n_processes, y_size, false);
                bcs_mpi(sum_phs_h, nx, ny, nlayers, 1, ng, comm, status, rank, n_processes, y_size, false);
            }

            cudaMemcpy(Upc_d, Upc_h, nx*ny*nlayers*4*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(sum_phs_d, sum_phs_h, nx*ny*nlayers*sizeof(float), cudaMemcpyHostToDevice);

            kx_offset = 0;
            ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

            for (int j = 0; j < kernels[rank].y; j++) {
                kx_offset = 0;
                for (int i = 0; i < kernels[rank].x; i++) {
                    evolve2<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(Uc_d,
                           Upc_d, Uc_half_d, sum_phs_d,
                           nx, ny, nlayers, ng, alpha,
                           dx, dy, dt, kx_offset, ky_offset);
                    kx_offset += blocks[k_offset + j * kernels[rank].x + i].x * threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
                }
                ky_offset += blocks[k_offset + j * kernels[rank].x].y * threads[k_offset + j * kernels[rank].x].y - 2*ng;
            }

            cudaDeviceSynchronize();

            err = cudaGetLastError();

            if (err != cudaSuccess)
                printf("Error: %s\n", cudaGetErrorString(err));

            // boundaries
            cudaMemcpy(Uc_h, Uc_d, nx*ny*nlayers*4*sizeof(float), cudaMemcpyDeviceToHost);
            if (n_processes == 1) {
                bcs_fv(Uc_h, nx, ny, nlayers, ng, 4);
            } else {
                int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
                bcs_mpi(Uc_h, nx, ny, nlayers, 4, ng, comm, status, rank, n_processes, y_size, false);
            }
            cudaMemcpy(Uc_d, Uc_h, nx*ny*nlayers*4*sizeof(float), cudaMemcpyHostToDevice);

            int mpi_err;

            if ((t+1) % dprint == 0) {
                if (rank == 0) {
                    printf("Printing t = %i\n", t+1);

                    if (n_processes > 1) { // only do MPI stuff if needed
                        float * buf = new float[nx*ny*nlayers*4];
                        int tag = 0;
                        for (int source = 1; source < n_processes; source++) {
                            mpi_err = MPI_Recv(buf, nx*ny*nlayers*4, MPI_FLOAT, source, tag, comm, &status);

                            check_mpi_error(mpi_err);

                            // copy data back to grid
                            ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;
                            // cheating slightly and using the fact that are moving from bottom to top to make calculations a bit easier.
                            for (int z = 0; z < nlayers; z++) {
                                for (int y = ky_offset; y < ny; y++) {
                                    for (int x = 0; x < nx; x++) {
                                        for (int i = 0; i < 4; i++) {
                                            Uc_h[((z * ny + y) * nx + x) * 4 + i] = buf[((z * ny + y) * nx + x) * 4 + i];
                                        }
                                    }
                                }
                            }
                        }
                        delete[] buf;
                    }

                    // receive data from other processes and copy to grid

                    // select a hyperslab
                    file_space = H5Dget_space(dset);
                    hsize_t start[] = {hsize_t((t+1)/dprint), 0, 0, 0, 0};
                    hsize_t hcount[] = {1, hsize_t(nlayers), hsize_t(ny), hsize_t(nx), 4};
                    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, hcount, NULL);
                    // write to dataset
                    H5Dwrite(dset, H5T_NATIVE_FLOAT, mem_space, file_space, H5P_DEFAULT, Uc_h);
                    // close file dataspae
                    H5Sclose(file_space);
                } else { // send data to rank 0
                    int tag = 0;
                    mpi_err = MPI_Ssend(Uc_h, ny*nx*nlayers*4, MPI_FLOAT, 0, tag, comm);
                    check_mpi_error(mpi_err);
                }
            }
        }

        if (rank == 0) {
            H5Sclose(mem_space);
            H5Fclose(outFile);
        }

    } else { // don't print

        for (int t = 0; t < nt; t++) {

            // prolong to fine grid
            prolong_grid(kernels, threads, blocks, cumulative_kernels, Uc_d,
                         Uf_d, nx, ny, nlayers, nxf, nyf, nz, dx, dy, dz,
                         dt, zmin, gamma_up,
                         rho_d, gamma, matching_indices_d, ng, rank, q_comp_d, old_phi_d);

            cudaMemcpy(Uf_h, Uf_d, nxf*nyf*nz*6*sizeof(float), cudaMemcpyDeviceToHost);

            // evolve fine grid through two subcycles
            for (int i = 0; i < 2; i++) {
                rk3(kernels, threads, blocks, cumulative_kernels,
                        beta_d, gamma_up_d, Uf_d, Uf_half_d, Upf_d,
                        qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
                        fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
                        nxf, nyf, nz, 6, ng, alpha, gamma,
                        dx*0.5, dy*0.5, dz, dt*0.5, Upf_h, Ff_h, Uf_h,
                        comm, status, rank, n_processes,
                        h_compressible_fluxes, true);

                // if not last step, copy output array to input array
                if (i < 1) {
                    for (int j = 0; j < nxf*nyf*nz*6; j++) {
                        Uf_h[j] = Upf_h[j];
                    }
                }
            }

            // restrict to coarse grid
            restrict_grid(kernels, threads, blocks, cumulative_kernels,
                          Uc_d, Uf_d, nx, ny, nlayers, nxf, nyf, nz,
                          dz, zmin, matching_indices_d,
                          rho_d, gamma, gamma_up_d, ng, rank, qf_swe);

            rk3(kernels, threads, blocks, cumulative_kernels,
                beta_d, gamma_up_d, Uc_d, Uc_half_d, Upc_d,
                qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
                fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
                nx, ny, nlayers, 4, ng, alpha, gamma,
                dx, dy, dz, dt, Upc_h, Fc_h, Uc_h,
                comm, status, rank, n_processes,
                h_shallow_water_fluxes, false);

            /*int k_offset = 0;
            if (rank > 0) {
                k_offset = cumulative_kernels[rank-1];
            }
            for (int j = 0; j < kernels[rank].y; j++) {
                kx_offset = 0;
                for (int i = 0; i < kernels[rank].x; i++) {
                    evolve_fv_heating<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(
                           gamma_up_d, Un_d,
                           Up_d, U_half_d,
                           qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                           fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                           sum_phs_d, rho_d, Q_d,
                           nx, ny, nlayers, alpha, gamma,
                           dx, dy, dt, burning, Cv, E_He,
                           kx_offset, ky_offset);
                    kx_offset += blocks[k_offset + j * kernels[rank].x + i].x * threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
                }
                ky_offset += blocks[k_offset + j * kernels[rank].x].y * threads[j * kernels[rank].x].y - 2*ng;
            }


            kx_offset = 0;
            ky_offset = (kernels[0].y * blocks[0].y * threads[0].y - 2*ng) * rank;

            for (int j = 0; j < kernels[rank].y; j++) {
                kx_offset = 0;
                for (int i = 0; i < kernels[rank].x; i++) {
                    evolve2<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(gamma_up_d, Un_d,
                           Up_d, U_half_d, sum_phs_d, rho_d, Q_d,
                           nx, ny, nlayers, ng, alpha,
                           dx, dy, dt, kx_offset, ky_offset);
                    kx_offset += blocks[k_offset + j * kernels[rank].x + i].x * threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
                }
                ky_offset += blocks[k_offset + j * kernels[rank].x].y * threads[k_offset + j * kernels[rank].x].y - 2*ng;
            }*/

            cudaDeviceSynchronize();

            // boundaries
            cudaMemcpy(Uc_h, Uc_d, nx*ny*nlayers*4*sizeof(float), cudaMemcpyDeviceToHost);
            if (n_processes == 1) {
                bcs_fv(Uc_h, nx, ny, nlayers, ng, 4);
            } else {
                int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
                bcs_mpi(Uc_h, nx, ny, nlayers, 4, ng, comm, status, rank, n_processes, y_size, false);
            }
            cudaMemcpy(Uc_d, Uc_h, nx*ny*nlayers*4*sizeof(float), cudaMemcpyHostToDevice);

            cudaError_t err = cudaGetLastError();

            if (err != cudaSuccess)
                printf("Error: %s\n", cudaGetErrorString(err));
        }
    }

    // delete some stuff
    cudaFree(beta_d);
    cudaFree(gamma_up_d);
    cudaFree(Uc_d);
    cudaFree(Uf_d);
    cudaFree(rho_d);
    cudaFree(Q_d);
    cudaFree(Upc_d);
    cudaFree(Uc_half_d);
    cudaFree(Upf_d);
    cudaFree(Uf_half_d);
    cudaFree(old_phi_d);
    cudaFree(sum_phs_d);

    cudaFree(qx_p_d);
    cudaFree(qx_m_d);
    cudaFree(qy_p_d);
    cudaFree(qy_m_d);
    cudaFree(qz_p_d);
    cudaFree(qz_m_d);
    cudaFree(fx_p_d);
    cudaFree(fx_m_d);
    cudaFree(fy_p_d);
    cudaFree(fy_m_d);
    cudaFree(fz_p_d);
    cudaFree(fz_m_d);
    cudaFree(q_comp_d);
    cudaFree(qf_swe);
    cudaFree(matching_indices_d);

    delete[] kernels;
    delete[] cumulative_kernels;
    delete[] threads;
    delete[] blocks;
    delete[] Upc_h;
    delete[] Fc_h;
    delete[] Upf_h;
    delete[] Ff_h;
    delete[] pphi;
    delete[] sum_phs_h;
}
