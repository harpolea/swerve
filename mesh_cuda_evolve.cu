/**
File containing routines which model the evolution.
**/

__global__ void evolve_fv(float * Un_d, flux_func_ptr flux_func,
                     float * qx_plus_half, float * qx_minus_half,
                     float * qy_plus_half, float * qy_minus_half,
                     float * fx_plus_half, float * fx_minus_half,
                     float * fy_plus_half, float * fy_minus_half,
                     int nx, int ny, int nz, int vec_dim, float alpha,
                     float gamma,
                     int kx_offset, int ky_offset) {
    /**
    First part of evolution through one timestep using finite volume methods.
    Reconstructs state vector to cell boundaries using slope limiter
    and calculates fluxes there.

    NOTE: we assume that beta is smooth so can get value at cell boundaries with simple averaging

    Parameters
    ----------
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
    kx_offset, ky_offset : int
        x, y offset for current kernel
    */

    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    int offset = ((z * ny + y) * nx + x) * vec_dim;

    if ((x > 0) && (x < (nx-1)) && (y > 0) && (y < (ny-1)) && (z < nz)) {

        float * q_p, *q_m, * f;
        q_p = (float *)malloc(vec_dim * sizeof(float));
        q_m = (float *)malloc(vec_dim * sizeof(float));
        f = (float *)malloc(vec_dim * sizeof(float));

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
        flux_func(q_p, f, 0, alpha, gamma);

        for (int i = 0; i < vec_dim; i++) {
            qx_plus_half[offset + i] = q_p[i];
            fx_plus_half[offset + i] = f[i];
        }

        flux_func(q_m, f, 0, alpha, gamma);

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

        flux_func(q_p, f, 1, alpha, gamma);

        for (int i = 0; i < vec_dim; i++) {
            qy_plus_half[offset + i] = q_p[i];
            fy_plus_half[offset + i] = f[i];
        }

        flux_func(q_m, f, 1, alpha, gamma);

        for (int i = 0; i < vec_dim; i++) {
            qy_minus_half[offset + i] = q_m[i];
            fy_minus_half[offset + i] = f[i];
            //if (nan_check(q_p[i]) || nan_check(q_m[i])) printf("(%d, %d, %d) i: %d, qy_p: %f, qy_m: %f\n", x, y, z, i, q_p[i], q_m[i]);
        }

        free(q_p);
        free(q_m);
        free(f);
    }
}

__global__ void evolve_z(float * Un_d, flux_func_ptr flux_func,
                     float * qz_plus_half, float * qz_minus_half,
                     float * fz_plus_half, float * fz_minus_half,
                     int nx, int ny, int nz, int vec_dim, float alpha,
                     float gamma,
                     int kx_offset, int ky_offset) {
    /**
    First part of evolution through one timestep using finite volume methods.
    Reconstructs state vector to cell boundaries using slope limiter
    and calculates fluxes there.

    NOTE: we assume that beta is smooth so can get value at cell boundaries with simple averaging

    Parameters
    ----------
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
    kx_offset, ky_offset : int
        x, y offset for current kernel
    */

    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    int offset = ((z * ny + y) * nx + x) * vec_dim;

    if ((x > 0) && (x < (nx-1)) &&
        (y > 0) && (y < (ny-1)) &&
        (z > 0) && (z < (nz-1))) {

        float * q_p, *q_m, * f;
        q_p = (float *)malloc(vec_dim * sizeof(float));
        q_m = (float *)malloc(vec_dim * sizeof(float));
        f = (float *)malloc(vec_dim * sizeof(float));

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
        flux_func(q_p, f, 2, alpha, gamma);

        for (int i = 0; i < vec_dim; i++) {
            qz_plus_half[offset + i] = q_p[i];
            fz_plus_half[offset + i] = f[i];
        }

        flux_func(q_m, f, 2, alpha, gamma);

        for (int i = 0; i < vec_dim; i++) {
            qz_minus_half[offset + i] = q_m[i];
            fz_minus_half[offset + i] = f[i];
        }
        free(q_p);
        free(q_m);
        free(f);
    }
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
    if ((x > 0) && (x < (nx-1)) &&
        (y > 0) && (y < (ny-1)) &&
        (z > 0) && (z < (nz-1))) {
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
            if (nan_check(F[((z * ny + y) * nx + x)*vec_dim + i]))
                F[((z * ny + y) * nx + x)*vec_dim + i] = old_F;
        }
    }
}

__global__ void evolve_fv_heating(float * Up, float * U_half,
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
        W = W_swe(q_swe);

        float * A, * phis;
        A = (float *)malloc(nlayers * sizeof(float));
        phis = (float *)malloc(nlayers * sizeof(float));
        for (int i = 0; i < nlayers; i++) {
            phis[i] = U_half[((i * ny + y) * nx + x)* 4];
        }

        calc_As(rho, phis, A, nlayers, gamma, phis[0], rho[0]);

        float p = p_from_swe(q_swe, rho[z], gamma, W, A[z]);
        float Y = q_swe[3] / q_swe[0];

        X_dot = calc_Q_swe(rho[z], p, gamma, Y, Cv) / E_He;

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
            deltaQx = Q_d[z] * (U_half[offset*4+1] -
                 U_half[(((z + 1) * ny + y) * nx + x)*4+1]) /
                (W * U_half[offset*4]);
            deltaQy = (Q_d[z]) * (U_half[offset*4+2] -
                 U_half[(((z + 1) * ny + y) * nx + x)*4+2]) /
                (W * U_half[offset*4]);
        }
        if (z > 0) {
            sum_qs += -rho[z-1] / rho[z] * (Q_d[z] - Q_d[z - 1]);
            deltaQx = rho[z-1] / rho[z] * Q_d[z] *
                (U_half[offset*4+1] -
                 U_half[(((z - 1) * ny + y) * nx + x)*4+1]) /
                 (W * U_half[offset*4]);
            deltaQy = rho[z-1] / rho[z] * Q_d[z] *
                (U_half[offset*4+2] -
                 U_half[(((z - 1) * ny + y) * nx + x)*4+2]) /
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
       int * cumulative_kernels, float * Un_d, float * F_d,
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
           evolve_fv<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(Un_d, h_flux_func,
                  qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                  fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                  nx, ny, nz, vec_dim, alpha, gamma,
                  kx_offset, ky_offset);
           if (do_z) {
               evolve_z<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(Un_d, h_flux_func,
                      qz_p_d, qz_m_d,
                      fz_p_d, fz_m_d,
                      nx, ny, nz, vec_dim, alpha, gamma,
                      kx_offset, ky_offset);
           }
           kx_offset += blocks[k_offset + j * kernels[rank].x + i].x *
                threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
       }
       ky_offset += blocks[k_offset + j * kernels[rank].x].y *
            threads[k_offset + j * kernels[rank].x].y - 2*ng;
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

            kx_offset += blocks[k_offset + j * kernels[rank].x + i].x *
                threads[k_offset + j * kernels[rank].x + i].x - 2*ng;
       }
       ky_offset += blocks[k_offset + j * kernels[rank].x].y *
            threads[k_offset + j * kernels[rank].x].y - 2*ng;
    }
}

void rk3(dim3 * kernels, dim3 * threads, dim3 * blocks,
       int * cumulative_kernels,
       float * Un_d, float * F_d, float * Up_d,
       float * qx_p_d, float * qx_m_d, float * qy_p_d, float * qy_m_d,
       float * qz_p_d, float * qz_m_d,
       float * fx_p_d, float * fx_m_d, float * fy_p_d, float * fy_m_d,
       float * fz_p_d, float * fz_m_d,
       int nx, int ny, int nz, int vec_dim, int ng, float alpha, float gamma,
       float dx, float dy, float dz, float dt,
       float * Up_h, float * F_h, float * Un_h,
       MPI_Comm comm, MPI_Status status, int rank, int n_processes,
       flux_func_ptr flux_func, bool do_z, bool periodic) {
    /**
    Integrates the homogeneous part of the ODE in time using RK3.

    Parameters
    ----------
    kernels, threads, blocks : dim3 *
        number of kernels, threads and blocks for each process/kernel
    cumulative_kernels : int *
        Cumulative total of kernels in ranks < rank of current MPI process
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
    periodic : bool
        do we use periodic or outflow boundary conditions?
    */
    //cout << "\nu1\n\n\n";
    // u1 = un + dt * F(un)
    homogeneuous_fv(kernels, threads, blocks, cumulative_kernels,
          Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
          nx, ny, nz, vec_dim, ng, alpha, gamma,
          dx, dy, dz, dt, rank, flux_func, do_z);

    // copy back flux
    cudaMemcpy(F_h, F_d, nx*ny*nz*vec_dim*sizeof(float),
        cudaMemcpyDeviceToHost);
    if (n_processes == 1) {
        bcs_fv(F_h, nx, ny, nz, ng, vec_dim, periodic, do_z);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(F_h, nx, ny, nz, vec_dim, ng, comm, status, rank, n_processes,
                y_size, do_z, periodic);
    }

    for (int n = 0; n < nx*ny*nz*vec_dim; n++) {
        Up_h[n] = Un_h[n] + dt * F_h[n];
    }
    // enforce boundaries and copy back
    if (n_processes == 1) {
        bcs_fv(Up_h, nx, ny, nz, ng, vec_dim, periodic, do_z);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(Up_h, nx, ny, nz, vec_dim, ng, comm, status, rank,
                n_processes, y_size, do_z, periodic);
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

    cudaMemcpy(Un_d, Up_h, nx*ny*nz*vec_dim*sizeof(float),
               cudaMemcpyHostToDevice);
    //cout << "\nu2\n\n\n";
    // u2 = 0.25 * (3*un + u1 + dt*F(u1))
    homogeneuous_fv(kernels, threads, blocks, cumulative_kernels,
          Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
          nx, ny, nz, vec_dim, ng, alpha, gamma,
          dx, dy, dz, dt, rank, flux_func, do_z);

    // copy back flux
    cudaMemcpy(F_h, F_d, nx*ny*nz*vec_dim*sizeof(float),
               cudaMemcpyDeviceToHost);

    if (n_processes == 1) {
        bcs_fv(F_h, nx, ny, nz, ng, vec_dim, periodic, do_z);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(F_h, nx, ny, nz, vec_dim, ng, comm, status, rank, n_processes,
                y_size, do_z, periodic);
    }

    for (int n = 0; n < nx*ny*nz*vec_dim; n++) {
        Up_h[n] = 0.25 * (3.0 * Un_h[n] + Up_h[n] + dt * F_h[n]);
    }

    // enforce boundaries and copy back
    if (n_processes == 1) {
        bcs_fv(Up_h, nx, ny, nz, ng, vec_dim, periodic, do_z);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(Up_h, nx, ny, nz, vec_dim, ng, comm, status, rank,
                n_processes, y_size, do_z, periodic);
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

    cudaMemcpy(Un_d, Up_h, nx*ny*nz*vec_dim*sizeof(float),
               cudaMemcpyHostToDevice);
    //cout << "\nun+1\n\n\n";
    // un+1 = (1/3) * (un + 2*u2 + 2*dt*F(u2))
    homogeneuous_fv(kernels, threads, blocks, cumulative_kernels,
          Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
          nx, ny, nz, vec_dim, ng, alpha, gamma,
          dx, dy, dz, dt, rank, flux_func, do_z);

    // copy back flux
    cudaMemcpy(F_h, F_d, nx*ny*nz*vec_dim*sizeof(float),
               cudaMemcpyDeviceToHost);

    if (n_processes == 1) {
        bcs_fv(F_h, nx, ny, nz, ng, vec_dim, periodic, do_z);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(F_h, nx, ny, nz, vec_dim, ng, comm, status, rank, n_processes,
                y_size, do_z, periodic);
    }

    for (int n = 0; n < nx*ny*nz*vec_dim; n++) {
        Up_h[n] = (1/3.0) * (Un_h[n] + 2.0*Up_h[n] + 2.0*dt * F_h[n]);
    }

    // enforce boundaries
    if (n_processes == 1) {
        bcs_fv(Up_h, nx, ny, nz, ng, vec_dim, periodic, do_z);
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(Up_h, nx, ny, nz, vec_dim, ng, comm, status, rank,
                n_processes, y_size, do_z, periodic);
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
        //if (!do_z) Un_h[j] = Up_h[j];
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

void cuda_run(float * beta, float * gamma_up,
         float ** Us_h, float * rho, float * Q,
         int * nxs, int * nys, int * nzs, int nlevels, char * models,
         int * vec_dims, int ng,
         int nt, float alpha, float gamma, float E_He, float Cv,
         float zmin,
         float dx, float dy, float dz, float dt, bool burning,
         bool periodic, int dprint, char * filename, char * param_filename,
         MPI_Comm comm, MPI_Status status, int rank, int n_processes,
         int * matching_indices, int r, int n_print_levels,
         int * print_levels, int tstart) {
    /**
    Evolve system through nt timesteps, saving data to filename every dprint timesteps.

    Parameters
    ----------
    beta : float *
        shift vector at each grid point
    gamma_up : float *
        gamma matrix at each grid point
    Us_h : float **
        state vector at each grid point in each layer at current timestep on host in grids
    rho : float *
        densities in each layer
    Q : float *
        heating rate at each point and in each layer
    nxs, nys, nzs : int *
        dimensions of grids
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
    periodic : bool
        do we use periodic or outflow boundary conditions?
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
    r : int
        ratio of resolutions
    print_level : int
        number of the level to be output to file
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
    if (rank >= count) return;

    // redefine - we only want to run on as many cores as we have GPUs
    if (n_processes > count) n_processes = count;

    if (rank == 0) {
        printf("Running on %i processor(s)\n", n_processes);
    }

    int maxThreads = 256;
    int maxBlocks = 160;

    dim3 *kernels = new dim3[n_processes];
    int *cumulative_kernels = new int[n_processes];

    getNumKernels(array_max(nxs, nlevels), array_max(nys, nlevels),
                  array_max(nzs, nlevels), ng, n_processes,
                  &maxBlocks, &maxThreads, kernels, cumulative_kernels);

    int total_kernels = cumulative_kernels[n_processes-1];

    dim3 *blocks = new dim3[total_kernels];
    dim3 *threads = new dim3[total_kernels];

    getNumBlocksAndThreads(array_max(nxs, nlevels), array_max(nys, nlevels),
                           array_max(nzs, nlevels), ng, maxBlocks, maxThreads,
                           n_processes, kernels, blocks, threads);

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
    float * rho_d, * Q_d;

    // set device
    cudaSetDevice(rank);

    // index of first multilayer SWE grid level
    int m_in = 0;
    while (models[m_in] != 'M') m_in += 1;
    // index of first compressible grid level
    int c_in = nlevels;
    if (models[nlevels-1] == 'C') {
        while(models[c_in-1] == 'C') c_in -= 1;
    }
    // allocate memory on device
    cudaMalloc((void**)&rho_d, nzs[m_in]*sizeof(float));
    cudaMalloc((void**)&Q_d, nzs[m_in]*sizeof(float));

    // copy stuff to GPU
    cudaMemcpyToSymbol(beta_d, beta, 3*sizeof(float));
    cudaMemcpyToSymbol(gamma_up_d, gamma_up, 9*sizeof(float));
    cudaMemcpy(rho_d, rho, nzs[m_in]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Q_d, Q, nzs[m_in]*sizeof(float), cudaMemcpyHostToDevice);

    int grid_size = nxs[0]*nys[0]*nzs[0]*vec_dims[0];
    for (int i = 1; i < nlevels; i++) {
        grid_size = max(nxs[i]*nys[i]*nzs[i]*vec_dims[i], grid_size);
    }

    int * nxs_d, * nys_d, * nzs_d;
    cudaMalloc((void**)&nxs_d, nlevels*sizeof(int));
    cudaMalloc((void**)&nys_d, nlevels*sizeof(int));
    cudaMalloc((void**)&nzs_d, nlevels*sizeof(int));
    cudaMemcpy(nxs_d, nxs, nlevels*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(nys_d, nys, nlevels*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(nzs_d, nzs, nlevels*sizeof(int), cudaMemcpyHostToDevice);

    float * U_h = new float[grid_size];
    float * Up_h = new float[grid_size];
    float * F_h = new float[grid_size];

    // initialise
    for (int i = 0; i < grid_size; i++) {
        U_h[i] = 0.0;
        Up_h[i] = 0.0;
        F_h[i] = 0.0;
    }

    float * U_d, * U_half_d, * Up_d, * F_d;

    cudaMalloc((void**)&U_d, grid_size*sizeof(float));
    cudaMalloc((void**)&U_half_d, grid_size*sizeof(float));
    cudaMalloc((void**)&Up_d, grid_size*sizeof(float));
    cudaMalloc((void**)&F_d, grid_size*sizeof(float));

    // initialise with coarsest grid
    for (int i = 0; i < nxs[0]*nys[0]*nzs[0]*vec_dims[0]; i++) {
        U_h[i] = Us_h[0][i];
    }
    cudaMemcpy(U_d, U_h, nxs[0]*nys[0]*nzs[0]*vec_dims[0]*sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(Up_d, Up_h, grid_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F_h, grid_size*sizeof(float), cudaMemcpyHostToDevice);

    float *qx_p_d, *qx_m_d, *qy_p_d, *qy_m_d, *qz_p_d, *qz_m_d, *fx_p_d,
          *fx_m_d, *fy_p_d, *fy_m_d, *fz_p_d, *fz_m_d;

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

    // find size of largest compressible and SWE grids
    int largest_comp_grid = 0;
    int largest_swe_grid = 0;
    for (int i = 0; i < nlevels; i++) {
        if (models[i] == 'M' || models[i] == 'S') {
            largest_swe_grid = max(largest_swe_grid, nxs[i]*nys[i]*nzs[i]);
        } else if (models[i] == 'C') {
            largest_comp_grid = max(largest_comp_grid, nxs[i]*nys[i]*nzs[i]);
        }
    }

    float * q_comp_d;
    cudaMalloc((void**)&q_comp_d, largest_swe_grid*6*sizeof(float));
    float * qf_swe;
    cudaMalloc((void**)&qf_swe, largest_comp_grid*4*sizeof(float));
    float *old_phi_d, *sum_phs_d;
    cudaMalloc((void**)&old_phi_d, largest_swe_grid*sizeof(float));
    cudaMalloc((void**)&sum_phs_d, largest_swe_grid*sizeof(float));

    // initialise old_phi with phi on coarsest multilayer SWE grid
    float *pphi = new float[largest_swe_grid];
    for (int j = 0; j < nxs[m_in]*nys[m_in]*nzs[m_in]; j++) {
        pphi[j] = Us_h[m_in][j*vec_dims[m_in]];
    }
    cudaMemcpy(old_phi_d, pphi, nxs[m_in]*nys[m_in]*nzs[m_in]*sizeof(float),
               cudaMemcpyHostToDevice);

    float * sum_phs_h = new float[largest_swe_grid];

    int * matching_indices_d;
    cudaMalloc((void**)&matching_indices_d, (nlevels-1)*4*sizeof(int));
    cudaMemcpy(matching_indices_d, matching_indices,
               (nlevels-1)*4*sizeof(int), cudaMemcpyHostToDevice);

    // make host-side function pointers to __device__ functions
    flux_func_ptr h_compressible_fluxes;
    flux_func_ptr h_shallow_water_fluxes;

    // copy function pointers to host equivalent
    cudaMemcpyFromSymbol(&h_compressible_fluxes, d_compressible_fluxes,
                         sizeof(flux_func_ptr));
    cudaMemcpyFromSymbol(&h_shallow_water_fluxes, d_shallow_water_fluxes,
                         sizeof(flux_func_ptr));

    cudaError_t err;

    // if first layer is single layer SWE, need to restrict multilayer SWE
    // data (where initial data has been defined) to this
    if (tstart == 0) {
        for (int i = min(c_in,nlevels-1); i > 0; i--) {
            // TODO: check if need to do BCS stuff here
            cudaMemcpy(U_d, Us_h[i-1],
                nxs[i-1]*nys[i-1]*nzs[i-1]*vec_dims[i-1]*sizeof(float),
                cudaMemcpyHostToDevice);
            cudaMemcpy(Up_d, Us_h[i],
                nxs[i]*nys[i]*nzs[i]*vec_dims[i]*sizeof(float),
                cudaMemcpyHostToDevice);
            // select restriction algorithm
            if (models[i-1] == 'M' && models[i] == 'C') {
              // compressible to multilayer SWE
              restrict_comp_to_swe(kernels, threads, blocks,
                        cumulative_kernels,
                        U_d, Up_d, nxs_d, nys_d, nzs_d,
                        dz/pow(r, i), zmin, matching_indices_d,
                        rho_d, gamma, ng, rank, qf_swe, i-1);
            } else if (models[i-1] == 'M' && models[i] == 'M') {
              // multilayer SWE to multilayer SWE
              restrict_multiswe_to_multiswe(kernels, threads, blocks,
                        cumulative_kernels,
                        U_d, Up_d, nxs_d, nys_d, nzs_d,
                        matching_indices_d,
                        ng, rank, i-1);
            } else if (models[i-1] == 'C' && models[i] == 'C') {
              // compressible to compressible
              restrict_comp_to_comp(kernels, threads, blocks,
                        cumulative_kernels,
                        U_d, Up_d, nxs_d, nys_d, nzs_d,
                        matching_indices_d,
                        ng, rank, i-1);
            } else if (models[i-1] == 'S' && (models[i] == 'S' || models[i] == 'M')) {
              // multilayer SWE to single layer SWE
              restrict_swe_to_swe(kernels, threads, blocks,
                        cumulative_kernels,
                        U_d, Up_d, nxs_d, nys_d, nzs_d,
                        matching_indices_d,
                        ng, rank, i-1);
            }
            cudaMemcpy(Us_h[i-1], U_d,
                  nxs[i-1]*nys[i-1]*nzs[i-1]*vec_dims[i-1]*sizeof(float),
                  cudaMemcpyDeviceToHost);

            // enforce boundaries
            for (int x = 0; x < matching_indices[(i-1)*4]+ng; x++) {
                for (int y = 0; y < nys[i-1]*nzs[i-1]; y++) {
                    for (int n = 0; n < vec_dims[i-1]; n++) {
                        Us_h[i-1][(y * nxs[i-1] + x) * vec_dims[i-1] + n] =
                            Us_h[i-1][(y * nxs[i-1] + matching_indices[(i-1)*4]+ng) * vec_dims[i-1] + n];
                    }
                }
            }
            for (int x = matching_indices[(i-1)*4+1]; x < nxs[i-1]; x++) {
                for (int y = 0; y < nys[i-1]*nzs[i-1]; y++) {
                    for (int n = 0; n < vec_dims[i-1]; n++) {
                        Us_h[i-1][(y * nxs[i-1] + x) * vec_dims[i-1] + n] =
                            Us_h[i-1][(y * nxs[i-1] + matching_indices[(i-1)*4+1]-ng) * vec_dims[i-1] + n];
                    }
                }
            }
            for (int z = 0; z < nzs[i-1]; z++) {
                for (int x = 0; x < nxs[i-1]; x++) {
                    for (int y = 0; y < matching_indices[(i-1)*4+2]+ng; y++) {
                        for (int n = 0; n < vec_dims[i-1]; n++) {
                            Us_h[i-1][((z * nys[i-1] + y) * nxs[i-1] + x) * vec_dims[i-1] + n] =
                                Us_h[i-1][((z * nys[i-1] + matching_indices[(i-1)*4+2]+ng) * nxs[i-1] + x) * vec_dims[i-1] + n];
                        }
                    }
                }
                for (int x = 0; x < nxs[i-1]; x++) {
                    for (int y = matching_indices[(i-1)*4+3]; y < nys[i-1]; y++) {
                        for (int n = 0; n < vec_dims[i-1]; n++) {
                            Us_h[i-1][((z * nys[i-1] + y) * nxs[i-1] + x) * vec_dims[i-1] + n] =
                                Us_h[i-1][((z * nys[i-1] + matching_indices[(i-1)*4+3]-ng) * nxs[i-1] + x) * vec_dims[i-1] + n];
                        }
                    }
                }
            }
        }

        //for (int i = 0; i < nxs[0]*nys[0]*nzs[0]; i++) {
            //cout << Us_h[0][i*4+1] << ' ' << Us_h[0][i*4+1] << '\n';
        //}

        // prolong data from coarsest swe grid to finer grids
        for (int i = c_in; i < (nlevels-1); i++) {
            cudaMemcpy(U_d, Us_h[i],
                    nxs[i]*nys[i]*nzs[i]*vec_dims[i]*sizeof(float),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(Up_d, Us_h[i+1],
                    nxs[i+1]*nys[i+1]*nzs[i+1]*vec_dims[i+1]*sizeof(float),
                    cudaMemcpyHostToDevice);

            // select prolongation algorithm
            if (models[i] == 'M' && models[i+1] == 'C') {
                // multilayer SWE to compressible
                for (int j = 0; j < nxs[i]*nys[i]*nzs[i]; j++) {
                    pphi[j] = Us_h[i][j*4];
                }
                cudaMemcpy(old_phi_d, pphi,
                        nxs[i]*nys[i]*nzs[i]*sizeof(float),
                        cudaMemcpyHostToDevice);
                prolong_swe_to_comp(kernels, threads, blocks,
                             cumulative_kernels,
                             U_d, Up_d, nxs_d, nys_d, nzs_d,
                             dz/pow(r, i), dt/pow(r, i), zmin,
                             rho_d, gamma, matching_indices_d, ng, rank,
                             q_comp_d, old_phi_d, i, false);
            } else if (models[i] == 'M' && models[i+1] == 'M') {
                // multilayer SWE to multilayer SWE
                prolong_multiswe_to_multiswe(kernels, threads, blocks,
                             cumulative_kernels,
                             U_d, Up_d, nxs_d, nys_d, nzs_d,
                             matching_indices_d, ng, rank, i, false);
            } else if (models[i] == 'C' && models[i+1] == 'C') {
                // compressible to compressible
                prolong_comp_to_comp(kernels, threads, blocks,
                             cumulative_kernels,
                             U_d, Up_d, nxs_d, nys_d, nzs_d,
                             matching_indices_d, ng, rank, i, false);
            } else if (models[i] == 'S' && (models[i+1] == 'S' || models[i+1] == 'M')) {
                // single layer SWE to multilayer SWE
                prolong_swe_to_swe(kernels, threads, blocks,
                             cumulative_kernels,
                             U_d, Up_d, nxs_d, nys_d, nzs_d,
                             matching_indices_d, ng, rank, i, false);
            }

            err = cudaGetLastError();
            if (err != cudaSuccess){
                cout << "After prolonging\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }

            cudaMemcpy(Us_h[i+1], Up_d,
                    nxs[i+1]*nys[i+1]*nzs[i+1]*vec_dims[i+1]*sizeof(float),
                    cudaMemcpyDeviceToHost);

            err = cudaGetLastError();
            if (err != cudaSuccess){
                cout << "After copying\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }

            bool do_z = true;
            if (models[i+1] == 'M' || models[i+1] == 'S') {
                do_z = false;
            }

            // enforce boundaries
            if (n_processes == 1) {
                bcs_fv(Us_h[i+1], nxs[i+1], nys[i+1], nzs[i+1],
                        ng, vec_dims[i+1], false, do_z);
            } else {
                int y_size = kernels[0].y*blocks[0].y*threads[0].y - 2*ng;
                bcs_mpi(Us_h[i+1], nxs[i+1], nys[i+1], nzs[i+1],
                        vec_dims[i+1], ng, comm, status, rank,
                        n_processes, y_size, do_z, false);
            }
        }
        // NOTE: Initial conditions for multiscale test
        /*if (models[nlevels-1] == 'C') { // there's at least one compressible level
            for (int z = 0; z < nzs[nlevels-1]; z++) {
                for (int y = 0; y < nys[nlevels-1]; y++) {
                    for (int x = 0; x < nxs[nlevels-1]; x++) {
                        float max_v = 0.3;
                        float r = sqrt(
                            (x - 0.5*nxs[nlevels-1])*(x - 0.5*nxs[nlevels-1]) +
                            (y - 0.5*nys[nlevels-1])*(y - 0.5*nys[nlevels-1]));
                        float v = 0.0;
                        if (r < 0.05 * nxs[nlevels-1]) {
                            v = 20.0 * max_v * r / nxs[nlevels-1];
                        } else if (r < 0.1 * nxs[nlevels-1]) {
                            v = 2.0 * 20.0 * max_v * 0.05 - 20.0 * max_v * r / nxs[nlevels-1];
                        }
                        float D = Us_h[nlevels-1][((z*nys[nlevels-1] + y) * nxs[nlevels-1] + x) * vec_dims[nlevels-1]];

                        if (r > 0.0) {
                            // Sx
                            Us_h[nlevels-1][((z * nys[nlevels-1] + y) * nxs[nlevels-1] + x) * vec_dims[nlevels-1] + 1]
                                = - D * v * (y - 0.5*nys[nlevels-1]) / r;
                            Us_h[nlevels-1][((z * nys[nlevels-1] + y) * nxs[nlevels-1] + x) * vec_dims[nlevels-1] + 2]
                                = D * v * (x - 0.5*nxs[nlevels-1]) / r;
                        }
                    }
                }
            }
        }*/
    }

    hid_t outFile;
    hid_t * dset = new hid_t[n_print_levels];
    hid_t * mem_space = new hid_t[n_print_levels];
    hid_t * file_space = new hid_t[n_print_levels];

    if (rank == 0) {
        initialise_hdf5_file(filename, nt, dprint,
            nzs, nys, nxs, vec_dims, n_print_levels,
            print_levels, Us_h, &outFile, dset, mem_space, file_space,
            param_filename);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess){
        cout << "Before evolution\n";
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    // main loop
    for (int t = tstart; t < nt; t++) {
        cout << "Evolving t = " << t << '\n';

        // Do evolutions on grids
        for (int i = (nlevels-1); i >= 0; i--) {

            flux_func_ptr flux_func = h_compressible_fluxes;
            // HACK - set back to true
            bool do_z = true;
            if (models[i] == 'M' || models[i] == 'S') { // SWE
                flux_func = h_shallow_water_fluxes;
                do_z = false;
            }

            for (int j = 0; j < pow(r, i); j++) {
                cudaMemcpy(U_d, Us_h[i],
                        nxs[i]*nys[i]*nzs[i]*vec_dims[i]*sizeof(float),
                        cudaMemcpyHostToDevice);
                // TODO: fix dz calculation (i.e. need to work out how to store it in a way such that the compressible grids are getting the correct value)

                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    cout << "Before rk3\n";
                    printf("Error: %s\n", cudaGetErrorString(err));
                }

                rk3(kernels, threads, blocks, cumulative_kernels,
                        U_d, U_half_d, Up_d,
                        qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
                        fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
                        nxs[i], nys[i], nzs[i], vec_dims[i], ng,
                        alpha, gamma,
                        dx/pow(r, i), dy/pow(r, i), dz/pow(r, i),
                        dt/pow(r, i),
                        Up_h, F_h, Us_h[i],
                        comm, status, rank, n_processes,
                        flux_func, do_z, (i==0) ? periodic : false);

                cudaDeviceSynchronize();

                /*if (models[i] == 'C') {
                    // hack on the burning
                    float * H = new float[nxs[i]*nys[i]*nzs[i]];
                    calc_Q(rho, Us_h[i], nxs[i], nys[i], nzs[i], gamma,
                           H, Cv, gamma_up);
                    for (int z = 0; z < nzs[i]; z++) {
                        for (int y = ng; y < nys[i]-ng; y++) {
                            for (int x = ng; x < nxs[i] - ng; x++) {
                                // tau
                                Us_h[i][((z*nys[i]+y)*nxs[i]+x)*6 + 4] +=
                                    dt/pow(r, i) * 0.5 * alpha *
                                    Us_h[i][((z*nys[i]+y)*nxs[i]+x)*6] *
                                    H[(z * nys[i] + y) * nxs[i] + x];
                                float X_dot =
                                    H[(z*nys[i] + y)*nxs[i] + x] / E_He;
                                // DX
                                Us_h[i][((z*nys[i]+y)*nxs[i]+x)*6+5] +=
                                    dt/pow(r, i) * 0.5 * alpha * rho[0] *
                                    X_dot;
                            }
                        }
                    }
                    delete[] H;
                } else if (models[i] == 'M') { // SWE burning
                    // update old_phi
                    for (int j = 0; j < nxs[i]*nys[i]*nzs[i]; j++) {
                        pphi[j] = Us_h[i][j*4];
                    }
                    cudaMemcpy(old_phi_d, pphi,
                            nxs[i]*nys[i]*nzs[i]*sizeof(float),
                            cudaMemcpyHostToDevice);

                    cudaMemcpy(Up_d, Us_h[i],
                            nxs[i]*nys[i]*nzs[i]*4*sizeof(float),
                            cudaMemcpyHostToDevice);
                    cudaMemcpy(U_half_d, Us_h[i],
                            nxs[i]*nys[i]*nzs[i]*4*sizeof(float),
                            cudaMemcpyHostToDevice);

                    float kx_offset = 0;
                    ky_offset = (kernels[0].y * blocks[0].y *
                                 threads[0].y - 2*ng) * rank;

                    for (int l = 0; l < kernels[rank].y; l++) {
                        kx_offset = 0;
                        for (int k = 0; k < kernels[rank].x; k++) {
                            evolve_fv_heating<<<blocks[k_offset + l * kernels[rank].x + k], threads[k_offset + l * kernels[rank].x + k]>>>(
                                   Up_d, U_half_d,
                                   qx_p_d, qx_m_d, qy_p_d, qy_m_d,
                                   fx_p_d, fx_m_d, fy_p_d, fy_m_d,
                                   sum_phs_d, rho_d, Q_d,
                                   nxs[i], nys[i], nzs[i], alpha, gamma,
                                   dx/pow(r, i), dy/pow(r, i),
                                   dt/pow(r, i),
                                   burning, Cv, E_He,
                                   kx_offset, ky_offset);
                            kx_offset += blocks[k_offset + l *
                                kernels[rank].x + k].x *
                                threads[k_offset+l*kernels[rank].x+k].x -
                                2*ng;
                        }
                        ky_offset += blocks[k_offset + l *
                            kernels[rank].x].y *
                            threads[k_offset + l * kernels[rank].x].y -
                            2*ng;
                    }

                    cudaMemcpy(Up_h, Up_d,
                            nxs[i]*nys[i]*nzs[i]*4*sizeof(float),
                            cudaMemcpyDeviceToHost);
                    cudaMemcpy(sum_phs_h, sum_phs_d,
                            nxs[i]*nys[i]*nzs[i]*sizeof(float),
                            cudaMemcpyDeviceToHost);

                    // enforce boundaries
                    if (n_processes == 1) {
                        bcs_fv(Up_h, nxs[i], nys[i], nzs[i], ng, 4, (i==0) ? periodic : false, do_z);
                        bcs_fv(sum_phs_h, nxs[i], nys[i], nzs[i], ng, 1,
                               (i==0) ? periodic : false, do_z);
                    } else {
                        int y_size = kernels[0].y * blocks[0].y *
                                     threads[0].y - 2*ng;
                        bcs_mpi(Up_h, nxs[i], nys[i], nzs[i], 4, ng, comm,
                                status, rank, n_processes, y_size, false,
                                (i==0) ? periodic : false);
                        bcs_mpi(sum_phs_h, nxs[i], nys[i], nzs[i], 1, ng,
                                comm, status, rank, n_processes, y_size,
                                false, (i==0) ? periodic : false);
                    }

                    cudaMemcpy(Up_d, Up_h,
                               nxs[i]*nys[i]*nzs[i]*4*sizeof(float),
                               cudaMemcpyHostToDevice);
                    cudaMemcpy(sum_phs_d, sum_phs_h,
                               nxs[i]*nys[i]*nzs[i]*sizeof(float),
                               cudaMemcpyHostToDevice);

                    kx_offset = 0;
                    ky_offset = (kernels[0].y * blocks[0].y *
                                 threads[0].y - 2*ng) * rank;

                    for (int p = 0; p < kernels[rank].y; p++) {
                        kx_offset = 0;
                        for (int q = 0; q < kernels[rank].x; q++) {
                            evolve2<<<blocks[k_offset + p * kernels[rank].x + q], threads[k_offset + p * kernels[rank].x + q]>>>(U_d,
                                   Up_d, U_half_d, sum_phs_d,
                                   nxs[i], nys[i], nzs[i], ng, alpha,
                                   dx/pow(r, i), dy/pow(r, i), dt/pow(r, i),
                                   kx_offset, ky_offset);
                            kx_offset +=
                                blocks[k_offset+p*kernels[rank].x + q].x *
                                threads[k_offset+p*kernels[rank].x+q].x -
                                2*ng;
                        }
                        ky_offset += blocks[k_offset + p *
                            kernels[rank].x].y *
                            threads[k_offset + p * kernels[rank].x].y -
                            2*ng;
                    }

                    cudaDeviceSynchronize();

                    err = cudaGetLastError();

                    if (err != cudaSuccess)
                        printf("Error: %s\n", cudaGetErrorString(err));

                    // boundaries
                    cudaMemcpy(Us_h[i], U_d,
                               nxs[i]*nys[i]*nzs[i]*4*sizeof(float),
                               cudaMemcpyDeviceToHost);
                    if (n_processes == 1) {
                        bcs_fv(Us_h[i], nxs[i], nys[i], nzs[i], ng, 4,
                               (i==0) ? periodic : false, do_z);
                    } else {
                        int y_size = kernels[0].y * blocks[0].y *
                                     threads[0].y - 2*ng;
                        bcs_mpi(Us_h[i], nxs[i], nys[i], nzs[i], 4, ng,
                                comm, status, rank, n_processes, y_size,
                                false, (i==0) ? periodic : false);
                    }
                    cudaMemcpy(U_d, Us_h[i],
                               nxs[i]*nys[i]*nzs[i]*4*sizeof(float),
                               cudaMemcpyHostToDevice);
                }*/

                if (n_processes == 1) {
                    bcs_fv(Us_h[i], nxs[i], nys[i], nzs[i], ng,
                           vec_dims[i], (i==0) ? periodic : false, do_z);
                } else {
                    int y_size = kernels[0].y * blocks[0].y *
                                 threads[0].y - 2*ng;
                    bcs_mpi(Us_h[i], nxs[i], nys[i], nzs[i], vec_dims[i],
                            ng, comm, status, rank, n_processes, y_size,
                            false, (i==0) ? periodic : false);
                }
            }
        }

        //for (int i = 0; i < nxs[0]*nys[0]*nzs[0]; i++) {
            //cout << Us_h[1][i*6] << '\n';
        //}

        for (int i = (nlevels-1); i > 0; i--) {
            // restrict to coarse grid
            // copy to device
            cudaMemcpy(Up_d, Us_h[i-1],
                nxs[i-1]*nys[i-1]*nzs[i-1]*vec_dims[i-1]*sizeof(float),
                cudaMemcpyHostToDevice);
            cudaMemcpy(U_d, Us_h[i],
                nxs[i]*nys[i]*nzs[i]*vec_dims[i]*sizeof(float),
                cudaMemcpyHostToDevice);
            // select restriction algorithm
            if (models[i-1] == 'M' && models[i] == 'C') {
                // compressible to multilayer SWE
                restrict_comp_to_swe(kernels, threads, blocks,
                          cumulative_kernels,
                          Up_d, U_d, nxs_d, nys_d, nzs_d,
                          dz/pow(r, i), zmin, matching_indices_d,
                          rho_d, gamma, ng, rank, qf_swe, i-1);
            } else if (models[i-1] == 'M' && models[i] == 'M') {
                // multilayer SWE to multilayer SWE
                restrict_multiswe_to_multiswe(kernels, threads, blocks,
                          cumulative_kernels,
                          Up_d, U_d, nxs_d, nys_d, nzs_d,
                          matching_indices_d,
                          ng, rank, i-1);
            } else if (models[i-1] == 'C' && models[i] == 'C') {
                // compressible to compressible
                restrict_comp_to_comp(kernels, threads, blocks,
                          cumulative_kernels,
                          Up_d, U_d, nxs_d, nys_d, nzs_d,
                          matching_indices_d,
                          ng, rank, i-1);
            } else if (models[i-1] == 'S' && (models[i] == 'S' || models[i] == 'M')) {
                // multilayer SWE to single layer SWE
                restrict_swe_to_swe(kernels, threads, blocks,
                          cumulative_kernels,
                          Up_d, U_d, nxs_d, nys_d, nzs_d,
                          matching_indices_d,
                          ng, rank, i-1);
            }
            err = cudaGetLastError();
            if (err != cudaSuccess){
                cout << "After restricting\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }

            cudaMemcpy(Us_h[i-1], Up_d,
                nxs[i-1]*nys[i-1]*nzs[i-1]*vec_dims[i-1]*sizeof(float),
                cudaMemcpyDeviceToHost);

            err = cudaGetLastError();
            if (err != cudaSuccess){
                cout << "After copying\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }
        }

        /*for (int i = 0; i < nxs[0]*nys[0]*nzs[0]; i++) {
            cout << Us_h[0][i*4] << '\n';
        }*/

        // prolong data down from coarse grids to fine grids
        for (int i = 0; i < (nlevels-1); i++) {
            cudaMemcpy(U_d, Us_h[i],
                    nxs[i]*nys[i]*nzs[i]*vec_dims[i]*sizeof(float),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(Up_d, Us_h[i+1],
                    nxs[i+1]*nys[i+1]*nzs[i+1]*vec_dims[i+1]*sizeof(float),
                    cudaMemcpyHostToDevice);

            // select prolongation algorithm
            if (models[i] == 'M' && models[i+1] == 'C') {
                // multilayer SWE to compressible
                for (int j = 0; j < nxs[i]*nys[i]*nzs[i]; j++) {
                    pphi[j] = Us_h[i][j*4];
                }
                cudaMemcpy(old_phi_d, pphi,
                        nxs[i]*nys[i]*nzs[i]*sizeof(float),
                        cudaMemcpyHostToDevice);
                prolong_swe_to_comp(kernels, threads, blocks,
                             cumulative_kernels,
                             U_d, Up_d, nxs_d, nys_d, nzs_d,
                             dz/pow(r, i), dt/pow(r, i), zmin,
                             rho_d, gamma, matching_indices_d, 2*ng, rank,
                             q_comp_d, old_phi_d, i, true);
            } else if (models[i] == 'M' && models[i+1] == 'M') {
                // multilayer SWE to multilayer SWE
                prolong_multiswe_to_multiswe(kernels, threads, blocks,
                             cumulative_kernels,
                             U_d, Up_d, nxs_d, nys_d, nzs_d,
                             matching_indices_d, 2*ng, rank, i, true);
            } else if (models[i] == 'C' && models[i+1] == 'C') {
                // compressible to compressible
                prolong_comp_to_comp(kernels, threads, blocks,
                             cumulative_kernels,
                             U_d, Up_d, nxs_d, nys_d, nzs_d,
                             matching_indices_d, ng, rank, i, true);
            } else if (models[i] == 'S' && (models[i+1] == 'S' || models[i+1] == 'M')) {
                // single layer SWE to multilayer SWE
                prolong_swe_to_swe(kernels, threads, blocks,
                             cumulative_kernels,
                             U_d, Up_d, nxs_d, nys_d, nzs_d,
                             matching_indices_d, 2*ng, rank, i, true);
            }

            err = cudaGetLastError();
            if (err != cudaSuccess){
                cout << "After prolonging\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }

            cudaMemcpy(Us_h[i+1], Up_d,
                    nxs[i+1]*nys[i+1]*nzs[i+1]*vec_dims[i+1]*sizeof(float),
                    cudaMemcpyDeviceToHost);

            err = cudaGetLastError();
            if (err != cudaSuccess){
                cout << "After copying\n";
                printf("Error: %s\n", cudaGetErrorString(err));
            }

        }

        if ((t+1) % dprint == 0) {
            for (int i = 0; i < n_print_levels; i++) {
                print_timestep(rank, n_processes, print_levels[i],
                               nxs, nys, nzs, vec_dims, ng, t, comm, status,
                               kernels, threads, blocks, Us_h,
                               dset[i], mem_space[i], file_space[i], dprint);
            }
        }
    }

    if (rank == 0) {
        close_hdf5_file(n_print_levels, mem_space, outFile);
    }

    // delete some stuff
    cudaFree(rho_d);
    cudaFree(Q_d);
    cudaFree(old_phi_d);
    cudaFree(sum_phs_d);

    cudaFree(U_d);
    cudaFree(Up_d);
    cudaFree(U_half_d);
    cudaFree(F_d);

    cudaFree(nxs_d);
    cudaFree(nys_d);
    cudaFree(nzs_d);

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
    delete[] pphi;
    delete[] sum_phs_h;

    delete[] U_h;
    delete[] Up_h;
    delete[] F_h;

    delete[] dset;
    delete[] mem_space;
    delete[] file_space;
}
