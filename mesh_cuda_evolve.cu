/**
File containing routines which model the evolution.
**/

__global__ void evolve_fv(float * Un_d, flux_func_ptr flux_func,
                     float * qx_plus_half, float * qx_minus_half,
                     float * qy_plus_half, float * qy_minus_half,
                     float * fx_plus_half, float * fx_minus_half,
                     float * fy_plus_half, float * fy_minus_half,
                     int nx, int ny, int nz, int vec_dim, float alpha0,
                     float gamma, float zmin, float dz, float R,
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
        flux_func(q_p, f, 0, alpha0, gamma, zmin, dz, nz, z, R);

        for (int i = 0; i < vec_dim; i++) {
            qx_plus_half[offset + i] = q_p[i];
            fx_plus_half[offset + i] = f[i];
        }

        flux_func(q_m, f, 0, alpha0, gamma, zmin, dz, nz, z, R);

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

        flux_func(q_p, f, 1, alpha0, gamma, zmin, dz, nz, z, R);

        for (int i = 0; i < vec_dim; i++) {
            qy_plus_half[offset + i] = q_p[i];
            fy_plus_half[offset + i] = f[i];
        }

        flux_func(q_m, f, 1, alpha0, gamma, zmin, dz, nz, z, R);

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
                     int nx, int ny, int nz, int vec_dim, float alpha0,
                     float gamma, float zmin, float dz, float R,
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

    // NOTE: z numbering is from top to bottom, so need to invert
    // so that velocity points from bottom to top

    if ((x < nx) && (y < ny) && (z > 0) && (z < (nz-1))) {

        float * q_p, *q_m, * f;
        q_p = (float *)malloc(vec_dim * sizeof(float));
        q_m = (float *)malloc(vec_dim * sizeof(float));
        f = (float *)malloc(vec_dim * sizeof(float));

        // z-direction
        for (int i = 0; i < vec_dim; i++) {
            float S_upwind = (Un_d[(((z-1) * ny + y) * nx + x) * vec_dim + i] -
                Un_d[((z * ny + y) * nx + x) * vec_dim + i]);
            float S_downwind = (Un_d[((z * ny + y) * nx + x) * vec_dim + i] -
                Un_d[(((z+1) * ny + y) * nx + x) * vec_dim + i]);
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
        flux_func(q_p, f, 2, alpha0, gamma, zmin, dz, nz, z, R);

        for (int i = 0; i < vec_dim; i++) {
            qz_plus_half[offset + i] = q_p[i];
            fz_plus_half[offset + i] = f[i];
        }

        flux_func(q_m, f, 2, alpha0, gamma, zmin, dz, nz, z, R);

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
                     int nx, int ny, int nz, int vec_dim, float alpha0,
                     float dx, float dy, float dz, float dt, float zmin,
                     float R,
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
        float alpha;
        if (vec_dim < 6) {
            // shallow water
            alpha = sqrt(exp(-2.0 * 0.25 * (qx_plus_half[((z * ny + y) * nx + x) * vec_dim] + qx_minus_half[((z * ny + y) * nx + x) * vec_dim] + qy_plus_half[((z * ny + y) * nx + x) * vec_dim] + qy_minus_half[((z * ny + y) * nx + x) * vec_dim])));
        } else {
            float h = zmin + dz * (nz - 1 - z);
            float M = 1;
            alpha = alpha0 + M * h / (R*R * alpha0);
        }

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
                     int nx, int ny, int nz, int vec_dim, float alpha0,
                     float dz, float dt, float zmin, float R,
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
    if ((x > 1) && (x < (nx-2)) && (y > 1) && (y < (ny-2)) &&
        (z > 1) && (z < (nz-2))) {

        float h = zmin + dz * (nz - 1 - z);
        float M = 1;
        float alpha = alpha0 + M * h / (R*R * alpha0);

        // NOTE: z numbering is from top to bottom, so need to invert
        // so that velocity points from bottom to top

        for (int i = 0; i < vec_dim; i++) {
            // z-boundary
            // from below
            float fz_m = 0.5 * (
                fz_plus_half[((z * ny + y) * nx + x) * vec_dim + i] +
                fz_minus_half[(((z+1) * ny + y) * nx + x) * vec_dim + i] +
                qz_plus_half[((z * ny + y) * nx + x) * vec_dim + i] -
                qz_minus_half[(((z+1) * ny + y) * nx + x) * vec_dim + i]);
            // from above
            float fz_p = 0.5 * (
                fz_plus_half[(((z-1) * ny + y) * nx + x) * vec_dim + i] +
                fz_minus_half[((z * ny + y) * nx + x) * vec_dim + i] +
                qz_plus_half[(((z-1) * ny + y) * nx + x) * vec_dim + i] -
                qz_minus_half[((z * ny + y) * nx + x) * vec_dim + i]);

            float old_F = F[((z * ny + y) * nx + x)*vec_dim + i];

            F[((z * ny + y) * nx + x)*vec_dim + i] =
                F[((z * ny + y) * nx + x)*vec_dim + i]
                - alpha * (fz_p - fz_m) / dz;

            // hack?
            if (nan_check(F[((z * ny + y) * nx + x)*vec_dim + i]))
                F[((z * ny + y) * nx + x)*vec_dim + i] = old_F;
        }

        //printf("Fz before: %f, after: %f\n", before, F[((z * ny + y) * nx + x)*vec_dim + 3]);
    }
}

__global__ void grav_sources(float * q, float gamma,
    int nx, int ny, int nz, int vec_dim, float zmin, float R, float alpha0,
    float dz, float dt,
    int kx_offset, int ky_offset) {
    /**
    Calculate gravitational source terms
    **/
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;
    int offset = ((z * ny + y) * nx + x) * vec_dim;

    if ((x > 0) && (x < (nx-1)) &&
        (y > 0) && (y < (ny-1)) &&
        (z > 0) && (z < (nz-1))) {

        float h = zmin + dz * (nz - 1 - z);
        float M = 1;
        float alpha = alpha0 + M * h / (R*R * alpha0);
        float * gamma_up;
        gamma_up = (float *)malloc(9 * sizeof(float));
        for (int i = 0; i < 9; i++) {
            gamma_up[i] = 0.0;
        }
        gamma_up[0] = 1.0;
        gamma_up[4] = 1.0;
        gamma_up[8] = alpha*alpha;

        const float TOL = 1.0e-5;
        float D = q[offset];
        float Sx = q[offset+1];
        float Sy = q[offset+2];
        float Sz = q[offset+3];
        float tau = q[offset+4];

        float Ssq = Sx*Sx*gamma_up[0] + 2.0*Sx*Sy*gamma_up[1] +
            2.0*Sx*Sz*gamma_up[2] + Sy*Sy*gamma_up[4] + 2.0*Sy*Sz*gamma_up[5] +
            Sz*Sz*gamma_up[8];

        float pmin = (1.0 - Ssq) * (1.0 - Ssq) * tau * (gamma - 1.0);
        float pmax = (gamma - 1.0) * (tau + D) / (2.0 - gamma);

        if (pmin < 0.0) {
            pmin = 0.0;//1.0e-9;
        }
        if (pmax < 0.0 || pmax < pmin) {
            pmax = 1.0;
        }

        // check sign change
        if (f_of_p(pmin, D, Sx, Sy, Sz, tau, gamma, gamma_up) *
            f_of_p(pmax, D, Sx, Sy, Sz, tau, gamma, gamma_up) > 0.0) {
            pmin = 0.0;
        }
        if (f_of_p(pmin, D, Sx, Sy, Sz, tau, gamma, gamma_up) *
            f_of_p(pmax, D, Sx, Sy, Sz, tau, gamma, gamma_up) > 0.0) {
            pmax *= 10.0;
        }

        float p = zbrent((fptr)f_of_p, pmin, pmax, TOL, D, Sx, Sy, Sz,
                        tau, gamma, gamma_up);
        if (nan_check(p) || p < 0.0 || p > 1.0e9){
            p = abs((gamma - 1.0) * (tau + D) / (2.0 - gamma)) > 1.0 ? 1.0 :
                abs((gamma - 1.0) * (tau + D) / (2.0 - gamma));
        }

        float sq = sqrt(pow(tau + p + D, 2) - Ssq);
        if (nan_check(sq))
            sq = tau + p + D;

        float hh, W2;

        if (abs(D) < TOL) {
            hh = 1.0;
            W2 = 1.0;
        } else {
            hh = 1.0 + gamma * (sq - p * (tau + p + D)/sq - D) / D;
            W2 = 1.0 + Ssq / (D*D*hh*hh);
        }

        printf("source term/dt: %f, p: %f, Sx, Sy, Sz: (%f, %f, %f), D: %f\n", (-M / (R*R) * (Sz*Sz / W2 + (tau + p + D) / alpha)), p, Sx, Sy, Sz, D);

        q[offset+3] += dt *
            (-M / (R*R) * (Sz*Sz / W2 + (tau + p + D) / alpha));
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
        float * gamma_up;
        gamma_up = (float *)malloc(9 * sizeof(float));
        for (int i = 0; i < 9; i++) {
            gamma_up[i] = 0.0;
        }
        gamma_up[0] = 1.0;
        gamma_up[4] = 1.0;
        gamma_up[8] = exp(2.0 * q_swe[0]);
        W = W_swe(q_swe, gamma_up);

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
        free(gamma_up);

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
       int nx, int ny, int nz, int vec_dim, int ng, float alpha0, float gamma,
       float dx, float dy, float dz, float dt, int rank, float zmin, float R,
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
                  nx, ny, nz, vec_dim, alpha0, gamma,
                  zmin, dz, R,
                  kx_offset, ky_offset);
           if (do_z) {
               evolve_z<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(Un_d, h_flux_func,
                      qz_p_d, qz_m_d,
                      fz_p_d, fz_m_d,
                      nx, ny, nz, vec_dim, alpha0, gamma,
                      zmin, dz, R,
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
                  nx, ny, nz, vec_dim, alpha0,
                  dx, dy, dz, dt, zmin, R, kx_offset, ky_offset);

            if (do_z) {
                evolve_z_fluxes<<<blocks[k_offset + j * kernels[rank].x + i], threads[k_offset + j * kernels[rank].x + i]>>>(
                       F_d,
                       qz_p_d, qz_m_d,
                       fz_p_d, fz_m_d,
                       nx, ny, nz, vec_dim, alpha0,
                       dz, dt, zmin, R, kx_offset, ky_offset);
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
       int level,
       int *nxs, int *nys, int *nzs, int *vec_dims, int ng, float alpha0, float R, float gamma,
       float dx, float dy, float dz, float dt,
       float * Up_h, float * F_h, float * Un_h,
       MPI_Comm comm, MPI_Status status, int rank, int n_processes,
       flux_func_ptr flux_func, bool do_z, bool periodic,
       int m_in, float * U_swe, int * matching_indices, float zmin) {
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
          nxs[level], nys[level], nzs[level], vec_dims[level], ng, alpha0, gamma,
          dx, dy, dz, dt, rank, zmin, R, flux_func, do_z);

    // copy back flux
    cudaMemcpy(F_h, F_d, nxs[level]*nys[level]*nzs[level]*vec_dims[level]*sizeof(float),
        cudaMemcpyDeviceToHost);
    if (n_processes == 1) {
        bcs_fv(F_h, nxs[level], nys[level], nzs[level], ng, vec_dims[level], periodic, do_z);
        if (do_z) {
            enforce_hse(F_h, U_swe,
                            nxs, nys, nzs, ng,
                            level, m_in, zmin, dz,
                            matching_indices, gamma, R, alpha0);
        }
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(F_h, nxs[level], nys[level], nzs[level], vec_dims[level], ng, comm, status, rank, n_processes,
                y_size, do_z, periodic);
    }

    for (int n = 0; n < nxs[level]*nys[level]*nzs[level]*vec_dims[level]; n++) {
        Up_h[n] = Un_h[n] + dt * F_h[n];
    }
    // enforce boundaries and copy back
    if (n_processes == 1) {
        bcs_fv(Up_h, nxs[level], nys[level], nzs[level], ng, vec_dims[level], periodic, do_z);
        if (do_z) {
            enforce_hse(Up_h, U_swe,
                            nxs, nys, nzs, ng,
                            level, m_in, zmin, dz,
                            matching_indices, gamma, R, alpha0);
        }

    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(Up_h, nxs[level], nys[level], nzs[level], vec_dims[level], ng, comm, status, rank,
                n_processes, y_size, do_z, periodic);
    }

    if (do_z) {
        // HACK:
        // going to do some hacky data sanitisation here
        // NOTE: could argue that this is actually a form of artificial
        // dissipation to ensure stability (as it is just smoothing out
        // spikes in the data after all)
        for (int x = 0; x < nxs[level] * nys[level] * nzs[level]; x++) {
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

    cudaMemcpy(Un_d, Up_h, nxs[level]*nys[level]*nzs[level]*vec_dims[level]*sizeof(float),
               cudaMemcpyHostToDevice);
    //cout << "\nu2\n\n\n";
    // u2 = 0.25 * (3*un + u1 + dt*F(u1))
    homogeneuous_fv(kernels, threads, blocks, cumulative_kernels,
          Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
          nxs[level], nys[level], nzs[level], vec_dims[level], ng, alpha0, gamma,
          dx, dy, dz, dt, rank, zmin, R, flux_func, do_z);

    // copy back flux
    cudaMemcpy(F_h, F_d, nxs[level]*nys[level]*nzs[level]*vec_dims[level]*sizeof(float),
               cudaMemcpyDeviceToHost);

    if (n_processes == 1) {
        bcs_fv(F_h, nxs[level], nys[level], nzs[level], ng, vec_dims[level], periodic, do_z);
        if (do_z) {
            enforce_hse(F_h, U_swe,
                            nxs, nys, nzs, ng,
                            level, m_in, zmin, dz,
                            matching_indices, gamma, R, alpha0);
        }
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(F_h, nxs[level], nys[level], nzs[level], vec_dims[level], ng, comm, status, rank, n_processes,
                y_size, do_z, periodic);
    }

    for (int n = 0; n < nxs[level]*nys[level]*nzs[level]*vec_dims[level]; n++) {
        Up_h[n] = 0.25 * (3.0 * Un_h[n] + Up_h[n] + dt * F_h[n]);
    }

    // enforce boundaries and copy back
    if (n_processes == 1) {
        bcs_fv(Up_h, nxs[level], nys[level], nzs[level], ng, vec_dims[level], periodic, do_z);
        if (do_z) {
            enforce_hse(Up_h, U_swe,
                            nxs, nys, nzs, ng,
                            level, m_in, zmin, dz,
                            matching_indices, gamma, R, alpha0);
        }
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(Up_h, nxs[level], nys[level], nzs[level], vec_dims[level], ng, comm, status, rank,
                n_processes, y_size, do_z, periodic);
    }

    if (do_z) {
        // HACK:
        // going to do some hacky data sanitisation here
        for (int x = 0; x < nxs[level] * nys[level] * nzs[level]; x++) {
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

    cudaMemcpy(Un_d, Up_h, nxs[level]*nys[level]*nzs[level]*vec_dims[level]*sizeof(float),
               cudaMemcpyHostToDevice);
    //cout << "\nun+1\n\n\n";
    // un+1 = (1/3) * (un + 2*u2 + 2*dt*F(u2))
    homogeneuous_fv(kernels, threads, blocks, cumulative_kernels,
          Un_d, F_d,
          qx_p_d, qx_m_d, qy_p_d, qy_m_d, qz_p_d, qz_m_d,
          fx_p_d, fx_m_d, fy_p_d, fy_m_d, fz_p_d, fz_m_d,
          nxs[level], nys[level], nzs[level], vec_dims[level], ng, alpha0, gamma,
          dx, dy, dz, dt, rank, zmin, R, flux_func, do_z);

    // copy back flux
    cudaMemcpy(F_h, F_d, nxs[level]*nys[level]*nzs[level]*vec_dims[level]*sizeof(float),
               cudaMemcpyDeviceToHost);

    if (n_processes == 1) {
        bcs_fv(F_h, nxs[level], nys[level], nzs[level], ng, vec_dims[level], periodic, do_z);
        if (do_z) {
            enforce_hse(F_h, U_swe,
                            nxs, nys, nzs, ng,
                            level, m_in, zmin, dz,
                            matching_indices, gamma, R, alpha0);
        }
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(F_h, nxs[level], nys[level], nzs[level], vec_dims[level], ng, comm, status, rank, n_processes,
                y_size, do_z, periodic);
    }

    for (int n = 0; n < nxs[level]*nys[level]*nzs[level]*vec_dims[level]; n++) {
        Up_h[n] = (1/3.0) * (Un_h[n] + 2.0*Up_h[n] + 2.0*dt * F_h[n]);
    }

    // enforce boundaries
    if (n_processes == 1) {
        bcs_fv(Up_h, nxs[level], nys[level], nzs[level], ng, vec_dims[level], periodic, do_z);
        if (do_z) {
            enforce_hse(Up_h, U_swe,
                            nxs, nys, nzs, ng,
                            level, m_in, zmin, dz,
                            matching_indices, gamma, R, alpha0);
        }
    } else {
        int y_size = kernels[0].y * blocks[0].y * threads[0].y - 2*ng;
        bcs_mpi(Up_h, nxs[level], nys[level], nzs[level], vec_dims[level], ng, comm, status, rank,
                n_processes, y_size, do_z, periodic);
    }

    if (do_z) {
        // HACK: going to do some hacky data sanitisation here
        for (int x = 0; x < nxs[level] * nys[level] * nzs[level]; x++) {
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

    for (int j = 0; j < nxs[level]*nys[level]*nzs[level]*vec_dims[level]; j++) {
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
