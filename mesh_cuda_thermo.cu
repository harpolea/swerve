/**
File containing thermodynamic routines and auxiliary functions.
**/

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

__host__ __device__ bool nan_check(float a) {
    // check to see whether float a is a nan
    if (a != a || (abs(a) > 1.0e13)) {
        return true;
    } else {
        return false;
    }
}

__host__ __device__ float zbrent(fptr func, const float x1, float b,
             const float tol,
             float D, float Sx, float Sy, float Sz, float tau, float gamma,
             float * gamma_up) {
    /**
    Using Brent's method, return the root of a function or functor func known
    to lie between x1 and x2. The root will be regined until its accuracy is
    tol.

    Parameters
    ----------
    func : fptr
        function pointer to shallow water or compressible flux function.
    x1, b : const float
        limits of root
    tol : const float
        tolerance to which root shall be calculated to
    D, Sx, Sy, tau: float
        conserved variables
    gamma : float
        adiabatic index
    */

    const int ITMAX = 100;

    float a=x1;
    float c, d=0.0;
    float fa = func(a, D, Sx, Sy, Sz, tau, gamma, gamma_up);
    float fb = func(b, D, Sx, Sy, Sz, tau, gamma, gamma_up);
    float fc=0.0, fs, s;

    if (fa * fb >= 0.0) {
        //cout << "Root must be bracketed in zbrent.\n";
        //printf("Root must be bracketed in zbrent.\n");
        return b;
    }

    if (abs(fa) < abs(fb)) {
        // swap a, b
        d = a;
        a = b;
        b = d;

        d = fa;
        fa = fb;
        fb = d;
    }

    c = a;
    fc = fa;

    bool mflag = true;

    for (int i = 0; i < ITMAX; i++) {
        if (fa != fc && fb != fc) {
            s = a*fb*fc / ((fa-fb) * (fa-fc)) + b*fa*fc / ((fb-fa)*(fb-fc)) +
                c*fa*fb / ((fc-fa)*(fc-fb));
        } else {
            s = b - fb * (b-a) / (fb-fa);
        }

        // list of conditions
        bool con1 = false;
        if (0.25*(3.0 * a + b) < b) {
            if (s < 0.25*(3.0 * a + b) || s > b)
                con1 = true;
        } else if (s < b || s > 0.25*(3.0 * a + b)) {
            con1 = true;
        }
        bool con2 = (mflag && abs(s-b) >= 0.5*abs(b-c));
        bool con3 = (!(mflag) && abs(s-b) >= 0.5 * abs(c-d));
        bool con4 =  (mflag && abs(b-c) < tol);
        bool con5 = (!(mflag) && abs(c-d) < tol);

        if (con1 || con2 || con3 || con4 || con5) {
            s = 0.5 * (a+b);
            mflag = true;
        } else {
            mflag = false;
        }

        fs = func(s, D, Sx, Sy, Sz, tau, gamma, gamma_up);

        if (abs(fa) < abs(fb)) {
            d = a;
            a = b;
            b = d;

            d = fa;
            fa = fb;
            fb = d;
        }

        d = c;
        c = b;
        fc = fb;

        if (fa * fs < 0.0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        // test for convegence
        if (fb == 0.0 || fs == 0.0 || abs(b-a) < tol)
            return b;
    }
    //printf("Maximum number of iterations exceeded in zbrent.\n");
    return x1;
}

void check_mpi_error(int mpi_err) {
    /**
    Checks to see if the integer returned by an mpi function, mpi_err, is an MPI error. If so, it prints out some useful stuff to screen.
    */

    int errclass, resultlen;
    char err_buffer[MPI_MAX_ERROR_STRING];

    if (mpi_err != MPI_SUCCESS) {
        MPI_Error_class(mpi_err, &errclass);
        if (errclass == MPI_ERR_RANK) {
            fprintf(stderr,"%s","Invalid rank used in MPI send call\n");
            MPI_Error_string(mpi_err,err_buffer,&resultlen);
            fprintf(stderr,"%s",err_buffer);
            MPI_Finalize();
        } else {
            fprintf(stderr, "%s","Other MPI error\n");
            MPI_Error_string(mpi_err,err_buffer,&resultlen);
            fprintf(stderr,"%s",err_buffer);
            MPI_Finalize();
        }
    }
}

__host__ __device__ float W_swe(float * q, float * gamma_up) {
    /**
    calculate Lorentz factor for conserved swe state vector
    */
    return sqrt((q[1]*q[1] * gamma_up[0] +
            2.0 * q[1] * q[2] * gamma_up[1] +
            q[2] * q[2] * gamma_up[4]) / (q[0]*q[0]) + 1.0);
}

__host__ __device__ float phi(float r) {
    /**
    calculate superbee slope limiter Phi(r)
    */
    float ph = 0.0;
    if (r >= 1.0) {
        ph = min(float(2.0), min(r, float(2.0 / (1.0 + r))));
    } else if (r >= 0.5) {
        ph = 1.0;
    } else if (r > 0.0) {
        ph = 2.0 * r;
    }
    return ph;
}

__host__ __device__ float find_height(float ph, float R) {
    /**
    Finds r given Phi.
    */
    const float M = 1.0; // set this for convenience
    return 2.0 * M / (1.0 - exp(-2.0 * ph)) - R;
}

__device__ float find_pot(float r, float R) {
    /**
    Finds Phi given r.
    */
    const float M = 1.0; // set this for convenience
    return -0.5 * log(1.0 - 2.0 * M / (r + R));
}

__device__ float rhoh_from_p(float p, float rho, float gamma) {
    /**
    calculate rhoh using p for gamma law equation of state
    */
    return rho + gamma * p / (gamma - 1.0);
}

__device__ float p_from_rhoh(float rhoh, float rho, float gamma) {
    /**
    calculate p using rhoh for gamma law equation of state
    */
    return (rhoh - rho) * (gamma - 1.0) / gamma;
}

__device__ __host__ float p_from_rho_eps(float rho, float eps, float gamma) {
    /**
    calculate p using rho and epsilon for gamma law equation of state
    */
    return (gamma - 1.0) * rho * eps;
}

__device__ __host__ float phi_from_p(float p, float rho, float gamma, float A) {
    /**
    Calculate the metric potential Phi given p for gamma law equation of
    state

    Parameters
    ----------
    p, rho : float
        pressure and density
    gamma : float
        adiabatic index
    A : float
        constant used in Phi to p conversion
    */
    return (gamma - 1.0) / gamma *
        log((rho + gamma * p / (gamma - 1.0)) / A);
}

__host__ __device__ float f_of_p(float p, float D, float Sx, float Sy,
                        float Sz, float tau, float gamma, float * gamma_up) {
    /**
    Function of p whose root is to be found when doing conserved to
    primitive variable conversion

    Parameters
    ----------
    p : float
        pressure
    D, Sx, Sy, Sz, tau :float
        components of conserved state vector
    gamma : float
        adiabatic index
    */

    float sq = sqrt((tau + p + D) * (tau + p + D) -
        Sx*Sx*gamma_up[0] - 2.0*Sx*Sy*gamma_up[1] - 2.0*Sx*Sz*gamma_up[2] -
        Sy*Sy*gamma_up[4] - 2.0*Sy*Sz*gamma_up[5] - Sz*Sz*gamma_up[8]);

    //if (nan_check(sq)) cout << "sq is nan :(\n";

    //float rho = D * sq / (tau + p + D);
    //float eps = (sq - p * (tau + p + D) / sq - D) / D;

    //return (gamma - 1.0) * rho * eps - p;
    return (gamma - 1.0) * sq / (tau + p + D) * (sq - p * (tau + p + D) / sq - D) - p;
}


/*__host__ __device__ float alpha(float z, float alpha0, float R){
    float M = 1;
    return alpha0 + M * z / (alpha0 * R*R);
}*/

__device__ float h_dot(float phi, float old_phi, float dt, float R) {
    /**
    Calculates the time derivative of the height given the shallow water
    variable phi at current time and previous timestep
    NOTE: this is an upwinded approximation of hdot - there may be a better
    way to do this which will more accurately give hdot at current time.

    Parameters
    ----------
    phi : float
        Phi at current timestep
    old_phi : float
        Phi at previous timestep
    dt : float
        timestep
    */

    return -2.0 * find_height(phi, R) * (phi - old_phi) / (dt * (exp(2.0 * phi) - 1.0));
}

__device__ float calc_Q_swe(float rho, float p, float gamma, float Y, float Cv) {
    /**
    Calculate the heating rate per unit mass from the shallow water variables
    */

    float T = p / ((gamma - 1.0) * rho * Cv);
    float A = 1.0e8; // constant of proportionality

    float X_dot = A*rho*rho*Y*Y*Y / (T*T*T) * exp(-44.0 / T);

    if (nan_check(X_dot))
        X_dot = 0.0;

    return X_dot;

}

void calc_Q(float * rho, float * q_cons, int nx, int ny, int nz,
            float gamma, float * Q, float Cv, float * gamma_up) {
    /**
    Calculate the heating rate per unit mass
    */

    // hack: need to actually interpolate rho here rather than just assume it's constant

    float * q_prim = new float[nx*ny*nz*6];
    cons_to_prim_comp(q_cons, q_prim, nx, ny, nz, gamma, gamma_up);

    for (int i = 0; i < nx*ny*nz; i++) {
        float eps = q_prim[i*6+4];
        float Y = q_prim[i*6+5];
        float T = eps / Cv;
        float A = 1.0e8; // constant of proportionality

        Q[i] = A*rho[0]*rho[0]*Y*Y*Y / (T*T*T) * exp(-44.0 / T);

        //cout << "eps = " << eps << "  Y = " << Y << "  H = " << Q[i] << '\n';
    }

    delete[] q_prim;
}

__device__ void calc_As(float * rhos, float * phis, float * A,
                        int nlayers, float gamma, float surface_phi, float surface_rho) {
    /**
    Calculates the As used to calculate the pressure given Phi, given
    the surface pressure = 0

    Parameters
    ----------
    rhos : float array
        densities of layers
    phis : float array
        Vector of Phi for different layers
    A : float array
        vector of As for layers
    nlayers : int
        number of layers
    gamma : float
        adiabatic index
    surface_phi : float
        Phi at surface
    surface_rho : float
        density at surface
    */

    // define A at sea surface using condition that p = 0
    float A_surface = surface_rho * exp(-gamma * surface_phi / (gamma-1.0));
    A[0] = A_surface + exp(-gamma * phis[0] / (gamma-1.0)) * (rhos[0] - surface_rho);

    for (int n = 0; n < (nlayers-1); n++) {
        A[n+1] = A[n] +
            exp(-gamma * phis[n+1] / (gamma - 1.0)) * (rhos[n+1] - rhos[n]);
    }
}

/*__device__ void restrict_interpolate_swe(float * p_const, float gamma,
    float * q_comp, float * q_swe, float zmin, float dz,
    int * nxs, int * nys, int * nzs, int clevel,
    int kx_offset, int ky_offset, int * matching_indices) {

    Find phis of surfaces of constant pressure

    Parameters
    ----------
    p_const
        pressure on SWE surfaces
    gamma
        adiabatic index
    q_comp
        primitive compressible state vector on grid
    q_swe
        conserved SWE state vector on grid
    zmin
        height of sea floor
    dz
        compressible grid separation
    nxs, nys, nzs
        grid dimensions
    clevel
        level id of coarse SWE grid
    kx_offset, ky_offset
        MPI process offset
    matching_indices
        location of compressible grid wrt SWE grid


    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    if (((x > 0) && (x < int(round(nxs[clevel+1]*0.5))-1)) &&
        ((y > 0) && (y < int(round(nys[clevel+1]*0.5))-1)) &&
        (z > 0) && (z < nzs[clevel]-1)) {
        int c_x = x + matching_indices[clevel*4];
        int c_y = y + matching_indices[clevel*4+2];

        int neighbour_layer = 0;
        float layer_frac = 0.0;

        for (int i = 0; i < nzs[clevel+1]-1; i++) {
            float p = 0;
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    p += 0.25 * (gamma - 1.0) *
                        q_comp[((i * nys[clevel+1] + 2*y+j) * nxs[clevel+1] + 2*x+k)*6] *
                        q_comp[((i * nys[clevel+1] + 2*y+j) * nxs[clevel+1] + 2*x+k)*6+4];
                }
            }

            if (p_const[z] < p) {
                neighbour_layer = i+1;

                float p_below = 0;
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < 2; k++) {
                        p_below += 0.25 * (gamma - 1.0) *
                            q_comp[(((i+1) * nys[clevel+1] + 2*y+j) * nxs[clevel+1] + 2*x+k)*6] *
                            q_comp[(((i+1) * nys[clevel+1] + 2*y+j) * nxs[clevel+1] + 2*x+k)*6+4];
                    }
                }

                layer_frac = (p_const[z] - p) / (p_below - p);
                break;
            }
        }

        float layer_height = zmin + dz * (neighbour_layer + layer_frac);

        int l = neighbour_layer;
        float Ww[8];
        float us[8];
        float vs[8];
        float WX[8];
        for (int j = 0; j < 2; j++) {
            for (int i = 0; i < 2; i++) {
                float u, v, w;
                u=q_comp[((l*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+1];
                v=q_comp[((l*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+2];
                w=q_comp[((l*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+3];
                Ww[j*2+i] = 1.0 / sqrt(1.0 -
                        u*u*gamma_up[0] - 2.0 * u*v * gamma_up[1] -
                        2.0 * u*w * gamma_up[2] - v*v*gamma_up[4] -
                        2.0 * v*w*gamma_up[5] - w*w*gamma_up[8]);
                us[j*2+i] = u;
                vs[j*2+i] = v;
                WX[j*2+i] = Ww[j*2+i] *
                q_comp[((l*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+5];

                u=q_comp[(((l-1)*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+1];
                v=q_comp[(((l-1)*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+2];
                w=q_comp[(((l-1)*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+3];
                Ww[(2+j)*2+i] = 1.0 / sqrt(1.0 -
                        u*u*gamma_up[0] - 2.0 * u*v * gamma_up[1] -
                        2.0 * u*w * gamma_up[2] - v*v*gamma_up[4] -
                        2.0 * v*w*gamma_up[5] - w*w*gamma_up[8]);

                us[(2+j)*2+i] = u;
                vs[(2+j)*2+i] = v;
                WX[(2+j)*2+i] = Ww[(2+j)*2+i] *
                q_comp[(((l-1)*nys[clevel+1]+y*2+j)*nxs[clevel+1]+x*2+i)*6+5];

            }
        }

        float interp_W = layer_frac * 0.25 * (Ww[0] + Ww[1] + Ww[2] + Ww[3]) +
            (1.0 - layer_frac) * 0.25 * (Ww[4] + Ww[5] + Ww[6] + Ww[7]);
        float interp_u = layer_frac * 0.25 * (us[0] + us[1] + us[2] + us[3]) +
            (1.0 - layer_frac) * 0.25 * (us[4] + us[5] + us[6] + us[7]);
        float interp_v = layer_frac * 0.25 * (vs[0] + vs[1] + vs[2] + vs[3]) +
            (1.0 - layer_frac) * 0.25 * (vs[4] + vs[5] + vs[6] + vs[7]);
        float interp_WX = layer_frac * 0.25 * (WX[0] + WX[1] + WX[2] + WX[3]) +
            (1.0 - layer_frac) * 0.25 * (WX[4] + WX[5] + WX[6] + WX[7]);

        q_swe[((z * nys[clevel] + c_y) * nxs[clevel] + c_x) * 4] =
            find_pot(layer_height) * interp_W;
        q_swe[((z * nys[clevel] + c_y) * nxs[clevel] + c_x) * 4 + 1] =
            q_swe[((z * nys[clevel] + c_y) * nxs[clevel] + c_x) * 4] *
            interp_W * interp_u;
        q_swe[((z * nys[clevel] + c_y) * nxs[clevel] + c_x) * 4 + 2] =
            q_swe[((z * nys[clevel] + c_y) * nxs[clevel] + c_x) * 4] *
            interp_W * interp_v;
        q_swe[((z * nys[clevel] + c_y) * nxs[clevel] + c_x) * 4 + 3] =
            find_pot(layer_height) * interp_WX ;
    }
}*/

__device__ void enforce_hse_d(float * q_comp, float * q_swe,
                            int kx_offset, int ky_offset,
                            int * nxs, int * nys, int * nzs, int ng,
                            int level, int clevel, float zmin, float dz,
                            int * matching_indices_d, float gamma, float R, float alpha0) {
    /**
    Enforce hse on compressible grid by calculating height of fluid surface from swe grid.
    **/
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;

    // interpolate swe to get surface height
    if (x < int(round(nxs[level]/pow(2.0, level-clevel))) &&
        y < int(round(nys[level]/pow(2.0, level-clevel))) &&
        (z < ng || (z > (nzs[level] - 1 - ng) && z < nzs[level]))) {
        int c_x = x + matching_indices_d[clevel*4];
        int c_y = y + matching_indices_d[clevel*4+2];

        // do some slope limiting
        // x-dir
        float S_upwind =
            find_height(q_swe[(c_y * nxs[clevel] + c_x+1) * 4], R) -
            find_height(q_swe[(c_y * nxs[clevel] + c_x) * 4], R);
        float S_downwind =
            find_height(q_swe[(c_y * nxs[clevel] + c_x) * 4], R) -
            find_height(q_swe[(c_y * nxs[clevel] + c_x-1) * 4], R);

        float Sx = 0.5 * (S_upwind + S_downwind);

        float r = 1.0e6;
        if (abs(S_downwind) > 1.0e-10) {
            r = S_upwind / S_downwind;
        }

        Sx *= phi(r);

        // y-dir
        S_upwind =
            find_height(q_swe[((c_y+1) * nxs[clevel] + c_x) * 4], R) -
            find_height(q_swe[(c_y * nxs[clevel] + c_x) * 4], R);
        S_downwind =
            find_height(q_swe[(c_y * nxs[clevel] + c_x) * 4], R) -
            find_height(q_swe[((c_y-1) * nxs[clevel] + c_x) * 4], R);

        float Sy = 0.5 * (S_upwind + S_downwind);

        r = 1.0e6;
        if (abs(S_downwind) > 1.0e-10) {
            r = S_upwind / S_downwind;
        }

        Sy *= phi(r);

        int rr = pow(2.0, level-clevel); // ratio between fine compressible level and coarse swe level

        float zz_a = zmin + dz * (nzs[level] - z - 1);
        float zz_b = zmin + dz * (nzs[level] - z - 1);
        float zg_a = zmin + dz * (nzs[level] - ng - 1);
        float zg_b = zmin + dz * ng;
        float M = 1;
        float gamma_za = (1.0 - M * zz_a / (R*R*alpha0*alpha0)) / alpha0;
        float gamma_zb = (1.0 - M * zz_b / (R*R*alpha0*alpha0)) / alpha0;
        float gamma_zga = (1.0 - M * zg_a / (R*R*alpha0*alpha0)) / alpha0;
        float gamma_zgb = (1.0 - M * zg_b / (R*R*alpha0*alpha0)) / alpha0;

        for (int i = 0; i < rr; i++) {
            for (int j = 0; j < rr; j++) {
                float z_surface =
                    find_height(q_swe[(c_y * nxs[clevel] + c_x) * 4], R) +
                    ((2*i - rr+1) * Sx + (2*j - rr+1) * Sy) / (2.0*rr);
                float gamma_surf = (1.0 - M * z_surface / (R*R*alpha0*alpha0)) / alpha0;

                for (int n = 0; n < 6; n++) {
                    if (z < ng) {
                        q_comp[((z * nys[level] + rr*y+j) * nxs[level] + rr*x+i) * 6 + n] =
                        q_comp[((ng * nys[level] + rr*y+j) * nxs[level] + rr*x+i) * 6 + n] *
                        pow((gamma_za - gamma_surf) / (gamma_zga - gamma_surf), float(1.0)/gamma);
                    } else {
                        q_comp[((z * nys[level] + rr*y+j) * nxs[level] + rr*x+i) * 6 + n] =
                        q_comp[(((nzs[level] - 1 - ng) * nys[level] + rr*y+j) * nxs[level] + rr*x+i) * 6 + n] *
                        pow((gamma_zb - gamma_surf) / (gamma_zgb - gamma_surf), float(1.0)/gamma);
                    }

                }
            }
        }
    }
}

void enforce_hse(float * q_comp, float * q_swe,
                int * nxs, int * nys, int * nzs, int ng,
                int level, int clevel, float zmin, float dz,
                int * matching_indices, float gamma, float R, float alpha0) {
    /**
    Enforce hse on compressible grid by calculating height of fluid surface from swe grid.
    **/

    // interpolate swe to get surface height
    for (int x = 0; x < int(round(nxs[level]/pow(2, level-clevel))); x++){
        for(int y = 0; y < int(round(nys[level]/pow(2, level-clevel))); y++) {
            for (int z = 0; z < ng; z++) {
                int c_x = x;
                int c_y = y;
                for (int i = clevel; i < level; i++) {
                    c_x += matching_indices[i*4] / pow(2, i-clevel);
                    c_y += matching_indices[i*4+2] / pow(2, i-clevel);
                }

                // do some slope limiting
                // x-dir
                float S_upwind =
                    q_swe[(c_y * nxs[clevel] + c_x+1) * 4] -
                    q_swe[(c_y * nxs[clevel] + c_x) * 4];
                float S_downwind =
                    q_swe[(c_y * nxs[clevel] + c_x) * 4] -
                    q_swe[(c_y * nxs[clevel] + c_x-1) * 4];

                float Sx = 0.5 * (S_upwind + S_downwind);

                float r = 1.0e6;
                if (abs(S_downwind) > 1.0e-10) {
                    r = S_upwind / S_downwind;
                }

                Sx *= phi(r);

                // y-dir
                S_upwind =
                    q_swe[((c_y+1) * nxs[clevel] + c_x) * 4] -
                    q_swe[(c_y * nxs[clevel] + c_x) * 4];
                S_downwind =
                    q_swe[(c_y * nxs[clevel] + c_x) * 4] -
                    q_swe[((c_y-1) * nxs[clevel] + c_x) * 4];

                float Sy = 0.5 * (S_upwind + S_downwind);

                r = 1.0e6;
                if (abs(S_downwind) > 1.0e-10) {
                    r = S_upwind / S_downwind;
                }

                Sy *= phi(r);

                int rr = pow(2, level-clevel); // ratio between fine compressible level and coarse swe level

                float zz_a = zmin + dz * (nzs[level] - z - 1);
                float zz_b = zmin + dz * z;
                float zg_a = zmin + dz * (nzs[level] - ng - 1);
                float zg_b = zmin + dz * ng;

                float M = 1;
                float gamma_za = (1.0 - M * zz_a / (R*R*alpha0*alpha0)) / alpha0;
                float gamma_zb = (1.0 - M * zz_b / (R*R*alpha0*alpha0)) / alpha0;
                float gamma_zga = (1.0 - M * zg_a / (R*R*alpha0*alpha0)) / alpha0;
                float gamma_zgb = (1.0 - M * zg_b / (R*R*alpha0*alpha0)) / alpha0;

                for (int i = 0; i < rr; i++) {
                    for (int j = 0; j < rr; j++) {
                        float z_surface =
                            find_height(q_swe[(c_y * nxs[clevel] + c_x) * 4] +
                            ((2*i - rr+1) * Sx + (2*j - rr+1) * Sy) / (2.0*rr), R);
                        float gamma_surf = (1.0 - M * z_surface / (R*R*alpha0*alpha0)) / alpha0;

                        for (int n = 0; n < 6; n++) {
                            q_comp[((z * nys[level] + rr*y+j) * nxs[level] + rr*x+i) * 6 + n] =
                            q_comp[((ng * nys[level] + rr*y+j) * nxs[level] + rr*x+i) * 6 + n] *
                            pow((gamma_za - gamma_surf) / (gamma_zga - gamma_surf), 1.0/gamma);

                            q_comp[(((nzs[level] - 1 - z) * nys[level] + rr*y+j) * nxs[level] + rr*x+i) * 6 + n] =
                            q_comp[(((nzs[level] - 1 - ng) * nys[level] + rr*y+j) * nxs[level] + rr*x+i) * 6 + n] *
                            pow((gamma_zb - gamma_surf) / (gamma_zgb - gamma_surf), 1.0/gamma);
                        }
                    }
                }
            }
        }
    }
}

__device__ void cons_to_prim_comp_d(float * q_cons, float * q_prim,
                       float gamma, float * gamma_up) {
    /**
    Convert compressible conserved variables to primitive variables

    Parameters
    ----------
    q_cons : float *
        state vector of conserved variables
    q_prim : float *
        state vector of primitive variables
    gamma : float
        adiabatic index
    */

    const float TOL = 1.0e-5;
    float D = q_cons[0];
    float Sx = q_cons[1];
    float Sy = q_cons[2];
    float Sz = q_cons[3];
    float tau = q_cons[4];

    if (D == 0.0) {
        q_prim[0] = 0.0;
        q_prim[1] = 0.0;
        q_prim[2] = 0.0;
        q_prim[3] = 0.0;
        q_prim[4] = 0.0;
        q_prim[5] = 0.0;
    } else {
        // S^2
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
        //float eps = (sq - p * (tau + p + D)/sq - D) / D;
        float h = 1.0 + gamma * (sq - p * (tau + p + D)/sq - D) / D;
        float W2 = 1.0 + Ssq / (D*D*h*h);

        q_prim[0] = D * sq / (tau + p + D);//D / W;
        q_prim[1] = Sx / (W2 * h * q_prim[0]);
        q_prim[2] = Sy / (W2 * h * q_prim[0]);
        q_prim[3] = Sz / (W2 * h * q_prim[0]);
        q_prim[4] = (sq - p * (tau + p + D)/sq - D) / D;
        q_prim[5] = q_cons[5] / D;
    }
}

void cons_to_prim_comp(float * q_cons, float * q_prim, int nxf, int nyf,
                       int nz, float gamma, float * gamma_up) {
    /**
    Convert compressible conserved variables to primitive variables

    Parameters
    ----------
    q_cons : float *
        grid of conserved variables
    q_prim : float *
        grid where shall put the primitive variables
    nxf, nyf, nz : int
        grid dimensions
    gamma : float
        adiabatic index
    */

    const float TOL = 1.e-5;

    for (int i = 0; i < nxf*nyf*nz; i++) {
        float D = q_cons[i*6];
        float Sx = q_cons[i*6+1];
        float Sy = q_cons[i*6+2];
        float Sz = q_cons[i*6+3];
        float tau = q_cons[i*6+4];
        float DX = q_cons[i*6+5];

        // S^2
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

        float p;
        try {
            p = zbrent((fptr)f_of_p, pmin, pmax, TOL, D, Sx, Sy, Sz,
                        tau, gamma, gamma_up);
        } catch (char const*){
            p = abs((gamma - 1.0) * (tau + D) / (2.0 - gamma)) > 1.0 ? 1.0 :
                abs((gamma - 1.0) * (tau + D) / (2.0 - gamma));
        }

        float sq = sqrt((tau + p + D)*(tau + p + D) - Ssq);
        float eps = (sq - p * (tau + p + D)/sq - D) / D;
        float h = 1.0 + gamma * eps;
        float W = sqrt(1.0 + Ssq / (D*D*h*h));
        float X = DX / D;

        q_prim[i*6] = D * sq / (tau + p + D);//D / W;
        q_prim[i*6+1] = Sx / (W*W * h * q_prim[i*6]);
        q_prim[i*6+2] = Sy / (W*W * h * q_prim[i*6]);
        q_prim[i*6+3] = Sz / (W*W * h * q_prim[i*6]);
        q_prim[i*6+4] = eps;
        q_prim[i*6+5] = X;
    }
}

__device__ void shallow_water_fluxes(float * q, float * f, int dir,
                          float alpha0, float gamma, float zmin, float dz,
                          int nz, int layer, float R) {
    /**
    Calculate the flux vector of the shallow water equations

    Parameters
    ----------
    q : float *
        state vector
    f : float *
        grid where fluxes shall be stored
    dir : int
        0 if calculating flux in x-direction, 1 if in y-direction
    alpha0 : float at surface
        lapse function
    gamma : float
        adiabatic index
    */
    if (nan_check(q[0])) q[0] = 1.0;
    if (nan_check(q[1])) q[1] = 0.0;
    if (nan_check(q[2])) q[2] = 0.0;
    if (nan_check(q[3])) q[3] = 0.0;

    float alpha = sqrt(exp(-2.0 * q[0]));
    float * gamma_up;
    gamma_up = (float *)malloc(9 * sizeof(float));
    for (int i = 0; i < 9; i++) {
        gamma_up[i] = 0.0;
    }
    gamma_up[0] = 1.0;
    gamma_up[4] = 1.0;
    gamma_up[8] = exp(2.0 * q[0]);

    float W = W_swe(q, gamma_up);
    if (nan_check(W)) {
        printf("W is nan! q0, q1, q2: %f, %f, %f\n", q[0], q[1], q[2]);
        W = 1.0;
    }

    float u = q[1] / (q[0] * W);
    float v = q[2] / (q[0] * W);

    if (dir == 0) {
        float qx = u * gamma_up[0] + v * gamma_up[1] -
            beta_d[0] / alpha;

        f[0] = q[0] * qx;
        f[1] = q[1] * qx + 0.5 * q[0] * q[0] / (W * W);
        f[2] = q[2] * qx;
        f[3] = q[3] * qx;
    } else {
        float qy = v * gamma_up[4] + u * gamma_up[1] -
            beta_d[1] / alpha;

        f[0] = q[0] * qy;
        f[1] = q[1] * qy;
        f[2] = q[2] * qy + 0.5 * q[0] * q[0] / (W * W);
        f[3] = q[3] * qy;
    }

    free(gamma_up);
}

__device__ void compressible_fluxes(float * q, float * f, int dir,
                         float alpha0, float gamma, float zmin, float dz,
                         int nz, int layer, float R) {
    /**
    Calculate the flux vector of the compressible GR hydrodynamics equations

    Parameters
    ----------
    q : float *
        state vector
    f : float *
        grid where fluxes shall be stored
    dir : int
        0 if calculating flux in x-direction, 1 if in y-direction,
        2 if in z-direction
    alpha0 : float
        lapse function at surface
    gamma : float
        adiabatic index
    */

    // this is worked out on the fine grid
    float * q_prim;
    q_prim = (float *)malloc(6 * sizeof(float));

    float z = zmin + dz * (nz - 1 - layer);
    float M = 1;
    float alpha = alpha0 + M * z / (R*R * alpha0);
    float * gamma_up;
    gamma_up = (float *)malloc(9 * sizeof(float));
    for (int i = 0; i < 9; i++) {
        gamma_up[i] = 0.0;
    }
    gamma_up[0] = 1.0;
    gamma_up[4] = 1.0;
    gamma_up[8] = alpha*alpha;

    cons_to_prim_comp_d(q, q_prim, gamma, gamma_up);

    float p = p_from_rho_eps(q_prim[0], q_prim[4], gamma);

    //printf("p: %f, D: %f, rho: %f, u: %f, v: %f, w: %f, tau: %f, eps: %f\n", p, q[0], q_prim[0], u, v, w, q[4], q_prim[4]);

    if (dir == 0) {
        float qx = q_prim[1] * gamma_up[0] + q_prim[2] * gamma_up[1] + q_prim[3] * gamma_up[2] - beta_d[0] / alpha;

        f[0] = q[0] * qx;
        f[1] = q[1] * qx + p;
        f[2] = q[2] * qx;
        f[3] = q[3] * qx;
        f[4] = q[4] * qx + p * q_prim[1];
        f[5] = q[5] * qx;
    } else if (dir == 1){
        float qy = q_prim[2] * gamma_up[4] + q_prim[1] * gamma_up[1] + q_prim[3] * gamma_up[5] - beta_d[1] / alpha;

        f[0] = q[0] * qy;
        f[1] = q[1] * qy;
        f[2] = q[2] * qy + p;
        f[3] = q[3] * qy;
        f[4] = q[4] * qy + p * q_prim[2];
        f[5] = q[5] * qy;
    } else {
        float qz = q_prim[3] * gamma_up[8] + q_prim[1] * gamma_up[2] + q_prim[2] * gamma_up[5] - beta_d[2] / alpha;

        f[0] = q[0] * qz;
        f[1] = q[1] * qz;
        f[2] = q[2] * qz;
        f[3] = q[3] * qz + p;
        f[4] = q[4] * qz + p * q_prim[3];
        f[5] = q[5] * qz;
    }

    //printf("f(tau): %f\n", f[4]);

    // HACK
    //for (int i = 0; i < 6; i++) {
    //    f[i] = 0.0;
    //}

    free(q_prim);
    free(gamma_up);
}

void p_from_swe(float * q, float * p, int nx, int ny, int nz,
                 float rho, float gamma, float A, float * gamma_up) {
    /**
    Calculate p using SWE conserved variables

    Parameters
    ----------
    q : float *
        state vector
    p : float *
        grid where pressure shall be stored
    nx, ny, nz : int
        grid dimensions
    rho : float
        density
    gamma : float
        adiabatic index
    A : float
        variable required in p(Phi) calculation
    */

    for (int i = 0; i < nx*ny*nz; i++) {
        float W = W_swe(q, gamma_up);

        float ph = q[i*4] / W;

        p[i] = (gamma - 1.0) * (A * exp(gamma * ph /
            (gamma - 1.0)) - rho) / gamma;
    }
}

__device__ float p_from_swe(float * q, float rho,
                            float gamma, float W, float A) {
    /**
    Calculates p and returns using SWE conserved variables

    Parameters
    ----------
    q : float *
        state vector
    rho : float
        density
    gamma : float
        adiabatic index
    W : float
        Lorentz factor
    A : float
        variable required in p(Phi) calculation
    */

    float ph = q[0] / W;

    return (gamma - 1.0) * (A * exp(gamma * ph /
        (gamma - 1.0)) - rho) / gamma;
}

__global__ void compressible_from_swe(float * q, float * q_comp,
                           int * nxs, int * nys, int * nzs,
                           float * rho, float gamma,
                           int kx_offset, int ky_offset, float dt,
                           float * old_phi, int level, float R) {
    /**
    Calculates the compressible state vector from the SWE variables.

    Parameters
    ----------
    q : float *
        grid of SWE state vector
    q_comp : float *
        grid where compressible state vector to be stored
    nxs, nys, nzs : int *
        grid dimensions
    rho, gamma : float
        density and adiabatic index
    kx_offset, ky_offset : int
        kernel offsets in the x and y directions
    dt : float
        timestep
    old_phi : float *
        Phi at previous timestep
    level : int
        index of level
    */

    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;
    int offset = (z * nys[level] + y) * nxs[level] + x;

    if ((x < nxs[level]) && (y < nys[level]) && (z < nzs[level])) {
        //printf("(%d, %d, %d): %f, %f, %f\n", x, y, z, q[offset*4], q[offset*4+1], q[offset*4+2]);

        float * q_swe;
        q_swe = (float *)malloc(4 * sizeof(float));

        for (int i = 0; i < 4; i++) {
            q_swe[i] = q[offset * 4 + i];
        }

        float * gamma_up;
        gamma_up = (float *)malloc(9 * sizeof(float));
        for (int i = 0; i < 9; i++) {
            gamma_up[i] = 0.0;
        }
        gamma_up[0] = 1.0;
        gamma_up[4] = 1.0;
        gamma_up[8] = exp(2.0 * q_swe[0]);

        // calculate hdot = w (?)
        float hdot = h_dot(q[offset*4], old_phi[offset], dt, R);
        //printf("hdot(%d, %d, %d): %f, \n", x, y, z, hdot);

        float W = sqrt((q[offset*4+1] * q[offset*4+1] * gamma_up[0] +
                2.0 * q[offset*4+1] * q[offset*4+2] * gamma_up[1] +
                q[offset*4+2] * q[offset*4+2] * gamma_up[4]) /
                (q[offset*4] * q[offset*4]) +
                2.0 * hdot * (q[offset*4+1] * gamma_up[2] +
                q[offset*4+2] * gamma_up[5]) / q[offset*4] +
                hdot * hdot * gamma_up[8] + 1.0);
        //printf("%d\n",  gamma_up[8]);
        //printf("W(%d, %d, %d): %f, \n", x, y, z, W);
        // TODO: this is really inefficient as redoing the same calculation
        // on differnt layers
        float * A, * phis;
        A = (float *)malloc(nzs[level] * sizeof(float));
        phis = (float *)malloc(nzs[level] * sizeof(float));
        for (int i = 0; i < nzs[level]; i++) {
            phis[i] = q[((i * nys[level] + y) * nxs[level] + x) * 4];
        }

        calc_As(rho, phis, A, nzs[level], gamma, phis[0], rho[0]);

        float p = p_from_swe(q_swe, rho[z], gamma, W, A[z]);
        float rhoh = rhoh_from_p(p, rho[z], gamma);

        free(phis);
        free(A);
        free(gamma_up);

        q_comp[offset*6] = rho[z] * W;
        q_comp[offset*6+1] = rhoh * W * q[offset*4+1] / q[offset*4];
        q_comp[offset*6+2] = rhoh * W * q[offset*4+2] / q[offset*4];
        q_comp[offset*6+3] = rho[z] * W * hdot;
        q_comp[offset*6+4] = rhoh*W*W - p - rho[z] * W;
        q_comp[offset*6+5] = rho[z] * W * q[offset*4+3] / q[offset*4];

        //printf("s2c (%d, %d, %d): %f, %f\n", x, y, z, q_comp[offset*6+4], p);

        // NOTE: hack?
        if (q_comp[offset*6+4] < 0.0)
            q_comp[offset*6+4] = 0.0;

        // cannot have X < 0.0
        if (q_comp[offset*6+5] < 0.0)
            q_comp[offset*6+5] = 0.0;

        free(q_swe);
    }
}

__device__ float slope_limit(float layer_frac, float left, float middle, float right, float aleft, float amiddle, float aright) {
    /**
    Calculates slope limited verticle gradient at layer_frac between middle and amiddle.
    Left, middle and right are from row n, aleft, amiddle and aright are from row above it (n-1)
    */
    float S_upwind = (layer_frac * (right - middle) +
        (1.0 - layer_frac) * (aright - amiddle));
    float S_downwind = (layer_frac * (middle - left)
        + (1.0 - layer_frac) * (amiddle - aleft));

    float S = 0.5 * (S_upwind + S_downwind);

    float r = 1.0e6;
    if (abs(S_downwind) > 1.0e-10)
        r = S_upwind / S_downwind;

    return S * phi(r);
}

__global__ void calc_comp_prim(float * q, int * nxs, int * nys, int * nzs,
                               float gamma, int kx_offset, int ky_offset,
                               int coarse_level, float zmin, float dz,
                               float R, float alpha0) {
    /**
    Calculates the SWE state vector from the compressible variables.

    Parameters
    ----------
    q : float *
        grid of compressible state vector
    q_swe : float *
        grid where SWE state vector to be stored
    nxs, nys, nzs : int *
        grid dimensions
    rho, gamma : float
        density and adiabatic index
    kx_offset, ky_offset : int
        kernel offsets in the x and y directions
    qc : float *
        coarse grid
    matching_indices : int *
        indices of fine grid wrt coarse grid
    coarse_level : int
        index of coarser grid
    */
    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;
    int offset = (z * nys[coarse_level+1] + y) * nxs[coarse_level+1] + x;

    float * q_prim, * q_con, * gamma_up;
    q_con = (float *)malloc(6 * sizeof(float));
    q_prim = (float *)malloc(6 * sizeof(float));
    gamma_up = (float *)malloc(9 * sizeof(float));

    if ((x < nxs[coarse_level+1]) &&
        (y < nys[coarse_level+1]) &&
        (z < nzs[coarse_level+1])) {

        for (int i = 0; i < 6; i++) {
            q_con[i] = q[offset*6 + i];
        }

        float h = zmin + dz * (nzs[coarse_level+1] - 1 - z);
        float M = 1;
        float alpha = alpha0 + M * h / (R*R * alpha0);
        for (int i = 0; i < 9; i++) {
            gamma_up[i] = 0.0;
        }
        gamma_up[0] = 1.0;
        gamma_up[4] = 1.0;
        gamma_up[8] = alpha*alpha;

        //printf("alpha = %f\n", alpha);
        // NOTE: alpha ok here

        // find primitive variables
        cons_to_prim_comp_d(q_con, q_prim, gamma, gamma_up);

        for (int i = 0; i < 6; i++) {
            q[offset*6 + i] = q_prim[i];
        }
        //printf("D, tau: %f, %f, rho, eps: %f, %f\n", q_con[0], q_con[4], q_prim[0], q_prim[4]);
    }

    free(q_con);
    free(q_prim);
    free(gamma_up);
}
__global__ void swe_from_compressible(float * q_prim, float * q_swe,
                                      int * nxs, int * nys, int * nzs,
                                      float * rhos, float gamma,
                                      int kx_offset, int ky_offset,
                                      float * qc,
                                      int * matching_indices,
                                      int coarse_level, float zmin, float dz,
                                      float alpha0, float R) {

    int x = kx_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int y = ky_offset + blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;
    int offset = (z * nys[coarse_level+1] + y) * nxs[coarse_level+1] + x;

    float rho, u, v, w, eps, X, W, ph;

    if ((x < nxs[coarse_level+1]) &&
        (y < nys[coarse_level+1]) &&
        (z < nzs[coarse_level+1])) {

        rho = q_prim[offset*6];
        u = q_prim[offset*6+1];
        v = q_prim[offset*6+2];
        w = q_prim[offset*6+3];
        eps = q_prim[offset*6+4];
        X = q_prim[offset*6+5];

        float h = zmin + dz * (nzs[coarse_level+1] - 1 - z);
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

        W = 1.0 / sqrt(1.0 -
                u*u*gamma_up[0] - 2.0 * u*v * gamma_up[1] -
                2.0 * u*w * gamma_up[2] - v*v*gamma_up[4] -
                2.0 * v*w*gamma_up[5] - w*w*gamma_up[8]);

        float * A, * phis, * rho_column;
        A = (float *)malloc(nzs[coarse_level+1] * sizeof(float));
        phis = (float *)malloc(nzs[coarse_level+1] * sizeof(float));
        rho_column = (float *)malloc(nzs[coarse_level+1] * sizeof(float));

        for (int i = 0; i < nzs[coarse_level+1]; i++) {
            phis[i] = find_pot(zmin + (nzs[coarse_level+1]-i-1)*dz, R);
            rho_column[i] = q_prim[((i * nys[coarse_level+1] + y) *
                                   nxs[coarse_level+1] + x)*6];
        }

        // do some interpolation of surface SW layer to find surface phi at
        // same (x,y) as current compressible gridpoint
        int c_x = round(x*0.5) + matching_indices[coarse_level*4];
        int c_y = round(y*0.5) + matching_indices[coarse_level*4+2];
        float interp_q_comp = qc[(c_y * nxs[coarse_level] + c_x) * 4];

        float Sx = slope_limit(1.0,
            qc[(c_y * nxs[coarse_level] + c_x-1) * 4],
            qc[(c_y * nxs[coarse_level] + c_x) * 4],
            qc[(c_y * nxs[coarse_level] + c_x+1) * 4], 0.0, 0.0, 0.0);
        float Sy = slope_limit(1.0,
            qc[((c_y-1) * nxs[coarse_level] + c_x) * 4],
            qc[(c_y * nxs[coarse_level] + c_x) * 4],
            qc[((c_y+1) * nxs[coarse_level] + c_x) * 4], 0.0, 0.0, 0.0);

        float phi_surface = interp_q_comp;
        if (x % 2 == 1) {
            phi_surface += 0.25 * Sx;
        } else {
            phi_surface -= 0.25 * Sx;
        }

        if (y % 2 == 1) {
            phi_surface += 0.25 * Sy;
        } else {
            phi_surface -= 0.25 * Sy;
        }

        calc_As(rho_column, phis, A, nzs[coarse_level+1], gamma, phi_surface, rhos[0]);

        // NOTE: hack to get this to not nan
        if (nan_check(A[z]) || A[z] < 0.0) A[z] = 1.0;

        // calculate SWE conserved variables on fine grid.
        float p = p_from_rho_eps(rho, eps, gamma);

        ph = phi_from_p(p, rho, gamma, A[z]);

        free(phis);
        free(A);
        free(rho_column);
        free(gamma_up);

        //printf("W: %f, ph: %f, tau: %f, eps: %f, A[z]: %f, p: %f, rho: %f\n", W, ph, q_con[4], q_prim[4], A[z], p, q_prim[0]);
    }
    __syncthreads();
    if ((x < nxs[coarse_level+1]) &&
        (y < nys[coarse_level+1]) &&
        (z < nzs[coarse_level+1])) {

        q_swe[offset*4] = ph * W;
        q_swe[offset*4+1] = ph * W * W * u;
        q_swe[offset*4+2] = ph * W * W * v;
        q_swe[offset*4+3] = ph * W * X;

        //printf("(x,y,z): %d, %d, %d Phi, Sx, Sy: %f, %f, %f\n", x,y,z,q_swe[offset*4], q_swe[offset*4+1], q_swe[offset*4+2]);
    }
}
// device-side function pointers to __device__ functions
__device__ flux_func_ptr d_compressible_fluxes = compressible_fluxes;
__device__ flux_func_ptr d_shallow_water_fluxes = shallow_water_fluxes;
