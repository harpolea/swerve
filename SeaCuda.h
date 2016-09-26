#ifndef SEA_CUDA_H
#define SEA_CUDA_H

class SeaCuda {
public:
    SeaCuda(int n_layers, int _nx, int _ny, int _nt,
            float xmin, float xmax,
            float ymin, float ymax, float * _rho,
            float * _Q,
            float _alpha, float * _beta, float ** _gamma,
            bool _periodic);

    SeaCuda(const SeaCuda &); // copy constructor

    void U(float * grid, int l, int x, int y, int t, float * u);

    void initial_data(float * D0, float * Sx0, float * Sy0);

    void bcs(int t);
    void bcs(float * grid);

    //void evolve(int t, float * beta_d, float * gamma_up_d, float * U_grid_d, float * rho_d, float * Q_d);

    void run();

    void output(char * filename);

    ~SeaCuda();

    // these need to be public
    float *xs;
    float *ys;

private:
    int nlayers;
    int nx;
    int ny;
    int nt;

    float **U_grid;

    float dx;
    float dy;
    float dt;

    float *rho;
    float *Q;

    float alpha;
    float beta[2];
    float gamma[2][2];
    float gamma_up[2][2];

    bool periodic;

    //void Jx(float * u, float * beta_d, float * gamma_up_d, float * jx);
    //void Jy(float * u, float * beta_d, float * gamma_up_d, float * jx);
};

#endif
