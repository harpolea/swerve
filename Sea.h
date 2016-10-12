#ifndef SEA_H
#define SEA_H

// define some special vector/matrix classes to make things easier

class Vec {
public:
    static const int dim = 3;
    float vec[dim];

    Vec();

    Vec operator* (float a);

    //static float * vec_multiply(SquareMatrix a, float * v);
};

class SquareMatrix {
public:
    static const int dim = 3;
    float mat[dim][dim];

    Vec operator*(Vec v);
    SquareMatrix operator*(float a);
};



class Sea {
public:
    Sea(int n_layers, int _nx, int _ny, int _nt,
            float xmin, float xmax,
            float ymin, float ymax, float * _rho,
            float * _Q,
            float _alpha, float * _beta, float ** _gamma,
            bool _periodic);

    Sea(const Sea &); // copy constructor

    Vec U(int l, int x, int y, int t);

    void initial_data(float * D0, float * Sx0, float * Sy0);

    void bcs(int t);
    void bcs(float ** grid);
    void bcs_fv(float ** grid);

    void evolve(int t);
    void evolve_fv(int t);
    void evolve_cuda(int t, int numBlocks, int numThreads, float * beta_d, float * gamma_up_d, float * U_grid_d);

    void run();

    void output(char * filename);

    ~Sea();

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

    SquareMatrix Jx(Vec u);
    SquareMatrix Jy(Vec u);
};

#endif
