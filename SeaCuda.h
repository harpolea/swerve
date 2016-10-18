#ifndef SEA_CUDA_H
#define SEA_CUDA_H

void cuda_run(float * beta, float * gamma_up, float * U_grid,
         float * rho, float * Q, float mu,
         int nx, int ny, int nlayers, int ng,
         int nt, float alpha, float dx, float dy, float dt, int dprint, char * filename);


class SeaCuda {
public:
    SeaCuda(int n_layers, int _nx, int _ny, int _nt, int _ng,
            float xmin, float xmax,
            float ymin, float ymax, float * _rho,
            float * _Q, float mu,
            float _alpha, float * _beta, float * _gamma,
            bool _periodic, int _dprint);

    SeaCuda(char * filename); // constructor which takes input from file

    SeaCuda(const SeaCuda &); // copy constructor

    void initial_data(float * D0, float * Sx0, float * Sy0, float * _Q, float * _beta);

    void bcs();

    void print_inputs();

    void run();

    void output(char * filename);
    void output_hdf5(char * filename);
    void output();

    ~SeaCuda();

    // these need to be public
    int nlayers;
    int nx;
    int ny;
    int ng;
    float *xs;
    float *ys;
    float *U_grid;

private:

    int nt;

    float dx;
    float dy;
    float dt;

    float *rho;
    float *Q;
    float mu; // friction

    float alpha;
    float *beta;
    float gamma[2*2];
    float gamma_up[2*2];

    bool periodic;

    int dprint; // number of timesteps between printouts

    char outfile[200];
};

#endif
