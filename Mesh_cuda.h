#ifndef MESH_CUDA_H
#define MESH_CUDA_H

#include "cuda_runtime.h"
#include "mpi.h"

/** A class that manages the simulation.

Implements Sea class.
*/

class Sea {
public:
    /** Constructor from list of parameters.
    */
    Sea(int _nx, int _ny, int _nz, int _nlayers, int _nt, int _ng,
            int _r, float _df,
            float xmin, float xmax,
            float ymin, float ymax,
            float zmin, float zmax, float  * _rho,
            float  _Q, float _gamma, float _E_He, float _Cv,
            float _alpha, float * _beta, float * _gamma_down,
            bool _periodic, bool _burning, int _dprint);

    /** Constructor for Sea class using inputs from file.

    Data is validated: an error will be thrown and the program terminated if any of the inputs are found to be invalid.

    \param filename name of input file
    */
    Sea(char * filename);

    Sea(const Sea &); /**<Copy constructor */

    /** Initialise D, Sx, Sy and Q.

    \param D0 conserved density
    \param Sx0 conserved x-velocity
    \param Sy0 conserved y-velocity
    */
    void initial_data(float * D0, float * Sx0, float * Sy0);

    /**
    Enforce boundary conditions on grid of quantities with dimension vec_dim.

    \param grid grid on which boundary conditions are to be enforced
    \param n_x, n_y, n_z grid dimensions
    \param vec_dim dimension of state vector
    */
    void bcs(float * grid, int n_x, int n_y, int n_z, int vec_dim);

    /**
    Print some input and runtime parameters to screen.
    */
    void print_inputs();

    /**
    Invert the m x n matrix M in place using Gaussian elimination.

    \param A Matrix to be inverted
    \param m, n Dimensions of matrix
    */
    static void invert_mat(float * A, int m, int n);

    /**
    Run simulation.

    \param comm MPI communicator
    \param status MPI status flag
    \param rank MPI process rank number
    \param size Total number of MPI processes
    */
    void run(MPI_Comm comm, MPI_Status * status, int rank, int size);

    ~Sea(); /**<Deconstructor. Clean up member arrays. */

    // these need to be public
    int nx; /**<number of gridpoints in x-direction of coarsest grid*/
    int ny; /**< number of gridpoints in y-direction of coarsest grid */
    int * nxs; /**<number of gridpoints in x-direction of grids */
    int * nys; /**< number of gridpoints in y-direction of grids */
    int * nzs; /**< Number of layers to have in each grid*/
    int nz; /**< number of gridpoints in z-direction of fine (compressible) grid */
    int nlayers; /**< Number of shallow water layers*/
    int ng; /**< Number of ghost cells */
    int nlevels; /**< Number of levels of mesh refinement*/
    char * models; /**< Array describing the physical model to use on each level. S = single layer SWE, M = multilayer SWE, C = compressible, L = Low Mach*/
    int * vec_dims; /**< Dimensions of state vectors on each grid*/


    float dz; /**< Gridpoint separation in the z-direction of fine (compressible grid)*/
    float zmin; /**< Height of sea floor */
    float zmax; /**< Maximum height of sea surface */
    float * xs; /**< Vector of x-coordinates of coarsest gridpoints */
    float * ys; /**< Vector of y-coordinates of coarsest gridpoints  */
    float * U_coarse; /**< Conserved shallow water variables on coarse grid */
    float * U_fine; /**< Conserved compressible variables on fine grid.*/
    float ** Us; /**< Array of pointers to grids.*/

private:

    int nt; /**< Total number of timesteps to run simulation for */
    int r; /**< refinement ratio */
    int nxf; /**< Number of gridpoints in x-direction on fine grid */
    int nyf; /**< Number of gridpoints in y-direction on fine grid */
    int * matching_indices; /**< Location of fine grids wrt coarser grid coordinates */

    float dx; /**< Gridpoint separation in x-direction on coarsest grid */
    float dy; /**< Gridpoint separation in y-direction on coarsest grid */

    float dt; /**< Timestep */
    float df; /**< Fraction of coarse grid covered by fine grid */

    float *rho; /**< Vector of density in each of the shallow water layers  */
    float Q; /**< Mass transfer rate */
    float gamma; /**< Adiabatic index */
    float E_He; /**< Energy release per unit mass of helium burning*/
    float Cv; /**< Specific heat at constant volume*/

    float alpha; /**< Lapse function */
    float beta[3]; /**< Shift vector */
    float gamma_down[3*3]; /**< Covariant spatial metric */
    float gamma_up[3*3]; /**< Contravariant spatial metric */

    bool periodic; /**< Are the boundaries periodic (true) or outflow (false) */
    bool burning; /**< Do we include burning? (True)*/

    int dprint; /**< number of timesteps between printouts */

    char outfile[200]; /**< Name of (hdf5) file to print output data to */
};

#endif
