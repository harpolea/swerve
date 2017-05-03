#ifndef MESH_OUTPUT_H
#define MESH_OUTPUT_H

#include "mpi.h"
#include "cuda_runtime.h"
#include "H5Cpp.h"

/**
Initialises hdf5 file ready for output.

\param filename
    Name of hdf5 file to print to
\param nt
    Number of timesteps
\param dprint
    Number of timesteps between printouts
\param nxs, nys, nzs, vec_dims
    Grid dimensions
\param n_print_levels
    Number of levels to be printed
\param print_levels
    Indices of levels to be printed
\param Us
    Array containing all the data
\param outFile
    Reference to HDF5 file id
\param dset
    Array of datasets in HDF5 file containing levels' data
\param mem_space
    Array of memory spaces in HDF5 file containing levels' data
\param file_space
    Array of file spaces in HDF5 file containing levels' data
\param param_filename
    Name of parameter file
*/
void initialise_hdf5_file(char * filename, int nt, int dprint,
    int * nzs, int * nys, int * nxs, int * vec_dims, int n_print_levels,
    int * print_levels, float ** Us,
    hid_t * outFile, hid_t * dset, hid_t * mem_space, hid_t * file_space,
    char * param_filename);

/**
Closes hdf5 file.

\param nlevels
    Number of print levels
\param mem_space
    Array of memory spaces in HDF5 file containing levels' data
\param outFile
    Reference to HDF5 file id
*/
void close_hdf5_file(int nlevels, hid_t * mem_space, hid_t outFile);

/**
Output current timestep to file.

\param rank
    Rank of MPI process
\param n_processes
    Total number of MPI processes
\param print_level
    Index of level to be printed
\param nxs, nys, nzs, vec_dims
    Grid dimensions
\param ng
    Number of ghost cells
\param t
    Timestep
\param comm
    MPI communicator
\param status
    MPI status
\param kernels, threads, blocks
    Dimensions of GPU threads
\param Us
    Array containing all the data
\param dset
    Dataset in HDF5 file containing level's data
\param mem_space
    Memory space in HDF5 file containing level's data
\param file_space
    File space in HDF5 file containing level's data
\param dprint
    Interval between printouts
*/
void print_timestep(int rank, int n_processes, int print_level,
                    int * nxs, int * nys, int * nzs, int * vec_dims, int ng,
                    int t, MPI_Comm comm, MPI_Status status,
                    dim3 * kernels, dim3 * threads, dim3 * blocks,
                    float ** Us,
                    hid_t dset, hid_t mem_space, hid_t file_space, int dprint);

/**
Print a checkpoint file.

\param Us
    Array containing all the data
\param nxs, nys, nzs, vec_dims
    Grid dimensions
\param ng
    Number of ghost cells
\param nlevels
    Total number of grid levels
\param filename
    Name of HDF5 file data is normally printout out to
\param comm
    MPI communicator
\param status
    MPI status
\param rank
    Rank of MPI process
\param n_processes
    Total number of MPI processes
\param kernels, threads, blocks
    Dimensions of GPU threads
\param param_filename
    Name of input parameter file
*/
void print_checkpoint(float ** Us, int * nxs, int * nys, int * nzs,
         int * vec_dims, int ng, int nlevels, char * filename,
         MPI_Comm comm, MPI_Status status, int rank, int n_processes,
         int t, dim3 * kernels, dim3 * threads, dim3 * blocks,
         char * param_filename);

/**
Begin simulation from a checkpoint file

\param filename
    Name of checkpoint file
\param comm
    MPI communicator
\param status
    MPI status
\param rank
    Rank of MPI process
\param n_processes
    Total number of MPI processes
*/
void start_from_checkpoint(char * filename,
        MPI_Comm comm, MPI_Status status, int rank, int n_processes);

void mpi_error(int mpi_err);


#endif
