#ifndef MESH_OUTPUT_H
#define MESH_OUTPUT_H

#include "mpi.h"
#include "cuda_runtime.h"
#include "H5Cpp.h"

void initialise_hdf5_file(char * filename, int nt, int dprint,
    int * nzs, int * nys, int * nxs, int * vec_dims, int nlevels,
    int * print_levels, float ** Us_h,
    hid_t * outFile, hid_t * dset, hid_t * mem_space, hid_t * file_space);

void close_hdf5_file(int nlevels, hid_t * mem_space, hid_t outFile);

void print_timestep(int rank, int n_processes, int print_level,
                    int * nxs, int * nys, int * nzs, int * vec_dims, int ng,
                    int t, MPI_Comm comm, MPI_Status status,
                    dim3 * kernels, dim3 * threads, dim3 * blocks,
                    float ** Us_h,
                    hid_t dset, hid_t mem_space, hid_t file_space, int dprint);

void print_checkpoint(float ** Us_h, int * nxs, int * nys, int * nzs, int nlevels,
         int * vec_dims, int ng, int nt, int dprint, char * filename,
         MPI_Comm comm, MPI_Status status, int rank, int n_processes,
         int t, dim3 * kernels, dim3 * threads, dim3 * blocks);

void mpi_error(int mpi_err);


#endif
