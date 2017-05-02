#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <limits>
#include "Mesh_cuda.h"
#include "mesh_cuda_kernel.h"
#include <iostream>
#include <string.h>
#include <fstream>
#include <algorithm>
#include "mpi.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "output.h"

#ifndef H5_NO_NAMESPACE
#ifndef H5_NO_STD
    using std::cout;
    using std::endl;
#endif  // H5_NO_STD
#endif
#include "H5Cpp.h"
using namespace std;

void initialise_hdf5_file(char * filename, int nt, int dprint,
    int * nzs, int * nys, int * nxs, int * vec_dims, int nlevels,
    int * print_levels, float ** Us_h,
    hid_t * outFile, hid_t * dset, hid_t * mem_space, hid_t * file_space) {

    *outFile = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT,
                            H5P_DEFAULT);
    for (int i = 0; i < nlevels; i++) {
        int print_level = print_levels[i];

        // create dataspace
        int ndims = 5;
        hsize_t dims[] = {hsize_t((nt+1)/dprint+1), hsize_t(nzs[print_level]),
                          hsize_t(nys[print_level]), hsize_t(nxs[print_level]), hsize_t(vec_dims[print_level])};
        file_space[i] = H5Screate_simple(ndims, dims, NULL);

        hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_layout(plist, H5D_CHUNKED);
        hsize_t chunk_dims[] = {1, hsize_t(nzs[print_level]), hsize_t(nys[print_level]),
                                hsize_t(nxs[print_level]), hsize_t(vec_dims[print_level])};
        H5Pset_chunk(plist, ndims, chunk_dims);

        string dataset_name("level " + to_string(print_level));

        // create dataset
        dset[i] = H5Dcreate(*outFile,
                          (char *)dataset_name.c_str(), H5T_NATIVE_FLOAT,
                          file_space[i], H5P_DEFAULT, plist, H5P_DEFAULT);

        H5Pclose(plist);

        // make a memory dataspace
        mem_space[i] = H5Screate_simple(ndims, chunk_dims, NULL);

        // select a hyperslab
        file_space[i] = H5Dget_space(*dset);
        hsize_t start[] = {0, 0, 0, 0, 0};
        hsize_t hcount[] = {1, hsize_t(nzs[print_level]), hsize_t(nys[print_level]),
                            hsize_t(nxs[print_level]), hsize_t(vec_dims[print_level])};
        H5Sselect_hyperslab(file_space[i], H5S_SELECT_SET, start, NULL,
                            hcount, NULL);
        // write to dataset
        printf("Printing t = %i\n", 0);
        H5Dwrite(dset[i], H5T_NATIVE_FLOAT, mem_space[i], file_space[i],
                 H5P_DEFAULT, Us_h[print_level]);
        // close file dataspace
        H5Sclose(file_space[i]);
    }
}

void close_hdf5_file(int nlevels, hid_t * mem_space, hid_t outFile) {
    for (int i = 0; i < nlevels; i ++) {
        H5Sclose(mem_space[i]);
    }
    H5Fclose(outFile);
}

void print_timestep(int rank, int n_processes, int print_level,
                    int * nxs, int * nys, int * nzs, int * vec_dims, int ng,
                    int t, MPI_Comm comm, MPI_Status status,
                    dim3 * kernels, dim3 * threads, dim3 * blocks,
                    float ** Us_h,
                    hid_t dset, hid_t mem_space, hid_t file_space, int dprint) {
    if (rank == 0) {
        printf("Printing t = %i\n", t+1);

        if (n_processes > 1) { // only do MPI stuff if needed
            float * buf =
                new float[nxs[print_level]*nys[print_level]*nzs[print_level]*vec_dims[print_level]];
            int tag = 0;
            for (int source = 1; source < n_processes; source++) {
                int mpi_err = MPI_Recv(buf,
                    nxs[print_level]*nys[print_level]*nzs[print_level]*vec_dims[print_level], MPI_FLOAT,
                    source, tag, comm, &status);

                check_mpi_error(mpi_err);

                // copy data back to grid
                int ky_offset = (kernels[0].y * blocks[0].y *
                             threads[0].y - 2*ng) * rank;
                // cheating slightly and using the fact that are moving from bottom to top to make calculations a bit easier.
                for (int z = 0; z < nzs[print_level]; z++) {
                    for (int y = ky_offset; y < nys[print_level]; y++) {
                        for (int x = 0; x < nxs[print_level]; x++) {
                            for (int i = 0; i < vec_dims[print_level]; i++) {
                                Us_h[print_level][((z*nys[print_level]+y)*nxs[print_level]+x)*vec_dims[print_level]+i] =
                                    buf[((z*nys[print_level]+y)*nxs[print_level]+x)*vec_dims[print_level]+i];
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
        hsize_t hcount[] = {1, hsize_t(nzs[print_level]), hsize_t(nys[print_level]),
                            hsize_t(nxs[print_level]), hsize_t(vec_dims[print_level])};
        H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start,
                            NULL, hcount, NULL);
        // write to dataset
        H5Dwrite(dset, H5T_NATIVE_FLOAT, mem_space, file_space,
                 H5P_DEFAULT, Us_h[print_level]);
        // close file dataspae
        H5Sclose(file_space);
    } else { // send data to rank 0
        int tag = 0;
        int mpi_err = MPI_Ssend(Us_h[print_level],
                            nys[print_level]*nxs[print_level]*nzs[print_level]*vec_dims[print_level],
                            MPI_FLOAT, 0, tag, comm);
        check_mpi_error(mpi_err);
    }
}

void checkpoint(float ** Us_h, int * nxs, int * nys, int * nzs, int nlevels,
         int * vec_dims, int ng, int nt, int dprint, char * filename,
         MPI_Comm comm, MPI_Status status, int rank, int n_processes,
         int t, dim3 * kernels, dim3 * threads, dim3 * blocks) {

    string checkpoint_filename("checkpoint_" + to_string(t) + "_");
    checkpoint_filename.append(filename);

    hid_t outFile;
    hid_t dset[nlevels];
    hid_t mem_space[nlevels];
    hid_t file_space[nlevels];

    int print_levels[nlevels];

    for (int i = 0; i < nlevels; i++) {
        print_levels[i] = i;
    }

    initialise_hdf5_file((char*)checkpoint_filename.c_str(), nt, dprint, nzs, nys, nxs, vec_dims, nlevels, print_levels, Us_h, &outFile, dset, mem_space, file_space);

    for (int i = 0; i < nlevels; i++) {

        print_timestep(rank, n_processes, i,
                            nxs, nys, nzs, vec_dims, ng,
                            t, comm, status,
                            kernels, threads, blocks,
                            Us_h,
                            dset[i], mem_space[i], file_space[i], dprint);
    }

    close_hdf5_file(nlevels, mem_space, outFile);
}
