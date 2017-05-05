#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <limits>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <algorithm>
#include "mpi.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "mesh_output.h"
#include "Mesh_cuda.h"

#ifndef H5_NO_NAMESPACE
#ifndef H5_NO_STD
    using std::cout;
    using std::endl;
#endif  // H5_NO_STD
#endif
#include "H5Cpp.h"
using namespace std;

void initialise_hdf5_file(char * filename, int nt, int dprint,
    int * nzs, int * nys, int * nxs, int * vec_dims, int n_print_levels,
    int * print_levels, float ** Us,
    hid_t * outFile, hid_t * dset, hid_t * mem_space, hid_t * file_space,
    char * param_filename) {
    /*
    Initialise the HDF5 file. Uses walkthrough from https://stackoverflow.com/questions/15379399/writing-appending-arrays-of-float-to-the-only-dataset-in-hdf5-file-in-c/15396949#15396949?newreg=b254d01af38948159bd529d8d4f5f5b9
    */
    printf("Printing t = %i\n", 0);

    *outFile = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT,
                            H5P_DEFAULT);
    for (int i = 0; i < n_print_levels; i++) {
        int print_level = print_levels[i];

        // create dataspace
        int ndims = 5;
        hsize_t dims[] = {hsize_t((nt+1)/dprint+1), hsize_t(nzs[print_level]),
                          hsize_t(nys[print_level]), hsize_t(nxs[print_level]),
                          hsize_t(vec_dims[print_level])};
        file_space[i] = H5Screate_simple(ndims, dims, NULL);

        hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_layout(plist, H5D_CHUNKED);
        hsize_t chunk_dims[] = {1, hsize_t(nzs[print_level]),
                                hsize_t(nys[print_level]),
                                hsize_t(nxs[print_level]),
                                hsize_t(vec_dims[print_level])};
        H5Pset_chunk(plist, ndims, chunk_dims);

        stringstream ss;
        ss << print_level;

        string dataset_name("level_" + ss.str());

        // create dataset
        dset[i] = H5Dcreate(*outFile,
                          (char *)dataset_name.c_str(), H5T_NATIVE_FLOAT,
                          file_space[i], H5P_DEFAULT, plist, H5P_DEFAULT);

        H5Pclose(plist);

        // make a memory dataspace
        mem_space[i] = H5Screate_simple(ndims, chunk_dims, NULL);

        // select a hyperslab
        file_space[i] = H5Dget_space(dset[i]);
        hsize_t start[] = {0, 0, 0, 0, 0};
        hsize_t hcount[] = {1, hsize_t(nzs[print_level]),
                            hsize_t(nys[print_level]),
                            hsize_t(nxs[print_level]),
                            hsize_t(vec_dims[print_level])};
        H5Sselect_hyperslab(file_space[i], H5S_SELECT_SET, start, NULL,
                            hcount, NULL);
        // write to dataset
        H5Dwrite(dset[i], H5T_NATIVE_FLOAT, mem_space[i], file_space[i],
                 H5P_DEFAULT, Us[print_level]);
        // close file dataspace
        H5Sclose(file_space[i]);
    }

    // output parameter file as a long string
    ifstream inputFile(param_filename);
    string contents((istreambuf_iterator<char>(inputFile)),
                    istreambuf_iterator<char>());
    // create dataspace
    int ndims = 1;
    hsize_t dims[] = {1};
    hid_t param_file_space = H5Screate_simple(ndims, dims, NULL);

    // create dataset
    hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    hid_t param_dset = H5Dcreate(*outFile,
                      "Input_parameters", H5T_NATIVE_CHAR,
                      param_file_space, H5P_DEFAULT, plist, H5P_DEFAULT);

    H5Pclose(plist);

    // make a memory dataspace
    hid_t param_mem_space = H5Screate_simple(ndims, dims, NULL);

    // select a hyperslab
    param_file_space = H5Dget_space(param_dset);
    hsize_t start[] = {0};
    hsize_t hcount[] = {1};
    H5Sselect_hyperslab(param_file_space, H5S_SELECT_SET, start, NULL,
                        hcount, NULL);
    // write to dataset
    H5Dwrite(param_dset, H5T_NATIVE_FLOAT, param_mem_space, param_file_space,
             H5P_DEFAULT, contents.c_str());
    // close file dataspace
    H5Sclose(param_file_space);
    H5Sclose(param_mem_space);
}

void close_hdf5_file(int nlevels, hid_t * mem_space, hid_t outFile) {
    /*
    Close HDF5 file.
    */
    for (int i = 0; i < nlevels; i ++) {
        H5Sclose(mem_space[i]);
    }
    H5Fclose(outFile);
}

void print_timestep(int rank, int n_processes, int print_level,
                    int * nxs, int * nys, int * nzs, int * vec_dims, int ng,
                    int t, MPI_Comm comm, MPI_Status status,
                    dim3 * kernels, dim3 * threads, dim3 * blocks,
                    float ** Us,
                    hid_t dset, hid_t mem_space, hid_t file_space, int dprint) {
    /*
    Print timestep.
    */
    if (rank == 0) {
        printf("Printing t = %i ,level = %i\n", t+1, print_level);

        if (n_processes > 1) { // only do MPI stuff if needed
            float * buf =
                new float[nxs[print_level]*nys[print_level]*
                    nzs[print_level]*vec_dims[print_level]];
            int tag = 0;
            for (int source = 1; source < n_processes; source++) {
                int mpi_err = MPI_Recv(buf,
                    nxs[print_level]*nys[print_level]*
                    nzs[print_level]*vec_dims[print_level], MPI_FLOAT,
                    source, tag, comm, &status);

                mpi_error(mpi_err);

                // copy data back to grid
                int ky_offset = (kernels[0].y * blocks[0].y *
                             threads[0].y - 2*ng) * rank;
                // cheating slightly and using the fact that are moving from
                // bottom to top to make calculations a bit easier.
                for (int z = 0; z < nzs[print_level]; z++) {
                    for (int y = ky_offset; y < nys[print_level]; y++) {
                        for (int x = 0; x < nxs[print_level]; x++) {
                            for (int i = 0; i < vec_dims[print_level]; i++) {
                                Us[print_level][((z*nys[print_level]+y)*nxs[print_level]+x)*vec_dims[print_level]+i] =
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
        hsize_t hcount[] = {1, hsize_t(nzs[print_level]),
                            hsize_t(nys[print_level]),
                            hsize_t(nxs[print_level]),
                            hsize_t(vec_dims[print_level])};
        H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start,
                            NULL, hcount, NULL);
        // write to dataset
        H5Dwrite(dset, H5T_NATIVE_FLOAT, mem_space, file_space,
                 H5P_DEFAULT, Us[print_level]);
        // close file dataspae
        H5Sclose(file_space);
    } else { // send data to rank 0
        int tag = 0;
        int mpi_err = MPI_Ssend(Us[print_level],
                            nys[print_level]*nxs[print_level]*
                            nzs[print_level]*vec_dims[print_level],
                            MPI_FLOAT, 0, tag, comm);
        mpi_error(mpi_err);
    }
}

void print_checkpoint(float ** Us, int * nxs, int * nys, int * nzs,
         int * vec_dims, int ng, int nlevels, char * filename,
         MPI_Comm comm, MPI_Status status, int rank, int n_processes,
         int t, dim3 * kernels, dim3 * threads, dim3 * blocks,
         char * param_filename) {
    /*
    Print checkpoint to file.
    */

    stringstream ss;
    ss << t;

    string checkpoint_filename("checkpoint_" + ss.str() + "_");
    checkpoint_filename.append(filename);

    hid_t outFile;
    hid_t dset[nlevels];
    hid_t mem_space[nlevels];
    hid_t file_space[nlevels];

    int print_levels[nlevels];

    for (int i = 0; i < nlevels; i++) {
        print_levels[i] = i;
    }

    initialise_hdf5_file((char*)checkpoint_filename.c_str(), 1, 1,
                         nzs, nys, nxs, vec_dims, nlevels, print_levels,
                         Us, &outFile, dset, mem_space, file_space,
                         param_filename);

    for (int i = 0; i < nlevels; i++) {

        print_timestep(rank, n_processes, i, nxs, nys, nzs, vec_dims, ng,
                       t, comm, status, kernels, threads, blocks, Us,
                       dset[i], mem_space[i], file_space[i], 1);
    }

    close_hdf5_file(nlevels, mem_space, outFile);
}

void start_from_checkpoint(char * filename, MPI_Comm comm, MPI_Status status,
        int rank, int n_processes) {
    /**
    Reads checkpoint file and restarts simulation.
    */

    hid_t file_id, param_dataset;

    // open file
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

    // open param datset
    param_dataset = H5Dopen2(file_id, "/Input_parameters", H5P_DEFAULT);

    // read parameter file into a buffer char array
    char param_buf[2000];
    H5Dread(param_dataset, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL,
             H5P_DEFAULT, param_buf);

    // close datset
    H5Dclose(param_dataset);

    // process char array into something intelligible.
    stringstream s;
    s << param_buf;

    // initialise Sea object
    Sea sea(s, filename);

    // read data from file
    hid_t dset;

    for (int i = 0; i < sea.nlevels; i++) {
        stringstream ss;
        ss << i;
        string dataset_name("/level_" + ss.str());
        // open  datset
        dset = H5Dopen2(file_id, dataset_name.c_str(), H5P_DEFAULT);

        H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                sea.Us[i]);
        H5Dclose(dset);
    }

    if (rank == 0) {
        sea.print_inputs();
    }

    // Find timestep
    size_t t1 = string(filename).find("_");
    size_t t2 = string(filename).find("_", t1+1);
    stringstream ss;
    for (int i = t1+1; i < t2; i++) ss << filename[i];
    int tstart;
    ss >> tstart;

    cout << "Starting simulation from t = " << tstart << '\n';

    // run simulation
    sea.run(comm, &status, rank, n_processes, tstart);
}

void mpi_error(int mpi_err) {
    /**
    Checks to see if the integer returned by an mpi function, mpi_err, is an
    MPI error. If so, it prints out some useful stuff to screen.
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
