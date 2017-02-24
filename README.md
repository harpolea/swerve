[![Documentation Status](https://readthedocs.org/projects/swerve/badge/?version=mpi)](http://swerve.readthedocs.io/en/mpi/?badge=mpi)

# swerve
**S**hallow **W**ater **E**quations for **R**elati**V**istic **E**nvironments

`swerve` is a set of software designed to investigate the general relativistic form of the shallow water equations. The code is developed in the notebook `Shallow_Water_Equations.ipynb`, before being implemented in an optimized C++/CUDA version which runs on the GPU. MPI is used to run the code on multiple GPUs (if available).

## Installation and usage

The CUDA version can be built using the Makefile and run using the parameters in the file `input_file.txt`. Before compiling, make sure that the variables `CUDA_PATH` and `MPI_PATH` at the top of the Makefile point to the correct locations of CUDA and MPI on your system. The code can then be compiled by executing `make` (or `make debug` to include debug flags).

To run on e.g. 2 GPU's/processors, execute

    mpirun -np 2 ./gr_cuda

or to use the custom input file `custom_input.txt`,

    mpirun -np 2 ./gr_cuda custom_input.txt

This code outputs into an HDF5 file which can be viewed using the notebook `Plotting.ipynb` (inadvisable except for very small files) or using the python script `plot.py`.

A version of the code which evolves a section of the domain using the compressible fluid equations on a finer grid can be compiled and run using `make mesh` and `./mesh`.

## Testing

A set of tests can be compiled by executing

    make test

then a test case can be run:

    cd testing
    ./flat

This test case provides initial data that is flat with a static gravitational field and no burning. It then tests that this data remains unchanged after being evolved through 100 timesteps.

Unit tests can be run by compiling the tests then running

    cd testing
    ./unit_tests

This will run a set of tests on the majority of the individual functions used and output to screen whether each function tested has passed or failed.
