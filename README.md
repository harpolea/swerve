# swerve
**S**hallow **W**ater **E**quations for **R**elati**V**istic **E**nvironments

`swerve` is a set of software designed to investigate the general relativistic form of the shallow water equations. The code is developed in the notebook `Shallow_Water_Equations.ipynb`, before being implemented in an optimized C++/CUDA version which runs on the GPU. MPI is used to run the code on multiple GPUs (if available).

The CUDA version can be built using the Makefile and run using the parameters in the file `input_file.txt`. To run on e.g. 2 GPU's/processors, execute
    ```bash
    mpirun -np 2 ./gr_cuda
    ```
or to use the custom input file `custom_input.txt`,
    ```bash
    mpirun -np 2 ./gr_cuda custom_input.txt
    ```

This code outputs into an HDF5 file which can be viewed using the notebook `Plotting.ipynb` or using the python script `plot.py`.

A test case can be compiled by executing
    ```bash
    make test
    ```
then
    ```bash
    cd testing
    ./flat
    ```
This test case provides initial data that is flat with a static gravitational field and no burning. It then tests that this data remains unchanged after being evolved through 100 timesteps.
