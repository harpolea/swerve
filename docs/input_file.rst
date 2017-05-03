===========
Input Files
===========

Initial data in swerve is described in two ways: the first is an input file, describing the parameters of the system, the second is a C++ function which describes the initial data on the coarsest multilayer shallow water grid.

The input file is a text file which provides swerve with the system parameters. This input file is read in at the beginning of the program and used to set up the necessary data structures. Input data is validated at this point and will terminate if invalid parameters are encountered. The filename of this input file can be provided as an argument at runtime - if no argument is provided, then the program defaults to the file ``mesh_input.txt``. The standard form of the input file is as follows:

    :``nx``:      Number of grid points in the :math:`x` dimension of the coarsest grid
    :``ny``:      Number of grid points in the :math:`y` dimension of the coarsest grid
    :``nt``:      Number of timesteps
    :``ng``:      Number of ghost cells
    :``r``:       Refinement ratio
    :``nlevels``: Number of levels of mesh refinement
    :``models``:  List of physical models to be used on each level, where

        ``S`` = single layer shallow water,

        ``M`` = multilayer shallow water,

        ``C`` = compressible and

        ``L`` = Low Mach
    :``nzs``:     Number of layers / grid points in the vertical direction for each grid
    :``df``:      Fraction of the domain each level should cover with respect to the previous level
    :``xmin``:    Minimum :math:`x` coordinate of coarsest grid
    :``xmax``:    Maximum :math:`x` coordinate of coarsest grid
    :``ymin``:    Minimum :math:`y` coordinate of coarsest grid
    :``ymax``:    Maximum :math:`y` coordinate of coarsest grid
    :``zmin``:    Height of sea floor
    :``zmax``:    Maximum :math:`z` coordinate of coarsest compressible grid
    :``rho``:     List of densities :math:`\rho` of multilayer shallow water layers
    :``Q``:       Energy release rate
    :``E_He``:    Binding energy of reactant
    :``Cv``:      Specific heat capacity at constant volume
    :``gamma``:   Adiabatic index :math:`\gamma`
    :``alpha``:   Lapse function :math:`\alpha`
    :``beta``:    Shift vector :math:`\beta`
    :``gamma_down``:   Covariant spatial metric :math:`\gamma_{ij}`
    :``periodic``: Are the boundary conditions periodic (``t``) or outflow (``f``)
    :``burning``: Do we include burning reactions (``t``) or not (``f``)
    :``dprint``:  Number of timesteps between outputting data to file
    :``outfile``: Path to output file (must be HDF5)
    :``n_print_levels``: Number of levels to be printed to file
    :``print_levels``: List of indices of levels to be printed to file

The specific form of the initial data is described in the ``main`` function of the file ``run_mesh_cuda.cpp``. The initial state vector must be provided for all points in the coarsest multilayer shallow water grid.
