import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib import animation, cm
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
import tables as tb
import subprocess

def quick_plot(input_filename=None, filename=None, start=0):

    # the other version was really slow - this does it by hand, making a load of png files then using ffmpeg to stitch them together. It finishes by deleting all the pngs.

    # set defaults
    if input_filename is None:
        input_filename =  'input_file.txt'
    if filename is None:
        filename = '../../Documents/Work/swerve/iridis2'

    data_filename = filename + '.h5'

    # read input file
    input_file = open(input_filename, 'r')
    inputs = input_file.readlines()

    for line in inputs:
        name, *dat = line.split()

        if name == 'nx':
            nx = int(dat[0])
        elif name == 'ny':
            ny = int(dat[0])
        elif name == 'nt':
            nt = int(dat[0])
        elif name == 'nlayers':
            nlayers = int(dat[0])
        elif name == 'xmin':
            xmin = float(dat[0])
        elif name == 'xmax':
            xmax = float(dat[0])
        elif name == 'ymin':
            ymin = float(dat[0])
        elif name == 'ymax':
            ymax = float(dat[0])
        elif name == 'dprint':
            dprint = int(dat[0])


    dx = (xmax - xmin) / (nx-2)
    dy = (ymax - ymin) / (ny-2)
    dt = 0.1 * min(dx, dy)
    input_file.close()

    # read data
    f = tb.open_file(data_filename, 'r')
    table = f.root.SwerveOutput
    D_2d = np.swapaxes(table[:,:,:,:,0], 1, 3)
    zeta_2d = np.swapaxes(table[:,:,:,:,3], 1, 3)
    #D_2d[D_2d > 1.e3] = 0.
        #D_2d = D_2d[::dprint,:,:,:]
    #print(D_2d[:,:,2:-2,2:-2])


    x = np.linspace(0, xmax, num=nx-4, endpoint=False)
    y = np.linspace(0, ymax, num=ny-4, endpoint=False)

    X, Y = np.meshgrid(x,y)

    fig = plt.figure(figsize=(12,10), facecolor='w', dpi=100)
    ax = fig.gca(projection='3d')

    #print(np.shape(X), np.shape(Y), np.shape(D_2d[0,1,:,:].T))

    location = '/'.join(filename.split('/')[:-1])
    name = filename.split('/')[-1]

    for i in range(start, len(D_2d[:,0,0,0])):
        #if i % 10 == 0:
        print('Printing {}'.format(i))

        outname = location + '/plotting/' + name + '_' + format(i, '05') + '.png'
        ax.clear()
        ax.set_xlim(0,10)
        ax.set_ylim(0,10)
        ax.set_zlim(0.7,1.9)
        ax.plot_surface(X,Y,D_2d[i,1,2:-2,2:-2].T, rstride=1, cstride=2, lw=0, facecolors=cm.viridis_r(zeta_2d[i,1,2:-2,2:-2].T), antialiased=True)
        #ax.plot_wireframe(X,Y,D_2d[i,0,2:-2,2:-2].T, rstride=2, cstride=2, lw=0.1, cmap=cm.viridis, antialiased=True)
        ax.plot_surface(X,Y,D_2d[i,0,2:-2,2:-2].T, rstride=1, cstride=2, lw=0, facecolors=cm.viridis_r(zeta_2d[i,0,2:-2,2:-2].T), antialiased=True)
        plt.savefig(outname)

    # close hdf5 file
    f.close()

    # now make a video!
    """
    bashCommand = "ffmpeg -framerate 10 -pattern_type glob -i '../../Documents/Work/swerve/plotting/fv_?????.png' -c:v libx264 -r 10 " +  movie_filename
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # delete image files
    bashCommand = "rm ../../Documents/Work/swerve/plotting/fv_*.png"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()"""

def find_height(D, Sx, Sy, gamma_up, M=1.0):

    D[D < 1.e-6] = 1.e-6
    W = np.sqrt((Sx**2*gamma_up[0,0] + 2. * Sx * Sy * gamma_up[1,0]
        + Sy**2 * gamma_up[1,1]) / D**2 + 1.0)
    return 2. * M / (1. - np.exp(-2 * D / W))


def mesh_plot(input_filename=None, filename=None, start=0):

    # the other version was really slow - this does it by hand, making a load of png files then using ffmpeg to stitch them together. It finishes by deleting all the pngs.

    # set defaults
    if input_filename is None:
        input_filename =  'mesh_input.txt'
    if filename is None:
        filename = '../../Documents/Work/swerve/iridis2'

    data_filename = filename + '.h5'

    # read input file
    input_file = open(input_filename, 'r')
    inputs = input_file.readlines()

    for line in inputs:
        name, *dat = line.split()

        if name == 'nx':
            nx = int(dat[0])
        elif name == 'ny':
            ny = int(dat[0])
        elif name == 'nt':
            nt = int(dat[0])
        elif name == 'nlayers':
            nlayers = int(dat[0])
        elif name == 'xmin':
            xmin = float(dat[0])
        elif name == 'xmax':
            xmax = float(dat[0])
        elif name == 'ymin':
            ymin = float(dat[0])
        elif name == 'ymax':
            ymax = float(dat[0])
        elif name == 'gamma_down':
            gamma_down = np.array([float(i) for i in dat])
            if len(gamma_down) == 4:
                gamma_up = np.reshape(gamma_down, (2,2))
                gamma_up[0,0] = 1. / gamma_up[0,0]
                gamma_up[1,1] = 1. / gamma_up[1,1]
            else:
                n = int(np.sqrt(len(gamma_down)))
                gamma_up = np.reshape(gamma_down, (n,n))
                gamma_up = inv(gamma_up)
        elif name == 'dprint':
            dprint = int(dat[0])

    dx = (xmax - xmin) / (nx-2)
    dy = (ymax - ymin) / (ny-2)
    dt = 0.1 * min(dx, dy)
    input_file.close()

    # read data
    f = tb.open_file(data_filename, 'r')
    table = f.root.SwerveOutput
    D_2d = table[:,:,:,:,0]
    Sx = table[:,:,:,:,1]
    Sy = table[:,:,:,:,2]
    tau = table[:,:,:,:,4]

    v = np.sqrt(Sx**2 + Sy**2)

    #heights = find_height(D_2d, Sx, Sy, gamma_up)
    #D_2d[D_2d > 1.e3] = 0.
        #D_2d = D_2d[::dprint,:,:,:]
    #print(D_2d[:,:,2:-2,2:-2])

    x = np.linspace(0, xmax, num=nx-4, endpoint=False)
    y = np.linspace(0, ymax, num=ny-4, endpoint=False)

    X, Y = np.meshgrid(x,y)

    fig = plt.figure(figsize=(12,10), facecolor='w', dpi=100)
    ax = fig.gca(projection='3d')

    location = '/'.join(filename.split('/')[:-1])
    name = filename.split('/')[-1]

    #print('shapes: X {}, Y {}, D2d {}'.format(np.shape(X), np.shape(Y), np.shape(D_2d[0,2:-2,2:-2].T)))

    for i in range(start, len(D_2d[:,0,0,0])-1):
        #if i % 10 == 0:
        print('Printing {}'.format(i))

        outname = location + '/plotting/' + name + '_' + format(i, '05') + '.png'
        ax.clear()
        ax.set_xlim(0,10)
        ax.set_ylim(0,10)
        #ax.set_zlim(2.0,2.6)
        for l in range(nlayers-1):
            ax.plot_surface(X,Y,tau[i,l,2:-2,2:-2].T, rstride=1, cstride=2, lw=0, cmap=cm.viridis_r, antialiased=True)
        plt.savefig(outname)

    # close hdf5 file
    f.close()


if __name__ == '__main__':
    #quick_plot(filename="../../Documents/Work/swerve/mpi")

    mesh_plot(filename="../../Documents/Work/swerve/mpi_mesh")
