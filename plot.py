import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
import tables as tb
import subprocess

def quick_plot(input_filename=None, data_filename=None, movie_filename=None):
    # the other version was really slow - this does it by hand, making a load of png files then using ffmpeg to stitch them together. It finishes by deleting all the pngs.

    # set defaults
    if input_filename is None:
        input_filename =  'input_file.txt'
    if data_filename is None:
        data_filename = '../../Documents/Work/swerve/iridis2.h5'
    if movie_filename is None:
        movie_filename = '../../Documents/Work/swerve/iridis.mp4'

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
    #D_2d[D_2d > 1.e3] = 0.
        #D_2d = D_2d[::dprint,:,:,:]
    #print(D_2d[:,:,2:-2,2:-2])


    x = np.linspace(0, xmax, num=nx-4, endpoint=False)
    y = np.linspace(0, ymax, num=ny-4, endpoint=False)

    X, Y = np.meshgrid(x,y)

    fig = plt.figure(figsize=(12,10), facecolor='w', dpi=100)
    ax = fig.gca(projection='3d')

    #print(np.shape(X), np.shape(Y), np.shape(D_2d[0,1,:,:].T))

    for i in range(len(D_2d[:,0,0,0])):
        outname = '../../Documents/Work/swerve/plotting/tsunami_superbee_' + format(i, '05') + '.png'
        ax.clear()
        ax.set_xlim(0,10)
        ax.set_ylim(0,10)
        ax.set_zlim(0.7,1.4)
        ax.plot_surface(X,Y,D_2d[i,1,2:-2,2:-2].T, rstride=1, cstride=2, lw=0, cmap=cm.viridis, antialiased=True)
        ax.plot_wireframe(X,Y,D_2d[i,0,2:-2,2:-2].T, rstride=2, cstride=2, lw=0.1, cmap=cm.viridis, antialiased=True)
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


if __name__ == '__main__':
    #plotme()
    quick_plot(data_filename="../../Documents/Work/swerve/tsunami_superbee.h5", movie_filename="../../Documents/Work/swerve/tsunami_long.mp4")
