import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib import animation, cm
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
import tables as tb
import subprocess
from scipy.optimize import brentq

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

        if name == 'r':
            r = int(dat[0])
        elif name == 'df':
            df = float(dat[0])
        elif name == 'gamma':
            gamma = float(dat[0])
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

    #if (models[0] == 'S'):
        # coarsest layer is single layer SWE - adjust nx, ny to get multilayer dimensions
    #    nx *= r * df
    #    ny *= r * df

    # read data
    f = tb.open_file(data_filename, 'r')
    table = f.root.SwerveOutput

    if len(table[0,0,0,0,:]) == 4:
        # swe
        swe = True
        D = table[:,:,:,:,0]
        Sx = table[:,:,:,:,1]
        Sy = table[:,:,:,:,2]
        DX = table[:,:,:,:,3]
    else:
        swe = False
        D = table[:,:,:,:,0]
        Sx = table[:,:,:,:,1]
        Sy = table[:,:,:,:,2]
        Sz = table[:,:,:,:,3]
        tau = table[:,:,:,:,4]
        DX = table[:,:,:,:,5]
    # HACK
    nx = len(D[0,0,0,:])
    ny = len(D[0,0,:,0])
    nz = len(D[0,:,0,0])
    nt = len(D[:,0,0,0])

    dx = (xmax - xmin) / (nx-2)
    dy = (ymax - ymin) / (ny-2)
    dt = 0.1 * min(dx, dy)
    input_file.close()

    """S = np.sqrt(Sx**2 + Sy**2 + Sz**2)

    def f_of_p(p, tau, D, S):
        sq = np.sqrt((tau + p + D) * (tau + p + D) - S**2)
        return (gamma - 1.0) * sq / (tau + p + D) * (sq - p * (tau + p + D) / sq - D) - p

    ps = np.zeros_like(D[:,:,:,:])
    for t in range(nt):
        print(t)
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    #print(tau[t,z,y,x], D[t,z,y,x], S[t,z,y,x])
                    ps[t,z,y,x] = brentq(f_of_p, 0, tau[t,z,y,x] + D[t,z,y,x] + 1., args=(tau[t,z,y,x], D[t,z,y,x], S[t,z,y,x]))"""

    if swe:
        plot_var = find_height(D, Sx, Sy, gamma_up)#np.sqrt(Sx**2 + Sy**2)#
        if nz > 1:
            plot_range = range(1,2)
        else:
            plot_range = range(1)
    else:
        plot_var = D
        plot_range = range(2,3)
    #D_2d[D_2d > 1.e3] = 0.
        #D_2d = D_2d[::dprint,:,:,:]
    #print(D_2d[:,:,2:-2,2:-2])

    x = np.linspace(0, xmax, num=nx, endpoint=False)
    y = np.linspace(0, ymax, num=ny, endpoint=False)

    X, Y = np.meshgrid(x,y)

    fig = plt.figure(figsize=(12,10), facecolor='w', dpi=100)
    if swe:
        ax = fig.gca(projection='3d')
    else:
        ax = fig.gca()

    location = '/'.join(filename.split('/')[:-1])
    name = filename.split('/')[-1]

    #print('shapes: X {}, Y {}, D2d {}'.format(np.shape(X), np.shape(Y), np.shape(D_2d[0,2:-2,2:-2].T)))

    for i in range(start, len(D[:,0,0,0])-1):
        #if i % 10 == 0:
        print('Printing {}'.format(i))

        outname = location + '/plotting/' + name + '_' + format(i, '05') + '.png'
        ax.clear()
        #ax.set_xlim(0,10)
        #ax.set_ylim(0,10)
        #ax.set_zlim(2.2,2.35)

        for l in plot_range:
            #print(plot_var[i,l,:,15])
            face_colours = DX[i,l,:,:].T
            if abs(np.amax(face_colours)) > 0.:
                face_colours /= abs(np.amax(face_colours))
            face_colours = (face_colours - np.amin(face_colours)) / (np.amax(face_colours) - np.amin(face_colours))
            if swe:
                ax.plot_surface(X[:,:],Y[:,:],plot_var[i,l,:,:].T, rstride=1,
                cstride=2, lw=0, cmap=cm.viridis_r, antialiased=True,
                facecolors=cm.viridis_r(face_colours))
            else:
                plt.plot(Y[:,0], plot_var[i,l,:,15])#
        plt.savefig(outname)

    # close hdf5 file
    f.close()


if __name__ == '__main__':
    mesh_plot(input_filename="testing/multiscale_input.txt", filename="../../Documents/Work/swerve/multiscale_test")
    #mesh_plot(filename="../../Documents/Work/swerve/mesh_test")
