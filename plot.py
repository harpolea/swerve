import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
import tables as tb


def plotme():
    # see it now? (crashes except for small stuff)
    to_show = False
    # save it?
    to_save = True

    # read input file
    input_file = open('input_file.txt', 'r')
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
        elif name == 'rho':
            rho = [float(d) for d in dat]
        elif name == 'mu':
            mu = float(dat[0])
        elif name == 'alpha':
            alpha = float(dat[0])
        elif name == 'beta':
            beta = [float(d) for d in dat]
        elif name == 'gamma':
            gamma = [float(d) for d in dat]
        elif name == 'dprint':
            dprint = int(dat[0])
        elif name == 'outfile':
            outfile = dat[0]

    dx = (xmax - xmin) / (nx-2)
    dy = (ymax - ymin) / (ny-2)
    dt = 0.1 * min(dx, dy)
    input_file.close()

    # read data
    outfile = '../../Documents/Work/swerve/iridis.h5'
    if (outfile[-2:] == 'h5'): #hdf5
        f = tb.open_file(outfile, 'r')
        table = f.root.SwerveOutput
        D_2d = np.swapaxes(table[:,:,:,:,0], 1, 3)
    else: # assume some kind of csv
        data = np.loadtxt(outfile, delimiter=',')
        ts = data[:,0]
        xs = data[:,1]
        ys = data[:,2]
        ls = data[:,3]
        Ds = data[:,4]
        Sxs = data[:,5]
        Syx = data[:,6]
        #t = range(int((nt+1)/dprint))*dprint

        D_2d = np.zeros((int((nt+1)/dprint), nlayers, nx, ny))
        #Sx_2d = np.zeros((nt, nlayers, nx, ny))
        #Sy_2d = np.zeros((nt, nlayers, nx, ny))

        for i in range(int((nt+1)/dprint)*dprint*nlayers*nx*ny):
            #print(int(xs[i]*nx/xmax))
            D_2d[int(ts[i]), int(ls[i]), int(xs[i]), int(ys[i])] = Ds[i]

    x = np.linspace(0, xmax, num=nx, endpoint=False)
    y = np.linspace(0, ymax, num=ny, endpoint=False)

    X, Y = np.meshgrid(x,y)

    fig = plt.figure(figsize=(12,10))
    ax = fig.gca(projection='3d')

    #print(np.shape(X), np.shape(Y), np.shape(D_2d[0,1,:,:].T))

    surface_1 = ax.plot_surface(X,Y,D_2d[0,1,:,:].T, rstride=1, cstride=2, lw=0, cmap=cm.viridis, antialiased=True)
    surface_2 = ax.plot_wireframe(X,Y,D_2d[0,0,:,:].T, rstride=2, cstride=2, lw=0.1, cmap=cm.viridis, antialiased=True)

    def animate(i):
        ax.clear()
        ax.set_xlim(0,10)
        ax.set_ylim(0,10)
        ax.set_zlim(0.7,1.4)
        ax.plot_surface(X,Y,D_2d[i,1,:,:].T, rstride=1, cstride=2, lw=0, cmap=cm.viridis, antialiased=True)
        ax.plot_wireframe(X,Y,D_2d[i,0,:,:].T, rstride=2, cstride=2, lw=0.1, cmap=cm.viridis, antialiased=True)

    anim = animation.FuncAnimation(fig, animate, frames=len(D_2d[:,0,0,0]), interval=200)

    if to_show:
        plt.show()

    if to_save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
        anim.save('../../Documents/Work/swerve/iridis.mp4', writer=writer)

    if (outfile[-2:] == 'h5'): #hdf5
        f.close()

if __name__ == '__main__':
    plotme()
