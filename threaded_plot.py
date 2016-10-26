import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
import tables as tb
import subprocess
import threading
#import Queue


def quick_plot(input_filename=None, filename=None):
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

    # put the class inside here so can be lazy and not pass X,Y as member variables each time and therefore hopefully save some memory?
    class myThread (threading.Thread):
        def __init__(self, threadID, outname, D_2d, zeta_2d):
            threading.Thread.__init__(self)
            self.threadID = threadID
            self.outname = outname
            self.D = D_2d
            self.zeta = zeta_2d
        def run(self):
            print( "Starting " + format(self.threadID))

            # make plot and save
            #fig = plt.figure(figsize=(12,10), facecolor='w', dpi=100)
            #ax = fig.gca(projection='3d')
            fig, ax = plt.subplots()
            fig.set_dpi(100)
            fig.set_size_inches(12,10)
            fig.set_facecolor('w')
            ax.clear()
            ax.set_xlim(0,10)
            ax.set_ylim(0,10)
            ax.set_zlim(0.7,1.4)
            #print(np.shape(X), np.shape(self.D))
            ax.plot_surface(X,Y,self.D[:,:,1], rstride=1, cstride=2, lw=0, facecolors=cm.viridis_r(self.zeta[:,:,1]), antialiased=True)
            #ax.plot_wireframe(X,Y,D_2d[i,0,2:-2,2:-2].T, rstride=2, cstride=2, lw=0.1, cmap=cm.viridis, antialiased=True)
            ax.plot_surface(X,Y,self.D[:,:,0], rstride=1, cstride=2, lw=0, facecolors=cm.viridis_r(self.zeta[:,:,0]), antialiased=True)

            plt.savefig(self.outname)

            print( "Exiting " + format(self.threadID))

    threads = []
    for i in range(159, len(D_2d[:,0,0,0])):
        outname = location + '/plotting/' + name + '_' + format(i, '05') + '.png'
        thread = myThread(i, outname, D_2d[i,:,2:-2,2:-2].T, zeta_2d[i,:,2:-2,2:-2].T)
        thread.start()
        threads.append(thread)

    # wait for them to finish
    for t in threads:
        t.join()


    """for i in range(len(D_2d[:,0,0,0])):
        outname = location + '/plotting/' + name + '_' + format(i, '05') + '.png'
        ax.clear()
        ax.set_xlim(0,10)
        ax.set_ylim(0,10)
        ax.set_zlim(0.7,1.4)
        ax.plot_surface(X,Y,D_2d[i,1,2:-2,2:-2].T, rstride=1, cstride=2, lw=0, facecolors=cm.viridis_r(zeta_2d[i,1,2:-2,2:-2].T), antialiased=True)
        #ax.plot_wireframe(X,Y,D_2d[i,0,2:-2,2:-2].T, rstride=2, cstride=2, lw=0.1, cmap=cm.viridis, antialiased=True)
        ax.plot_surface(X,Y,D_2d[i,0,2:-2,2:-2].T, rstride=1, cstride=2, lw=0, facecolors=cm.viridis_r(zeta_2d[i,0,2:-2,2:-2].T), antialiased=True)
        plt.savefig(outname)"""

    # close hdf5 file
    f.close()



if __name__ == '__main__':
    #plotme()
    quick_plot(filename="../../Documents/Work/swerve/burning5")
