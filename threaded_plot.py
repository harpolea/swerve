import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
import tables as tb
import subprocess
import multiprocessing
from tkinter import *


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

    #fig = plt.figure(figsize=(12,10), facecolor='w', dpi=100)
    #ax = fig.gca(projection='3d')

    #print(np.shape(X), np.shape(Y), np.shape(D_2d[0,1,:,:].T))

    location = '/'.join(filename.split('/')[:-1])
    name = filename.split('/')[-1]

    number_of_processes = 6

    windows = []
    for i in range(number_of_processes):
        windows.append(Tk())

    # put the class inside here so can be lazy and not pass X,Y as member variables each time and therefore hopefully save some memory?

    def draw_canvases(canvases, axes, n_processes):
        #global ax,canvas
        for i in range(n_processes):
            fig = plt.figure(figsize=(12,10), facecolor='w', dpi=100)
            ax = fig.gca(projection='3d')
            canvas = FigureCanvasTkAgg(fig, master=windows[i])
            canvas.show()
            canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
            canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)
            #line, = ax.plot([1,2,3], [1,2,10])
            canvases.append(canvas)
            axes.append(ax)

    def data(q):
        for i in range(start, len(D_2d[:,0,0,0])):
            outname = location + '/plotting/' + name + '_' + format(i, '05') + '.png'
            D = D_2d[i,:,2:-2,2:-2].T
            zeta = zeta_2d[i,:,2:-2,2:-2].T
            q.put((outname, D, zeta))
        q.put('Q')

    def plot_me(q, process):
        try:

            dat = q.get_nowait()

            if dat != 'Q':
                outname, D, zeta = dat
                print('Printing {}'.format(outname))

                #fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
                #fig.set_dpi(100)
                #fig.set_size_inches(12,10)
                #fig.set_facecolor('w')
                ax = axes[process]
                ax.clear()
                ax.set_xlim(0,10)
                ax.set_ylim(0,10)
                ax.set_zlim(0.7,1.4)
                #print(np.shape(X), np.shape(self.D))
                ax.plot_surface(X,Y,D[:,:,1], rstride=1, cstride=2, lw=0, facecolors=cm.viridis_r(zeta[:,:,1]), antialiased=True)
                #ax.plot_wireframe(X,Y,D_2d[i,0,2:-2,2:-2].T, rstride=2, cstride=2, lw=0.1, cmap=cm.viridis, antialiased=True)
                ax.plot_surface(X,Y,D[:,:,0], rstride=1, cstride=2, lw=0, facecolors=cm.viridis_r(zeta[:,:,0]), antialiased=True)
                canvases[process].draw()

                plt.savefig(outname)

                windows[process].after(10, plot_me, q, process)
            else:
                print('done?')
        except:
            print('empty :(')
            windows[process].after(10, plot_me, q, process)

    q = multiprocessing.Queue()

    # submit tasks
    task_queue = data(q)

    canvases = []
    axes = []

    draw_canvases(canvases, axes, number_of_processes)

    for i in range(number_of_processes):
        print("Starting process {}".format(i))
        multiprocessing.Process(target=plot_me, args=(q,i)).start()

    #plotting = multiprocessing.Process(None, data, args=(q,))
    #plotting.start()


    #plot_me(q)
    for i in range(number_of_processes):
        windows[i].mainloop()
    print('Done?')


    """class myThread (threading.Thread):
        def __init__(self, threadID, outname, D_2d, zeta_2d, queue):
            threading.Thread.__init__(self)
            self.threadID = threadID
            self.outname = outname
            self.D = D_2d
            self.zeta = zeta_2d
            self.queue = queue
        def run(self):
            print( "Starting " + format(self.threadID))

            while not exit_flag:
                queue_lock.acquire()

                if not work_queue.empty():
                    print("Plotting {}".format(self.threadID))
                    queue_obj = self.queue.get()
                    queue_lock.release()
                    # make plot and save
                    #fig = plt.figure(figsize=(12,10), facecolor='w', dpi=100)
                    #ax = fig.gca(projection='3d')
                    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
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

                else:
                    queue_lock.release()

            print( "Exiting " + format(self.threadID))"""

    """exit_flag = 0
    queue_lock = threading.Lock()
    work_queue = queue.Queue()
    threads = []
    for i in range(start, len(D_2d[:,0,0,0])):
        outname = location + '/plotting/' + name + '_' + format(i, '05') + '.png'
        thread = myThread(i, outname, D_2d[i,:,2:-2,2:-2].T, zeta_2d[i,:,2:-2,2:-2].T, work_queue)
        thread.start()
        threads.append(thread)

    # Fill queue
    queue_lock.acquire()
    for i in range(start, len(D_2d[:,0,0,0])):
        work_queue.put(i)
    queue_lock.release()

    # wait for queue to empty
    while not work_queue.empty():
        pass

    # notify threads it's time to exit
    exit_flag = 1

    # wait for them to finish
    for t in threads:
        t.join()"""


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
    quick_plot(filename="../../Documents/Work/swerve/burning5", start=109)
