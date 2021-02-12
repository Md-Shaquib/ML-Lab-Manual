import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from matplotlib import gridspec
import numpy as np 

def sig_f(w, b, x):
    return 1.0/(1.0 + np.exp(-(w*x+b)))

def error(w,b):
    err = 0.0
    for x,y in zip(X,Y):
        f_x= sig_f(w,b,x)
        err += 0.5+(f_x -y)**2
    return err

def grad_b(w,b,x,y):
    f_x= sig_f(w,b,x)
    return (f_x - y)* f_x*(1-f_x)

def grad_w(w,b,x,y):
    f_x= sig_f(w,b,x)
    return (f_x - y)* f_x*(1-f_x)*x

def do_grad_desc(w, b, eta, epoch):
    err=[ ]
    wt=[ ]
    bs=[ ]
    for i in range (epoch):
        dw, db = 0,0
        for x,y in zip(X,Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
        w = w-eta*dw
        b= b - eta*db
        err.append(error(w,b))
        wt.append(w)
        bs.append(b)
    return err, wt, bs

def plot_error_surface(biases, weights, errors):
    axis = np.linspace(-3, 3, 100, dtype=np.float)
    s_error, wt_range, bs_range = [ ], [ ], [ ]
    for wt in axis:
        for bs in axis:
            wt_range.append(wt)
            bs_range.append(bs)
            s_error.append(error(wt, bs))
    
    e_surface = np.reshape(np.array(s_error), (axis.shape[0], axis.shape[0]))
    _X = np.reshape(np.array(wt_range), (axis.shape[0], axis.shape[0]))
    _Y = np.reshape(np.array(bs_range), (axis.shape[0], axis.shape[0]))
    
    # plot
    fig = plt.figure(figsize=(14,8))
    gs = gridspec.GridSpec(2, 3, wspace=0.4, hspace=0.3) 
    ax = fig.add_subplot(gs[0:2, 0:2], projection='3d')
    ax.view_init(20, azim = -20)
    ax.scatter(weights, biases, errors, c = 'r', marker = '.')
    surf = ax.plot_surface(_X, _Y, e_surface, 
                cmap=cm.coolwarm,
                linewidth=0,
                antialiased=False,
                alpha = 0.5)
    
    # Customize the z axis.
    ax.set_zlim(-1, 1)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Set labels
    ax.set_xlabel(r'weight')
    ax.set_ylabel(r'bias')
    ax.set_zlabel(r'Error')
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, orientation="horizontal", shrink=0.6, aspect=10)
    
    ax1 = fig.add_subplot(gs[0, 2])
    ax1.scatter(weights, biases, errors, c = 'r',  marker = 'o', lw=3)
    ax1.contourf(_X, _Y, e_surface, 20,
                 cmap=cm.coolwarm,
                 alpha = 0.5)
    ax1.set_xlabel(r'weight')
    ax1.set_ylabel(r'bias')
    
    ax2 = fig.add_subplot(gs[1, 2])
    #eps = np.linspace(0, 200, 200, dtype=np.int)
    #ax2.scatter(eps, errors, c = 'b',  marker = '.', alpha=0.5)
    ax2.plot(errors,'r-', lw = 2.0)
    ax2.set_xlabel(r'epoch')
    ax2.set_ylabel(r'error')
    
    plt.show()
    fig.savefig('Error_Surface.pdf')

 

X = [0.5, 2.5]
Y = [0.2, 0.9]
w1, w0, l_rate, n_epoch = -2, -2, 1.0, 200
errors, wts, bias = do_grad_desc(w1, w0, l_rate, n_epoch)
plot_error_surface(bias, wts, errors)
