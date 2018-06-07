import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm

def Plot3dSurface(M, ax = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    X = np.arange(M.shape[0])
    Y = np.arange(M.shape[1])

    X, Y = np.meshgrid(X, Y)

    surf = ax.plot_surface(X, Y, transpose(M), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()
    return surf
