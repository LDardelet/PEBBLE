import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, PathPatch
# register Axes3D class with matplotlib by importing Axes3D
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D

class Map3D:
    def __init__(self, FlowManager, Z_speed = 20.):
        self.PointsList = []
        self.FlowManager = FlowManager

        Center = np.array(FlowMap.shape[:2])/2

        self.Defined_z_speed = Z_speed
        self.Pixel_angular_aperture = 0.1

        self.CurrentZ = 0.
        self.InitialTs = None

    def OnEvent(self, event, R_threshold = 0.95):
        R_Value = self.FlowManager.RegValue[event.location[0], event.location[1]]
        if R_Value < R_threshold:
            return None

        if self.InitialTs == None:
            self.InitialTs = event.timestamp
        self.CurrentZ = (event.timestamp - self.InitialTs) * self.Defined_z_speed

        centered_location = event.location - self.Center

        Flow = np.array(self.FlowManager.FlowMap[event.location[0], event.location[1], :])
        FlowNorm = np.sqrt(Flow[0]**2 + Flow[1]**2)
        NormalUnitaryVector = np.array([-Flow[1]/FlowNorm, Flow[0]/FlowNorm])

        D = -self.Defined_z_speed*(centered_location[1]*NormalUnitaryVector[0] - centered_location[0]*NormalUnitaryVector[1])/FlowNorm

        if D < 0:
            print D

        Thetas = self.Pixel_angular_aperture * centered_location
        X, Y = np.tan(Thetas)*D

        FinalPosition = np.array([X, Y, self.CurrentZ + D])

        self.PointsList += [FinalPosition]

def DrawRectangle(CenterPosition, Variances, Alpha, ax = None, Limits = None):
    if ax == None:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.axis('equal')
        #ax.autoscale(True)
    Boundaries = np.array([CenterPosition - Variances, CenterPosition + Variances], dtype = float)
    if Limits is None:
        print "Generating new boundaries"
        Limits = np.array(Boundaries)
    else:
        print "Updating current boundaries"
        Limits[0,:] = np.minimum(Limits[0,:], Boundaries[0,:])
        Limits[1,:] = np.maximum(Limits[1,:], Boundaries[1,:])
    Lenghts = 2*Variances

    nAxes = range(3)
    axes = ['x', 'y', 'z']
    for n_axis in nAxes:
        Remainings = list(nAxes)
        Remainings.pop(n_axis)

        r1 = Rectangle((Boundaries[0, Remainings[0]],Boundaries[0, Remainings[1]]), Lenghts[Remainings[0]],Lenghts[Remainings[1]], alpha = Alpha)
        ax.add_patch(r1)
        art3d.pathpatch_2d_to_3d(r1, z=Boundaries[0, n_axis], zdir=axes[n_axis])
        r2 = Rectangle((Boundaries[0, Remainings[0]],Boundaries[0, Remainings[1]]), Lenghts[Remainings[0]],Lenghts[Remainings[1]], alpha = Alpha)
        ax.add_patch(r2)
        art3d.pathpatch_2d_to_3d(r2, z=Boundaries[1, n_axis], zdir=axes[n_axis])

    DeltaMax = (Limits[1,:] - Limits[0,:]).max()
    Centers = (Limits[1,:] + Limits[0,:])/2
    Borders = np.array([Centers - DeltaMax/2, Centers + DeltaMax/2])
    print "Limits : "
    print Limits
    print "Borders : "
    print Borders

    functions = [ax.set_xlim, ax.set_ylim, ax.set_zlim]
    for nfunction in range(3):
        functions[nfunction](Borders[0, nfunction], Borders[1, nfunction])

    plt.show()
    return ax, Limits
