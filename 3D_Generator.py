import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, PathPatch
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D

class Map3D:
    def __init__(self, argsCreationDict):
        '''
        Creates a 3d Map from the optical flow, incase of a constant translation along the optical axis of the camera.
        Expects:
        'FlowComputer.Self' as 'FlowComputer' -> Accesses the diverse optical flow results
        '''
        for key in argsCreationDict.keys():
            self.__dict__[key.split('.')[2]] = argsCreationDict[key]

        self.Z_speed = 1.
        self.Pixel_angular_aperture = 0.00245
        self.R_threshold = 0.92
        self.N_threshold = 20

        self._Type = 'Computation'

    def _Initialize(self, argsInitializationDict):
        '''
        Expects:
        '''

        self.PointsList = []
        self.PointsVariance = []

        self.Center = np.array(self.FlowComputer.FlowMap.shape[:2])/2

        self.CurrentZ = 0.
        self.InitialTs = None

        self.PointsCreationFailuresReasons = {'R':0, 'N' :0, 'Dneg': 0, 'Outside':0}

    def _OnEvent(self, event):
        R_Value = self.FlowComputer.RegMap[event.location[0], event.location[1], event.polarity]
        if R_Value < self.R_threshold:
            self.PointsCreationFailuresReasons['R'] += 1
            return event
        N_Value = self.FlowComputer.NEventsMap[event.location[0], event.location[1], event.polarity]
        if N_Value < self.N_threshold:
            self.PointsCreationFailuresReasons['N'] += 1
            return event


        if self.InitialTs is None:
            self.InitialTs = event.timestamp
        self.CurrentZ = (event.timestamp - self.InitialTs) * self.Z_speed

        centered_location = event.location - self.Center

        Flow = np.array(self.FlowComputer.FlowMap[event.location[0], event.location[1], event.polarity, :])
        FlowNorm = self.FlowComputer.NormMap[event.location[0], event.location[1], event.polarity]
        NormalUnitaryVector = np.array([-Flow[1]/FlowNorm, Flow[0]/FlowNorm])

        D = -self.Z_speed*(centered_location[1]*NormalUnitaryVector[0] - centered_location[0]*NormalUnitaryVector[1])/FlowNorm

        if D < 0:
            self.PointsCreationFailuresReasons['Dneg'] += 1
            return event

        Thetas = self.Pixel_angular_aperture * centered_location
        X, Y = np.tan(Thetas)*D

        FinalPosition = np.array([Y, self.CurrentZ + D, X])

        if (abs(FinalPosition) < 20).all():
            self.PointsList += [FinalPosition]
            self.PointsVariance += [0.05*D]
        else:
            self.PointsCreationFailuresReasons['Outside'] += 1

        return event

    def PlotCurrentMap(self, ):
        default_variances = np.array([1.,1.,1.])
        default_alpha_value = 0.01

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        Limits = None
        ax.set_xlabel('Transverse')
        ax.set_ylabel('Depth')
        ax.set_zlabel('Height')

        for Point, PointVariance in zip(self.PointsList, self.PointsVariance):
            Variance  = default_variances*PointVariance
            ax, Limits = DrawRectangle(Point, Variance, default_alpha_value, ax, Limits)
        return ax, Limits

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
