import numpy as np
import random

import matplotlib.pyplot as plt

from Framework import Module, Event, TrackerEvent

_SPACE_DIM = 2
_VOXEL_SIZE = 0.01
_MAP_DIMENSIONS = np.array([5., 5.]) # In Meters
_GAUSSIAN_LOOKUP_TABLE_LOCATIONS = {1: [0.],
                                    2: [-0.6744897501961, 0.6744897501961],
                                    3: [-0.9674215661017, 0., 0.9674215661017],
                                    4: [-1.150349380376, -0.3186393639644, 0.3186393639644, 1.150349380376],
                                    5: [-1.281551565545, -0.5244005127080, 0., 0.5244005127080, 1.281551565545],
                                    6: [-1.382994127101, -0.6744897501961, -0.2104283942479, 0.2104283942479, 0.6744897501961, 1.382994127101]}
_MAX_T = 100.
_SIMULATED_MAP_DENSITY = 0.0003
_MAP_RELATIVE_BIGINING_OFFSET = np.array([0., 0.5])

# Movement estimators
_ANGULAR_SIGMA_ACC = 1. # rads/s^2
_TRANSLATION_SIGMA_ACC = 1. # m/s^2

_ANGULAR_SIGMA_INITIAL_SPD = 1.
_TRANSLATION_SIGMA_INITIAL_SPD = 1.

def GetVoxelIndexes(Location):
    return np.array(Location/_VOXEL_SIZE + 0.5, dtype = int)

def ComputeGaussian(xs, ws):
    n = ws.sum()
    mu = (xs*ws).sum() / n
    sigma2 = (ws * (xs - mu)**2).sum() / ((ws.shape[0] - 1.) / ws.shape[0] * n)
    return mu, np.sqrt(sigma2)

def ComputeAngularGaussian(thetas, ws):
# Tricky as its a circular distribution
# This version assumes that all angles are close

    # First put all angles into same quadrant
    RefTheta = thetas[0]
    Mod = np.array(thetas < RefTheta - np.pi, dtype = int) - np.array(thetas > RefTheta + np.pi, dtype = int)
    thetas = thetas + 2*np.pi * Mod
    
    theta_mean, theta_sigma = ComputeGaussian(thetas, ws)
    if theta_mean > np.pi:
        theta_mean -= 2*np.pi
    return theta_mean, theta_sigma

class PoseClass:
    def __init__(self, X = None, Theta = None, XSigma = None, ThetaSigma = None, PreviousPose = None, Graphs = None, Marker = 'x'):
        if not PreviousPose is None:
            self = PreviousPose
            return None

        # Position values
        if not X is None:
            self.X = np.array(X)
        else:
            self.X = np.zeros(_SPACE_DIM)
        if not Theta is None:
            self.Theta = Theta
        else:
            self.Theta = 0 
        if not XSigma is None:
            self.XSigma = np.array(XSigma)
        else:
            self.XSigma = 0.05 * np.ones(_SPACE_DIM) # This writing has semantic purpose.
        if not ThetaSigma is None:
            self.ThetaSigma = ThetaSigma
        else:
            self.ThetaSigma = 0.01

        # Speed values
        self.dTheta = np.zeros(_SPACE_DIM)
        self.dThetaSigma = _ANGULAR_SIGMA_INITIAL_SPD
        self.dX = np.zeros(_SPACE_DIM)
        self.dXSigma = _TRANSLATION_SIGMA_INITIAL_SPD * np.ones(_SPACE_DIM)

        self.UpToDate = False
        self.TransformationMatrix = np.identity(_SPACE_DIM)

        self._Graphs = Graphs
        self._GraphsMarker = Marker
        self._GraphsDtUpdate = 0.02
        self._MaxGraphTimespan = 2.
        self.GraphsLastUpdate = -np.inf

    def UpdatePose(self, NewX, NewTheta, t = None):
        self.UpToDate = False
        self.X = NewX
        self.Theta = NewTheta
        self._UpdateGraphs(t)
    def UpdatePoseX(self, NewX, t = None):
        self.UpToDate = False
        self.X = NewX
        self._UpdateGraphs(t)
    def UpdatePoseTheta(self, NewTheta, t = None):
        self.UpToDate = False
        self.Theta = NewTheta
        self._UpdateGraphs
        self._UpdateGraphs(t)

    def _UpdateGraphs(self, t):
        if self._Graphs is None or t < self.GraphsLastUpdate + self._GraphsDtUpdate:
            return None
        self.GraphsLastUpdate = t
        for Dim in range(3):
            if self.XSigma[Dim] == 0.:
                self._Graphs[0].plot(t, self.X[Dim], marker = self._GraphsMarker, color = ['r', 'g', 'b'][Dim])
            else:
                self._Graphs[0].errorbar(t, self.X[Dim], self.XSigma[Dim], color = ['r', 'g', 'b'][Dim])
            if self.ThetaSigma[Dim] == 0.:
                self._Graphs[1].plot(t, self.Theta[Dim], marker = self._GraphsMarker, color = ['r', 'g', 'b'][Dim])
            else:
                self._Graphs[1].errorbar(t, self.Theta[Dim], self.ThetaSigma[Dim], color = ['r', 'g', 'b'][Dim])

        tMin = max(0, t - self._MaxGraphTimespan)
        self._Graphs[0].set_xlim(tMin,  t + self._GraphsDtUpdate/2)
        self._Graphs[1].set_xlim(tMin,  t + self._GraphsDtUpdate/2)
        plt.pause(0.001)

    def ComputeCenterVoxelLocation(self, CameraFrameVector):
        return self.X + self._ComputeTransformationMatrix().dot(CameraFrameVector)

    def _ComputeTransformationMatrix(self):
        if not self.UpToDate:
            C, S = np.cos(self.Theta), np.sin(self.Theta)
            self.TransformationMatrix = np.array([[C,  S],
                                                  [-S, C]]).T
            self.UpToDate = True
        return self.TransformationMatrix
    def Uv(self):
        return self._ComputeTransformationMatrix()[:,0]
    def Ux(self):
        return self._ComputeTransformationMatrix()[:,1]

    def _ComputeLocalTransformationMatrix(self, LocalTheta):
        C, S = np.cos(LocalTheta), np.sin(LocalTheta)
        return np.array([[C,  S],
                        [-S, C]]).T

class Map2DClass:
    def __init__(self, MapType = '', BiCameraSystem = None):
        self.Dimensions = np.array(_MAP_DIMENSIONS)
        self._MapType = MapType

        self.Voxels = 0.5*np.ones(tuple(np.array(self.Dimensions / _VOXEL_SIZE, dtype = int)))
        self.shape = self.Voxels.shape

        self.EventsGenerators = []

        if MapType:
            self._Density = _SIMULATED_MAP_DENSITY
            self.Voxels[:,:] = 0.

            if self._MapType == 'random':
                self._GenerateRandomMap(BiCameraSystem)
            if self._MapType == 'cubes':
                self._GenerateCubesMap(BiCameraSystem)

    def _GenerateRandomMap(self, BiCameraSystem):
        NVoxels = self.shape[0] * self.shape[1]
        NCube = int(self._Density * NVoxels)

        xs = np.random.randint(self.shape[0], size = NCube)
        ys = np.random.randint(self.shape[1], size = NCube)

        self.BaseMap.Voxels[xs, ys] = 1.
        for nObject in range(NCube):
            self.EventsGenerators += [EventGeneratorClass(nObject, np.array([xs[nObject], ys[nObject]]) * _VOXEL_SIZE, np.array([xs, ys]), BiCameraSystem, self.Voxels)]

    def _GenerateCubesMap(self, BiCameraSystem):
        LinearInterval = int((1. / self._Density)**(1./_SPACE_DIM))
        Locations = []
        for Dim in range(_SPACE_DIM):
            Locations += [list(range(0, self.shape[Dim], LinearInterval))]
        #print(Locations)
        for x in Locations[0]:
            for y in Locations[1]:
                Indexes = np.array([x,y])
                #print(Indexes)
                X = Indexes * _VOXEL_SIZE
                self.Voxels[x, y] = 1.
                self.EventsGenerators += [EventGeneratorClass(len(self.EventsGenerators), X, np.array([x,y]), BiCameraSystem, self.Voxels)]

class MovementSimulatorClass(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to emulate stereo system moving with artificial map
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Input'

        self._MapType = 'cubes'
        self._dt = 0.0001
        self._RelativeBeginingOffset = np.array(_MAP_RELATIVE_BIGINING_OFFSET)

        self._Sequence = []
        self._TranslationSpeed = [0., 0.]
        self._RotationSpeed = 0.

        self._SingleCameraMode = False
        self._CreateTrackerEvents = True
        self._TrackersLocationGaussianNoise = 0.

        self._MaxStepsNoEvents = 100
        self._AddAxes = False

    def _InitializeModule(self, **kwargs):

        if self._AddAxes:
            self.PoseGraphs, self.PoseAxs = plt.subplots(2,1)
            self.PoseAxs[0].set_title('X')
            self.PoseAxs[1].set_title('Theta')
        else:
            self.PoseAxs = None
        self._BiCameraSystem = BiCameraClass(self, self.PoseAxs, self._CreateTrackerEvents, self._TrackersLocationGaussianNoise, self._SingleCameraMode)
        self.BaseMap = Map2DClass(self._MapType, self._BiCameraSystem)

        self.StreamName = self.__Framework__._GetStreamFormattedName(self)
        self.Geometry = BiCameraClass.Definition.tolist() + [2]

        self.nEvent = 0
        self.Current2DEvent = None
        self.T = 0.
        self._BiCameraSystem.Pose.UpdatePoseX(_MAP_DIMENSIONS * self._RelativeBeginingOffset, t = 0.)

        self.nSequenceStep = 0
        if not self._Sequence:
            self._TranslationSpeed = np.array(self._TranslationSpeed)
            self._RotationSpeed = self._RotationSpeed
        else:
            self._TranslationSpeed = np.array(self._Sequence[self.nSequenceStep][0])
            self._RotationSpeed = self._Sequence[self.nSequenceStep][1]

        if (self._TranslationSpeed == 0).all() and self._RotationSpeed == 0:
            self.LogError("Null speeds. No events will be produced.")
            return False

        return True

    def _OnEventModule(self, event):

        NoEventsSteps = 0
        while self.T < _MAX_T and not self._BiCameraSystem.Events and NoEventsSteps < self._MaxStepsNoEvents:
            self.T += self._dt
            self._BiCameraSystem.Pose.UpdatePose(self._BiCameraSystem.Pose.X + self._dt * self._TranslationSpeed, self._BiCameraSystem.Pose.Theta + self._dt * self._RotationSpeed, t = self.T)

            for EG in self.BaseMap.EventsGenerators:
                EG.ComputeCameraLocations()
            NoEventsSteps += 1
            if len(self._Sequence) > self.nSequenceStep + 1:
                if self.T >= self._Sequence[self.nSequenceStep][2]:
                    self.nSequenceStep += 1
                    self._TranslationSpeed = np.array(self._Sequence[self.nSequenceStep][0])
                    self._RotationSpeed = self._Sequence[self.nSequenceStep][1]

        if not self._BiCameraSystem.Events:
            self.__Framework__.Running = False
            self.LogWarning("No displayed object in virtual scene left.")
            return None
        NewEvent = self._BiCameraSystem.Events.pop(0)
        NewEvent.timestamp = self.T

        self.Current2DEvent = self._BiCameraSystem.Events2D.pop(0)

        if self.__Framework__.Running:
            self.nEvent += 1
            if (self.nEvent & 2047) == 0:
                self.Log("Current pose:")
                self.Log(" X = {0}".format(self._BiCameraSystem.Pose.X))
                self.Log(" Theta = {0}".format(self._BiCameraSystem.Pose.Theta))
            return NewEvent
        else:
            return None

    def EventToEvent2DMatch(self, event):
        return self.Current2DEvent

    def UpdateEventsGeneratorsLocations(self):
        for EG in self.BaseMap.EventsGenerators:
            EG.ComputeCameraLocations(self._BiCameraSystem)

class EventGeneratorClass:
    def __init__(self, ID, Location, Index, BiCamera, BaseVoxelsMap):
        self.ID = ID
        self.Location = Location
        self.Index = Index
        self.BiCamera = BiCamera
        self.BaseVoxelsMap = BaseVoxelsMap # Used for Occlusion

        self.DisplacementEventTrigger = 1. # In Px
        self.EventRatio = 2.
        self.IgnoreObjectsCloserThan = 0.5 # In meters
        self.EpsilonSideVoxels = 0.1 # In meters

        self.OnBiCameraPresence = np.array([False, False])
        self.Occlusion = np.array([False, False])
        self.OnBiCameraLocations = np.array([0., 0.])
        self.BiCamera2DDistance = np.array([0., 0.])
        self.OnBiCameraRadius = np.array([1, 1])
        self.BiCameraFocalDistance = 0.
        self.CameraFrame2DLocation = np.array([0., 0.])

        self.CheckForOcclusions = False

    def ComputeCameraLocations(self, OnlyUpdateBoth = False):
        if self.BiCamera.SingleCameraMode:
            CamerasRandomOrder = [0]
        else:
            FirstCamera = random.randint(0,1)
            CamerasRandomOrder = [FirstCamera, 1-FirstCamera]
        Spikes = 0
        
        ObjectBiCameraVector = self.Location - self.BiCamera.Pose.X
        self.CameraFrame2DLocation = np.array([(ObjectBiCameraVector * self.BiCamera.Pose.Uv()).sum(),
                                              (ObjectBiCameraVector * self.BiCamera.Pose.Ux()).sum()])

        for CameraIndex in CamerasRandomOrder:
            BiCameraFrameObjectCameraVector = self.CameraFrame2DLocation - self.BiCamera.CameraFrameCamerasOffsets[CameraIndex]
            self.BiCamera2DDistance[CameraIndex] = np.linalg.norm(BiCameraFrameObjectCameraVector)
            self.BiCameraFocalDistance = BiCameraFrameObjectCameraVector[0]

            if self.BiCameraFocalDistance <= self.IgnoreObjectsCloserThan:
                self.OnBiCameraPresence[CameraIndex] = False
                continue
            
            Xi = self.BiCamera.K.dot(BiCameraFrameObjectCameraVector)
            OnCameraLocation = Xi[:-1] / Xi[-1]
            if (OnCameraLocation < 0) or (OnCameraLocation > self.BiCamera.Definition[0]):
                if self.OnBiCameraPresence[CameraIndex]:
                    self.BiCamera.Module.Log("Tracker {0} went off screen for camera {1}".format(self.ID, CameraIndex))
                self.OnBiCameraPresence[CameraIndex] = False
                continue

            self.OnBiCameraPresence[CameraIndex] = True
            HalfScalarAperture = ((_VOXEL_SIZE / 2) / self.BiCamera2DDistance[CameraIndex])
            self.OnBiCameraRadius[CameraIndex] = max(int(HalfScalarAperture / self.BiCamera.ScalarAperturePerPixel), 1)

            if (abs(OnCameraLocation - self.OnBiCameraLocations[CameraIndex]) > self.DisplacementEventTrigger / (2*self.OnBiCameraRadius[CameraIndex] * self.EventRatio)).any():
                self.OnBiCameraLocations[CameraIndex] = OnCameraLocation
                Spikes += 1
                if not OnlyUpdateBoth:
                    self.Spike(CameraIndex)
        if Spikes:
            if not OnlyUpdateBoth and self.OnBiCameraPresence.all():
                self.Spike2D() # We have to add an offset due to the camera, as we want to give a 2D space location of the spike in the BiCamera system frame
            else:
                self.Spike2D(False)

            if Spikes == 2: # Create another 2D event incase both cameras fired
                self.Spike2D(False)

    def Spike2D(self, HasSpike = True):
        if HasSpike:
            self.BiCamera.AddEvent2D(np.array(self.CameraFrame2DLocation))
        else:
            self.BiCamera.AddEvent2D(None)
    def Spike(self, CameraIndex):
        Angle = random.random() * np.pi * 2
        Loc = np.array([self.OnBiCameraLocations[CameraIndex] + self.OnBiCameraRadius[CameraIndex] * 2 * (np.random.rand()-0.5), 0], dtype = int)
        if (Loc < 0).any() or (Loc >= self.BiCamera.Definition).any():
            return None
        self.BiCamera.AddEvent(Loc, CameraIndex, self.OnBiCameraLocations[CameraIndex], self.ID)

class BiCameraClass: # Allows to emulate events from events generator
    Definition = np.array([640, 1])
    def __init__(self, Module, PoseGraphs = None, CreateTrackerEvents = False, TrackerGaussianNoise = 0., SingleCameraMode = False):
        self.Module = Module
        self.Pose = PoseClass(Graphs = PoseGraphs)
        self.CreateTrackerEvents = CreateTrackerEvents
        self.TrackersLocationGaussianNoise = TrackerGaussianNoise
        self.SingleCameraMode = SingleCameraMode

        self.InterCamDistance = 0.3
        self.HalfAngularApertureWidth = np.pi / 6

        self.ScalarAperturePerPixel = np.tan(self.HalfAngularApertureWidth) / self.Definition[0]

        self.K = np.array([[self.Definition[0]/2, self.Definition[0]], [1, 0]])

        self.ProjectionMethod = 0 # 0 for K matrix, 1 for scalar values

        self.CameraNames = ['Left', 'Right']
        if self.SingleCameraMode:
            self.CameraFrameCamerasOffsets = [np.array([0., 0.])]
            self.CamerasIndexesUsed = [0]
        else:
            self.CameraFrameCamerasOffsets = [np.array([0., self.InterCamDistance/2]), np.array([0., -self.InterCamDistance/2])] # WARNING : Ux is oriented alog negative x cameras pixels values.
            self.CamerasIndexesUsed = [0,1]
        
        self.Events2D = []
        self.Events = []

    def AddEvent2D(self, CameraFrame2DLocation):
        self.Events2D += [CameraFrame2DLocation]
    def AddEvent(self, EventOnCameraLocation, CameraIndex, EGOnCameraLocation = None, EGID = None):
        NewEvent = Event(location = self.Definition - 1 - EventOnCameraLocation, polarity = 0, cameraIndex = CameraIndex)#BiCameraSystem doesn't know the time.
        if self.CreateTrackerEvents:
            TrackerLocation = np.array([self.Definition[0] - 1 - EGOnCameraLocation, 0.]) + np.array([np.random.normal(0, self.TrackersLocationGaussianNoise), 0.])
            if not ((TrackerLocation < 0).any() or (TrackerLocation >= self.Definition).any()):
                self.Events += [TrackerEvent(original = NewEvent, TrackerLocation = TrackerLocation, TrackerID = EGID)]
            else:
                self.Events += [NewEvent]
        else:
            self.Events += [NewEvent] 
