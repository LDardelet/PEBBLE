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
_SIMULATED_MAP_DENSITY = 0.0002
_MAP_RELATIVE_BEGINING_OFFSET = np.array([0., 0.])
_MAP_OBJECTS_CENTER = 'visible'

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

class BaseMoveClass:
    def __init__(self, Duration):
        self.Duration = Duration
        self.T = 0
    
    def Start(self, PoseClass):
        pass

    def Step(self, PoseClass, dt):
        PoseClass.UpToDate = False
        self.T += dt
        self.SpecialMovement(PoseClass, dt)
        return self.T < self.Duration

class RotationClass(BaseMoveClass):
    def __init__(self, Duration, AngularVelocity, Center, ReferenceFrame):
        self.Center = Center # Center can be a np array or 'camera'
        if AngularVelocity == 0.:
            raise Exception("Null rotational speed")
        self.AngularVelocity = AngularVelocity
        self.ReferenceFrame = ReferenceFrame
        BaseMoveClass.__init__(self, Duration)

    def Start(self, PoseClass):
        BaseMoveClass.Start(self, PoseClass)

    def SpecialMovement(self, PoseClass, dt):
        AngularVar = dt * self.AngularVelocity
        X = PoseClass.CameraToWorldLocation(0)
        if self.ReferenceFrame == 'world':
            R = self.Center - X
        elif self.ReferenceFrame == 'camera':
            R = PoseClass.CameraToWorldVector(self.Center)
        PoseClass.Theta += AngularVar
        PoseClass.UpToDate = False
        PoseClass.T = PoseClass.WorldToCameraVector(X + AngularVar * np.array([R[1], -R[0]]))

class TranslationClass(BaseMoveClass):
    def __init__(self, Duration, TranslationSpeed, ReferenceFrame):
        if (TranslationSpeed == 0).all():
            raise Exception("Null translational speed")
        self.TranslationSpeed = TranslationSpeed # Center can be a np array or 'camera'
        self.ReferenceFrame = ReferenceFrame
        BaseMoveClass.__init__(self, Duration)

    def Start(self, PoseClass):
        BaseMoveClass.Start(self, PoseClass)

    def SpecialMovement(self, PoseClass, dt):
        if self.ReferenceFrame == 'camera':
            PoseClass.T += self.TranslationSpeed * dt
        elif self.ReferenceFrame == 'world':
            PoseClass.T += dt * (PoseClass.WorldToCameraVector(self.TranslationSpeed))

class PoseClass:
    def __init__(self, T = None, Theta = None, PreviousPose = None, Marker = 'x'):
        if not PreviousPose is None:
            self = PreviousPose
            return None

        # Position values
        if not T is None:
            self.T = np.array(X)
        else:
            self.T = np.zeros(_SPACE_DIM)
        if not Theta is None:
            self.Theta = Theta
        else:
            self.Theta = 0 

        self.UpToDate = False
        self._RT = np.zeros((2, 3))
        self._R = np.identity(2) # Reports world vector in camera frame vectors !

    def R(self):
        if not self.UpToDate:
            C, S = np.cos(self.Theta), np.sin(self.Theta)
            self._R = np.array([[C,-S],
                                [S, C]])
            self.UpToDate = True
        return self._R

    def RT(self):
        self._RT[:, :2] = self.R().T
        self._RT[:,2] = -self.T
        return self._RT

    def CameraToWorldVector(self, Vector):
        return self.R().dot(Vector)
    def CameraToWorldLocation(self, Location):
        return self.CameraToWorldVector(Location + self.T)
    def WorldToCameraVector(self, Vector):
        return self.R().T.dot(Vector)
    def WorldToCameraLocation(self, Location):
        return self.WorldToCameraVector(Location) - self.T

    def _ComputeLocalTransformationMatrix(self, LocalTheta):
        C, S = np.cos(LocalTheta), np.sin(LocalTheta)
        return np.array([[C,-S],
                         [S, C]])

class Map2DClass:
    def __init__(self, MapType = '', BiCameraSystem = None):
        self.Dimensions = np.array(_MAP_DIMENSIONS)
        self._MapType = MapType

        if _MAP_OBJECTS_CENTER == 'visible':
            Uv = BiCameraSystem.Pose.CameraToWorldVector(np.array([0., 1.]))
            self.ObjectsCenter = (self.Dimensions / 2).dot(Uv) * Uv
        elif _MAP_OBJECTS_CENTER == 'centered':
            self.ObjectsCenter = np.array([0., 0.])

        self.Voxels = np.zeros(tuple(np.array(self.Dimensions / _VOXEL_SIZE, dtype = int)))
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
            self.EventsGenerators += [EventGeneratorClass(nObject, np.array([xs[nObject], ys[nObject]]) * _VOXEL_SIZE + self.ObjectsCenter, np.array([xs, ys]), BiCameraSystem, self.Voxels)]

    def _GenerateCubesMap(self, BiCameraSystem):
        LinearInterval = int((1. / self._Density)**(1./_SPACE_DIM))
        Locations = []
        for Dim in range(_SPACE_DIM):
            Locations += [list(range(int(LinearInterval/2), self.shape[Dim], LinearInterval))]
        #print(Locations)
        for x in Locations[0]:
            for y in Locations[1]:
                Indexes = np.array([x,y])
                #print(Indexes)
                X = Indexes * _VOXEL_SIZE - _MAP_DIMENSIONS / 2 + self.ObjectsCenter
                self.Voxels[x, y] = 1.
                self.EventsGenerators += [EventGeneratorClass(len(self.EventsGenerators), X, np.array([x,y]), BiCameraSystem, self.Voxels)]

    def Plot(self, fax = None, CameraPresence = 'any'):
        if fax is None:
            f, ax = plt.subplots(1,1)
        else:
            f, ax = fax
        ax.set_xlim((self.ObjectsCenter - 1.05*self.Dimensions/2)[0], (self.ObjectsCenter + 1.05*self.Dimensions/2)[0])
        ax.set_ylim((self.ObjectsCenter - 1.05*self.Dimensions/2)[1], (self.ObjectsCenter + 1.05*self.Dimensions/2)[1])
        for EG in self.EventsGenerators:
            if (CameraPresence == 'any' and EG.OnBiCameraPresence.any()) or (CameraPresence == 'all' and EG.OnBiCameraPresence.all()) or (CameraPresence == 'none'):
                c = 'b'
            else:
                c = 'r'
            ax.plot(EG.Location[0], EG.Location[1], marker = 'o', color = c)
            ax.text(EG.Location[0]+0.05, EG.Location[1], str(EG.ID), color = c)
        return f, ax


class MovementSimulatorClass(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to emulate stereo system moving with artificial map
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Input'

        self._MapType = 'cubes'
        self._dt = 0.0001
        self._RelativeBeginingOffset = np.array(_MAP_RELATIVE_BEGINING_OFFSET)

        self._Sequence = []

        self._SingleCameraMode = False
        self._CreateTrackerEvents = True
        self._TrackersLocationGaussianNoise = 0.

        self._MaxStepsNoEvents = 100
        self._AddAxes = False

        self.Translation = TranslationClass
        self.Rotation = RotationClass

    def _InitializeModule(self, **kwargs):

        self.BiCameraSystem = BiCameraClass(self, self._CreateTrackerEvents, self._TrackersLocationGaussianNoise, self._SingleCameraMode)
        self.BaseMap = Map2DClass(self._MapType, self.BiCameraSystem)

        self.StreamName = self.__Framework__._GetStreamFormattedName(self)
        self.Geometry = BiCameraClass.Definition.tolist() + [2]

        self.nEvent = 0
        self.Current2DEvent = None
        self.T = 0.
        self.BiCameraSystem.Pose.T = self.BiCameraSystem.Pose.WorldToCameraLocation(_MAP_DIMENSIONS * self._RelativeBeginingOffset)
        self.BiCameraSystem.Pose.UpToDate = False

        try:
            self.CurrentSequenceStep = self._Sequence.pop(0)
        except IndexError:
            self.LogError("Movement Sequence is empty. Aborting run")
            return False
        self.CurrentSequenceStep.Start(self.BiCameraSystem.Pose)

        return True

    def _OnEventModule(self, event):

        NoEventsSteps = 0
        while self.T < _MAX_T and not self.BiCameraSystem.Events and NoEventsSteps < self._MaxStepsNoEvents:
            self.T += self._dt
            if not self.CurrentSequenceStep.Step(self.BiCameraSystem.Pose, self._dt):
                try:
                    self.CurrentSequenceStep = self._Sequence.pop(0)
                except IndexError:
                    self.__Framework__.Running = False
                    self.LogWarning("End of Sequence reached")
                    return None
                self.CurrentSequenceStep.Start(self.BiCameraSystem.Pose)
                self.Log("Next sequence step")

            RandomGenerators = list(self.BaseMap.EventsGenerators)
            random.shuffle(RandomGenerators)
            for EG in RandomGenerators:
                EG.ComputeCameraLocations()

        if not self.BiCameraSystem.Events:
            self.__Framework__.Running = False
            self.LogWarning("No displayed object in virtual scene left.")
            return None
        NewEvent = self.BiCameraSystem.Events.pop(0)
        NewEvent.timestamp = self.T

        self.Current2DEvent = self.BiCameraSystem.Events2D.pop(0)

        if self.__Framework__.Running:
            self.nEvent += 1
            if (self.nEvent & 2047) == 0:
                self.Log("Current pose:")
                self.Log(" T = {0}, X = {1}".format(self.BiCameraSystem.Pose.T, self.BiCameraSystem.Pose.CameraToWorldLocation(np.array([0., 0.]))))
                self.Log(" Theta = {0}".format(self.BiCameraSystem.Pose.Theta))
            return NewEvent
        else:
            return None

    def EventToEvent2DMatch(self, event):
        return self.Current2DEvent

    def UpdateEventsGeneratorsLocations(self):
        for EG in self.BaseMap.EventsGenerators:
            EG.ComputeCameraLocations(self.BiCameraSystem)

    def Plot(self, fax = None, CameraPresence = 'any'):
        if fax is None:
            f, ax = plt.subplots(1,1)
        else:
            f, ax = fax
        ax.set_aspect('equal')
        self.BaseMap.Plot((f, ax), CameraPresence)
        self.BiCameraSystem.Plot((f, ax))
        return f, ax

class EventGeneratorClass:
    def __init__(self, ID, Location, Index, BiCamera, BaseVoxelsMap):
        self.ID = ID
        self.Location = Location
        self.HLocation = np.concatenate((self.Location, [1]))
        self.Index = Index
        self.BiCamera = BiCamera
        self.BaseVoxelsMap = BaseVoxelsMap # Used for Occlusion

        self.DisplacementEventTrigger = 1. # In Px
        self.EventRatio = 5.
        self.IgnoreObjectsCloserThan = 0.2 # In meters
        self.EpsilonSideVoxels = 0.1 # In meters

        self.OnBiCameraPresence = np.array([False, False])
        self.Occlusion = np.array([False, False])
        self.OnBiCameraLocations = np.array([0., 0.])
        self.BiCamera2DDistance = np.array([0., 0.])
        self.OnBiCameraRadius = np.array([1, 1])
        self.BiCameraFocalDistance = 0.
        self.BiCameraFrame2DLocation = np.array([0., 0.])

        self.CheckForOcclusions = False

    def ComputeCameraLocations(self, OnlyUpdateBoth = False):
        if self.BiCamera.SingleCameraMode:
            CamerasRandomOrder = [0]
        else:
            FirstCamera = random.randint(0,1)
            CamerasRandomOrder = [FirstCamera, 1-FirstCamera]
        Spikes = 0
        
        self.BiCameraFrame2DLocation = self.BiCamera.Pose.WorldToCameraLocation(self.Location)

        for CameraIndex in CamerasRandomOrder:
            self.BiCameraFocalDistance = self.BiCameraFrame2DLocation[1]

            if self.BiCameraFocalDistance <= self.IgnoreObjectsCloserThan:
                self.OnBiCameraPresence[CameraIndex] = False
                continue
            
            Xi = self.BiCamera.KRT().dot(self.HLocation)
            OnCameraLocation = Xi[:-1] / Xi[-1]
            if (OnCameraLocation < 0) or (OnCameraLocation > self.BiCamera.Definition[0]):
                if self.OnBiCameraPresence[CameraIndex]:
                    self.BiCamera.Module.Log("Tracker {0} went off screen for camera {1}".format(self.ID, CameraIndex))
                self.OnBiCameraPresence[CameraIndex] = False
                continue

            self.OnBiCameraPresence[CameraIndex] = True
            XiSide = self.BiCamera.K.dot(self.BiCameraFrame2DLocation - self.BiCamera.CameraFrameCamerasOffsets[CameraIndex] + np.array([_VOXEL_SIZE, 0.]))
            ApparentRadius = abs(XiSide[0] / XiSide[-1] - OnCameraLocation) # Should be > 0 anyway, but to make sure
            self.OnBiCameraRadius[CameraIndex] = max(ApparentRadius, 1)

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
            self.BiCamera.AddEvent2D(np.array(self.BiCameraFrame2DLocation))
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
    def __init__(self, Module, CreateTrackerEvents = False, TrackerGaussianNoise = 0., SingleCameraMode = False):
        self.Module = Module
        self.Pose = PoseClass()
        self.CreateTrackerEvents = CreateTrackerEvents
        self.TrackersLocationGaussianNoise = TrackerGaussianNoise
        self.SingleCameraMode = SingleCameraMode

        self.InterCamDistance = 0.3

        self.K = np.array([[self.Definition[0], self.Definition[0]/2], [0, 1]])

        self.CameraNames = ['Left', 'Right']
        if self.SingleCameraMode:
            self.CameraFrameCamerasOffsets = [np.array([0., 0.])]
            self.CamerasIndexesUsed = [0]
        else:
            self.CameraFrameCamerasOffsets = [np.array([-self.InterCamDistance/2, 0.]), np.array([self.InterCamDistance/2, 0.])]
            self.CamerasIndexesUsed = [0,1]
        
        self.Events2D = []
        self.Events = []

    def KRT(self):
        return self.K.dot(self.Pose.RT())

    def AddEvent2D(self, BiCameraFrame2DLocation):
        self.Events2D += [BiCameraFrame2DLocation]
    def AddEvent(self, EventOnCameraLocation, CameraIndex, EGOnCameraLocation = None, EGID = None):
        NewEvent = Event(timestamp = None, location = self.Definition - 1 - EventOnCameraLocation, polarity = 0, cameraIndex = CameraIndex)#BiCameraSystem doesn't know the time.
        if self.CreateTrackerEvents:
            TrackerLocation = np.array([EGOnCameraLocation, 0.]) + np.array([np.random.normal(0, self.TrackersLocationGaussianNoise), 0.])
            if not ((TrackerLocation < 0).any() or (TrackerLocation >= self.Definition).any()):
                NewEvent.Attach(TrackerEvent, TrackerLocation = TrackerLocation, TrackerID = EGID)
        self.Events += [NewEvent] 

    def Plot(self, fax):
        if fax is None:
            f, ax = plt.subplots(1,1)
        else:
            f, ax = fax
        for CamOffset in self.CameraFrameCamerasOffsets:
            CamLoc = self.Pose.CameraToWorldLocation(CamOffset)
            Ux, Uv = self.Pose.CameraToWorldVector(np.array([1., 0.])), self.Pose.CameraToWorldVector(np.array([0., 1.]))
            ax.plot(CamLoc[0], CamLoc[1], 'og')
            ax.plot([CamLoc[0], (CamLoc+Uv)[0]], [CamLoc[1], (CamLoc+Uv)[1]], 'g')
            ax.plot((CamLoc-Ux)[0], (CamLoc-Ux)[1], 'ob')
            ax.plot([(CamLoc-Ux)[0], (CamLoc+Ux)[0]], [(CamLoc-Ux)[1], (CamLoc+Ux)[1]], 'b')
        return f, ax
