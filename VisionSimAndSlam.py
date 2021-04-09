import numpy as np
import random

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, PathPatch
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D

from PEBBLE import Module, CameraEvent, TrackerEvent

_SPACE_DIM = 3
_VOXEL_SIZE = 0.1
_MAP_DIMENSIONS = np.array([5., 5., 5.]) # In Meters
_GAUSSIAN_LOOKUP_TABLE_LOCATIONS = {1: [0.],
                                    2: [-0.6744897501961, 0.6744897501961],
                                    3: [-0.9674215661017, 0., 0.9674215661017],
                                    4: [-1.150349380376, -0.3186393639644, 0.3186393639644, 1.150349380376],
                                    5: [-1.281551565545, -0.5244005127080, 0., 0.5244005127080, 1.281551565545],
                                    6: [-1.382994127101, -0.6744897501961, -0.2104283942479, 0.2104283942479, 0.6744897501961, 1.382994127101]}
_MAX_T = 100.
_SIMULATED_MAP_DENSITY = 0.0003
_MAP_RELATIVE_BIGINING_OFFSET = np.array([0., 0.5, 0.5])

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
    def __init__(self, X = None, Thetas = None, XSigma = None, ThetasSigma = None, PreviousPose = None, Graphs = None, Marker = 'x'):
        if not PreviousPose is None:
            self = PreviousPose
            return None

        # Position values
        if not X is None:
            self.X = np.array(X)
        else:
            self.X = np.zeros(_SPACE_DIM)
        if not Thetas is None:
            self.Thetas = np.array(Thetas)
        else:
            self.Thetas = np.zeros(_SPACE_DIM) 
        if not XSigma is None:
            self.XSigma = np.array(XSigma)
        else:
            self.XSigma = 0.05 * np.ones(_SPACE_DIM) # This writing has semantic purpose.
        if not ThetasSigma is None:
            self.ThetasSigma = np.array(ThetasSigma)
        else:
            self.ThetasSigma = 0.01 * np.ones(_SPACE_DIM)

        # Speed values
        self.dThetas = np.zeros(_SPACE_DIM)
        self.dThetasSigma = _ANGULAR_SIGMA_INITIAL_SPD * np.ones(_SPACE_DIM)
        self.dX = np.zeros(_SPACE_DIM)
        self.dXSigma = _TRANSLATION_SIGMA_INITIAL_SPD * np.ones(_SPACE_DIM)

        self.UpToDate = False
        self.TransformationMatrix = np.zeros((3,3))

        self._Graphs = Graphs
        self._GraphsMarker = Marker
        self._GraphsDtUpdate = 0.02
        self._MaxGraphTimespan = 2.
        self.GraphsLastUpdate = -np.inf

    def UpdatePose(self, NewX, NewThetas, t = None):
        self.UpToDate = False
        self.X = NewX
        self.Thetas = NewThetas
        self._UpdateGraphs(t)
    def UpdatePoseX(self, NewX, t = None):
        self.UpToDate = False
        self.X = NewX
        self._UpdateGraphs(t)
    def UpdatePoseThetas(self, NewThetas, t = None):
        self.UpToDate = False
        self.Thetas = NewThetas
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
            if self.ThetasSigma[Dim] == 0.:
                self._Graphs[1].plot(t, self.Thetas[Dim], marker = self._GraphsMarker, color = ['r', 'g', 'b'][Dim])
            else:
                self._Graphs[1].errorbar(t, self.Thetas[Dim], self.ThetasSigma[Dim], color = ['r', 'g', 'b'][Dim])

        tMin = max(0, t - self._MaxGraphTimespan)
        self._Graphs[0].set_xlim(tMin,  t + self._GraphsDtUpdate/2)
        self._Graphs[1].set_xlim(tMin,  t + self._GraphsDtUpdate/2)
        plt.pause(0.001)

    def ComputeCenterVoxelLocation(self, CameraFrameVector):
        return self.X + self._ComputeTransformationMatrix().dot(CameraFrameVector)

    def _ComputeTransformationMatrix(self):
        if not self.UpToDate:
            Tx, Ty, Tw = self.Thetas
            Cx, Sx = np.cos(Tx), np.sin(Tx)
            Cy, Sy = np.cos(Ty), np.sin(Ty)
            Cw, Sw = np.cos(Tw), np.sin(Tw)
            self.TransformationMatrix = np.array([[Cx*Cy, Sx*Cy, Sy], 
                            [-Sx*Cw - Cx*Sy*Sw, Cx*Cw - Sx*Sy*Sw, Cy*Sw],
                            [Sx*Sw - Cx*Sy*Cw, -Sx*Sy*Cw - Cx*Sw, Cy*Cw]]).T
            self.UpToDate = True
        return self.TransformationMatrix
    def Uv(self):
        return self._ComputeTransformationMatrix()[:,0]
    def Ux(self):
        return self._ComputeTransformationMatrix()[:,1]
    def Uy(self):
        return self._ComputeTransformationMatrix()[:,2]

    def _ComputeLocalTransformationMatrix(self, LocalThetas):
        Tx, Ty, Tw = LocalThetas
        Cx, Sx = np.cos(Tx), np.sin(Tx)
        Cy, Sy = np.cos(Ty), np.sin(Ty)
        Cw, Sw = np.cos(Tw), np.sin(Tw)
        return np.array([[Cx*Cy, Sx*Cy, Sy], 
                        [-Sx*Cw - Cx*Sy*Sw, Cx*Cw - Sx*Sy*Sw, Cy*Sw],
                        [Sx*Sw - Cx*Sy*Cw, -Sx*Sy*Cw - Cx*Sw, Cy*Cw]]).T
            
    def ComputeVoxelsProbability(self, CameraFrameVector, Quantilation = 3):
        VoxelsOriginPoses = {}
        DX = np.array([0., 0., 0.])
        DTheta = np.array([0., 0., 0.])
        QuantileValue = 1. / (Quantilation**6)
        for nTx in range(Quantilation):
            DTheta[0] = _GAUSSIAN_LOOKUP_TABLE_LOCATIONS[Quantilation][nTx]
            for nTy in range(Quantilation):
                DTheta[1] = _GAUSSIAN_LOOKUP_TABLE_LOCATIONS[Quantilation][nTy]
                for nTw in range(Quantilation):
                    DTheta[2] = _GAUSSIAN_LOOKUP_TABLE_LOCATIONS[Quantilation][nTw]

                    LocalMatrix = self._ComputeLocalTransformationMatrix(self.Thetas + self.ThetasSigma*DTheta)

                    for nX in range(Quantilation):
                        DX[0] = _GAUSSIAN_LOOKUP_TABLE_LOCATIONS[Quantilation][nX]
                        for nY in range(Quantilation):
                            DX[1] = _GAUSSIAN_LOOKUP_TABLE_LOCATIONS[Quantilation][nY]
                            for nZ in range(Quantilation):
                                DX[2] = _GAUSSIAN_LOOKUP_TABLE_LOCATIONS[Quantilation][nZ]

                                LocalPoint = self.X + self.XSigma*DX + LocalMatrix.dot(CameraFrameVector)

                                LocalIndexes = tuple(GetVoxelIndexes(LocalPoint))
                                if LocalIndexes not in VoxelsOriginPoses.keys():
                                    VoxelsOriginPoses[LocalIndexes] = []
                                VoxelsOriginPoses[LocalIndexes] += [tuple(DX) + tuple(DTheta)]
        return VoxelsOriginPoses

class Map3DClass:
    def __init__(self, MapType = '', BiCameraSystem = None):
        self.Dimensions = np.array(_MAP_DIMENSIONS)
        self._MapType = MapType

        self.Voxels = 0.5*np.ones(tuple(np.array(self.Dimensions / _VOXEL_SIZE, dtype = int)))
        self.shape = self.Voxels.shape

        self.EventsGenerators = []

        if MapType:
            self._Density = _SIMULATED_MAP_DENSITY
            self.Voxels[:,:,:] = 0.

            if self._MapType == 'random':
                self._GenerateRandomMap(BiCameraSystem)
            if self._MapType == 'cubes':
                self._GenerateCubesMap(BiCameraSystem)

    def _GenerateRandomMap(self, BiCameraSystem):
        NVoxels = self.shape[0] * self.shape[1] * self.shape[2]
        NCube = int(self._Density * NVoxels)

        xs = np.random.randint(self.shape[0], size = NCube)
        ys = np.random.randint(self.shape[1], size = NCube)
        zs = np.random.randint(self.shape[2], size = NCube)

        self.BaseMap.Voxels[xs, ys, zs] = 1.
        for nObject in range(NCube):
            self.EventsGenerators += [EventGeneratorClass(nObject, np.array([xs[nObject], ys[nObject], zs[nObject]]) * _VOXEL_SIZE, np.array([xs, ys, zs]), BiCameraSystem, self.Voxels)]

    def _GenerateCubesMap(self, BiCameraSystem):
        LinearInterval = int((1. / self._Density)**(1./3))
        Locations = []
        for Dim in range(3):
            Locations += [list(range(0, self.shape[Dim], LinearInterval))]
        #print(Locations)
        for x in Locations[0]:
            for y in Locations[1]:
                for z in Locations[2]:
                    Indexes = np.array([x,y,z])
                    #print(Indexes)
                    X = Indexes * _VOXEL_SIZE
                    self.Voxels[x, y, z] = 1.
                    self.EventsGenerators += [EventGeneratorClass(len(self.EventsGenerators), X, np.array([x,y,z]), BiCameraSystem, self.Voxels)]

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
        self._TranslationSpeed = [0., 0., 0.]
        self._RotationSpeed = [0., 0., 0.]

        self._SingleCameraMode = False
        self._CreateTrackerEvents = True
        self._TrackersLocationGaussianNoise = 0.

        self._MaxStepsNoEvents = 100
        self._AddAxes = False

    def _InitializeModule(self, **kwargs):
        self.LogError("Events handling not updated to containers. Unable to proceed")

        if self._AddAxes:
            self.PoseGraphs, self.PoseAxs = plt.subplots(2,1)
            self.PoseAxs[0].set_title('X')
            self.PoseAxs[1].set_title('Thetas')
        else:
            self.PoseAxs = None
        self._BiCameraSystem = BiCameraClass(self.PoseAxs, self._CreateTrackerEvents, self._TrackersLocationGaussianNoise, self._SingleCameraMode)
        self.BaseMap = Map3DClass(self._MapType, self._BiCameraSystem)

        self.StreamName = self.__Framework__._GetStreamFormattedName(self)
        self.Geometry = BiCameraClass.Definition.tolist() + [2]

        self.nEvent = 0
        self.Current3DEvent = None
        self.T = 0.
        self._BiCameraSystem.Pose.UpdatePoseX(_MAP_DIMENSIONS * self._RelativeBeginingOffset, t = 0.)

        self.nSequenceStep = 0
        if not self._Sequence:
            self._TranslationSpeed = np.array(self._TranslationSpeed)
            self._RotationSpeed = np.array(self._RotationSpeed)
        else:
            self._TranslationSpeed = np.array(self._Sequence[self.nSequenceStep][0])
            self._RotationSpeed = np.array(self._Sequence[self.nSequenceStep][1])

        if (self._TranslationSpeed == 0).all() and (self._RotationSpeed == 0).all():
            self.LogError("Null speeds. No events will be produced.")
            return False

        return True

    def _OnEventModule(self, event):

        NoEventsSteps = 0
        while self.T < _MAX_T and not self._BiCameraSystem.Events and NoEventsSteps < self._MaxStepsNoEvents:
            self.T += self._dt
            self._BiCameraSystem.Pose.UpdatePose(self._BiCameraSystem.Pose.X + self._dt * self._TranslationSpeed, self._BiCameraSystem.Pose.Thetas + self._dt * self._RotationSpeed, t = self.T)

            for EG in self.BaseMap.EventsGenerators:
                EG.ComputeCameraLocations()
            NoEventsSteps += 1
            if len(self._Sequence) > self.nSequenceStep + 1:
                if self.T >= self._Sequence[self.nSequenceStep][2]:
                    self.nSequenceStep += 1
                    self._TranslationSpeed = np.array(self._Sequence[self.nSequenceStep][0])
                    self._RotationSpeed = np.array(self._Sequence[self.nSequenceStep][1])

        if not self._BiCameraSystem.Events:
            self.__Framework__.Running = False
            self.LogWarning("No displayed object in virtual scene left.")
            return None
        NewEvent = self._BiCameraSystem.Events.pop(0)
        NewEvent.timestamp = self.T

        self.Current3DEvent = self._BiCameraSystem.Events3D.pop(0)

        if self.__Framework__.Running:
            self.nEvent += 1
            if (self.nEvent & 2047) == 0:
                self.Log("Current pose:")
                self.Log(" X = {0}".format(self._BiCameraSystem.Pose.X))
                self.Log(" Thetas = {0}".format(self._BiCameraSystem.Pose.Thetas))
            return NewEvent
        else:
            return None

    def EventToEvent3DMatch(self, event):
        return self.Current3DEvent

    def GenerateBiCameraView(self):
        if not self.__Initialized__:
            self._BiCameraSystem = BiCameraClass()
            self.BaseMap = Map3DClass(self._MapType, self._BiCameraSystem)
            self._BiCameraSystem.Pose.UpdatePoseX(_MAP_DIMENSIONS * self._RelativeBeginingOffset, t = 0.)
        self.UpdateEventsGeneratorsLocations()

        return self._BiCameraSystem._GenerateView(self.BaseMap)

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
        self.EventRatio = 4.
        self.IgnoreObjectsCloserThan = 0.5 # In meters
        self.EpsilonSideVoxels = 0.1 # In meters

        self.OnBiCameraPresence = np.array([False, False])
        self.Occlusion = np.array([False, False])
        self.OnBiCameraLocations = [np.array([0., 0.]), np.array([0., 0.])]
        self.BiCamera3DDistance = np.array([0., 0.])
        self.OnBiCameraRadius = np.array([1, 1])
        self.BiCameraFocalDistance = 0.
        self.CameraFrame3DLocation = np.array([0., 0., 0.])

        self.CheckForOcclusions = False

    def ComputeCameraLocations(self, OnlyUpdateBoth = False):
        if self.BiCamera.SingleCameraMode:
            CamerasRandomOrder = [0]
        else:
            FirstCamera = random.randint(0,1)
            CamerasRandomOrder = [FirstCamera, 1-FirstCamera]
        Spikes = 0
        
        ObjectBiCameraVector = self.Location - self.BiCamera.Pose.X
        self.CameraFrame3DLocation = np.array([(ObjectBiCameraVector * self.BiCamera.Pose.Uv()).sum(),
                                              (ObjectBiCameraVector * self.BiCamera.Pose.Ux()).sum(),
                                              (ObjectBiCameraVector * self.BiCamera.Pose.Uy()).sum()])

        
        for CameraIndex in CamerasRandomOrder:
            BiCameraFrameObjectCameraVector = self.CameraFrame3DLocation - self.BiCamera.CameraFrameCamerasOffsets[CameraIndex]
            self.BiCamera3DDistance[CameraIndex] = np.linalg.norm(BiCameraFrameObjectCameraVector)
            self.BiCameraFocalDistance = BiCameraFrameObjectCameraVector[0]

            if self.BiCameraFocalDistance <= self.IgnoreObjectsCloserThan:
                self.OnBiCameraPresence[CameraIndex] = False
                continue
            
            if self.BiCamera.ProjectionMethod == 1:
                AxisScalars = BiCameraFrameObjectCameraVector[1:] / (self.BiCamera3DDistance[CameraIndex] * self.BiCamera.SinApertureValues)
                if (abs(AxisScalars) > 1.0).any():
                    self.OnBiCameraPresence[CameraIndex] = False
                    continue

                self.OnBiCameraPresence[CameraIndex] = True
                OnCameraLocation = self.BiCamera.Definition/2 * (np.array([-1, 1]) * AxisScalars + 1)
            elif self.BiCamera.ProjectionMethod == 0:
                Xi = self.BiCamera.K.dot(BiCameraFrameObjectCameraVector)
                OnCameraLocation = Xi[:-1] / Xi[-1]
                if (OnCameraLocation < 0).any() or (OnCameraLocation > self.BiCamera.Definition).any():
                    self.OnBiCameraPresence[CameraIndex] = False
                    continue
            else:
                raise Exception("Non implemented screen projection method")

            if self.CheckForOcclusions:
                self.OcclusionCheck(CameraIndex, StopAtFirstEncounter = True)

            HalfScalarAperture = ((_VOXEL_SIZE / 2) / self.BiCamera3DDistance[CameraIndex])
            self.OnBiCameraRadius[CameraIndex] = max(int(HalfScalarAperture / self.BiCamera.ScalarAperturePerPixel), 1)

            if (abs(OnCameraLocation - self.OnBiCameraLocations[CameraIndex]) > self.DisplacementEventTrigger / (2*np.pi*self.OnBiCameraRadius[CameraIndex] * self.EventRatio)).any():
                self.OnBiCameraLocations[CameraIndex] = OnCameraLocation
                Spikes += 1
                if not OnlyUpdateBoth:
                    self.Spike(CameraIndex)
        if Spikes:
            if not OnlyUpdateBoth and self.OnBiCameraPresence.all():
                self.Spike3D() # We have to add an offset due to the camera, as we want to give a 3D space location of the spike in the BiCamera system frame
            else:
                self.Spike3D(False)

            if Spikes == 2: # Create another 3D event incase both cameras fired
                self.Spike3D(False)

    def OcclusionCheck(self, CameraIndex, StopAtFirstEncounter = False):
        CameraLocation = self.BiCamera.Pose.X + self.BiCamera.CameraFrameCamerasOffsets[CameraIndex][1] * self.BiCamera.Pose.Ux()
        ObjectCameraVector = self.Location - CameraLocation
        NormalizedObjectCameraVector = ObjectCameraVector / np.linalg.norm(ObjectCameraVector)
        MainAxisXYZ = abs(ObjectCameraVector).argmax()

        Displacement = ObjectCameraVector / abs(ObjectCameraVector[MainAxisXYZ]) * _VOXEL_SIZE
        StartLocation = CameraLocation + NormalizedObjectCameraVector * self.IgnoreObjectsCloserThan
        Offset = (int(StartLocation[MainAxisXYZ] / _VOXEL_SIZE) * _VOXEL_SIZE - StartLocation[MainAxisXYZ]) * Displacement

        StartLocation = StartLocation + Offset
        Location  = np.array(StartLocation)
        VoxelIndex = None

        EspsilonDisplacements = [-self.EpsilonSideVoxels * Displacement, self.EpsilonSideVoxels*Displacement]
        VoxelsEncountered = []
        while VoxelIndex is None or ((VoxelIndex != self.Index).any() and (VoxelIndex >= 0).all() and (VoxelIndex < self.BaseVoxelsMap.shape).all()):
            Location = Location + Displacement
            for LocalDisplacement in EspsilonDisplacements:
                CurrentLocation = Location + LocalDisplacement
                CurrentVoxelIndex = np.minimum(np.array(self.BaseVoxelsMap.shape) - 1, GetVoxelIndexes(CurrentLocation))
                if (CurrentVoxelIndex == self.Index).all():
                    VoxelIndex = CurrentVoxelIndex
                    break
                if (CurrentVoxelIndex != VoxelIndex).any():
                    if (CurrentVoxelIndex < 0).any() or (CurrentVoxelIndex >= np.array(self.BaseVoxelsMap.shape)).any():
                        print("WARNING : Out of bounds")
                    VoxelIndex = CurrentVoxelIndex
                    VoxelsEncountered += [np.array(VoxelIndex)]
                    if StopAtFirstEncounter and self.BaseVoxelsMap[VoxelIndex[0], VoxelIndex[1], VoxelIndex[2]] != 0:
                        self.Occlusion[CameraIndex] = True
                        return True

        return VoxelsEncountered

    def Spike3D(self, HasSpike = True):
        if HasSpike:
            self.BiCamera.AddEvent3D(np.array(self.CameraFrame3DLocation))
        else:
            self.BiCamera.AddEvent3D(None)
    def Spike(self, CameraIndex):
        Angle = random.random() * np.pi * 2
        Loc = np.array(self.OnBiCameraLocations[CameraIndex] + self.OnBiCameraRadius[CameraIndex] * np.array([np.cos(Angle), np.sin(Angle)]), dtype = int)
        if (Loc < 0).any() or (Loc >= self.BiCamera.Definition).any():
            return None
        self.BiCamera.AddEvent(Loc, CameraIndex, self.OnBiCameraLocations[CameraIndex], self.ID)

class BiCameraClass: # Allows to emulate events from events generator
    Definition = np.array([640, 480])
    def __init__(self, PoseGraphs = None, CreateTrackerEvents = False, TrackerGaussianNoise = 0., SingleCameraMode = False):
        self.Pose = PoseClass(Graphs = PoseGraphs)
        self.CreateTrackerEvents = CreateTrackerEvents
        self.TrackersLocationGaussianNoise = TrackerGaussianNoise
        self.SingleCameraMode = SingleCameraMode

        self.InterCamDistance = 0.3
        self.HalfAngularApertureWidth = np.pi / 6

        self.ScalarAperturePerPixel = np.tan(self.HalfAngularApertureWidth) / self.Definition[0]
        self.HalfAngularApertureHeight = np.arctan(self.ScalarAperturePerPixel * self.Definition[1])

        self.SinApertureValues = np.array([np.sin(self.HalfAngularApertureWidth), np.sin(self.HalfAngularApertureHeight)])

        self.K = np.array([[self.Definition[0]/2, self.Definition[0], 0], [self.Definition[1]/2, 0, self.Definition[0]], [1, 0, 0]])

        self.ProjectionMethod = 0 # 0 for K matrix, 1 for scalar values

        self.CameraNames = ['Left', 'Right']
        if self.SingleCameraMode:
            self.CameraFrameCamerasOffsets = [np.array([0., 0., 0.])]
            self.CamerasIndexesUsed = [0]
        else:
            self.CameraFrameCamerasOffsets = [np.array([0., self.InterCamDistance/2, 0.]), np.array([0., -self.InterCamDistance/2, 0.])] # WARNING : Ux is oriented alog negative x cameras pixels values.
            self.CamerasIndexesUsed = [0,1]
        
        self.Events3D = []
        self.Events = []

    def AddEvent3D(self, CameraFrame3DLocation):
        self.Events3D += [CameraFrame3DLocation]
    def AddEvent(self, EventOnCameraLocation, CameraIndex, EGOnCameraLocation = None, EGID = None):
        NewEvent = Event(timestamp = None, location = EventOnCameraLocation, polarity = 0, SubStreamIndex = CameraIndex)#BiCameraSystem doesn't know the time.
        if self.CreateTrackerEvents:
            TrackerLocation = EGOnCameraLocation + np.random.normal(0, self.TrackersLocationGaussianNoise, size = 2)
            if not ((TrackerLocation < 0).any() or (TrackerLocation >= self.Definition).any()):
                NewEvent.Attach(TrackerEvent, TrackerLocation = TrackerLocation, TrackerID = EGID)
        self.Events += [NewEvent] 

    def _GenerateView(self, Map, Views = ['3d', 'Left', 'Right'], AlphaOffset = True):
        f = plt.figure()
        ax_3d = f.add_subplot(131, projection='3d')
        axs = [f.add_subplot(100 + 10*(1+len(self.CamerasIndexesUsed)) + 2+i) for i in range(len(self.CamerasIndexesUsed))]
        BiCameraMaps = np.zeros(list(reversed(self.Definition)) + [3, 2])

        for EG in Map.EventsGenerators:
            for CameraIndex in self.CamerasIndexesUsed:
                if not self.CameraNames[CameraIndex] in Views:
                    continue
                if EG.OnBiCameraPresence[CameraIndex]:
                    Center = np.array(EG.OnBiCameraLocations[CameraIndex], dtype = int)
                    PixelRadius = EG.OnBiCameraRadius[CameraIndex]
                    #print(PixelRadius, Center)
                    PixelRadius2 = PixelRadius**2
                    for dx in range(-PixelRadius, PixelRadius):
                        for dy in range(-PixelRadius, PixelRadius):
                            if dx**2 + dy**2 < PixelRadius2:
                                x = dx + Center[0]
                                y = dy + Center[1]
                                if x > 0 and y > 0 and x < self.Definition[0] and y < self.Definition[1]:
                                    if EG.Occlusion[CameraIndex].any():
                                        BiCameraMaps[y, x, 0, CameraIndex] = 1.
                                    else:
                                        BiCameraMaps[y, x, 1, CameraIndex] = 1.
        for CameraIndex in self.CamerasIndexesUsed:
            axs[CameraIndex].imshow(BiCameraMaps[:,:,:,CameraIndex], origin = 'lower')
            axs[CameraIndex].set_title(self.CameraNames[CameraIndex])

        if not '3d' in Views:
            return ax_3d, axs
        xs, ys, zs = np.where(Map.Voxels > 0.1)
        NVoxels = xs.shape[0]
        Colors = np.zeros((NVoxels, 4))
        Colors[:,2] = 1.
        if AlphaOffset:
            Colors[:,3] = np.maximum(0., Map.Voxels[(xs, ys, zs)] - 0.5) * 2
        else:
            Colors[:,3] = Map.Voxels[(xs, ys, zs)]

        ax_3d.scatter(xs, ys, zs, c = Colors)

        BiCameraCenter = self.Pose.X
        for CameraIndex in self.CamerasIndexesUsed:
            CameraLocation = (BiCameraCenter + self.Pose.Ux() * self.CameraFrameCamerasOffsets[CameraIndex][1]) / _VOXEL_SIZE
            for BaseVector, Color in zip([self.Pose.Uv() / _VOXEL_SIZE, self.Pose.Ux() / _VOXEL_SIZE, self.Pose.Uy() / _VOXEL_SIZE], ['r', 'g', 'b']):
                ax_3d.plot([CameraLocation[0], (CameraLocation + BaseVector)[0]], [CameraLocation[1], (CameraLocation + BaseVector)[1]], [CameraLocation[2], (CameraLocation + BaseVector)[2]], Color, lw = 5)
        plt.show()

        ax_3d.set_aspect('equal')
        ax_3d.set_xlabel('x')
        ax_3d.set_ylabel('y')
        ax_3d.set_zlabel('z')

        return axs, ax_3d
