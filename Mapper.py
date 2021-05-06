from PEBBLE import Module, CameraEvent, TrackerEvent, PoseEvent
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Mapper(Module):
    def __init__(self, Name, Framework, ModulesLinked):
        '''
        Module that creates a stable 2D map from trackers
        '''
        Module.__init__(self, Name, Framework, ModulesLinked)

        self._MinActiveTrackers = 8
        self._MaxErrorAllowed = 4.
        self._MaxMapValue = 10

        self._MonitorDt = 0.001 # By default, a module does not stode any date over time.
        self._MonitoredVariables = [('Pose.Pose', np.array),
                                    ('CamToWH', np.array),
                                    ('AverageErrorVector', np.array),
                                    ('SigmaErrorVector', np.array),
                                    ('AverageErrorNorm', float),
                                    ('SigmaErrorNorm', float),
                                    ('TrustedAverageErrorNorm', float),
                                    ('TrustedSigmaErrorNorm', float)]

    def _InitializeModule(self):
        self.Trackers = {}
        self.ActiveTrackers = set()
        self.Pose = PoseClass(np.array(self.Geometry), self)
        self.IsComputing = False
        self.Maps = []
        self.CurrentMap = None
        self.AverageErrorVector = np.array([0., 0.])
        self.SigmaErrorVector = np.array([0., 0.])
        self.AverageErrorNorm = 0.
        self.SigmaErrorNorm = 0.
        self.TrustedAverageErrorNorm = 0.
        self.TrustedSigmaErrorNorm = 0.

        self.TrustedTrackers = set()

        self.WToCamH = np.identity(3)
        self.CamToWH = np.identity(3)

        return True

    def _OnEventModule(self, event):
        if event.Has(TrackerEvent):
            if event.TrackerColor == 'g' and event.TrackerMarker == 'o':
                if event.TrackerID in self.ActiveTrackers:
                    self.Trackers[event.TrackerID].Update(event.TrackerLocation)
                    if self.IsComputing and self.Trackers[event.TrackerID].TrustWorthy:
                        self.ComputeNewPose()
                        self.AttachCentralPointPose(event)
                else:
                    self.Trackers[event.TrackerID] = ControlPointClass(event.TrackerLocation, event.TrackerID, self)
                    if not self.IsComputing and len(self.TrustedTrackers) >= self._MinActiveTrackers:
                        self.StartMap(event)
                    elif self.IsComputing:
                        self.Trackers[event.TrackerID].Activate()
            elif event.TrackerColor == 'k' and event.TrackerID in self.ActiveTrackers:
                self.Trackers[event.TrackerID].Disable()
                if self.IsComputing and len(self.TrustedTrackers) < self._MinActiveTrackers:
                    self.StopMap(event)

        if event.Has(CameraEvent) and self.IsComputing:
            self.CurrentMap.OnEvent(event.location)

    def AttachCentralPointPose(self, event):
        Pose = np.array(self.Pose.Pose)
        event.Attach(TrackerEvent, TrackerLocation = Pose[:2], TrackerID = 'C', TrackerAngle = Pose[2], TrackerScaling = Pose[3], TrackerColor = 'g', TrackerMarker = 'o')
        event.Attach(CameraPoseEvent, poseHomography = self.WToCamH, reprojectionError = self.TrustedAverageErrorNorm, worldHomography = None)

    def StartMap(self, event):
        self.LogSuccess("Starting central point tracking")
        self.AttachCentralPointPose(event)
        self.IsComputing = True
        for TrackerID in self.TrustedTrackers:
            self.Trackers[TrackerID].Activate()
        self.CurrentMap = MapClass(self, np.array(self.Geometry), event.timestamp)
        self.Maps += [self.CurrentMap]
        self.LogSuccess("Generating map #{0}".format(len(self.Maps)))
    def StopMap(self, event):
        self.LogWarning("Not enough trackers, unable to continue")
        self.IsComputing = False
        self.CurrentMap.EndTs = event.timestamp
        self.CurrentMap = None

    def ComputeNewPose(self):
        XsIni, XsNow = np.empty((len(self.TrustedTrackers), 2)), np.empty((len(self.TrustedTrackers), 2))
        for nTracker, TrackerID in enumerate(self.TrustedTrackers):
            ControlPoint = self.Trackers[TrackerID]
            XsIni[nTracker, :] = ControlPoint.WLocation
            XsNow[nTracker,:] = ControlPoint.CurrentLocation
        
        self.WToCamH = cv2.findHomography(XsIni, XsNow)[0]
        self.CamToWH = np.linalg.inv(self.WToCamH)

        #N = np.zeros((len(self.TrustedTrackers)*2,5))
        #for nTracker, TrackerID in enumerate(self.TrustedTrackers):
        #    ControlPoint = self.Trackers[TrackerID]
        #    N[2*nTracker,0] = ControlPoint.WLocation[1]
        #    N[2*nTracker+1,0] = ControlPoint.WLocation[0]
        #    N[2*nTracker,1] = ControlPoint.WLocation[0]
        #    N[2*nTracker+1,1] = -ControlPoint.WLocation[1]
        #    N[2*nTracker,3] = 1
        #    N[2*nTracker+1,2] = 1
        #    N[2*nTracker,4] = -ControlPoint.CurrentLocation[1]
        #    N[2*nTracker+1,4] = -ControlPoint.CurrentLocation[0]
        #U, S, V = np.linalg.svd(N)
        #V = V[-1,:] / V[-1,-1]
        #self.WToCamH = np.array([[V[0], -V[1], V[2]], [V[1], V[0], V[3]], [0, 0, 1]])
        #self.CamToWH = np.linalg.inv(self.WToCamH) # Should be done in smarter way

        self.ComputeAverageReprojectionError()

    def ComputeAverageReprojectionError(self):
        Errors = np.empty((len(self.ActiveTrackers),2))
        TrustedBools = np.empty(len(self.ActiveTrackers), dtype = int)
        for nTracker, TrackerID in enumerate(self.ActiveTrackers):
            Errors[nTracker,:] = self.Trackers[TrackerID].ReprojectionError
            TrustedBools[nTracker] = int(self.Trackers[TrackerID].TrustWorthy)
        self.AverageErrorVector = Errors.mean(axis = 0)
        DeltaVectors = np.array(Errors)
        DeltaVectors[:,0] -= self.AverageErrorVector[0]
        DeltaVectors[:,1] -= self.AverageErrorVector[1]
        self.SigmaErrorVector = (DeltaVectors**2).mean(axis = 0)
        ErrorsNorms = np.linalg.norm(Errors, axis = 1)
        self.AverageErrorNorm = ErrorsNorms.mean()
        self.SigmaErrorNorm = np.sqrt(((ErrorsNorms - self.AverageErrorNorm)**2).mean())
        TrustedErrors = (ErrorsNorms * TrustedBools)
        self.TrustedAverageErrorNorm = TrustedErrors.sum() / TrustedBools.sum()
        self.TrustedSigmaErrorNorm = np.sqrt(((TrustedBools * (TrustedErrors - self.TrustedAverageErrorNorm))**2).sum() / TrustedBools.sum())

    def ToW(self, Location):
        PLocation = self.CamToWH.dot(np.array([Location[0], Location[1], 1]))
        return PLocation[:2] / PLocation[-1]
    def ToCam(self, Location):
        PLocation = self.WToCamH.dot(np.array([Location[0], Location[1], 1]))
        return PLocation[:2] / PLocation[-1]

import matplotlib.animation as animation

class MapClass:
    def __init__(self, Mapper, Geometry, t):
        self.StartTs = t
        self.EndTs = None
        self.Mapper = Mapper
        self.Geometry = Geometry
        self.SubMaps = {}

    def OnEvent(self, location):
        WLocation = self.Mapper.ToW(location).astype(int)
        SubMapArray = (WLocation // self.Geometry)
        SubMapTuple = tuple(SubMapArray)
        if not SubMapTuple in self.SubMaps.keys():
            self.SubMaps[SubMapTuple] = np.zeros(self.Geometry)
            self.Mapper.Log("Created submap {0}".format(SubMapTuple))
        SubMapOffset = self.Geometry * SubMapArray
        SubMapEventLocation = WLocation - SubMapOffset
        x = self.SubMaps[SubMapTuple][SubMapEventLocation[0], SubMapEventLocation[1]]
        if x < self.Mapper._MaxMapValue:
            self.SubMaps[SubMapTuple][SubMapEventLocation[0], SubMapEventLocation[1]] = min( x + np.e**(-self.Mapper.AverageErrorNorm), self.Mapper._MaxMapValue)

    @property
    def XMin(self):
        SubMapIndexes = np.array(list(self.SubMaps.keys()))
        return SubMapIndexes[:,0].min()
    @property
    def YMin(self):
        SubMapIndexes = np.array(list(self.SubMaps.keys()))
        return SubMapIndexes[:,1].min()
    @property
    def XMax(self):
        SubMapIndexes = np.array(list(self.SubMaps.keys()))
        return SubMapIndexes[:,0].max()
    @property
    def YMax(self):
        SubMapIndexes = np.array(list(self.SubMaps.keys()))
        return SubMapIndexes[:,1].max()
    @property
    def FullMap(self):
        FullMap = np.zeros(self.Geometry * np.array([self.XMax - self.XMin + 1, self.YMax - self.YMin + 1]))
        for Indexes, SubMap in self.SubMaps.items():
            FullMap[(Indexes[0] - self.XMin)*self.Geometry[0]:(Indexes[0] - self.XMin + 1)*self.Geometry[0], (Indexes[1] - self.YMin)*self.Geometry[1]:(Indexes[1] - self.YMin + 1)*self.Geometry[1]] = SubMap
        return FullMap

    def ShowMeWhatYouGot(self, tMin = 0., AddReprojection = True, MapTransformationFunction = None):
        f, ax = plt.subplots(1,1)
        XMin, YMin, XMax, YMax = self.XMin, self.YMin, self.XMax, self.YMax
        if not MapTransformationFunction is None:
            FM = MapTransformationFunction(self.FullMap)
        else:
            FM = self.FullMap
        ax.imshow(np.transpose(FM), origin = 'lower', cmap = 'binary', extent = (XMin * self.Geometry[0], (XMax+1) * self.Geometry[0], YMin * self.Geometry[1], (YMax+1) * self.Geometry[1]))

        for nX in range(XMin, XMax):
            ax.plot([(nX+1)*self.Geometry[0], (nX+1)*self.Geometry[0]], [YMin * self.Geometry[1], (YMax+1) * self.Geometry[1]], '--k')
        for nY in range(YMin, YMax):
            ax.plot([XMin * self.Geometry[0], (XMax+1) * self.Geometry[0]], [(nY+1)*self.Geometry[1], (nY+1)*self.Geometry[1]], '--k')

        ax.plot(self.Mapper.Pose.InitialLocation[0], self.Mapper.Pose.InitialLocation[1], 'ob', markersize = 10)
        for TrackerID, Tracker in self.Mapper.Trackers.items():
            if not Tracker.Enabled:
                continue
            if Tracker.TrustWorthy:
                color = 'g'
            else:
                color = 'r'
            ax.plot(Tracker.WLocation[0], Tracker.WLocation[1], marker = 'o', color = color, markersize = 6)
            ax.text(Tracker.WLocation[0]+3, Tracker.WLocation[1]+3, str(TrackerID), color=color, fontsize = 10)
            if AddReprojection:
                RP = np.array(Tracker.WorldReprojection)
                ax.plot([Tracker.WLocation[0], RP[0]], [Tracker.WLocation[1], RP[1]], color)

        Corners = list(self.Mapper.Pose.Corners)
        for C1, C2 in zip(Corners, [Corners[-1]] + Corners[:-1]):
            ax.plot([C1[0], C2[0]], [C1[1], C2[1]], 'g')

class ControlPointClass:
    def __init__(self, InitialObservation, ID, Mapper):
        self.Enabled = False
        self.TrustWorthy = True
        self.ID = ID
        self.CurrentLocation = np.array(InitialObservation)
        self.Mapper = Mapper
        self.Mapper.TrustedTrackers.add(self.ID)

    def Activate(self):
        self.WLocation = self.Mapper.ToW(self.CurrentLocation)
        self.Enabled = True
        self.Mapper.ActiveTrackers.add(self.ID)
    def Disable(self):
        self.Mapper.ActiveTrackers.remove(self.ID)
        self.Enabled = False
        if self.TrustWorthy:
            self.Mapper.TrustedTrackers.remove(self.ID)
            self.TrustWorthy = False

    def Update(self, TrackerLocation):
        self.CurrentLocation = np.array(TrackerLocation)

    @property
    def CameraReprojection(self):
        return self.Mapper.ToCam(self.WLocation)
    @property
    def ReprojectionError(self):
        Error = self.CameraReprojection - self.CurrentLocation
        ErrorNorm = np.linalg.norm(Error)
        if self.TrustWorthy and ErrorNorm > self.Mapper._MaxErrorAllowed:
            self.Mapper.TrustedTrackers.remove(self.ID)
            self.TrustWorthy = False
            self.Mapper.LogWarning("CP {0} can't be trusted any more".format(self.ID))
        elif not self.TrustWorthy and ErrorNorm <= self.Mapper._MaxErrorAllowed:
            self.Mapper.TrustedTrackers.add(self.ID)
            self.TrustWorthy = True
        return Error
    @property
    def WorldReprojection(self):
        return self.Mapper.ToW(self.CurrentLocation)

class PoseClass:
    def __init__(self, Geometry, Mapper):
        self.InitialLocation = Geometry / 2
        self.OffsetNorm = 100
        self.ControlPoint = self.InitialLocation + np.array([1, 0]) * self.OffsetNorm
        self.Mapper = Mapper

        self.StaticCorners = [np.array([0, 0]), np.array([0, Geometry[1]]), np.array([Geometry[0], Geometry[1]]), np.array([Geometry[0], 0])]
    def __repr__(self):
        return self.Pose, self.Vector

    @property
    def Center(self):
        return self.Mapper.ToCam(self.InitialLocation)
    @property
    def Vector(self):
        return self.Mapper.ToCam(self.ControlPoint)
    @property
    def Pose(self):
        Pose = np.empty(4)
        Pose[:2] = self.Center
        ControlVector = (self.Vector - Pose[:2])
        Pose[3] = np.linalg.norm(ControlVector) / self.OffsetNorm
        if ControlVector[0] == 0:
            if ControlVector[1] > 0:
                Pose[2] = np.pi/2
            else:
                Pose[2] = -np.pi/2
        else:
            Pose[2] = np.arctan(ControlVector[1]/ControlVector[0])
            if ControlVector[0] < 0:
                Pose[2] += np.pi
        return Pose

    @property
    def Corners(self):
        return [self.Mapper.ToW(Corner) for Corner in self.StaticCorners]
