from Framework import Module, TrackerEvent
import numpy as np
import matplotlib.pyplot as plt

class Mapper(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Module that creates a stable 2D map from trackers
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Computation'

        self._MinActiveTrackers = 8

        self.__ReferencesAsked__ = []
        self._MonitorDt = 0.001 # By default, a module does not stode any date over time.
        self._MonitoredVariables = [('CentralPoint.Pose', np.array)]

    def _InitializeModule(self, **kwargs):
        self.Trackers = {}
        self.ActiveTrackers = set()
        self.CentralPoint = CentralPointClass(self.Geometry[:2])
        self.IsComputing = False
        self.Maps = []
        self.CurrentMap = None
        return True

    def _OnEventModule(self, event):
        if event.Has(TrackerEvent):
            for TrackerAttached in event.Get(TrackerEvent):
                if event.TrackerColor == 'g' and event.TrackerMarker == 'o':
                    if event.TrackerID in self.ActiveTrackers:
                        self.Trackers[event.TrackerID].Update(event.TrackerLocation)
                        if self.IsComputing:
                            self.ComputeNewPose()
                            self.AttachCentralPointPose(event)
                    else:
                        self.Trackers[event.TrackerID] = ControlPointClass(event.TrackerLocation, self.CentralPoint)
                        self.ActiveTrackers.add(event.TrackerID)
                        if not self.IsComputing and len(self.ActiveTrackers) >= self._MinActiveTrackers:
                            self.StartMap(event)
                        elif self.IsComputing:
                            self.Trackers[event.TrackerID].Activate()
                elif event.TrackerColor == 'k' and event.TrackerID in self.ActiveTrackers:
                    self.ActiveTrackers.remove(event.TrackerID)
                    self.Trackers[event.TrackerID].Enabled = False
                    if self.IsComputing and len(self.ActiveTrackers) < self._MinActiveTrackers:
                        self.StopMap(event)

        if self.IsComputing:
            self.CurrentMap.OnEvent(event.location)
        return event

    def AttachCentralPointPose(self, event):
        event.Attach(TrackerEvent, TrackerLocation = np.array(self.CentralPoint.Location), TrackerID = 'C', TrackerAngle = self.CentralPoint.Angle, TrackerScaling = self.CentralPoint.Scaling, TrackerColor = 'g', TrackerMarker = 'X')

    def StartMap(self, event):
        self.LogSuccess("Starting central point tracking")
        self.AttachCentralPointPose(event)
        self.IsComputing = True
        for TrackerID in self.ActiveTrackers:
            self.Trackers[TrackerID].Activate()
        self.CurrentMap = MapClass(self, np.array(self.Geometry[:2]), event.timestamp)
        self.Maps += [self.CurrentMap]
        self.LogSuccess("Generating map #{0}".format(len(self.Maps)))
    def StopMap(self, event):
        self.LogWarning("Not enough trackers, unable to continue")
        self.IsComputing = False
        self.CurrentMap.EndTs = event.timestamp
        self.CurrentMap = None

    def ComputeNewPose(self):
        Xs, Deltas = np.empty((len(self.ActiveTrackers), 2)), np.empty((len(self.ActiveTrackers), 2))
        for nTracker, TrackerID in enumerate(self.ActiveTrackers):
            ControlPoint = self.Trackers[TrackerID]
            Xs[nTracker, :] = ControlPoint.CurrentLocation
            Deltas[nTracker,:] = ControlPoint.DeltaCenter
        AverageX = Xs.mean(axis = 0)
        AverageDelta = Deltas.mean(axis = 0)

        XOffsets = np.empty(Xs.shape)
        DeltaOffsets = np.empty(Deltas.shape)
        XOffsets[:,0] = Xs[:,0] - AverageX[0]
        XOffsets[:,1] = Xs[:,1] - AverageX[1]
        DeltaOffsets[:,0] = Deltas[:,0] - AverageDelta[0]
        DeltaOffsets[:,1] = Deltas[:,1] - AverageDelta[1]

        Scaling = np.sqrt((XOffsets**2).sum() / (DeltaOffsets**2).sum())

        XOffsetsNorms = np.linalg.norm(XOffsets, axis = 1)
        DeltaOffsetsNorms = np.linalg.norm(DeltaOffsets, axis = 1)
        XOffsetsUnit = np.empty(Xs.shape)
        DeltaOffsetsUnit = np.empty(Deltas.shape)
        XOffsetsUnit[:,0] = XOffsets[:,0] / XOffsetsNorms
        XOffsetsUnit[:,1] = XOffsets[:,1] / XOffsetsNorms
        DeltaOffsetsUnit[:,0] = DeltaOffsets[:,0] / DeltaOffsetsNorms
        DeltaOffsetsUnit[:,1] = DeltaOffsets[:,1] / DeltaOffsetsNorms

        CosValue = (XOffsetsUnit * DeltaOffsetsUnit).sum(axis = 1).mean()
        Angle = np.arccos(CosValue)
        SinValue = (DeltaOffsetsUnit[:,0] * XOffsetsUnit[:,1] - DeltaOffsetsUnit[:,1] * XOffsetsUnit[:,0]).mean()
        if SinValue <= 0:
            Angle *= -1
        self.CentralPoint.SetRS(Angle, Scaling)

        self.CentralPoint.Location = AverageX - self.CentralPoint.WToCamMatrix.dot(AverageDelta)

import matplotlib.animation as animation

class MapClass:
    def __init__(self, Mapper, Geometry, t):
        self.StartTs = t
        self.EndTs = None
        self.Mapper = Mapper
        self.CP = Mapper.CentralPoint
        self.Geometry = Geometry
        self.SubMaps = {}

    def OnEvent(self, location):
        WLocation = (self.CP.CamToWMatrix.dot(location - self.CP.Location)).astype(int)
        SubMapArray = (WLocation // self.Geometry)
        SubMapTuple = tuple(SubMapArray)
        if not SubMapTuple in self.SubMaps.keys():
            self.SubMaps[SubMapTuple] = np.zeros(self.Geometry, dtype = int)
            self.Mapper.Log("Created submap {0}".format(SubMapTuple))
        SubMapOffset = self.Geometry * SubMapArray
        SubMapEventLocation = WLocation - SubMapOffset
        self.SubMaps[SubMapTuple][SubMapEventLocation[0], SubMapEventLocation[1]] += 1

    def ShowMeWhatYouGot(self, Animate = True):
        SubMapIndexes = np.array(list(self.SubMaps.keys()))
        XMin, YMin = SubMapIndexes.min(axis = 0)
        XMax, YMax = SubMapIndexes.max(axis = 0)
        FullMap = np.empty(self.Geometry * np.array([XMax - XMin + 1, YMax - YMin + 1]), dtype = int)
        for Indexes, SubMap in self.SubMaps.items():
            FullMap[(Indexes[0] - XMin)*self.Geometry[0]:(Indexes[0] - XMin + 1)*self.Geometry[0], (Indexes[1] - YMin)*self.Geometry[1]:(Indexes[1] - YMin + 1)*self.Geometry[1]] = SubMap
        f, ax = plt.subplots(1,1)
        ax.imshow(np.transpose(FullMap), origin = 'lower', cmap = 'binary', extent = (XMin * self.Geometry[0], (XMax+1) * self.Geometry[0], YMin * self.Geometry[1], (YMax+1) * self.Geometry[1]))

        StaticCorners = [np.array([0, 0]), np.array([0, self.Geometry[1]]), np.array([self.Geometry[0], self.Geometry[1]]), np.array([self.Geometry[0], 0])]
        
        for nX in range(XMin, XMax):
            ax.plot([(nX+1)*self.Geometry[0], (nX+1)*self.Geometry[0]], [YMin * self.Geometry[1], (YMax+1) * self.Geometry[1]], '--k')
        for nY in range(YMin, YMax):
            ax.plot([XMin * self.Geometry[0], (XMax+1) * self.Geometry[0]], [(nY+1)*self.Geometry[1], (nY+1)*self.Geometry[1]], '--k')

        CornersLines = []
        for nCornerr in range(4):
            CornersLines += ax.plot([0, 0], [0, 0], 'g')
        self._Animate = Animate
        def handle_close(event):
            self._Animate = False
        f.canvas.mpl_connect('close_event', handle_close)
        while self._Animate:
            for t, CPPose in zip(self.Mapper.History['t'], self.Mapper.History['CentralPoint.Pose']):
                Angle, Scaling = CPPose[2:]
                c, s = np.cos(Angle), np.sin(Angle)
                CamToWMatrix = np.array([[c, s], [-s, c]]) / Scaling
                Corners = [CamToWMatrix.dot(Corner) - CPPose[:2] for Corner in StaticCorners]
                for nCorner, Corner in enumerate(Corners):
                    CornersLines[nCorner].set_data([Corner[0], Corners[(nCorner+1)%4][0]], [Corner[1], Corners[(nCorner+1)%4][1]])
                ax.set_title("t = {0:.3f}".format(t))
                plt.pause(0.001)



class CentralPointClass:
    def __init__(self, ScreenGeometry):
        self.InitialLocation = np.array(ScreenGeometry) / 2

        self.Location = np.array(self.InitialLocation)
        self.SetRS(0., 1.)

    def SetRS(self, Angle, Scaling):
        self.Angle = Angle
        self.Scaling = Scaling

        c, s = np.cos(self.Angle), np.sin(self.Angle)
        self.WToCamMatrix = np.array([[c, -s], [s, c]]) * self.Scaling
        self.CamToWMatrix = np.array([[c, s], [-s, c]]) / self.Scaling

    def __repr__(self):
        return self.Pose
    @property
    def Pose(self):
        return np.array([self.Location[0], self.Location[1], self.Angle, self.Scaling])

class ControlPointClass:
    def __init__(self, InitialObservation, CentralPoint):
        self.Enabled = True
        self.CurrentLocation = np.array(InitialObservation)
        self.CentralPoint = CentralPoint

    def Activate(self):
        self.DeltaCenter = self.CentralPoint.CamToWMatrix.dot(self.CurrentLocation - self.CentralPoint.Location)

    def Update(self, TrackerLocation):
        self.CurrentLocation = np.array(TrackerLocation)
