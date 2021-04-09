import numpy as np
from PEBBLE import Module, CameraEvent, TrackerEvent

class StereoPatternMatcher(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Module that given a tracker tries to match the corresponding location on another camera
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Computation'
        self.__ReferencesAsked__ = ['Tracker']

        self._ActivityRadius = 6
        self._CleanupEvery = 0.1
        self._MetricMatch = 0.8
        self._TCRatio = 1. / 3
        self._EpipolarMatrix = [[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
        self._MonitorDt = 0. # By default, a module does not stode any date over time.
        self._MonitoredVariables = []

        self._CameraTrackersIndex = 0
        self._MatchedCameraIndex = 0

    def _InitializeModule(self, **kwargs):
        self.TrackerModule = self.__Framework__.Tools[self.__CreationReferences__['Tracker']]

        self._EpipolarMatrix = np.array(self._EpipolarMatrix)

        self.TrackersMatchers = {}
        self.LastCleanup = 0.
        self.UsedGeometry = np.array(self.Geometry)
        return True

    def _OnEventModule(self, event):
        if event.SubStreamIndex == self._CameraTrackersIndex:
            self.OnEventTracker(event)
        if event.SubStreamIndex == self._MatchedCameraIndex:
            self.OnEventMatched(event)

    def OnEventTracker(self, event):
        if event.Has(TrackerEvent):
            for TrackerAttached in event.Get(TrackerEvent):
                Tracker = self.TrackerModule.Trackers[event.TrackerID]
                if Tracker.State.Locked:
                    if not Tracker.ID in self.TrackersMatchers.keys():
                        self.TrackersMatchers[Tracker.ID] = TrackerMatcherClass(Tracker, self)
                        self.Log("Added matcher for tracker {0}".format(Tracker.ID))
        for TrackerMatcher in self.TrackersMatchers.values():
            TrackerMatcher.RunTrackerEvent(event.timestamp, event.location)
        if event.timestamp - self.LastCleanup > self._CleanupEvery:
            self.CleanupDeadTrackers()
            self.LastCleanup = event.timestamp

    def OnEventMatched(self, event):
        for TrackerMatcher in self.TrackersMatchers.values():
            TrackerMatcher.RunMatcherEvent(event)

    def CleanupDeadTrackers(self):
        for TrackerID in list(self.TrackersMatchers.keys()):
            if self.TrackersMatchers[TrackerID].Tracker.State.Dead:
                del self.TrackersMatchers[TrackerID]

class TrackerMatcherClass:
    def __init__(self, Tracker, StereoModule):
        self.StereoModule = StereoModule
        self.Tracker = Tracker
        self.StatedAt = self.Tracker.LastUpdate

        self.TrackerActivity = np.zeros(int(2*self.StereoModule._ActivityRadius+1))
        self.EpipolarActivity = np.zeros(self.StereoModule.UsedGeometry[0])
        self.TrackerDistance = np.zeros(int(2*self.StereoModule._ActivityRadius+1))
        self.EpipolarDistance = np.zeros(self.StereoModule.UsedGeometry[0])
        self.TrackerSigma = np.zeros(int(2*self.StereoModule._ActivityRadius+1))
        self.EpipolarSigma = np.zeros(self.StereoModule.UsedGeometry[0])
        self.TrackerLastUpdate = -np.inf * np.ones(int(2*self.StereoModule._ActivityRadius+1))
        self.EpipolarLastUpdate = -np.inf * np.ones(self.StereoModule.UsedGeometry[0])
        self.EpipolarEquation = self.StereoModule._EpipolarMatrix.dot(np.array([self.Tracker.Position[0], self.Tracker.Position[1], 1]))
        self.EpipolarEquation /= np.linalg.norm(self.EpipolarEquation[:2])

        self.Matches = []
        self.BestMatchValue = 0
        self.BestMatchError = None

    def UpdateTracker(self, t, X_Proj):
        self.EpipolarEquation = self.StereoModule._EpipolarMatrix.dot(np.array([self.Tracker.Position[0], self.Tracker.Position[1], 1]))
        self.EpipolarEquation /= np.linalg.norm(self.EpipolarEquation[:2])

        DeltaTracker = self.TrackerLastUpdate[X_Proj] - t
        DecayTracker = np.e**(DeltaTracker / (self.Tracker.TimeConstant * self.StereoModule._TCRatio))
        self.TrackerActivity[X_Proj] *= DecayTracker
        self.TrackerDistance[X_Proj] *= DecayTracker
        self.TrackerSigma[X_Proj] *= DecayTracker
        self.TrackerLastUpdate[X_Proj] = t

    def DecayTracker(self, t):
        DeltaTracker = self.TrackerLastUpdate - t
        DecayTracker = np.e**(DeltaTracker / (self.Tracker.TimeConstant * self.StereoModule._TCRatio))
        self.TrackerActivity *= DecayTracker
        self.TrackerDistance *= DecayTracker
        self.TrackerSigma *= DecayTracker
        self.TrackerLastUpdate[:] = t

    def UpdateEpipolar(self, t, X_Proj):
        DeltaX_Proj = self.EpipolarLastUpdate[X_Proj] - t
        DecayX_Proj = np.e**(DeltaX_Proj / (self.Tracker.TimeConstant * self.StereoModule._TCRatio))
        self.EpipolarActivity[X_Proj] *= DecayX_Proj
        self.EpipolarDistance[X_Proj] *= DecayX_Proj
        self.EpipolarSigma[X_Proj] *= DecayX_Proj
        self.EpipolarLastUpdate[X_Proj] = t

    def DecayEpipolarPatch(self, t, X_Proj):
        DeltaPatch = self.EpipolarLastUpdate[X_Proj - self.StereoModule._ActivityRadius:X_Proj + self.StereoModule._ActivityRadius+1] - t
        DecayPatch = np.e**(DeltaPatch / (self.Tracker.TimeConstant * self.StereoModule._TCRatio))
        self.EpipolarActivity[X_Proj - self.StereoModule._ActivityRadius:X_Proj + self.StereoModule._ActivityRadius+1] *= DecayPatch
        self.EpipolarDistance[X_Proj - self.StereoModule._ActivityRadius:X_Proj + self.StereoModule._ActivityRadius+1] *= DecayPatch
        self.EpipolarSigma[X_Proj - self.StereoModule._ActivityRadius:X_Proj + self.StereoModule._ActivityRadius+1] *= DecayPatch
        self.EpipolarLastUpdate[X_Proj - self.StereoModule._ActivityRadius:X_Proj + self.StereoModule._ActivityRadius+1] = t

    def RunTrackerEvent(self, t, Location):
        VecDiff = Location - self.Tracker.Position[:2]
        if (abs(VecDiff)).max() <= self.StereoModule._ActivityRadius:
            X_Proj = int(VecDiff[0] + self.StereoModule._ActivityRadius)
            self.UpdateTracker(t, X_Proj)
            self.TrackerActivity[X_Proj] += 1
            self.TrackerDistance[X_Proj] += VecDiff[1]
            self.TrackerSigma[X_Proj] += (VecDiff[1] - self.TrackerDistance[X_Proj] / self.TrackerActivity[X_Proj]) ** 2

    def GetActivityMetric(self, X_Proj):
        LocalActivities = self.EpipolarActivity[X_Proj - self.StereoModule._ActivityRadius:X_Proj + self.StereoModule._ActivityRadius+1]
        return (self.TrackerActivity * LocalActivities).sum() / (np.linalg.norm(self.TrackerActivity) * np.linalg.norm(LocalActivities))

    def GetDistanceMetric(self, X_Proj):
        LocalDistances = np.array(self.EpipolarDistance[X_Proj - self.StereoModule._ActivityRadius:X_Proj + self.StereoModule._ActivityRadius+1])
        LocalActivities = self.EpipolarActivity[X_Proj - self.StereoModule._ActivityRadius:X_Proj + self.StereoModule._ActivityRadius+1]
        LocalActivities = np.maximum(LocalActivities, 0.01)
        LocalDistances /= LocalActivities

        TrackerDistances = self.TrackerDistance /  np.maximum(self.TrackerActivity, 0.01)
        return (TrackerDistances.dot(LocalDistances)) / (np.linalg.norm(TrackerDistances) * np.linalg.norm(LocalDistances))

    def RunMatcherEvent(self, event):
        Location = event.location
        Distance = np.array([Location[0], Location[1], 1]).dot(self.EpipolarEquation)
        if abs(Distance) <= self.StereoModule._ActivityRadius:
            X_Proj = int(Location[0] + Distance * self.EpipolarEquation[0])
            self.UpdateEpipolar(event.timestamp, X_Proj)
            if X_Proj >=0 and X_Proj < self.EpipolarActivity.shape[0]:
                self.EpipolarActivity[X_Proj] += 1
                self.EpipolarDistance[X_Proj] += Distance
                self.EpipolarSigma[X_Proj] += (Distance - self.EpipolarDistance[X_Proj] / self.EpipolarActivity[X_Proj]) ** 2
        else:
            return
        
        if event.timestamp - self.StatedAt < self.Tracker.TimeConstant:
            return

        if X_Proj >= self.StereoModule._ActivityRadius and X_Proj < self.EpipolarActivity.shape[0] - self.StereoModule._ActivityRadius:
            self.DecayEpipolarPatch(event.timestamp, X_Proj)
            self.DecayTracker(event.timestamp)
            ActivityMetric = self.GetActivityMetric(X_Proj)
            DistanceMetric = self.GetDistanceMetric(X_Proj)
            MatchValue = ActivityMetric*DistanceMetric
            self.Matches += [[abs(X_Proj - self.Tracker.Position[0]), MatchValue]]

            if MatchValue > self.BestMatchValue:
                self.BestMatchValue = MatchValue
                self.BestMatchError = abs(X_Proj - self.Tracker.Position[0])
                
            if MatchValue > self.StereoModule._MetricMatch:
                event.Attach(TrackerEvent, TrackerLocation = np.array([X_Proj, self.Tracker.Position[1]]), TrackerID = -self.Tracker.ID, TrackerAngle = 0, TrackerScaling = 0, TrackerColor = self.Tracker.State.GetColor(), TrackerMarker = self.Tracker.State.GetMarker())
                self.StereoModule.LogSuccess("Matched tracker {0}".format(self.Tracker.ID))
