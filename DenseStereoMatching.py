import numpy as np
from Framework import Module, Event, DisparityEvent

class DenseStereo(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Module that given two input streams creates a dense disparity map.
        For now, works only with rectified square cameras?
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Computation'
        self.__ReferencesAsked__ = []

        self._ComparisonRadius = 6
        self._Tau = 0.01
        self._MaxSimultaneousComparisonPoints = 10

        self._LifespanTauRatio = 5
        self._CleanupTauRatio = 0.5 #After 5 tau, we consider no match as a failure and remove that comparison point

        self._MatchThreshold = 0.8
        self._MaxDistanceCluster = 1
        self._MaxMatches = 2

        self._EpipolarMatrix = [[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
        self._MonitorDt = 0. # By default, a module does not stode any date over time.
        self._MonitoredVariables = []

    def _InitializeModule(self, **kwargs):
        self._EpipolarMatrix = np.array(self._EpipolarMatrix)
        self.UsedGeometry = np.array(self.Geometry)[:2]

        self.CPLifespan = self._Tau * self._LifespanTauRatio
        self.CleanupDt = self._Tau * self._CleanupTauRatio
        self.LastCleanup = [0,0]

        self.Maps = [AnalysisMapClass(self.UsedGeometry, self._ComparisonRadius, self._Tau), AnalysisMapClass(self.UsedGeometry, self._ComparisonRadius, self._Tau)]
        self.ComparisonPoints = [[],[]]
        self.DisparitiesFound = []

        return True

    def _OnEventModule(self, event):
        self.Maps[event.cameraIndex].OnEvent(event)
        if len(self.ComparisonPoints[event.cameraIndex]) < self._MaxSimultaneousComparisonPoints and ((event.location >= self._ComparisonRadius).all() and (event.location < self.UsedGeometry - self._ComparisonRadius).all()):
            Add = True
            for CP in self.ComparisonPoints[event.cameraIndex]:
                if abs(event.location - CP[0]).max() <= self._ComparisonRadius:
                    Add = False
                    break
            if Add:
                EpipolarEquation = self._EpipolarMatrix.dot(np.array([event.location[0], event.location[1], 1]))
                EpipolarEquation /= np.linalg.norm(EpipolarEquation[:2])
                self.ComparisonPoints[event.cameraIndex] += [(np.array(event.location), event.timestamp, EpipolarEquation, [])]
                self.Log("Added CP for camera {0} at {1}".format(event.cameraIndex, event.location))
        
        if event.location[0] < self._ComparisonRadius or event.location[0] >= self.UsedGeometry[0] - self._ComparisonRadius: # This event cannot match at it is too close from screen border
            return event

        Matched = []
        for nCP, CP in enumerate(self.ComparisonPoints[1-event.cameraIndex]):
            if event.timestamp - CP[1] < self.Maps[1-event.cameraIndex].Tau: # No match can be trusted under 1 Tau
                continue
            Distance = np.array([event.location[0], event.location[1], 1]).dot(CP[2]) # Compute distance from this point to the compared point epipolar line
            if abs(Distance) <= self._ComparisonRadius:
                ProjectedEvent = (event.location - Distance * CP[2][:2]).astype(int) # Recover the location of point on the epipolar line that corresponds to this event
                if ProjectedEvent[0] == CP[0][0]: # We just want to avoid infinite disparity values
                    continue

                EventSignatures = self.Maps[event.cameraIndex].GetSignatures(ProjectedEvent, event.timestamp) # Recover specific signatures of the map on that epipolar line for that CP on the other camera
                CPSignatures = self.Maps[1-event.cameraIndex].GetSignatures(CP[0], event.timestamp)

                if self.Match(EventSignatures, CPSignatures):
                    GlobalMatch, Location = self.AddLocalMatch(event.location[0], CP)
                    if GlobalMatch:
                        event.Attach(DisparityEvent, disparity = 1./abs(Location - CP[0][0]))
                        self.LogSuccess("Matched y = {0}, x[0] = {{{1}}} & x[1] = {{{2}}}, d = {3:.3f}".format(ProjectedEvent[1], 1-event.cameraIndex, event.cameraIndex, event.disparity).format(CP[0][0], Location))
                        self.DisparitiesFound += [(event.timestamp, np.array(event.location), event.disparity)]
                        Matched += [nCP]
        for nCP in reversed(Matched):
            self.ComparisonPoints[1-event.cameraIndex].pop(nCP)

        if event.timestamp - self.LastCleanup[event.cameraIndex] > self.CleanupDt:
            self.ComparisonPoints[event.cameraIndex] = [CP for CP in self.ComparisonPoints[event.cameraIndex] if event.timestamp - CP[1] < self.CPLifespan]
            self.LastCleanup[event.cameraIndex] = event.timestamp
        return event

    def AddLocalMatch(self, X, CP):
        for PossibleLocation in CP[3]:
            if abs(PossibleLocation[0] - X) <= self._MaxDistanceCluster:
                PossibleLocation[0] = (PossibleLocation[0] * PossibleLocation[1] + X) / (PossibleLocation[1]+1)
                PossibleLocation[1] += 1
                if PossibleLocation[1] == self._MaxMatches:
                    return True, PossibleLocation[0]
                return False, None
        CP[3].append([X,1])
        return False, None

    def Match(self, SigsA, SigsB):
        MatchValue = 1.
        for ItemA, ItemB in zip(SigsA, SigsB):
            NA, NB = np.linalg.norm(ItemA), np.linalg.norm(ItemB)
            if NA == 0 or NB == 0:
                return False
            MatchValue *= ItemA.dot(ItemB) / (NA * NB)
        return MatchValue > self._MatchThreshold

class AnalysisMapClass:
    def __init__(self, Geometry, Radius, Tau):
        self.Geometry = Geometry
        self.Radius = int(Radius)

        self.Tau = Tau # Default time constant for now

        self.UsableGeometry = self.Geometry - np.array([0, 2*self.Radius]) # Remove blind zones at the top and bottom of the screen
        self.MaxY = self.UsableGeometry[1]

        self.ActivityMap = np.zeros(self.UsableGeometry)
        self.DistanceMap = np.zeros(self.UsableGeometry)
        self.SigmaMap = np.zeros(self.UsableGeometry)
        self.LastUpdateMap = -np.inf * np.ones(self.UsableGeometry)

    def OnEvent(self, event):
        YMin, YMax = max(0, event.location[1] - self.Radius), min(self.UsableGeometry[1], event.location[1] + self.Radius+1)
        DeltaColumn = self.LastUpdateMap[event.location[0],YMin:YMax] - event.timestamp
        DecayColumn = np.e**(DeltaColumn / self.Tau)

        self.LastUpdateMap[event.location[0],YMin:YMax] = event.timestamp

        Offset = event.location[1] - np.arange(YMin, YMax)

        self.ActivityMap[event.location[0],YMin:YMax] = self.ActivityMap[event.location[0],YMin:YMax] * DecayColumn + 1
        self.DistanceMap[event.location[0],YMin:YMax] = self.DistanceMap[event.location[0],YMin:YMax] * DecayColumn + Offset
        self.SigmaMap[event.location[0],YMin:YMax] = self.SigmaMap[event.location[0],YMin:YMax] * DecayColumn + (Offset - self.DistanceMap[event.location[0],YMin:YMax] / self.ActivityMap[event.location[0],YMin:YMax])**2

    def GetSignatures(self, location, t):
        XMin, XMax = max(0, location[0] - self.Radius), min(self.UsableGeometry[0], location[0] + self.Radius+1)
        YUsed = location[1] - self.Radius
        DeltaRow = self.LastUpdateMap[XMin:XMax,YUsed] - t
        DecayRow = np.e**(DeltaRow / self.Tau)
        
        return self.ActivityMap[XMin:XMax,YUsed] * DecayRow, self.DistanceMap[XMin:XMax,YUsed] / np.maximum(0.001, self.ActivityMap[XMin:XMax,YUsed]), self.SigmaMap[XMin:XMax,YUsed] / np.maximum(0.001, self.ActivityMap[XMin:XMax,YUsed])

