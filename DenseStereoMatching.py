import numpy as np
from Framework import Module, Event, DisparityEvent
import matplotlib.pyplot as plt

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
        self._MaxSimultaneousComparisonPoints = 20
        self._MaxPriorityCPs = 20

        self._MinCPAverageActivity = 3.
        self._LifespanTauRatio = 2 #After N tau, we consider no match as a failure and remove that comparison point
        self._CleanupTauRatio = 0.5 

        self._MatchThreshold = 0.8
        self._MaxDistanceCluster = 0
        self._MaxMatches = 2

        self._NDisparitiesStored = 200
        self._DisparityPatchRadius = 2
        self._DisparitySearchRadius = 1

        self._EpipolarMatrix = [[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
        self._MonitorDt = 0. # By default, a module does not stode any date over time.
        self._MonitoredVariables = []

    def _InitializeModule(self, **kwargs):
        self._EpipolarMatrix = np.array(self._EpipolarMatrix)
        self.UsedGeometry = np.array(self.Geometry)[:2]

        self.CPLifespan = self._Tau * self._LifespanTauRatio
        self.CleanupDt = self._Tau * self._CleanupTauRatio
        self.LastCleanup = [0,0]

        self.Maps = (AnalysisMapClass(self.UsedGeometry, self._ComparisonRadius, self._Tau), AnalysisMapClass(self.UsedGeometry, self._ComparisonRadius, self._Tau))
        self.ComparisonPoints = ([],[])
        self.DisparitiesStored = ([],[])
        self.DisparityMap = np.zeros(tuple(self.UsedGeometry) + (2,2)) # 0 default value seems legit as it pushes all points to infinity
        self.DisparityMap[:,:,1,:] = -np.inf

        return True

    def _OnEventModule(self, event):
        self.Maps[event.cameraIndex].OnEvent(event)
        if (event.location < self._ComparisonRadius).any() or (event.location >= self.UsedGeometry - self._ComparisonRadius).any(): # This event cannot be set as CP, nor can it be matched to anything as its outside the borders
            return event

        self.ConsiderAddingCP(event)
        
        self.TryMatchingEventToStereoCPs(event)

        self.GetDisparityEvent(event)

        if event.timestamp - self.LastCleanup[event.cameraIndex] > self.CleanupDt:
            L = len(self.ComparisonPoints[event.cameraIndex])
            for nCP, CP in enumerate(reversed(self.ComparisonPoints[event.cameraIndex])):
                if not self.KeepCP(event.timestamp, CP):
                    self.ComparisonPoints[event.cameraIndex].pop(L-(nCP+1))
            self.LastCleanup[event.cameraIndex] = event.timestamp
        return event

    def KeepCP(self, t, CP):
        if t - CP[1] > self.CPLifespan:
            self.LogWarning("Removed CP at {0} as its still unmatched".format(CP[0]))
            return False
        else:
            return True

    def PlotDisparitiesMaps(self):
        f, axs = plt.subplots(1,2)
        for i in range(2):
            axs[i].imshow(np.transpose(self.DisparityMap[:,:,0,i]), origin = 'lower', cmap = 'inferno')

    def ConsiderAddingCP(self, event):
        if len(self.ComparisonPoints[event.cameraIndex]) < self._MaxSimultaneousComparisonPoints:
            AverageActivity = self.Maps[event.cameraIndex].GetAverageActivity(event.location, event.timestamp)
            if AverageActivity >= self._MinCPAverageActivity:
                Add = True
                for CP in self.ComparisonPoints[event.cameraIndex]:
                    if abs(event.location - CP[0]).max() <= self._ComparisonRadius:
                        Add = False
                        break
                if Add:
                    for DS in self.DisparitiesStored[event.cameraIndex]:
                        if abs(event.location - DS[0]).max() <= self._ComparisonRadius:
                            Add = False
                            break
                if Add:
                    self.AddCP(event)
                    self.Log("Added CP for camera {0} at {1}, average activity is {2:.1f}".format(event.cameraIndex, event.location, AverageActivity))

    def AddCP(self, event):
        EpipolarEquation = self._EpipolarMatrix.dot(np.array([event.location[0], event.location[1], 1]))
        EpipolarEquation /= np.linalg.norm(EpipolarEquation[:2])
        self.ComparisonPoints[event.cameraIndex].append((np.array(event.location), event.timestamp, EpipolarEquation, []))

    def TryMatchingEventToStereoCPs(self, event):
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

                if self.Match(EventSignatures, CPSignatures) > self._MatchThreshold:
                    GlobalMatch, Location = self.AddLocalMatch(event.location[0], CP)
                    if GlobalMatch:
                        Offset = Location - CP[0][0]
                        self.LogSuccess("Matched y = {0}, x[0] = {{{1}}} & x[1] = {{{2}}}, offset = {3}".format(ProjectedEvent[1], 1-event.cameraIndex, str(event.cameraIndex)+":.1f", Offset).format(CP[0][0], Location))
                        if len(self.DisparitiesStored[event.cameraIndex]) == self._NDisparitiesStored:
                            self.DisparitiesStored[event.cameraIndex].pop(0)
                        self.DisparitiesStored[event.cameraIndex].append((np.array(CP[0]), event.timestamp, Offset))
                        self.DisparityMap[CP[0][0],CP[0][1],:,1-event.cameraIndex] = np.array([Offset, event.timestamp])
                        self.DisparityMap[event.location[0],CP[0][1],:,event.cameraIndex] = np.array([-Offset, event.timestamp])
                        Matched += [nCP]
        for nCP in reversed(Matched):
            self.ComparisonPoints[1-event.cameraIndex].pop(nCP)

    def GetDisparityEvent(self, event):
        if event.timestamp - self.DisparityMap[tuple(event.location) +(1,event.cameraIndex)] < self._Tau / 2: # We assume the disparity here still valid and dont update it
            return
        LocalPatch = self.DisparityMap[event.location[0]-self._DisparityPatchRadius:event.location[0]+self._DisparityPatchRadius+1,event.location[1]-self._DisparityPatchRadius:event.location[1]+self._DisparityPatchRadius+1,:,event.cameraIndex]
        DisparitiesSearched = set()
        for Disparity, t in LocalPatch.reshape(((self._DisparityPatchRadius*2 + 1)**2 , 2)):
            if event.timestamp - t < 4*self._Tau:
                for delta in range(-self._DisparitySearchRadius, self._DisparitySearchRadius+1):
                    DisparitiesSearched.add(int(Disparity) + delta)
        if len(DisparitiesSearched) == 0:
            return

        EventSignatures = self.Maps[event.cameraIndex].GetSignatures(event.location, event.timestamp)
        BestMatch = [0, None]
        for Disparity in DisparitiesSearched:
            XLocation = event.location[0] + Disparity
            if XLocation < self._ComparisonRadius or XLocation >= self.UsedGeometry[0] - self._ComparisonRadius:
                continue
            StereoSignatures = self.Maps[1-event.cameraIndex].GetSignatures(np.array([XLocation, event.location[1]]), event.timestamp)
            MatchValue = self.Match(EventSignatures, StereoSignatures)
            if MatchValue > BestMatch[0]:
                BestMatch = [MatchValue, Disparity]
        if BestMatch[0] > self._MatchThreshold:
            Disparity = BestMatch[1]
            XLocation = event.location[0] + Disparity
            event.Attach(DisparityEvent, disparity = 1./abs(Disparity))
            self.DisparityMap[event.location[0], event.location[1],:,event.cameraIndex] = np.array([Disparity, event.timestamp])
            self.DisparityMap[XLocation, event.location[1],:,1-event.cameraIndex] = np.array([-Disparity, event.timestamp])
        else:
            if len(self.ComparisonPoints[event.cameraIndex]) < self._MaxSimultaneousComparisonPoints + self._MaxPriorityCPs:
                self.AddCP(event)

    def AddLocalMatch(self, X, CP):
        for PossibleLocation in CP[3]:
            if abs(PossibleLocation[0] - X) <= self._MaxDistanceCluster:
                #PossibleLocation[0] = (PossibleLocation[0] * PossibleLocation[1] + X) / (PossibleLocation[1]+1)
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
        return MatchValue

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

    def GetLineSignatures(self, Y, t):
        YUsed = Y - self.Radius
        DeltaRow = self.LastUpdateMap[:,YUsed] - t
        DecayRow = np.e**(DeltaRow / self.Tau)
        
        return self.ActivityMap[:,YUsed] * DecayRow, self.DistanceMap[:,YUsed] / np.maximum(0.001, self.ActivityMap[:,YUsed]), self.SigmaMap[:,YUsed] / np.maximum(0.001, self.ActivityMap[:,YUsed])

    def GetAverageActivity(self, location, t):
        XMin, XMax = max(0, location[0] - self.Radius), min(self.UsableGeometry[0], location[0] + self.Radius+1)
        YUsed = location[1] - self.Radius
        DeltaRow = self.LastUpdateMap[XMin:XMax,YUsed] - t
        DecayRow = np.e**(DeltaRow / self.Tau)
        
        return (self.ActivityMap[XMin:XMax,YUsed] * DecayRow).mean()

    def GetSignatures(self, location, t):
        XMin, XMax = max(0, location[0] - self.Radius), min(self.UsableGeometry[0], location[0] + self.Radius+1)
        YUsed = location[1] - self.Radius
        DeltaRow = self.LastUpdateMap[XMin:XMax,YUsed] - t
        DecayRow = np.e**(DeltaRow / self.Tau)
        
        return self.ActivityMap[XMin:XMax,YUsed] * DecayRow, self.DistanceMap[XMin:XMax,YUsed] / np.maximum(0.001, self.ActivityMap[XMin:XMax,YUsed]), self.SigmaMap[XMin:XMax,YUsed] / np.maximum(0.001, self.ActivityMap[XMin:XMax,YUsed])

