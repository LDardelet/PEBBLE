import numpy as np
from Framework import Module, Event, DisparityEvent, TrackerEvent
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

        self._ComparisonRadius = 10
        self._Tau = 0.005
        self._MaxSimultaneousPointsPerType = np.array([100, 100])

        self._MinCPAverageActivity = 3.
        self._LifespanTauRatio = 2 #After N tau, we consider no match as a failure and remove that comparison point
        self._CleanupTauRatio = 0.5 
        self._UsedSignatures = ['Activity', 'Distance', 'Sigma']

        self._MatchThreshold = 0.7
        self._MaxDistanceCluster = 0
        self._MaxMatches = 2

        self._AverageRadius = 1

        self._NDisparitiesStored = 200
        self._DisparityPatchRadius = 2
        self._DisparitySearchRadius = 1

        self._DisparityRange = [0, np.inf]

        self._EpipolarMatrix = [[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
        self._MonitorDt = 0.0001 # By default, a module does not stode any date over time.
        self._MonitoredVariables = [('DisparityMap', np.array),
                                    ('InputEvents', float),
                                    ('MatchedEvents', float)]

    def _InitializeModule(self, **kwargs):
        self._EpipolarMatrix = np.array(self._EpipolarMatrix)
        self.UsedGeometry = np.array(self.Geometry)[:2]

        self.MetricIndexes = [['Activity', 'Distance', 'Sigma'].index(SignatureName) for SignatureName in self._UsedSignatures]
        self.MetricThreshold = self._MatchThreshold ** len(self._UsedSignatures)

        self.CPLifespan = self._Tau * self._LifespanTauRatio
        self.CleanupDt = self._Tau * self._CleanupTauRatio
        self.LastCleanup = [0,0]

        self.Maps = (AnalysisMapClass(self.UsedGeometry, self._ComparisonRadius, self._Tau, self._AverageRadius), AnalysisMapClass(self.UsedGeometry, self._ComparisonRadius, self._Tau, self._AverageRadius))
        self.ComparisonPoints = ([],[])
        self.DisparitiesStored = ([],[])
        self.DisparityMap = np.zeros(tuple(self.UsedGeometry) + (2,2)) # 0 default value seems legit as it pushes all points to infinity
        self.DisparityMap[:,:,1,:] = -np.inf
        self.ComparedPoints = np.array([[0,0],[0,0]], dtype = int)

        self.MatchedCPs = np.array([0,0])
        self.InputEvents = 0
        self.MatchedEvents = 0
        self.LastTDecay = -np.inf

        self._DisparityRanges = [[-self._DisparityRange[1], -self._DisparityRange[0]], [self._DisparityRange[0], self._DisparityRange[1]]]
        
        return True

    def _OnEventModule(self, event):
        self.Maps[event.cameraIndex].OnEvent(event)
        if (event.location < self._ComparisonRadius).any() or (event.location >= self.UsedGeometry - self._ComparisonRadius).any(): # This event cannot be set as CP, nor can it be matched to anything as its outside the borders
            return event

        self.ConsiderAddingCP(event, CPType = 0)
        
        Match = self.TryMatchingEventToStereoCPs(event)

        if not Match:
            Match = self.GetDisparityEvent(event)
        
        Decay = np.e**((self.LastTDecay - event.timestamp)/self._Tau)
        self.InputEvents *= Decay
        self.MatchedEvents *= Decay
        self.InputEvents += 1
        self.MatchedEvents += int(Match)
        self.LastTDecay = event.timestamp

        if event.timestamp - self.LastCleanup[event.cameraIndex] > self.CleanupDt:
            L = len(self.ComparisonPoints[event.cameraIndex])
            for nCP, CP in enumerate(reversed(self.ComparisonPoints[event.cameraIndex])):
                if not self.KeepCP(event.timestamp, CP):
                    self.ComparedPoints[event.cameraIndex,CP[1]] -= 1
                    self.ComparisonPoints[event.cameraIndex].pop(L-(nCP+1))
            self.LastCleanup[event.cameraIndex] = event.timestamp
        return event

    def KeepCP(self, t, CP):
        if t - CP[2] > self.CPLifespan:
            self.LogWarning("Removed CP at {0} as its still unmatched".format(CP[0]))
            return False
        else:
            return True

    def PlotDisparitiesMaps(self, Tau = None, Corrected = True):
        f, axs = plt.subplots(2,2)
        NPixels = self.UsedGeometry.prod()
        MinDisp = int(self.DisparityMap[:,:,0,:].min())
        MaxDisp = int(self.DisparityMap[:,:,0,:].max())
        for Disparity in range(MinDisp, MaxDisp + 1):
            if (self.DisparityMap[:,:,0,:] < Disparity).sum() < 0.01 * NPixels:
                MinDisp = Disparity
            if (self.DisparityMap[:,:,0,:] > Disparity).sum() < 0.01 * NPixels:
                MaxDisp = Disparity
                break
        print("Ranging disparities from {0} to {1}".format(MinDisp, MaxDisp))
        if Tau is None:
            Tau = self._Tau
        for i in range(2):
            DMap = np.array(self.DisparityMap[:,:,0,i])
            DMap[np.where(self.DisparityMap[:,:,1,i] - self.DisparityMap[:,:,1,i].max() < -Tau)] = 0
            I = axs[0,i].imshow(np.transpose(DMap), origin = 'lower', cmap = 'inferno', vmin = MinDisp, vmax = MaxDisp)
            Map = np.e**((self.DisparityMap[:,:,1,i] - self.DisparityMap[:,:,1,i].max())/(Tau/2))
            axs[1,i].imshow(np.transpose(Map), origin = 'lower', cmap = 'binary')
        f.colorbar(I, ax = axs[0,1])
        return f, axs

    def AnimatedDisparitiesMaps(self, MaxDepth = 1., DepthSigma = 0.03, NSteps = 30, Tau = 0.05, cmap = 'binary'):
        f, axs = plt.subplots(1,2)
        
        MinDepth = 1/min(self._DisparityRange[1], abs(self.DisparityMap[:,:,0,:]).max())
        Mod = (MaxDepth - MinDepth)/NSteps
        DMaps = [-(-1)**i * self.DisparityMap[:,:,0,i] for i in range(2)]
        print("Ranging disparities from {0} to {1}".format(1/MaxDepth, 1/MinDepth))
        for i in range(2):
            DMaps[i][np.where(self.DisparityMap[:,:,1,i] < self.DisparityMap[:,:,1,i].max() - Tau)] = np.inf
        def GetMaps(Depth):
            Disparity = 1/Depth
            return [np.transpose(np.e**(-((DMaps[i] - Disparity)**2 / DepthSigma**2))) for i in range(2)]
        CurrentDepth = MinDepth
        Maps = GetMaps(CurrentDepth)
        Images = [axs[i].imshow(Maps[i], origin = 'lower', cmap = cmap) for i in range(2)]
        while True:
            CurrentDepth += Mod
            if CurrentDepth >= MaxDepth or CurrentDepth <= max(MinDepth, abs(Mod)):
                Mod *= -1
            Maps = GetMaps(CurrentDepth)
            for i in range(2):
                Images[i].set_data(Maps[i])
            axs[0].set_title('Depth = {0:.3f}, Disp. = {1:.1f}'.format(CurrentDepth, 1/CurrentDepth))
            plt.pause(0.1)



    def ConsiderAddingCP(self, event, CPType):
        if self.ComparedPoints[event.cameraIndex, CPType] < self._MaxSimultaneousPointsPerType[CPType]:
            AverageActivity = self.Maps[event.cameraIndex].GetAverageActivity(event.location, event.timestamp)
            if AverageActivity >= self._MinCPAverageActivity:
                Add = True
                for CP in self.ComparisonPoints[event.cameraIndex]:
                    if CPType != CP[1]:
                        continue
                    if abs(event.location - CP[0]).max() <= self._ComparisonRadius:
                        Add = False
                        break
                if Add:
                    self.AddCP(event, CPType)

    def AddCP(self, event, CPType):
        EpipolarEquation = self._EpipolarMatrix.dot(np.array([event.location[0], event.location[1], 1]))
        EpipolarEquation /= np.linalg.norm(EpipolarEquation[:2])
        self.ComparedPoints[event.cameraIndex, CPType] += 1
        self.ComparisonPoints[event.cameraIndex].append((np.array(event.location), CPType, event.timestamp, EpipolarEquation, []))
        event.Attach(TrackerEvent, TrackerLocation = np.array(event.location))
        self.Log("Added CPType {0} for camera {1} at {2}".format(CPType, event.cameraIndex, event.location))

    def TryMatchingEventToStereoCPs(self, event):
        Matched = []
        for nCP, CP in enumerate(self.ComparisonPoints[1-event.cameraIndex]):
            if event.timestamp - CP[2] < self.Maps[1-event.cameraIndex].Tau: # No match can be trusted under 1 Tau
                continue
            delta = event.location[0] - CP[0][0]
            if delta < self._DisparityRanges[event.cameraIndex][0] or delta > self._DisparityRanges[event.cameraIndex][1]:
                continue
            Distance = np.array([event.location[0], event.location[1], 1]).dot(CP[3]) # Compute distance from this point to the compared point epipolar line
            if abs(Distance) <= self._ComparisonRadius:
                ProjectedEvent = (event.location - Distance * CP[3][:2]).astype(int) # Recover the location of point on the epipolar line that corresponds to this event
                #if ProjectedEvent[0] == CP[0][0]: # We just want to avoid infinite disparity values
                #    continue

                EventSignatures = self.Maps[event.cameraIndex].GetSignatures(ProjectedEvent, event.timestamp) # Recover specific signatures of the map on that epipolar line for that CP on the other camera
                CPSignatures = self.Maps[1-event.cameraIndex].GetSignatures(CP[0], event.timestamp)

                if self.Match(EventSignatures, CPSignatures) > self.MetricThreshold:
                    GlobalMatch, Location = self.AddLocalMatch(event.location[0], CP)
                    if GlobalMatch:
                        Offset = Location - CP[0][0]
                        self.LogSuccess("Matched y = {0}, x[0] = {{{1}}} & x[1] = {{{2}}}, offset = {3}".format(ProjectedEvent[1], 1-event.cameraIndex, str(event.cameraIndex), Offset).format(CP[0][0], Location))
                        if len(self.DisparitiesStored[event.cameraIndex]) == self._NDisparitiesStored:
                            self.DisparitiesStored[event.cameraIndex].pop(0)
                        self.DisparitiesStored[event.cameraIndex].append((np.array(CP[0]), event.timestamp, Offset))
                        self.DisparityMap[CP[0][0],CP[0][1],:,1-event.cameraIndex] = np.array([-Offset, event.timestamp])
                        self.DisparityMap[Location,CP[0][1],:,event.cameraIndex] = np.array([Offset, event.timestamp])
                        Matched += [nCP]
                        self.MatchedCPs[CP[1]] += 1
        for nCP in reversed(Matched):
            self.ComparedPoints[1-event.cameraIndex,self.ComparisonPoints[1-event.cameraIndex][nCP][1]] -= 1
            self.ComparisonPoints[1-event.cameraIndex].pop(nCP)
        return bool(Matched)

    def GetDisparityEvent(self, event):
        if event.timestamp - self.DisparityMap[tuple(event.location) +(1,event.cameraIndex)] < self._Tau / 2: # We assume the disparity here still valid and dont update it
            return True
        LocalPatch = self.DisparityMap[event.location[0]-self._DisparityPatchRadius:event.location[0]+self._DisparityPatchRadius+1,event.location[1]-self._DisparityPatchRadius:event.location[1]+self._DisparityPatchRadius+1,:,event.cameraIndex]
        DisparitiesSearched = set()
        for Disparity, t in LocalPatch.reshape(((self._DisparityPatchRadius*2 + 1)**2 , 2)):
            if event.timestamp - t < 4*self._Tau:
                for delta in range(int(Disparity)-self._DisparitySearchRadius, int(Disparity)+self._DisparitySearchRadius+1):
                    if delta < self._DisparityRanges[event.cameraIndex][0] or delta > self._DisparityRanges[event.cameraIndex][1]:
                        continue
                    DisparitiesSearched.add(delta)
        if len(DisparitiesSearched) == 0:
            return False

        EventSignatures = self.Maps[event.cameraIndex].GetSignatures(event.location, event.timestamp)
        BestMatch = [0, None]
        for Disparity in DisparitiesSearched:
            if Disparity == 0:
                continue
            XLocation = event.location[0] - Disparity
            if XLocation < self._ComparisonRadius or XLocation >= self.UsedGeometry[0] - self._ComparisonRadius:
                continue
            StereoSignatures = self.Maps[1-event.cameraIndex].GetSignatures(np.array([XLocation, event.location[1]]), event.timestamp)
            MatchValue = self.Match(EventSignatures, StereoSignatures)
            if MatchValue > BestMatch[0]:
                BestMatch = [MatchValue, Disparity]
        if BestMatch[0] > self.MetricThreshold:
            Disparity = BestMatch[1]
            XLocation = event.location[0] - Disparity
            event.Attach(DisparityEvent, disparity = 1./abs(Disparity))
            self.DisparityMap[event.location[0], event.location[1],:,event.cameraIndex] = np.array([Disparity, event.timestamp])
            self.DisparityMap[XLocation, event.location[1],:,1-event.cameraIndex] = np.array([-Disparity, event.timestamp])
            return True
        else:
            self.ConsiderAddingCP(event, 1)
            return False

    def AddLocalMatch(self, X, CP):
        for PossibleLocation in CP[4]:
            if abs(PossibleLocation[0] - X) <= self._MaxDistanceCluster:
                #PossibleLocation[0] = (PossibleLocation[0] * PossibleLocation[1] + X) / (PossibleLocation[1]+1)
                PossibleLocation[1] += 1
                if PossibleLocation[1] == self._MaxMatches:
                    return True, PossibleLocation[0]
                return False, None
        CP[4].append([X,1])
        return False, None

    def Match(self, SigsA, SigsB):
        MatchValue = 1.
        for nMetric in self.MetricIndexes:
            NA, NB = np.linalg.norm(SigsA[nMetric]), np.linalg.norm(SigsB[nMetric])
            if NA == 0 or NB == 0:
                return False
            MatchValue *= SigsA[nMetric].dot(SigsB[nMetric]) / (NA * NB)
        return MatchValue

    def FullMapping(self):
        t = self.__Framework__.PropagatedEvent.timestamp
        FullDisparityMaps = np.zeros(tuple(self.UsedGeometry) + (2,), dtype = int)
        for y in range(self._ComparisonRadius, self.UsedGeometry[1]-self._ComparisonRadius):
            print("y = {0}/{1}".format(y - self._ComparisonRadius, self.UsedGeometry[1]-2*self._ComparisonRadius))
            Distances = np.zeros((self.UsedGeometry[0], 2), dtype = float)
            for nCam in range(2):
                for xPoint in range(self._ComparisonRadius, self.UsedGeometry[0]-self._ComparisonRadius):
                    PointSigs = self.Maps[nCam].GetSignatures((xPoint, y), t)
                    if PointSigs[0].mean() < 1:
                        continue
                    for xMatch in range(self._ComparisonRadius, self.UsedGeometry[0]-self._ComparisonRadius):
                        MatchSigs = self.Maps[1-nCam].GetSignatures((xMatch, y), t)
                        V = self.Match(PointSigs, MatchSigs)
                        if V > Distances[xPoint, nCam]:
                            Distances[xPoint, nCam] = V
                            FullDisparityMaps[xPoint, y, nCam] = xMatch - xPoint
        return FullDisparityMaps


class AnalysisMapClass:
    def __init__(self, Geometry, Radius, Tau, AveragingRadius = 0):
        self.Geometry = Geometry
        self.Radius = int(Radius)
        self.AveragingRadius = AveragingRadius

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
        
        Signatures = [self.ActivityMap[:,YUsed] * DecayRow, self.DistanceMap[:,YUsed] / np.maximum(0.001, self.ActivityMap[:,YUsed]), self.SigmaMap[:,YUsed] / np.maximum(0.001, self.ActivityMap[:,YUsed])]
        for nAverage in range(self.AveragingRadius):
            for nSig, Sig in enumerate(Signatures):
                Signatures[nSig] = (Sig[1:] + Sig[:-1])/2
        return Signatures

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
        
        Signatures = [self.ActivityMap[XMin:XMax,YUsed] * DecayRow, self.DistanceMap[XMin:XMax,YUsed] / np.maximum(0.001, self.ActivityMap[XMin:XMax,YUsed]), self.SigmaMap[XMin:XMax,YUsed] / np.maximum(0.001, self.ActivityMap[XMin:XMax,YUsed])]

        for nAverage in range(self.AveragingRadius):
            for nSig, Sig in enumerate(Signatures):
                Signatures[nSig] = (Sig[1:] + Sig[:-1])/2
        return Signatures

