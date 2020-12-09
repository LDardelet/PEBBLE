import numpy as np
from Framework import Module, Event, DisparityEvent, TrackerEvent
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter

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
        self._KeyPointsComparisonRadiusRatio = 0.5
        self._Tau = 0.005
        self._MaxSimultaneousPointsPerType = np.array([100, 100])

        self._MinCPAverageActivityRadiusRatio = 3. / 10
        self._LifespanTauRatio = 2 #After N tau, we consider no match as a failure and remove that comparison point
        self._CleanupTauRatio = 0.5 
        self._SignaturesExponents = [1,1,1]
        self._yAverageSignatureNullAverage = True

        self._MatchThreshold = 0.9
        self._MaxDistanceCluster = 0
        self._MinMatchMargin = 0.9

        self._AverageRadius = 0

        self._NDisparitiesStored = 200
        self._DisparityPatchRadius = 2
        self._DisparitySearchRadius = 1

        self._DisparityRange = [0, np.inf]

        self._EpipolarMatrix = [[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
        self._MonitorDt = 0. # By default, a module does not stode any date over time.
        self._MonitoredVariables = [('DisparityMap', np.array),
                                    ('InputEvents', float),
                                    ('MatchedEvents', float)]

    def _InitializeModule(self, **kwargs):
        self._EpipolarMatrix = np.array(self._EpipolarMatrix)
        self.UsedGeometry = np.array(self.Geometry)[:2]

        self.MetricThreshold = self._MatchThreshold ** np.sum(self._SignaturesExponents)

        self._KeyPointsComparisonRadius = self._ComparisonRadius * self._KeyPointsComparisonRadiusRatio
        self._MinCPAverageActivity = self._ComparisonRadius * self._MinCPAverageActivityRadiusRatio

        self.CPLifespan = self._Tau * self._LifespanTauRatio
        self.CleanupDt = self._Tau * self._CleanupTauRatio

        self.Maps = (AnalysisMapClass(self.UsedGeometry, self._ComparisonRadius, self._Tau, self._yAverageSignatureNullAverage, self._AverageRadius), AnalysisMapClass(self.UsedGeometry, self._ComparisonRadius, self._Tau, self._yAverageSignatureNullAverage, self._AverageRadius))
        self.ComparisonPoints = ([],[])
        self.DisparitiesStored = ([],[])
        self.DisparityMap = np.zeros(tuple(self.UsedGeometry) + (2,2)) # 0 default value seems legit as it pushes all points to infinity
        self.DisparityMap[:,:,1,:] = -np.inf
        self.ComparedPoints = np.array([[0,0],[0,0]], dtype = int)

        self.MatchedCPs = np.array([0,0])
        self.InputEvents = 0
        self.MatchedEvents = 0
        self.CPEvents = 0
        self.LastTDecay = -np.inf

        self._DisparityRanges = [[-self._DisparityRange[1], -self._DisparityRange[0]], [self._DisparityRange[0], self._DisparityRange[1]]]
        
        return True

    def _OnEventModule(self, event):
        self.Maps[event.cameraIndex].OnEvent(event)
        if (event.location < self._ComparisonRadius).any() or (event.location >= self.UsedGeometry - self._ComparisonRadius).any(): # This event cannot be set as CP, nor can it be matched to anything as its outside the borders
            return event


        #self.ConsiderAddingCP(event, CPType = 0)
        
        Decay = np.e**((self.LastTDecay - event.timestamp)/self._Tau)
        self.InputEvents *= Decay
        self.MatchedEvents *= Decay
        self.CPEvents *= Decay
        self.LastTDecay = event.timestamp

        self.InputEvents += 1
        Match = self.TryMatchingEventToStereoCPs(event)
        self.CPEvents += int(Match)

        if not Match:
            Match = self.GetDisparityEvent(event)
            self.MatchedEvents += int(Match)

        return event

    def KeepCP(self, t, CP):
        if t - CP[2] > self.CPLifespan:
            self.LogWarning("Removed CP at {0} as its still unmatched".format(CP[0]))
            return False
        else:
            return True

    def PlotDisparitiesMaps(self, PastIndex = None, f_axs_images = None, Tau = None, MinDispInput = None, MaxDispInput = None, origin = 'lower', loop = True, BiCamera = False, AddSTContext = False, SaveLocationPrefix = '', blurrSigma = 0):
        if SaveLocationPrefix and loop:
            loop = False
            print("Canceling loop to ensure data saving")
        NRows = 1+AddSTContext
        NCols = 1+BiCamera
        if f_axs_images is None:
            f, axs = plt.subplots(NRows,NCols)
            f.tight_layout()
            if NCols == 1 and NRows ==1:
                axs = np.array([[axs]])
            elif NRows == 1 and NCols == 2:
                axs = axs.reshape((1,2))
            elif NCols == 1 and NRows == 2:
                axs = axs.reshape((2,1))
            Images = None
            for nRow in range(NRows):
                for nCol in range(NCols):
                    axs[nRow,nCol].tick_params('both', bottom = False, labelbottom = False, left = False, labelleft = False)
        else:
            f, axs, Images = f_axs_images
        NPixels = self.UsedGeometry.prod()
        if MaxDispInput is None:
            MaxDisp = int(Map[:,:,0,:].max())
        else:
            MaxDisp = MaxDispInput
        if MinDispInput is None:
            MinDisp = int(Map[:,:,0,:].min())
        else:
            MinDisp = MinDispInput
        if MinDispInput is None:
            for Disparity in range(MinDisp, MaxDisp + 1):
                if (Map[:,:,0,:] < Disparity).sum() < 0.01 * NPixels:
                    MinDisp = Disparity
        if MaxDispInput is None:
            for Disparity in range(MinDisp, MaxDisp + 1):
                if (Map[:,:,0,:] > Disparity).sum() < 0.01 * NPixels:
                    MaxDisp = Disparity
                    break
        if MaxDispInput is None or MinDispInput is None:
            print("Ranging disparities from {0} to {1}".format(MinDisp, MaxDisp))
        if Tau is None:
            Tau = self._Tau

        if PastIndex == 'all':
            Images = None
            while True:
                for Index in range(len(self.History['t'])):
                    f, axs, Images = self.PlotDisparitiesMaps(Index, (f, axs, Images), Tau, MinDisp, MaxDisp, origin, False, BiCamera, AddSTContext, SaveLocationPrefix, blurrSigma)
                    plt.pause(0.1)
                if not loop:
                    return
            return
        elif PastIndex is None:
            Map = self.DisparityMap
        else:
            Map = self.History['DisparityMap'][PastIndex]

        if Images is None:
            StoredImages = [[],[]]
        for nax, nCam in enumerate([0,1][1-BiCamera:]):
            DMap = np.array(Map[:,:,0,nCam])
            DMap[np.where(Map[:,:,1,nCam] - Map[:,:,1,nCam].max() < -Tau)] = 0
            if blurrSigma:
                DMap = gaussian_filter(DMap, blurrSigma)
            if AddSTContext:
                SMap = np.e**((Map[:,:,1,nCam] - Map[:,:,1,nCam].max())/(Tau/2))
                if blurrSigma:
                    DMap = gaussian_filter(DMap, blurrSigma)
            if PastIndex is None:
                t = self.__Framework__.PropagatedEvent.timestamp
            else:
                t = self.History['t'][PastIndex]
            axs[0,nax].set_title("t: {0:.3f}".format(t))
            if Images is None:
                StoredImages[nax] += [axs[0,nax].imshow(np.transpose(DMap), origin = origin, cmap = 'inferno', vmin = MinDisp, vmax = MaxDisp)]
                if AddSTContext:
                    StoredImages[nax] += [axs[1,nax].imshow(np.transpose(SMap), origin = origin, cmap = 'binary')]
            else:
                Images[nax][0].set_data(np.transpose(DMap))
                if AddSTContext:
                    Images[nax][1].set_data(np.transpose(SMap))
                StoredImages = Images

        if Images is None:
            StickToImage = StoredImages[int(BiCamera)][0]
            StickOnAx = axs[0,int(BiCamera)]
            f.colorbar(StickToImage, ax = StickOnAx)
        if SaveLocationPrefix:
            if PastIndex is None:
                suffix = "_{0:.3f}".format(self.__Framework__.PropagatedEvent.timestamp)
            else:
                suffix = "_{0}".format(PastIndex)
            f.savefig(SaveLocationPrefix + suffix + ".png")
        return f, axs, StoredImages

    def AnimatedDisparitiesMaps(self, MaxDepth = 1., MinDepth = None, MaxDisp = None, DepthSigma = 0.03, NSteps = 30, Tau = 0.05, cmap = 'binary'):
        f, axs = plt.subplots(1,2)
        
        if MinDepth is None and MaxDisp is None:
            MinDepth = 1/min(self._DisparityRange[1], abs(self.DisparityMap[:,:,0,:]).max())
        elif not MaxDisp is None:
            MinDepth = 1/MaxDisp
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

    def TryMatchingEventToStereoCPs(self, event):
        MatchedAndRemoved = []
        Matched = False
        for nCP, CP in enumerate(self.ComparisonPoints[1-event.cameraIndex]):
            if event.timestamp - CP[2] < self.Maps[1-event.cameraIndex].Tau: # No match can be trusted under 1 Tau
                continue
            delta = event.location[0] - CP[0][0]
            if not self.KeepCP(event.timestamp, CP):
                MatchedAndRemoved += [nCP]
                continue
            if Matched:
                continue
            if delta < self._DisparityRanges[event.cameraIndex][0] or delta > self._DisparityRanges[event.cameraIndex][1]:
                continue
            Distance = np.array([event.location[0], event.location[1], 1]).dot(CP[3]) # Compute distance from this point to the compared point epipolar line
            if abs(Distance) <= self._KeyPointsComparisonRadius:
                ProjectedEvent = (event.location - Distance * CP[3][:2]).astype(int) # Recover the location of point on the epipolar line that corresponds to this event
                #if ProjectedEvent[0] == CP[0][0]: # We just want to avoid infinite disparity values
                #    continue

                EventSignatures = self.Maps[event.cameraIndex].GetSignatures(ProjectedEvent, event.timestamp) # Recover specific signatures of the map on that epipolar line for that CP on the other camera
                CPSignatures = self.Maps[1-event.cameraIndex].GetSignatures(CP[0], event.timestamp)

                M = self.Match(EventSignatures, CPSignatures)
                if M >= self.MetricThreshold * self._MinMatchMargin:
                    GlobalMatch, Location = self.AddLocalMatch(event.location[0], CP, M)
                if M >= self.MetricThreshold:
                    if GlobalMatch:
                        Offset = Location - CP[0][0]
                        self.LogSuccess("Matched y = {0}, x[0] = {{{1}}} & x[1] = {{{2}}}, offset = {3}".format(ProjectedEvent[1], 1-event.cameraIndex, str(event.cameraIndex), Offset).format(CP[0][0], Location))
                        if len(self.DisparitiesStored[event.cameraIndex]) == self._NDisparitiesStored:
                            self.DisparitiesStored[event.cameraIndex].pop(0)
                        self.DisparitiesStored[event.cameraIndex].append((np.array(CP[0]), event.timestamp, Offset))
                        self.DisparityMap[CP[0][0],CP[0][1],:,1-event.cameraIndex] = np.array([-Offset, event.timestamp])
                        self.DisparityMap[Location,CP[0][1],:,event.cameraIndex] = np.array([Offset, event.timestamp])
                        event.Attach(DisparityEvent, disparity = abs(Offset), disparityLocation = np.array(event.location))
                        MatchedAndRemoved += [nCP]
                        self.MatchedCPs[CP[1]] += 1
                        Matched = True
        for nCP in reversed(MatchedAndRemoved):
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
            self.ConsiderAddingCP(event, 0)
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
            event.Attach(DisparityEvent, disparity = abs(Disparity), disparityLocation = np.array(event.location))
            self.DisparityMap[event.location[0], event.location[1],:,event.cameraIndex] = np.array([Disparity, event.timestamp])
            self.DisparityMap[XLocation, event.location[1],:,1-event.cameraIndex] = np.array([-Disparity, event.timestamp])
            return True
        else:
            self.ConsiderAddingCP(event, 1)
            return False

    def AddCP(self, event, CPType):
        EpipolarEquation = self._EpipolarMatrix.dot(np.array([event.location[0], event.location[1], 1]))
        EpipolarEquation /= np.linalg.norm(EpipolarEquation[:2])
        self.ComparedPoints[event.cameraIndex, CPType] += 1
        self.ComparisonPoints[event.cameraIndex].append((np.array(event.location), CPType, event.timestamp, EpipolarEquation, ([None, 0], [None, 0])))
        event.Attach(TrackerEvent, TrackerLocation = np.array(event.location))
        self.Log("Added CPType {0} for camera {1} at {2}".format(CPType, event.cameraIndex, event.location))

    def AddLocalMatch(self, X, CP, Value):
        if CP[4][0][0] is None:
            CP[4][0][:] = [X, Value]
            return False, None
        if CP[4][1][0] is None:
            CP[4][1][:] = [X, Value]
            if Value > CP[4][0][1]:
                CP[4][0][:], CP[4][1][:] = CP[4][1][:], CP[4][0][:]
            return False, None

        if abs(X - CP[4][0][0]) <= self._MaxDistanceCluster: # If we match again the best one
            CP[4][0][1] = max(CP[4][0][1], Value)
            if CP[4][0][1] >= self.MetricThreshold and CP[4][1][1] <= CP[4][0][1] * self._MinMatchMargin:
                return True, X
            return False, None
        if abs(X - CP[4][1][0]) <= self._MaxDistanceCluster:
            CP[4][1][1] = max(CP[4][1][1], Value)
            if CP[4][1][1] > CP[4][0][1]:
                CP[4][0][:], CP[4][1][:] = CP[4][1][:], CP[4][0][:]
            if CP[4][0][1] >= self.MetricThreshold and CP[4][1][1] <= CP[4][0][1] * self._MinMatchMargin:
                return True, X
            return False, None
        if Value >= CP[4][1][1]:
            CP[4][1][:] = [X, Value]
            if Value > CP[4][0][1]:
                CP[4][0][:], CP[4][1][:] = CP[4][1][:], CP[4][0][:]
        return False, None

    def Match(self, SigsA, SigsB):
        MatchValue = 1.
        for nMetric, Exponent in enumerate(self._SignaturesExponents):
            NA, NB = np.linalg.norm(SigsA[nMetric]), np.linalg.norm(SigsB[nMetric])
            if NA == 0 or NB == 0:
                return False
            MatchValue *= ((1+(SigsA[nMetric].dot(SigsB[nMetric]) / (NA * NB)))/2)**Exponent
        return MatchValue

    def FullMapping(self, yOffset = 0, AutoMatch = False, UseRanges = True):
        t = self.__Framework__.PropagatedEvent.timestamp
        DisparityMaps = np.inf * np.ones(tuple(self.UsedGeometry) + (2,), dtype = int)
        CertaintyMaps = np.inf * np.ones(tuple(self.UsedGeometry) + (2,), dtype = float)
        for y in range(self._ComparisonRadius + max(0, yOffset), self.UsedGeometry[1]-self._ComparisonRadius + min(0, yOffset)):
            print("y = {0}/{1}".format(y - self._ComparisonRadius - max(0, yOffset), self.UsedGeometry[1]-2*self._ComparisonRadius - abs(yOffset)))
            for nCam in range(2):
                if UseRanges:
                    minOffset, maxOffset = self._DisparityRanges[nCam]
                else:
                    minOffset, maxOffset = -np.inf, np.inf
                if AutoMatch:
                    nOpp = nCam
                else:
                    nOpp = 1-nCam
                for xPoint in range(self._ComparisonRadius, self.UsedGeometry[0]-self._ComparisonRadius):
                    DBest = 0
                    PointSigs = self.Maps[nCam].GetSignatures((xPoint, y), t)
                    if PointSigs[0].mean() < 1:
                        continue
                    for xMatch in range(max(self._ComparisonRadius, xPoint + minOffset), min(self.UsedGeometry[0]-self._ComparisonRadius, xPoint + maxOffset)):
                        MatchSigs = self.Maps[nOpp].GetSignatures((xMatch, y - yOffset), t)
                        V = self.Match(PointSigs, MatchSigs)
                        if V > DBest:
                            DBest, CertaintyMaps[xPoint, y, nCam] = V, V - DBest
                            DisparityMaps[xPoint, y, nCam] = xMatch - xPoint
                        else:
                            CertaintyMaps[xPoint, y, nCam] = min(CertaintyMaps[xPoint, y, nCam], DBest - V)
        return DisparityMaps, CertaintyMaps

class AnalysisMapClass:
    def __init__(self, Geometry, Radius, Tau, ySigNullMean, AveragingRadius = 0):
        self.Geometry = Geometry
        self.Radius = int(Radius)
        self.AveragingRadius = AveragingRadius
        self.ySigNullMean = ySigNullMean

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
        
        Signatures = self._SignaturesCreation(self.ActivityMap[:,YUsed], self.DistanceMap[:,YUsed], self.SigmaMap[:,YUsed], DecayRow)

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
        
        Signatures = self._SignaturesCreation(self.ActivityMap[XMin:XMax,YUsed], self.DistanceMap[XMin:XMax,YUsed], self.SigmaMap[XMin:XMax,YUsed], DecayRow)

        return Signatures

    def _SignaturesCreation(self, AMap, DMap, SMap, DecayMap):
        MaxedAMap = np.maximum(0.001, AMap)
        Signatures = [AMap * DecayMap, DMap / MaxedAMap, SMap / MaxedAMap]
        if self.ySigNullMean:
            Signatures[1] -= Signatures[1].mean()
        for nAverage in range(self.AveragingRadius):
            for nSig, Sig in enumerate(Signatures):
                Signatures[nSig] = (Sig[1:] + Sig[:-1])/2
        return Signatures
