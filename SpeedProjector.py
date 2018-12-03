import numpy as np
import geometry
import sys
import matplotlib.pyplot as plt

from HoughCornerDetector import GetCorners
from scipy import ndimage
import Plotting_methods

from Framework import Module

class LocalProjector(Module):

    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Tool to create a 2D projection of the STContext onto a plane t = cst along different speeds.
        Should allow to recove the actual speed and generator of the movement
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__ReferencesAsked__ = ['Memory']
        self.__Type__ = 'Computation'

        self._R_Projection = 0.5
        self._ExpansionFactor = 1.

        self._SelectionMode = 'grid'
        self._DetectorPointsGrid = (5, 4)
        self._DetectorDefaultWindowsLength = 30
        self._DetectorMinActivityForStart = 50

        self._HalfNumberOfSpeeds = 3
        self._Initial_dv_MAX = 800
        self._Relative_precision_aimed = 0.01

        self._DensityDefinition = 3 # In dpp
        self._MaskDensityRatio = 0.3

        self._AskLocationAtTS = 0.020
        self._TW = 0.02
        self._SnapshotDt = 0.01

        self._DecayRatio = 5.

        self._KeepBestSpeeds = 3

        self._ModSpeeds = True
        self._SpeedModRatio = 0.02
        self._RelativeCorrectionThreshold = 0.0 # Relative distance (to ObservationRadius) over which correction speed is made. 0 implies that mod speed is applied in any situation
        self._UpdatePT = True

        self._DynamicPositionReference = False
        self._AutoSpeedRestart = True
    
    def _Initialize(self, **kwargs):
        Module._Initialize(self, **kwargs)

        self.MapSize = self.__Framework__.Tools[self.__CreationReferences__['Memory']].STContext.shape[:2]

        self.__Started__ = False
        self._Precision_aimed = self._Relative_precision_aimed*self._Initial_dv_MAX
        self.ActiveSpeeds = []
        self.Speeds = []
        self.IsActive = []
        self.DefaultObservationWindows = []

        self.DetectorNumberOfPoints = self._DetectorPointsGrid[0] * self._DetectorPointsGrid[1]

        self.ToInitializeSpeed = []
        self.SpeedStartTime = []
        self.SpeedStopTime = []
        self.SpeedProjectionTime = []
        self.SpeedNorms = []
        self.SpeedErrors = []
        self.AimedSpeeds = []
        self.SpeedTimeConstants = []
        self.Displacements = []
        self.ActiveSpeedsInZone = {}

        self.OWAPT = [] # Stands for "Observation Window at Projection Time". Means to stabilize any object present at start time, without adding new ones yet to appear. It is given at start time as it moves as time goes on.
        self.ObservationRadiuses = []
        self.DecayingMaps = []
        self.StreaksMaps = []

        self.MeanPositionsReferences = []
        self.DMReferences = []
        self.DMGradientReferences = []
        self.SpeedAngleReferences = []
        self.SpeedReferences = []
        self.ModSpeedPixels = []
        self.LowerPartModSpeed = []
        self.MeanPositionTimeReferences = []

        self.CurrentMeanPositions = []

        self.NZones = 0
        self.NActiveZones = 0
        self.Zones = {}
        self.ToCleanZones = []
        self.RegenNeeded = False

        self.LastConsideredEventsTs = []
        self.LocalPaddings = []
        self.InitialSpeeds = []

        self.LastSnapshotT = 0.
        self.DMSnaps = []
        self.SMSnaps = []
        self.PosSnaps = []
        self.TsSnaps = []
        self.DisplacementSnaps = []
        self.SpeedErrorSnaps = []
        self.AimedSpeedSnaps = []
        self.ProjectedEvents = []

        self.SpeedsChangesHistory = []
        self.ProjectionTimesHistory = []
        self.CurrentBaseDisplacement = []

        return True

    def _OnEvent(self, event):
        if not self.__Started__:
            if event >= self._AskLocationAtTS:
                if self._SelectionMode == 'ask':
                    if self.AskLocationAndStart():
                        self.LastSnapshotT = event.timestamp
                elif self._SelectionMode == 'grid':
                    self._GridSelectCorners()
                    self.LastSnapshotT = event.timestamp
                elif self._SelectionMode == 'auto':
                    self._AutoSelectCorners()
                    self.LastSnapshotT = event.timestamp
            return event

        if self.ToInitializeSpeed > 0:
            for speed_id in self.ToInitializeSpeed:
                self.SpeedStartTime[speed_id] = event.timestamp
                self.SpeedProjectionTime[speed_id] = event.timestamp
                self.ProjectionTimesHistory[speed_id][0] = event.timestamp
                self.SpeedsChangesHistory[speed_id] += [(event.timestamp, np.array(self.Speeds[speed_id]))]
            self.ToInitializeSpeed = []

        for speed_id in self.ActiveSpeeds:
            self._ProjectEventWithSpeed(event, speed_id)

        if event.timestamp - self.LastSnapshotT >= self._SnapshotDt:
            self.DMSnaps += [[]]
            self.SMSnaps += [[]]
            self.PosSnaps += [[]]
            self.DisplacementSnaps += [[]]
            self.SpeedErrorSnaps += [[]]
            self.AimedSpeedSnaps += [[]]

            for speed_id in range(len(self.Speeds)):
                if self.IsActive[speed_id]:
                    DeltaT = event.timestamp - self.LastConsideredEventsTs[speed_id]
                    if DeltaT:
                        self.LastConsideredEventsTs[speed_id] = event.timestamp
                        self.DecayingMaps[speed_id] = self.DecayingMaps[speed_id]*np.e**(-(DeltaT/self.SpeedTimeConstants[speed_id]))
                    self.DMSnaps[-1] += [np.array(self.DecayingMaps[speed_id])]
                    self.SMSnaps[-1] += [np.array(self.StreaksMaps[speed_id])]
                    self.PosSnaps[-1] += [np.array(self.CurrentMeanPositions[speed_id])]
                    self.DisplacementSnaps[-1] += [self.Displacements[speed_id]]
                    self.SpeedErrorSnaps[-1] += [np.array(self.SpeedErrors[speed_id])]
                    self.AimedSpeedSnaps[-1] += [np.array(self.AimedSpeeds[speed_id])]
                else:
                    self.DMSnaps[-1] += [None]
                    self.SMSnaps[-1] += [None]
                    self.PosSnaps[-1] += [None]
                    self.DisplacementSnaps[-1] += [None]
                    self.SpeedErrorSnaps[-1] += [np.array([0., 0.])]
                    self.AimedSpeedSnaps[-1] += [np.array([0., 0.])]

            self.LastSnapshotT = event.timestamp
            self.TsSnaps += [event.timestamp]

            self.__Framework__.Tools[self.__CreationReferences__['Memory']].GetSnapshot()

        for Zone in self.ToCleanZones:
            CanBeCleansed = True
            for speed_id in self.Zones[Zone]:
                if event.timestamp - self.SpeedStartTime[speed_id] < self.SpeedTimeConstants[speed_id]:
                    CanBeCleansed = False
                    break
            if CanBeCleansed:
                SMSum = (self.StreaksMaps[speed_id] > 0).sum()
                if SMSum == 0:
                    continue
                MeanValues = [float((self.StreaksMaps[speed_id]).sum())/SMSum for speed_id in self.Zones[Zone] if self.IsActive[speed_id]]
                IDs = [speed_id for speed_id in self.Zones[Zone] if self.IsActive[speed_id]]
                SortedIDs = np.argsort(MeanValues)
                for local_speed_id in SortedIDs[:-self._KeepBestSpeeds]:
                    speed_id = IDs[local_speed_id]
                    self.ActiveSpeeds.remove(speed_id)
                    self.IsActive[speed_id] = False
                    self.ActiveSpeedsInZone[Zone] -= 1
                    self.SpeedStopTime[speed_id] = event.timestamp
                    if not self.ActiveSpeedsInZone[Zone]:
                        self.NActiveZones -= 1
                        self.RegenNeeded = True
                print "Cleansed {0} speed considered wrong for zone {1} at t = {2}".format(len(SortedIDs[:-self._KeepBestSpeeds].tolist()), Zone, event.timestamp)
                self.ToCleanZones.remove(Zone)
        if self.RegenNeeded and self._AutoSpeedRestart:
            self._RegenSpeeds()
        return event

    def _ProjectEventWithSpeed(self, event, speed_id):
        self._UpdateDisplacement(event.timestamp, speed_id)

        if self.Displacements[speed_id][0] + self.OWAPT[speed_id][0] < 0 or self.Displacements[speed_id][1] + self.OWAPT[speed_id][1] < 0 or self.Displacements[speed_id][0] + self.OWAPT[speed_id][2] >= self.MapSize[0] or self.Displacements[speed_id][1] + self.OWAPT[speed_id][3] >= self.MapSize[1]:
            self.ActiveSpeeds.remove(speed_id)
            self.ActiveSpeedsInZone[self.OWAPT[speed_id]] -= 1
            self.SpeedStopTime[speed_id] = event.timestamp
            if not self.ActiveSpeedsInZone[self.OWAPT[speed_id]]:
                self.NActiveZones -= 1
                self.RegenNeeded = True
            self.IsActive[speed_id] = False
            return None

        x0 = event.location[0] - self.Displacements[speed_id][0]
        y0 = event.location[1] - self.Displacements[speed_id][1]
        
        OW = self.OWAPT[speed_id]
        if not (OW[0] <= x0 <= OW[2] and OW[1] <= y0 <= OW[3]):
            return None

        self.ProjectedEvents[speed_id] += 1
        DeltaT = event.timestamp - self.LastConsideredEventsTs[speed_id]
        EvolutionFactor = np.e**(-(DeltaT/self.SpeedTimeConstants[speed_id]))

        self.LastConsideredEventsTs[speed_id] = event.timestamp

        self.DecayingMaps[speed_id] = self.DecayingMaps[speed_id] * EvolutionFactor

        for x in range(int(np.floor(self._DensityDefinition*((x0 - self._R_Projection) - OW[0]))), int(np.ceil(self._DensityDefinition*((x0 + self._R_Projection) - OW[0])))):
            for y in range(int(np.floor(self._DensityDefinition*((y0 - self._R_Projection) - OW[1]))), int(np.ceil(self._DensityDefinition*((y0 + self._R_Projection) - OW[1])))):
                self.DecayingMaps[speed_id][x,y] += 1
                self.StreaksMaps[speed_id][x,y] += 1

        if self._UpdatePT:
            CanUpdatePT = False
        if self._ModSpeeds:
            self.CurrentMeanPositions[speed_id] = self.GetMeanPosition(speed_id)

            if self.MeanPositionsReferences[speed_id] is None:
                if event.timestamp - self.SpeedStartTime[speed_id] > self.SpeedTimeConstants[speed_id] and (self._SelectionMode != 'grid' or self.DecayingMaps[speed_id].sum() > self._DetectorMinActivityForStart) : # Hopefully we can do better. Still, the feature should construct itself over self.SpeedTimeConstants[speed_id] for each speed
                    self.DMReferences[speed_id] = np.array(self.DecayingMaps[speed_id])

                    self._GetModSpeedValuesFor(speed_id)

                    #self.MeanPositionsReferences[speed_id] = self.ObservationRadiuses[speed_id]
                    self.MeanPositionsReferences[speed_id] = np.array(self.CurrentMeanPositions[speed_id])
                    self.MeanPositionTimeReferences[speed_id] = event.timestamp


                return None
    
            #if event.timestamp - self.SpeedStartTime[speed_id] < self.SpeedTimeConstants[speed_id]:
            if event.timestamp - self.MeanPositionTimeReferences[speed_id] < self.SpeedTimeConstants[speed_id]:
                return None
            if self._DynamicPositionReference:
                PositionReference = self._ComputeDynamicPositionReference(speed_id)
            else:
                PositionReference = self.MeanPositionsReferences[speed_id]

            CurrentError = self.CurrentMeanPositions[speed_id] - PositionReference
            if (abs(CurrentError) < self.ObservationRadiuses[speed_id] * self._RelativeCorrectionThreshold).all():
                return None
            TimeEllapsed = event.timestamp - self.MeanPositionTimeReferences[speed_id]
            #TimeEllapsed = event.timestamp - self.SpeedProjectionTime[speed_id]

            self.SpeedErrors[speed_id] = CurrentError / TimeEllapsed
            self.AimedSpeeds[speed_id] = self.Speeds[speed_id] + self.SpeedErrors[speed_id]
            if (abs(self.SpeedErrors[speed_id]) < 0.01*abs(self.Speeds[speed_id])).all():
                CanUpdatePT = True
    
            self._ModifySpeed(speed_id, event.timestamp)
        
        if self._UpdatePT and CanUpdatePT and event.timestamp - self.SpeedProjectionTime[speed_id] > 6*self.SpeedTimeConstants[speed_id]: #TODO This 6 should be changed to something ... meaningful
            self.CurrentBaseDisplacement[speed_id] = np.array(self.Displacements[speed_id])
            #self.MeanPositionTimeReferences[speed_id] = event.timestamp - self.SpeedTimeConstants[speed_id]
            self.MeanPositionTimeReferences[speed_id] = event.timestamp

            self.SpeedProjectionTime[speed_id] = event.timestamp
            self.ProjectionTimesHistory[speed_id] += [event.timestamp]

    def RecoverCurrentBestSpeeds(self, OnlyAlive = False):
        MeanValues = [[self.IsActive[speed_id] * float((self.StreaksMaps[speed_id]).sum())/max(1, (self.StreaksMaps[speed_id] > 0).sum()) for speed_id in self.Zones[Zone]] for Zone in self.Zones.keys() if (not OnlyAlive or self.ActiveSpeedsInZone[Zone])]
        SortedIDs = [np.array(IDs)[np.argsort(MV)] for MV, IDs in zip(MeanValues, self.Zones.values())]
        FinalIDs = [V[-1] for V in SortedIDs]

        SortedIndexes = np.argsort(FinalIDs)
        Zones = [Zone[4] for Zone in self.Zones.keys()]
        SortedFinalDuos = [(FinalIDs[index], Zones[index]) for index in SortedIndexes]

        return SortedFinalDuos

    def _ModifySpeed(self, speed_id, t):
        #NewSpeed = self.Speeds[speed_id] + SpeedError * self.SpeedModRatio
        NewSpeed = self.Speeds[speed_id] + self.SpeedErrors[speed_id] / (0.3 * (self.DecayingMaps[speed_id].sum() / self._DensityDefinition ** 2))
        self.Speeds[speed_id] = NewSpeed
        self.SpeedNorms[speed_id] = np.linalg.norm(NewSpeed)
        self.SpeedTimeConstants[speed_id] = min(1./self._Precision_aimed, self._DecayRatio/self.SpeedNorms[speed_id])
        self.SpeedsChangesHistory[speed_id] += [(t, np.array(self.Speeds[speed_id]))]

    def _ComputeDynamicPositionReference(self, speed_id):
        NewTheta = GetSpeedAngle(self.Speeds[speed_id])
        Gx, Gy = self.DMGradientReferences[speed_id]

        RefMap = self.DMReferences[speed_id]
        ModSpeedMap = np.zeros(RefMap.shape)
        TopPart = abs(np.cos(NewTheta) * Gx + np.sin(NewTheta) * Gy)
        ModSpeedMap[self.ModSpeedPixels[speed_id]] = RefMap[self.ModSpeedPixels[speed_id]] * TopPart[self.ModSpeedPixels[speed_id]] / self.LowerPartModSpeed[speed_id]

        return self.GetMeanPosition(None, ModSpeedMap)
    
    def _GetModSpeedValuesFor(self, speed_id, KSize = 1):
        Kernel = np.ones((2 * KSize + 1, 2 * KSize + 1))/((2 * KSize + 1)**2)
        SmoothedMap = ndimage.convolve(self.DecayingMaps[speed_id], np.array(Kernel), mode='constant', cval=0.0)

        self.DMGradientReferences[speed_id] = np.gradient(SmoothedMap)
        self.SpeedAngleReferences[speed_id] = GetSpeedAngle(self.Speeds[speed_id])
        self.SpeedReferences[speed_id] = np.array(self.Speeds[speed_id])
        UsefulPixels = np.where(SmoothedMap > 0)

        LowerPart = abs(np.cos(self.SpeedAngleReferences[speed_id]) * self.DMGradientReferences[speed_id][0][UsefulPixels] + np.sin(self.SpeedAngleReferences[speed_id]) * self.DMGradientReferences[speed_id][1][UsefulPixels])
        AllowedPixels = np.where(LowerPart != 0)[0]

        self.ModSpeedPixels[speed_id] = [UsefulPixels[0][AllowedPixels], UsefulPixels[1][AllowedPixels]]
        self.LowerPartModSpeed[speed_id] = LowerPart[AllowedPixels]

    def GetMeanPosition(self, speed_id, Map = None, threshold = 0.): # Threshold should allow to get only the relevant points in case the speed is correcly estimated (e**-1 \simeq 0.3)
        if Map is None:
            Map = self.DecayingMaps[speed_id]
        if Map.max() <= threshold:
            return None, None
        Xs, Ys = np.where(Map > threshold)
        Weights = Map[Xs, Ys]
        return np.array([(Weights*Xs).sum() / (self._DensityDefinition*Weights.sum()), (Weights*Ys).sum() / (self._DensityDefinition*Weights.sum())])

    def _UpdateDisplacement(self, t, speed_id):
        self.Displacements[speed_id] = self.CurrentBaseDisplacement[speed_id] + (t - self.SpeedProjectionTime[speed_id])*self.Speeds[speed_id]

    def _RegenSpeeds(self):
        ConsideredSpeeds = self.RecoverCurrentBestSpeeds(OnlyAlive = True)
        Positions = []

        for speed_id, zone_id in ConsideredSpeeds:
            OW = self.OWAPT[speed_id]
            CenterPos = np.array([OW[2] + OW[0], OW[3] + OW[1]], dtype = float) / 2
            Positions += [CenterPos + self.Displacements[speed_id]]

        Positions = np.array(Positions)

        Padding = np.array(self.MapSize) / np.array(self._DetectorPointsGrid)
        FirstCorner = Padding / 2
        NewPoints = []
        MinDistances = []
        for x in range(self._DetectorPointsGrid[0]):
            for y in range(self._DetectorPointsGrid[1]):
                NewPoints += [FirstCorner + np.array([x, y]) * Padding]
                MinDistances += [np.linalg.norm((NewPoints[-1] - Positions), axis = 1).min()]

        for LocalIndex in np.argsort(MinDistances)[-(self.DetectorNumberOfPoints - self.NActiveZones):]:
            P = NewPoints[LocalIndex]
            OW = [P[0] - self._DetectorDefaultWindowsLength / 2, P[1] - self._DetectorDefaultWindowsLength / 2, P[0] + self._DetectorDefaultWindowsLength / 2, P[1] + self._DetectorDefaultWindowsLength / 2]
            self.DefaultObservationWindows += [list(OW)]
            self._AddExponentialSeeds(vx_center = 0., vy_center = 0., add_center = True)

        self.RegenNeeded = False


    def AskLocationAndStart(self, TW = None):
        if TW is None:
            TW = self._TW

        STContext = self.__Framework__.Tools[self.__CreationReferences__['Memory']].STContext.max(axis = 2)
        Mask = STContext > STContext.max() - TW

        self.SelectionLocation = None
        self.SelectionCorner = None

        self.CenterLocationPoint = None
        self.WindowLines = []

        self.SelectionFigure, self.SelectionAx = plt.subplots(1,1)
        self.SelectionAx.imshow(np.transpose(Mask), origin = 'lower')
        
        cid = self.SelectionFigure.canvas.mpl_connect('button_press_event', self._LocationSelection)

        print ""
        print "Pressed 'Enter' to resume once the windows are confirmed, or enter one of the following options: "
        print "'grid' -> Switch to grid mode"
        print "'auto' -> Detects corners with Hough detector"
        print "'quit' -> Stops framework in current state"
        print "'t=VALUE' -> Asks windows location at t"
        print "'delta=VALUE' -> Asks windows location after delta"
        ans = raw_input(" -> ")
        plt.pause(0.1)
        plt.close(self.SelectionFigure.number)
        del self.SelectionFigure
        del self.SelectionAx
        del self.CenterLocationPoint
        del self.WindowLines
        del self.SelectionLocation
        del self.SelectionCorner
        if not len(ans):
            if not self.__Started__:
                print "No window selected. Restarting ASAP."
                return False
            else:
                return True
        else:
            if ans == 'grid':
                print "Switching to grid mode"
                self._SelectionMode = 'grid'
                self._GridSelectCorners()
                return True
            elif ans == 'auto':
                print "Switching to auto mode"
                self._SelectionMode = 'auto'
                self._AutoSelectCorners()
                return True
            elif ans == 'quit':
                self.__Framework__.Paused = self.__Name__
                return False
            elif '=' in ans:
                if ans.split('=')[0] == 't':
                    Value = float(ans.split('=')[1])
                    self._AskLocationAtTS = Value
                    print "Set new asking time to {0:.3f}".format(self._AskLocationAtTS)
                    return False
                elif ans.split('=')[0] == 'delta':
                    Value = float(ans.split('=')[1])
                    self._AskLocationAtTS += Value
                    print "Set new asking time to {0:.3f}".format(self._AskLocationAtTS)
                    return False
                else:
                    print "Unrecognized answer. Restarting ASAP."
                    return False
            else:
                print "Unrecognized answer. Restarting ASAP."
                return False

    def _GridSelectCorners(self):
        Padding = np.array(self.MapSize) / np.array(self._DetectorPointsGrid)
        FirstCorner = Padding / 2
        NewPoints = []
        for x in range(self._DetectorPointsGrid[0]):
            for y in range(self._DetectorPointsGrid[1]):
                NewPoints += [FirstCorner + np.array([x, y]) * Padding]
        for P in NewPoints:
            OW = [P[0] - self._DetectorDefaultWindowsLength / 2, P[1] - self._DetectorDefaultWindowsLength / 2, P[0] + self._DetectorDefaultWindowsLength / 2, P[1] + self._DetectorDefaultWindowsLength / 2]
            self.DefaultObservationWindows += [list(OW)]
            self._AddExponentialSeeds(vx_center = 0., vy_center = 0., add_center = True)

            self.__Started__ = True

    def _AutoSelectCorners(self):
        STContext = self.__Framework__.Tools[self.__CreationReferences__['Memory']].STContext.max(axis = 2)
        Corners, edges = GetCorners(np.array(np.transpose(STContext > STContext.max() - 0.004)*200, dtype = np.uint8), Tini = 100)
        NewPoints = []

        Feature0Index = np.array(Corners)[:,0].argmax()
        NewPoints += [Corners[Feature0Index]]
        Corners.pop(Feature0Index)
        
        Feature1Index = np.array(Corners)[:,1].argmax()
        NewPoints += [Corners[Feature1Index]]
        Corners.pop(Feature1Index)
        
        Feature2Index = np.array(Corners)[:,0].argmin()
        NewPoints += [Corners[Feature2Index]]
        Corners.pop(Feature2Index)
        
        NewPoints += [Corners[0]]

        for P in NewPoints:
            self.SelectionLocation = np.array(P)

            Ranges = np.array([15,15])
            self.SelectionCorner = self.SelectionLocation + Ranges

            OW = [self.SelectionLocation[0] - Ranges[0], self.SelectionLocation[1] - Ranges[1], self.SelectionLocation[0] + Ranges[0], self.SelectionLocation[1] + Ranges[1]]

            print "Confirmed Window : "
            print "x : {0} -> {1}".format(OW[0], OW[2])
            print "y : {0} -> {1}".format(OW[1], OW[3])
            self.DefaultObservationWindows += [list(OW)]
            self._AddExponentialSeeds(vx_center = 0., vy_center = 0., add_center = True)
            
            self.__Started__ = True
            
            self.SelectionLocation = None
            self.SelectionCorner = None
            
            self.CenterLocationPoint = None

    def _LocationSelection(self, ev):
        if ev.button == 1:
            print "Selecting x = {0}, y = {1} for center location".format(int(ev.xdata + 0.5), int(ev.ydata + 0.5))
            if not self.SelectionLocation is None:
                self.CenterLocationPoint.set_data(int(ev.xdata + 0.5), int(ev.ydata + 0.5))
            else:
                self.CenterLocationPoint = self.SelectionAx.plot(int(ev.xdata + 0.5), int(ev.ydata + 0.5), 'rv')[0]
            self.SelectionLocation = np.array([int(ev.xdata + 0.5), int(ev.ydata + 0.5)])
            self.SelectionCorner = self.SelectionLocation + np.array([11,11])
            if not self.SelectionCorner is None:
                Ranges = abs(self.SelectionCorner - self.SelectionLocation)
                print "Current window Size : dx = {0}, dy = {1}".format(Ranges[0], Ranges[1])
                OW = [self.SelectionLocation[0] - Ranges[0], self.SelectionLocation[1] - Ranges[1], self.SelectionLocation[0] + Ranges[0], self.SelectionLocation[1] + Ranges[1]]
                self._UpdateOWDrawing(OW)
        elif ev.button == 3:
            print "Selecting x = {0}, y = {1} as a corner".format(int(ev.xdata + 0.5), int(ev.ydata + 0.5))
            self.SelectionCorner = np.array([int(ev.xdata + 0.5), int(ev.ydata + 0.5)])
            if not self.SelectionLocation is None:
                Ranges = abs(self.SelectionCorner - self.SelectionLocation)
                print "Current window Size : dx = {0}, dy = {1}".format(Ranges[0], Ranges[1])
                OW = [self.SelectionLocation[0] - Ranges[0], self.SelectionLocation[1] - Ranges[1], self.SelectionLocation[0] + Ranges[0], self.SelectionLocation[1] + Ranges[1]]
                self._UpdateOWDrawing(OW)
        elif ev.button == 2:
            if not self.SelectionLocation is None and not self.SelectionCorner is None:
                Ranges = abs(self.SelectionCorner - self.SelectionLocation)
                OW = [self.SelectionLocation[0] - Ranges[0], self.SelectionLocation[1] - Ranges[1], self.SelectionLocation[0] + Ranges[0], self.SelectionLocation[1] + Ranges[1]]
                print "Confirmed Window : "
                print "x : {0} -> {1}".format(OW[0], OW[2])
                print "y : {0} -> {1}".format(OW[1], OW[3])
                self.DefaultObservationWindows += [list(OW)]
                self._AddExponentialSeeds(vx_center = 0., vy_center = 0., add_center = True)

                self.__Started__ = True

                self.SelectionLocation = None
                self.SelectionCorner = None

                self.CenterLocationPoint = None
                self.WindowLines = []

            else:
                print "AutoSelecting corners"
        self.SelectionFigure.canvas.show()

    def _UpdateOWDrawing(self, OW):
        OW[0] -= 0.5
        OW[1] -= 0.5
        OW[2] += 0.5
        OW[3] += 0.5
        if len(self.WindowLines) == 0:
            self.WindowLines += [self.SelectionAx.plot([OW[0], OW[2]], [OW[1], OW[1]], 'r')[0]]
            self.WindowLines += [self.SelectionAx.plot([OW[2], OW[2]], [OW[1], OW[3]], 'r')[0]]
            self.WindowLines += [self.SelectionAx.plot([OW[2], OW[0]], [OW[3], OW[3]], 'r')[0]]
            self.WindowLines += [self.SelectionAx.plot([OW[0], OW[0]], [OW[3], OW[1]], 'r')[0]]
        else:
            self.WindowLines[0].set_data([OW[0], OW[2]], [OW[1], OW[1]])
            self.WindowLines[1].set_data([OW[2], OW[2]], [OW[1], OW[3]])
            self.WindowLines[2].set_data([OW[2], OW[0]], [OW[3], OW[3]])
            self.WindowLines[3].set_data([OW[0], OW[0]], [OW[3], OW[1]])

    def _AddSeed(self, speed, localpadding, OW):

        speed_id = len(self.Speeds)
        self.ActiveSpeeds += [speed_id]
        self.IsActive += [True]
        self.ToInitializeSpeed += [speed_id]
        self.Speeds += [speed]
        self.InitialSpeeds += [np.array(speed)]
        self.LocalPaddings += [localpadding]

        self.SpeedStartTime += [0]
        self.SpeedStopTime += [None]
        self.SpeedProjectionTime += [0]
        self.ProjectionTimesHistory += [[0]]

        self.SpeedNorms += [np.linalg.norm(speed)]
        self.SpeedErrors += [np.array([0., 0.])]
        self.AimedSpeeds += [np.array(speed)]

        if self.SpeedNorms[-1] != 0.:
            self.SpeedTimeConstants += [self._DecayRatio/self.SpeedNorms[-1]]
        else:
            self.SpeedTimeConstants += [self._DecayRatio/abs(np.array(self.LocalPaddings[-1])).min()]
        self.OWAPT += [tuple(OW)]
        self.ObservationRadiuses += [np.array([OW[2] - OW[0], OW[3] - OW[1]], dtype = float)/2]

        self.Displacements += [None]
        self.LastConsideredEventsTs += [0]
        self.DecayingMaps += [0*self._CreateUnitaryMap(OW[2] - OW[0], OW[3] - OW[1])]
        self.StreaksMaps += [0*self._CreateUnitaryMap(OW[2] - OW[0], OW[3] - OW[1])]

        self.CurrentMeanPositions += [None]
        self.MeanPositionsReferences += [None]
        self.DMReferences += [None]
        self.DMGradientReferences += [None]
        self.SpeedAngleReferences += [None]
        self.SpeedReferences += [None]
        self.ModSpeedPixels += [None]
        self.LowerPartModSpeed += [None]
        self.MeanPositionTimeReferences += [None]
        self.ProjectedEvents += [0]

        self.CurrentBaseDisplacement += [np.array([0.,0.])]

        self.SpeedsChangesHistory += [[]]

        self.Zones[tuple(OW)] += [speed_id]

        return True

    def _CreateUnitaryMap(self, SizeX, SizeY):
        return np.ones((int((SizeX + 2*self._R_Projection) * self._DensityDefinition), int((SizeY + 2*self._R_Projection) * self._DensityDefinition)))

    def _AddExponentialSeeds(self, vx_center = 0., vy_center = 0., add_center = True): #observed are referenced to the center

        Zone = self.DefaultObservationWindows[-1] + [self.NZones]
        self.NZones += 1
        self.ToCleanZones += [tuple(Zone)]
        self.Zones[tuple(Zone)] = []

        center_speed = np.array([vx_center, vy_center])

        NSpeedsBefore = len(self.Speeds)

        seeds = [0.]
        for i in range(self._HalfNumberOfSpeeds):
            seeds += [float(self._Initial_dv_MAX)/(2**i)]
        self.V_seed = abs(seeds[-1])
        
        if add_center:
            self._AddSeed(center_speed, (-self.V_seed, -self.V_seed, self.V_seed, self.V_seed), OW = Zone)
        
        for dvx in seeds:
            for dvy in seeds:
                if (dvx > 0 or dvy > 0) and dvx**2 + dvy**2 <= 1.01*self._Initial_dv_MAX**2:
                    self._AddSeed(center_speed + np.array([dvx, dvy]), (-max(dvx/2, self.V_seed), -max(dvy/2, self.V_seed), max(dvx, self.V_seed), max(dvy, self.V_seed)), OW = Zone)
                    if dvx != 0.:
                        self._AddSeed(center_speed + np.array([-dvx, dvy]), (-max(dvx, self.V_seed), -max(dvy/2, self.V_seed), max(dvx/2, self.V_seed), max(dvy, self.V_seed)), OW = Zone)
                    if dvy != 0:
                        self._AddSeed(center_speed + np.array([dvx, -dvy]), (-max(dvx/2, self.V_seed), -max(dvy, self.V_seed), max(dvx, self.V_seed), max(dvy/2, self.V_seed)), OW = Zone)
                        if dvx != 0:
                            self._AddSeed(center_speed + np.array([-dvx, -dvy]), (-max(dvx, self.V_seed), -max(dvy, self.V_seed), max(dvx/2, self.V_seed), max(dvy/2, self.V_seed)), OW = Zone)
        print "Initialized {0} new speed seeds.".format(len(self.Speeds) - NSpeedsBefore)
        self.ActiveSpeedsInZone[tuple(Zone)] = len(self.Speeds) - NSpeedsBefore
        self.NActiveZones += 1

def GetSpeedAngle(Speed):
    if Speed[0] == 0:
        Theta = np.pi/2 + np.pi * (Speed[1] < 0)  
    else:                                       
        Theta = np.arctan(Speed[1] / Speed[0]) + np.pi * (Speed[0] < 0)
    return Theta

