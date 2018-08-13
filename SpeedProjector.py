import numpy as np
import geometry
import sys
import matplotlib.pyplot as plt

from HoughCornerDetector import GetCorners
from scipy import ndimage
import Plotting_methods

class LocalProjector:

    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Tool to create a 2D projection of the STContext onto a plane t = cst along different speeds.
        Should allow to recove the actual speed and generator of the movement
        '''
        self._ReferencesAsked = ['Memory']
        self._Name = Name
        self._Framework = Framework
        self._Type = 'Computation'
        self._CreationReferences = dict(argsCreationReferences)


        self.R_Projection = 0.5
        self.ExpansionFactor = 1.

        self.BinDt = 0.100

        self.HalfNumberOfSpeeds = 3
        self.Initial_dv_MAX = 800
        self.Relative_precision_aimed = 0.01
        self.Precision_aimed = self.Relative_precision_aimed*self.Initial_dv_MAX

        self.DensityDefinition = 3 # In dpp
        self.MaskDensityRatio = 0.3

        self.AskLocationAtTS = 0.010
        self.SnapshotDt = 0.005

        self.DecayRatio = 5.

        self.KeepBestSpeeds = 3

        self.ModSpeeds = True
        self.SpeedModRatio = 0.02
        self.RelativeCorrectionThreshold = 0.0 # Relative distance (to ObservationRadius) over which correction speed is made. 0 implies that mod speed is applied in any situation
        self.UpdatePT = True

        self.DynamicPositionReference = True
    
    def _Initialize(self):

        self.ActiveSpeeds = []
        self.Speeds = []
        self.DefaultObservationWindows = []

        self.ToInitializeSpeed = []
        self.SpeedStartTime = []
        self.SpeedProjectionTime = []
        self.SpeedNorms = []
        self.SpeedTimeConstants = []
        self.Displacements = []

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
        #self.SecondOrderMeanPositionsReferences = []
        #self.SecondOrderWeightsReferences = []
        self.MeanPositionTimeReferences = []

        self.CurrentMeanPositions = []
        #self.CurrentSecondOrderMeanPositions = []

        self.NZones = 0
        self.Zones = {}
        self.ToCleanZones = []

        StreamGeometry = self._Framework.StreamsGeometries[self._Framework.StreamHistory[-1]]

        self.LastConsideredEventsTs = []
        self.LocalPaddings = []
        self.InitialSpeeds = []

        self.LastSnapshotT = 0.
        self.DMSnaps = []
        self.SMSnaps = []
        self.PosSnaps = []
        self.TsSnaps = []
        self.DisplacementSnaps = []

        self.SpeedsChangesHistory = []
        self.ProjectionTimesHistory = []
        self.CurrentBaseDisplacement = []

        self.Started = False

    def _OnEvent(self, event):
        if not self.Started:
            if event.timestamp >= self.AskLocationAtTS:
                self.AskLocationAndStart()
                self.LastSnapshotT = event.timestamp
            return event

        if len(self.ToInitializeSpeed) > 0:
            for speed_id in self.ToInitializeSpeed:
                self.SpeedStartTime[speed_id] = event.timestamp
                self.SpeedProjectionTime[speed_id] = event.timestamp
                self.ProjectionTimesHistory[speed_id][0] = event.timestamp
                self.SpeedsChangesHistory[speed_id] += [(event.timestamp, np.array(self.Speeds[speed_id]))]
            self.ToInitializeSpeed = []

        for speed_id in self.ActiveSpeeds:
            self.ProjectEventWithSpeed(event, speed_id)

        if event.timestamp - self.LastSnapshotT >= self.SnapshotDt:
            self.DMSnaps += [[]]
            self.SMSnaps += [[]]
            self.PosSnaps += [[]]
            self.DisplacementSnaps += [[]]

            for speed_id in range(len(self.Speeds)):
                if speed_id in self.ActiveSpeeds:
                    DeltaT = event.timestamp - self.LastConsideredEventsTs[speed_id]
                    if DeltaT != 0.:
                        self.LastConsideredEventsTs[speed_id] = event.timestamp
                        self.DecayingMaps[speed_id] = self.DecayingMaps[speed_id]*np.e**(-(DeltaT/self.SpeedTimeConstants[speed_id]))
                    self.DMSnaps[-1] += [np.array(self.DecayingMaps[speed_id])]
                    self.SMSnaps[-1] += [np.array(self.StreaksMaps[speed_id])]
                    self.PosSnaps[-1] += [np.array(self.CurrentMeanPositions[speed_id])]
                    self.DisplacementSnaps[-1] += [self.Displacements[speed_id]]
                else:
                    self.DMSnaps[-1] += [None]
                    self.SMSnaps[-1] += [None]
                    self.PosSnaps[-1] += [None]
                    self.DisplacementSnaps[-1] += [None]

            self.LastSnapshotT = event.timestamp
            self.TsSnaps += [event.timestamp]

            self._Framework.Tools[self._CreationReferences['Memory']].GetSnapshot()

        for Zone in self.ToCleanZones:
            CanBeCleansed = True
            for speed_id in self.Zones[Zone]:
                if event.timestamp - self.SpeedStartTime[speed_id] < self.SpeedTimeConstants[speed_id]:
                    CanBeCleansed = False
                    break
            if CanBeCleansed:
                MeanValues = [float((self.StreaksMaps[speed_id]).sum())/(self.StreaksMaps[speed_id] > 0).sum() for speed_id in self.Zones[Zone]]
                SortedIDs = np.argsort(MeanValues)
                for local_speed_id in SortedIDs[:-self.KeepBestSpeeds]:
                    self.ActiveSpeeds.remove(self.Zones[Zone][local_speed_id])
                print "Cleansed {0} speed considered wrong for zone {1} at t = {2}".format(len(SortedIDs[:-self.KeepBestSpeeds].tolist()), Zone, event.timestamp)
                self.ToCleanZones.remove(Zone)
        return event

    def ProjectEventWithSpeed(self, event, speed_id):
        self.UpdateDisplacement(event.timestamp, speed_id)
        x0 = event.location[0] - self.Displacements[speed_id][0]
        y0 = event.location[1] - self.Displacements[speed_id][1]
        
        OW = self.OWAPT[speed_id]
        if not (OW[0] <= x0 <= OW[2] and OW[1] <= y0 <= OW[3]):
            return None

        DeltaT = event.timestamp - self.LastConsideredEventsTs[speed_id]
        EvolutionFactor = np.e**(-(DeltaT/self.SpeedTimeConstants[speed_id]))

        self.LastConsideredEventsTs[speed_id] = event.timestamp

        self.DecayingMaps[speed_id] = self.DecayingMaps[speed_id] * EvolutionFactor

        for x in range(int(np.floor(self.DensityDefinition*((x0 - self.R_Projection) - OW[0]))), int(np.ceil(self.DensityDefinition*((x0 + self.R_Projection) - OW[0])))):
            for y in range(int(np.floor(self.DensityDefinition*((y0 - self.R_Projection) - OW[1]))), int(np.ceil(self.DensityDefinition*((y0 + self.R_Projection) - OW[1])))):
                self.DecayingMaps[speed_id][x,y] += 1
                self.StreaksMaps[speed_id][x,y] += 1

        if self.ModSpeeds:

            self.CurrentMeanPositions[speed_id] = self.GetMeanPosition(speed_id)

            if self.MeanPositionsReferences[speed_id] is None:
                if event.timestamp - self.SpeedStartTime[speed_id] > self.SpeedTimeConstants[speed_id]: # Hopefully we can do better. Still, the feature should construct itself over self.SpeedTimeConstants[speed_id] for each speed
                    self.DMReferences[speed_id] = np.array(self.DecayingMaps[speed_id])

                    self.GetModSpeedValuesFor(speed_id)

                    #self.MeanPositionsReferences[speed_id] = self.ObservationRadiuses[speed_id]
                    self.MeanPositionsReferences[speed_id] = np.array(self.CurrentMeanPositions[speed_id])
                    self.MeanPositionTimeReferences[speed_id] = event.timestamp

                    #self.SecondOrderMeanPositionsReferences[speed_id], self.SecondOrderWeightsReferences[speed_id] = self.GetSecondOrderMeanPosition(speed_id, self.MeanPositionsReferences[speed_id])

                return None
    
            if event.timestamp - self.SpeedStartTime[speed_id] < self.SpeedTimeConstants[speed_id]:
                return None
            if self.DynamicPositionReference:
                PositionReference = self.ComputeDynamicPositionReference(speed_id)
            else:
                PositionReference = self.MeanPositionsReferences[speed_id]

            CurrentError = self.CurrentMeanPositions[speed_id] - PositionReference
            if (abs(CurrentError) < self.ObservationRadiuses[speed_id] * self.RelativeCorrectionThreshold).all():
                return None
            TimeEllapsed = event.timestamp - self.MeanPositionTimeReferences[speed_id]
            SpeedError = CurrentError / TimeEllapsed
    
            self.ModifySpeed(speed_id, SpeedError, event.timestamp)
        
        if self.UpdatePT and event.timestamp - self.SpeedProjectionTime[speed_id] > 6*self.SpeedTimeConstants[speed_id]:
            self.CurrentBaseDisplacement[speed_id] = np.array(self.Displacements[speed_id])
            self.SpeedProjectionTime[speed_id] = event.timestamp
            self.ProjectionTimesHistory[speed_id] += [event.timestamp]

    def ModifySpeed(self, speed_id, SpeedError, t):
        #NewSpeed = self.Speeds[speed_id] + SpeedError * self.SpeedModRatio
        NewSpeed = self.Speeds[speed_id] + SpeedError * self.DecayRatio / (self.DecayingMaps[speed_id].sum() / self.DensityDefinition ** 2)
        self.Speeds[speed_id] = NewSpeed
        self.SpeedNorms[speed_id] = np.linalg.norm(NewSpeed)
        self.SpeedTimeConstants[speed_id] = min(1./self.Precision_aimed, self.DecayRatio/self.SpeedNorms[speed_id])
        self.SpeedsChangesHistory[speed_id] += [(t, np.array(self.Speeds[speed_id]))]

    def ComputeDynamicPositionReference(self, speed_id):
        NewTheta = GetSpeedAngle(self.Speeds[speed_id])
        Gx, Gy = self.DMGradientReferences[speed_id]

        RefMap = self.DMReferences[speed_id]
        ModSpeedMap = np.zeros(RefMap.shape)
        TopPart = abs(np.cos(NewTheta) * Gx + np.sin(NewTheta) * Gy)
        ModSpeedMap[self.ModSpeedPixels[speed_id]] = RefMap[self.ModSpeedPixels[speed_id]] * TopPart[self.ModSpeedPixels[speed_id]] / self.LowerPartModSpeed[speed_id]

        return self.GetMeanPosition(None, ModSpeedMap)
    
    def GetModSpeedValuesFor(self, speed_id, KSize = 1):
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
        return np.array([(Weights*Xs).sum() / (self.DensityDefinition*Weights.sum()), (Weights*Ys).sum() / (self.DensityDefinition*Weights.sum())])

    def _GetSecondOrderMeanPosition(self, speed_id, CuttingMeanPosition = None, threshold = 0.):
        if not CuttingMeanPosition is None:
            Xm, Ym = np.array(np.rint(self.DensityDefinition * CuttingMeanPosition), dtype = int)
        else:
            Xm, Ym = np.array(np.rint(self.DensityDefinition * self.GetMeanPosition(speed_id)), dtype = int)
        if Xm is None:
            return None, None

        Map = self.DecayingMaps[speed_id]
        LocalMaps = [Map[:Xm, :Ym], Map[Xm:, :Ym], Map[Xm:, Ym:], Map[:Xm, Ym:]]
        LocalShifts = [np.array([0., 0.]), np.array([Xm, 0.]), np.array([Xm, Ym]), np.array([0., Ym])]
        
        MeansPos = []
        WeightsSums = []
        for LocalMap, LocalShift in zip(LocalMaps, LocalShifts):
            Xs, Ys = np.where(LocalMap > threshold)
            if Xs.shape[0] == 0:
                MeansPos += [None]
                WeightsSums += [0.]
            else:
                Weights = LocalMap[Xs, Ys]
                MeansPos += [(LocalShift + np.array([(Weights*Xs).sum() / Weights.sum(), (Weights*Ys).sum() / Weights.sum()])) / self.DensityDefinition]
                WeightsSums += [Weights.sum()]
        return MeansPos, WeightsSums

    def UpdateDisplacement(self, t, speed_id):
        self.Displacements[speed_id] = self.CurrentBaseDisplacement[speed_id] + (t - self.SpeedProjectionTime[speed_id])*self.Speeds[speed_id]

    def AskLocationAndStart(self, TW = 0.01):
        STContext = self._Framework.Tools[self._CreationReferences['Memory']].STContext.max(axis = 2)
        Mask = STContext > STContext.max() - TW

        self.SelectionLocation = None
        self.SelectionCorner = None

        self.CenterLocationPoint = None
        self.WindowLines = []

        self.SelectionFigure, self.SelectionAx = plt.subplots(1,1)
        self.SelectionAx.imshow(np.transpose(Mask), origin = 'lower')
        
        cid = self.SelectionFigure.canvas.mpl_connect('button_press_event', self._LocationSelection)

        raw_input("Pressed 'Enter' to resume, once the windows are confirmed")
        if not self.Started:
            print "Autoselecting corners."
            self.AutoSelectCorners()
            plt.pause(0.1)
            del self.CenterLocationPoint
            del self.WindowLines
            del self.SelectionLocation
            del self.SelectionCorner
        else:
            self.SelectionFigure.canvas.mpl_disconnect(cid)
            plt.close(self.SelectionFigure.number)
            del self.SelectionFigure
            del self.SelectionAx
            del self.CenterLocationPoint
            del self.WindowLines
            del self.SelectionLocation
            del self.SelectionCorner

    def AutoSelectCorners(self):
        STContext = self._Framework.Tools[self._CreationReferences['Memory']].STContext.max(axis = 2)
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
            self.CenterLocationPoint = self.SelectionAx.plot(P[0], P[1], 'rv')[0]

            Ranges = np.array([15,15])
            self.SelectionCorner = self.SelectionLocation + Ranges

            OW = [self.SelectionLocation[0] - Ranges[0], self.SelectionLocation[1] - Ranges[1], self.SelectionLocation[0] + Ranges[0], self.SelectionLocation[1] + Ranges[1]]
            self._UpdateOWDrawing(OW)

            print "Confirmed Window : "
            print "x : {0} -> {1}".format(OW[0], OW[2])
            print "y : {0} -> {1}".format(OW[1], OW[3])
            self.DefaultObservationWindows += [list(OW)]
            self.AddExponentialSeeds(vx_center = 0., vy_center = 0., add_center = True)
            
            self.Started = True
            
            self.SelectionLocation = None
            self.SelectionCorner = None
            
            self.CenterLocationPoint = None
            self.WindowLines = []


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
                self.AddExponentialSeeds(vx_center = 0., vy_center = 0., add_center = True)

                self.Started = True

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

    def AddSeed(self, speed, localpadding, OW):

        speed_id = len(self.Speeds)
        self.ActiveSpeeds += [speed_id]
        self.ToInitializeSpeed += [speed_id]
        self.Speeds += [speed]
        self.InitialSpeeds += [np.array(speed)]
        self.LocalPaddings += [localpadding]

        self.SpeedStartTime += [0]
        self.SpeedProjectionTime += [0]
        self.ProjectionTimesHistory += [[0]]

        self.SpeedNorms += [np.linalg.norm(speed)]

        if self.SpeedNorms[-1] != 0.:
            self.SpeedTimeConstants += [self.DecayRatio/self.SpeedNorms[-1]]
        else:
            self.SpeedTimeConstants += [self.DecayRatio/abs(np.array(self.LocalPaddings[-1])).min()]
        self.OWAPT += [OW]
        self.ObservationRadiuses += [np.array([OW[2] - OW[0], OW[3] - OW[1]], dtype = float)/2]

        self.Displacements += [None]
        self.LastConsideredEventsTs += [0]
        self.DecayingMaps += [0*self.CreateUnitaryMap(OW[2] - OW[0], OW[3] - OW[1])]
        self.StreaksMaps += [0*self.CreateUnitaryMap(OW[2] - OW[0], OW[3] - OW[1])]

        self.CurrentMeanPositions += [None]
        #self.CurrentSecondOrderMeanPositions += [None]
        self.MeanPositionsReferences += [None]
        self.DMReferences += [None]
        self.DMGradientReferences += [None]
        self.SpeedAngleReferences += [None]
        self.SpeedReferences += [None]
        self.ModSpeedPixels += [None]
        self.LowerPartModSpeed += [None]
        #self.SecondOrderMeanPositionsReferences += [None]
        #self.SecondOrderWeightsReferences += [None]
        self.MeanPositionTimeReferences += [None]

        self.CurrentBaseDisplacement += [np.array([0.,0.])]

        self.SpeedsChangesHistory += [[]]

        self.Zones[tuple(OW)] += [speed_id]

        return True

    def CreateUnitaryMap(self, SizeX, SizeY):
        return np.ones((int((SizeX + 2*self.R_Projection) * self.DensityDefinition), int((SizeY + 2*self.R_Projection) * self.DensityDefinition)))

    def AddExponentialSeeds(self, vx_center = 0., vy_center = 0., add_center = True): #observed are referenced to the center

        Zone = self.DefaultObservationWindows[-1] + [self.NZones]
        self.NZones += 1
        self.ToCleanZones += [tuple(Zone)]
        self.Zones[tuple(Zone)] = []

        center_speed = np.array([vx_center, vy_center])

        seeds = [0.]
        for i in range(self.HalfNumberOfSpeeds):
            seeds += [float(self.Initial_dv_MAX)/(2**i)]
        self.V_seed = abs(seeds[-1])
        
        print "Seeds : ", seeds
        if add_center:
            self.AddSeed(center_speed, (-self.V_seed, -self.V_seed, self.V_seed, self.V_seed), OW = Zone)
        
        for dvx in seeds:
            for dvy in seeds:
                if (dvx > 0 or dvy > 0) and dvx**2 + dvy**2 <= 1.01*self.Initial_dv_MAX**2:
                    self.AddSeed(center_speed + np.array([dvx, dvy]), (-max(dvx/2, self.V_seed), -max(dvy/2, self.V_seed), max(dvx, self.V_seed), max(dvy, self.V_seed)), OW = Zone)
                    if dvx != 0.:
                        self.AddSeed(center_speed + np.array([-dvx, dvy]), (-max(dvx, self.V_seed), -max(dvy/2, self.V_seed), max(dvx/2, self.V_seed), max(dvy, self.V_seed)), OW = Zone)
                    if dvy != 0:
                        self.AddSeed(center_speed + np.array([dvx, -dvy]), (-max(dvx/2, self.V_seed), -max(dvy, self.V_seed), max(dvx, self.V_seed), max(dvy/2, self.V_seed)), OW = Zone)
                        if dvx != 0:
                            self.AddSeed(center_speed + np.array([-dvx, -dvy]), (-max(dvx, self.V_seed), -max(dvy, self.V_seed), max(dvx/2, self.V_seed), max(dvy/2, self.V_seed)), OW = Zone)
        print "Initialized {0} speed seeds, going from vx = {1} and vy = {2} to vx = {3} and vy = {4}.".format(len(self.Speeds), np.array(self.Speeds)[:,0].min(), np.array(self.Speeds)[:,1].min(), np.array(self.Speeds)[:,0].max(), np.array(self.Speeds)[:,1].max())

def GetSpeedAngle(Speed):
    if Speed[0] == 0:
        Theta = np.pi/2 + np.pi * (Speed[1] < 0)  
    else:                                       
        Theta = np.arctan(Speed[1] / Speed[0]) + np.pi * (Speed[0] < 0)
    return Theta

class ProjectorV4:
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Tool to create a 2D projection of the STContext onto a plane t = cst along different speeds.
        Should allow to recove the actual speed and generator of the movement
        '''
        self._ReferencesAsked = []
        self._Name = Name
        self._Framework = Framework
        self._Type = 'Computation'
        self._CreationReferences = dict(argsCreationReferences)

        self.Precision_aimed = 2.
        self.Initial_dv_MAX = 80.

        self.GroundTS = 0.

    def _Initialize(self):

        self.ActiveSpeeds = []
        self.Speeds = []

        self.ToInitializeSpeed = []
        self.SpeedStartTime = []
        self.SpeedNorms = []
        self.AllowedDeltaTRanges = []
        self.LocalPaddings = []
        self.OWAST = [] # Stands for "Observation Window at Start Time". Means to stabilize any object present at start time, without adding new ones yet to appear. It is given at start time as it moves as time goes on.

        StreamGeometry = self._Framework.StreamsGeometries[self._Framework.StreamHistory[-1]]
        self.DefaultObservationWindow = [0., 0., float(StreamGeometry[0]+1), float(StreamGeometry[1]+1)]

        self.Displacements = []

        self.ConfirmedProjections = []
        self.CurrentProjections = []
        self.CumulatedDisplacement = []
        self.CumulatedError = []
        self.CumulatedTimeDisplacement = []
        self.NEventsMatched = []

        self.AddExponentialSeeds(vx_center = 0., vy_center = 0., v_max_observed = self.Initial_dv_MAX, v_min_observed = self.Precision_aimed*2, add_center = True)

    def _OnEvent(self, event):
        if event.timestamp > self.GroundTS:
            if len(self.ToInitializeSpeed) > 0:
                for speed_id in self.ToInitializeSpeed:
                    self.SpeedStartTime[speed_id] = event.timestamp
                self.ToInitializeSpeed = []

            for speed_id in self.ActiveSpeeds:
                self.ProjectEventWithSpeed(event, speed_id)
        return event

    def ProjectEventWithSpeed(self, event, speed_id):
        self.UpdateDisplacement(event.timestamp, speed_id)
        x0 = event.location[0] - self.Displacements[speed_id][0]
        y0 = event.location[1] - self.Displacements[speed_id][1]
        
        OW = self.OWAST[speed_id]
        if not (OW[0] <= x0 <= OW[2] and OW[1] <= y0 <= OW[3]):
            return None

        StartID = 0
        for CurrentProjection in self.CurrentProjections[speed_id]:
            if event.timestamp - CurrentProjection[2] >= self.AllowedDeltaTRanges[speed_id][1]:
                StartID += 1
            else:
                break
        self.CurrentProjections[speed_id] = self.CurrentProjections[speed_id][StartID:]

        Projection = np.array([x0, y0, event.timestamp, 0])

        if len(self.CurrentProjections[speed_id]) > 0:
            AvailableProjections = np.array(self.CurrentProjections[speed_id])
            TDiffs = event.timestamp - AvailableProjections[:,2]
            TimeCoherentProjections = np.where((TDiffs >= self.AllowedDeltaTRanges[speed_id][0]))[0]
            if TimeCoherentProjections.shape[0] > 1:
                PositionsDiffs = Projection[:2] - AvailableProjections[TimeCoherentProjections, :2]
                NormsDiffs = np.linalg.norm(PositionsDiffs, axis = 1)
                
                ChoosenLocalID = NormsDiffs.argmin()
                if NormsDiffs[ChoosenLocalID] < 1.:
                    ChoosenGlobalID = TimeCoherentProjections[ChoosenLocalID]
                    if AvailableProjections[ChoosenGlobalID, 3] == 0:
                        self.ConfirmedProjections[speed_id] += [np.array(AvailableProjections[ChoosenGlobalID, :3])]
                        AvailableProjections[ChoosenGlobalID, 3] = 1
                    self.CumulatedDisplacement[speed_id] += PositionsDiffs[ChoosenLocalID]
                    self.CumulatedTimeDisplacement[speed_id] += TDiffs[ChoosenGlobalID]
                    self.CumulatedError[speed_id] += np.linalg.norm(PositionsDiffs[ChoosenLocalID]) / (TDiffs[ChoosenGlobalID])

                    self.NEventsMatched[speed_id] += 1
        self.CurrentProjections[speed_id] += [Projection]

    def UpdateDisplacement(self, t, speed_id):
        self.Displacements[speed_id] = (t - self.SpeedStartTime[speed_id])*self.Speeds[speed_id]

    def AddSeed(self, speed, localpadding, OW, force = False):
        if force:
            for PreviousSpeed in np.array(self.Speeds)[self.ActiveSpeeds]:
                if (abs(speed - PreviousSpeed) < 0.001).all():
                    return False
        else:
            for PreviousSpeed in self.Speeds:
                if (abs(speed - PreviousSpeed) < 0.001).all():
                    return False

        speed_id = len(self.Speeds)
        self.ActiveSpeeds += [speed_id]
        self.ToInitializeSpeed += [speed_id]
        self.Speeds += [speed]
        self.LocalPaddings += [localpadding]

        self.SpeedStartTime += [0]

        self.SpeedNorms += [np.linalg.norm(speed)]
        if self.SpeedNorms[-1] != 0.:
            self.AllowedDeltaTRanges += [[(1./2)/self.SpeedNorms[-1], (2./1)/self.SpeedNorms[-1]]]
        else:
            self.AllowedDeltaTRanges += [[1./min(self.LocalPaddings[-1]), np.inf]]
        self.OWAST += [OW]

        self.Displacements += [None]

        self.NEventsMatched += [0]
        self.CumulatedDisplacement += [np.array([0., 0.])]
        self.CumulatedTimeDisplacement += [0]
        self.CumulatedError += [0]
        self.CurrentProjections += [[]]
        self.ConfirmedProjections += [[]]

        return True

    def AddExponentialSeeds(self, vx_center = 0., vy_center = 0., v_min_observed = 1., v_max_observed = 100., add_center = True): #observed are referenced to the center
        center_speed = np.array([vx_center, vy_center])

        MaxSpeedTried = v_max_observed
        self.V_seed = 2*v_min_observed
        seeds = [0.]
        i = 0
        while self.V_seed**i <= MaxSpeedTried:
            seeds += [self.V_seed*(2**i)]
            i += 1
        print "Seeds : ", seeds
        if add_center:
            self.AddSeed(center_speed, (-self.V_seed, -self.V_seed, self.V_seed, self.V_seed), OW = self.DefaultObservationWindow)
        
        for dvx in seeds:
            for dvy in seeds:
                if (dvx > 0 or dvy > 0) and dvx**2 + dvy**2 <= MaxSpeedTried**2:
                    self.AddSeed(center_speed + np.array([dvx, dvy]), (-max(dvx/2, self.V_seed), -max(dvy/2, self.V_seed), max(dvx, self.V_seed), max(dvy, self.V_seed)), OW = self.DefaultObservationWindow)
                    if dvx != 0.:
                        self.AddSeed(center_speed + np.array([-dvx, dvy]), (-max(dvx, self.V_seed), -max(dvy/2, self.V_seed), max(dvx/2, self.V_seed), max(dvy, self.V_seed)), OW = self.DefaultObservationWindow)
                    if dvy != 0:
                        self.AddSeed(center_speed + np.array([-dvx, -dvy]), (-max(dvx, self.V_seed), -max(dvy, self.V_seed), max(dvx/2, self.V_seed), max(dvy/2, self.V_seed)), OW = self.DefaultObservationWindow)
                        self.AddSeed(center_speed + np.array([dvx, -dvy]), (-max(dvx/2, self.V_seed), -max(dvy, self.V_seed), max(dvx, self.V_seed), max(dvy/2, self.V_seed)), OW = self.DefaultObservationWindow)
        print "Initialized {0} speed seeds, going from vx = {1} and vy = {2} to vx = {3} and vy = {4}.".format(len(self.Speeds), np.array(self.Speeds)[:,0].min(), np.array(self.Speeds)[:,1].min(), np.array(self.Speeds)[:,0].max(), np.array(self.Speeds)[:,1].max())

class ProjectorV3:
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Tool to create a 2D projection of the STContext onto a plane t = cst along different speeds.
        Should allow to recove the actual speed and generator of the movement
        '''
        self._ReferencesAsked = []
        self._Name = Name
        self._Framework = Framework
        self._Type = 'Computation'
        self._CreationReferences = dict(argsCreationReferences)

        self.R_Projection = 0.5
        self.ExpansionFactor = 1.

        self.BinDt = 0.100
        self.Precision_aimed = 2.
        self.Initial_dv_MAX = 80.

        self.DensityDefinition = 3 # In dpp
        self.MaskDensityRatio = 0.3

        self.GroundTS = 0.

    def _Initialize(self):

        self.ActiveSpeeds = []
        self.Speeds = []

        self.ToInitializeSpeed = []
        self.SpeedStartTime = []
        self.SpeedNorms = []
        self.AllowedDeltaTRanges = []
        self.OWAST = [] # Stands for "Observation Window at Start Time". Means to stabilize any object present at start time, without adding new ones yet to appear. It is given at start time as it moves as time goes on.
        self.LandingTSsMaps = []
        self.StreaksMaps = []
        self.Masks = []

        StreamGeometry = self._Framework.StreamsGeometries[self._Framework.StreamHistory[-1]]
        self.DefaultObservationWindow = [0., 0., float(StreamGeometry[0]+1), float(StreamGeometry[1]+1)]

        self.LocalPaddings = []
        self.InitializedAt = []
        self.NEventsBySpeed = []
        self.Displacements = []
        self.Consistencies = []
        self.RelativeMeanStreaks = []

        self.NEventsMatched = []
        self.CumulatedAwaitingTime = []
        self.TimeFilteredEvents = []

        self.SpeedsChangesTs = []
        self.SpeedsChangesIndexes = [None]
        self.SpeedsExpanded  = [[]]

        self.AddExponentialSeeds(vx_center = 0., vy_center = 0., v_max_observed = self.Initial_dv_MAX, v_min_observed = self.Precision_aimed*2, add_center = True)

        self.LastTsSave = 0.
        self.TsHistory = []
        self.ConsistencyHistory = []
        self.RelativeMeanStreaksHistory = []

        self.SetTS = self.DynamicMinTSSet

    def _OnEvent(self, event):
        self.SetTS(event.timestamp)
        if event.timestamp > self.GroundTS:
            if len(self.ToInitializeSpeed) > 0:
                for speed_id in self.ToInitializeSpeed:
                    self.SpeedStartTime[speed_id] = event.timestamp
                self.ToInitializeSpeed = []

            for speed_id in self.ActiveSpeeds:
                if self.InitializedAt[speed_id] == 0:
                    if (event.timestamp - self.SpeedStartTime[speed_id])*max(1., abs(self.Speeds[speed_id]).min()) >= self.ExpansionFactor:
                        self.InitializedAt[speed_id] = event.timestamp
                self.ProjectEventWithSpeed(event, speed_id)
        return event

    def ProjectEventWithSpeed(self, event, speed_id):
        self.UpdateDisplacement(event.timestamp, speed_id)
        x0 = event.location[0] - self.Displacements[speed_id][0]
        y0 = event.location[1] - self.Displacements[speed_id][1]
        
        OW = self.OWAST[speed_id]
        if not (OW[0] <= x0 <= OW[2] and OW[1] <= y0 <= OW[3]):
            return None

        StreakMap = self.StreaksMaps[speed_id]
        LandingTsMap = self.LandingTSsMaps[speed_id]

        for x in range(int(np.floor(self.DensityDefinition*((x0 - self.R_Projection) - OW[0]))), int(np.ceil(self.DensityDefinition*((x0 + self.R_Projection) - OW[0])))):
            for y in range(int(np.floor(self.DensityDefinition*((y0 - self.R_Projection) - OW[1]))), int(np.ceil(self.DensityDefinition*((y0 + self.R_Projection) - OW[1])))):
                if self.AllowedDeltaTRanges[speed_id][0] <= event.timestamp - LandingTsMap[x,y] < self.AllowedDeltaTRanges[speed_id][1]:
                    StreakMap[x,y] += 1
                    self.CumulatedAwaitingTime[speed_id] -= LandingTsMap[x,y] - event.timestamp
                    self.NEventsMatched[speed_id] += 1
                else:
                    self.TimeFilteredEvents[speed_id] += 1
                LandingTsMap[x,y] = event.timestamp

    def UpdateDisplacement(self, t, speed_id):
        self.Displacements[speed_id] = (t - self.SpeedStartTime[speed_id])*self.Speeds[speed_id]

    def ComputeInterestBoxesBySpeeds(self, dSpaceRatio = 2.):
        Boxes = []
        for speed_id in self.ActiveSpeeds:
            Xs, Ys = np.where(self.Masks[speed_id])
            N_Points = Xs.shape[0]
            Xmean = Xs.mean()
            Ymean = Ys.mean()
            dX = dSpaceRatio * np.sqrt(float(((Xs - Xmean)**2).sum())/N_Points)
            dY = dSpaceRatio * np.sqrt(float(((Ys - Ymean)**2).sum())/N_Points)

            Boxes += [Xmean - dX, Ymean - dY, Xmean + dX, Ymean + dY]
        return Boxes

    def ComputeInterestCentersWithSpeeds(self, n_points, consistency_limit = 0.3, dSpaceRatio = 2.):
        PossibleIDsByConsistency = np.where(np.array(self.Consistencies) > consistency_limit)[0]

        InitialCenters = []
        InitialVariances = []
        InitialSpeedsAssociated = []

        for local_id in np.argsort(-np.array(self.RelativeMeanStreaks)[PossibleIDsByConsistency]): # We sort by -RelativeMeanStreaks to get the highest MeansStreaks first
            speed_id = PossibleIDsByConsistency[local_id]
            if self.RelativeMeanStreaks[speed_id] >= 1.:
                continue
            Xs, Ys = np.where(self.Masks[speed_id])
            Xs = np.array(Xs, dtype = float) / self.DensityDefinition + self.Displacements[speed_id][0] + self.OWAST[speed_id][0]
            Ys = np.array(Ys, dtype = float) / self.DensityDefinition + self.Displacements[speed_id][1] + self.OWAST[speed_id][1]
            N_Points = Xs.shape[0]
            Xmean = Xs.mean()
            Ymean = Ys.mean()
            dX = dSpaceRatio * np.sqrt(float(((Xs - Xmean)**2).sum())/N_Points)
            dY = dSpaceRatio * np.sqrt(float(((Ys - Ymean)**2).sum())/N_Points)
            
            InitialCenters += [np.array([Xmean, Ymean])]
            InitialVariances += [np.array([dX, dY])]
            InitialSpeedsAssociated += [speed_id]

        NObjectsFound = 0
        FinalCenters = []
        FinalVariances = []
        FinalSpeeds = []
        FinalSpeedsAverage = []

        for Center, Variances, SpeedId in zip(InitialCenters, InitialVariances, InitialSpeedsAssociated):
            FoundSimilarObject = False
            for nObject in range(NObjectsFound):
                nSpeedsInObject = len(FinalSpeeds[nObject])
                if 0.8 <= Variances[0] / FinalVariances[nObject][0] <= 1.2 and 0.8 <= Variances[1] / FinalVariances[nObject][1] <= 1.2:
                    if (abs(Center - FinalCenters[nObject]) < FinalVariances[nObject]/dSpaceRatio).all():
                        if np.linalg.norm(self.Speeds[SpeedId] - FinalSpeedsAverage[nObject])/max(1, np.linalg.norm(FinalSpeedsAverage[nObject])) <= 1.:
                            FinalCenters[nObject] = (FinalCenters[nObject]*nSpeedsInObject + Center)/(nSpeedsInObject+1)
                            FinalVariances[nObject] = (FinalVariances[nObject]*nSpeedsInObject + Variances)/(nSpeedsInObject+1)
                            FinalSpeeds[nObject] += [SpeedId]
                            FinalSpeedsAverage[nObject] = (FinalSpeedsAverage[nObject]*nSpeedsInObject + self.Speeds[SpeedId])/(nSpeedsInObject+1)

                            FoundSimilarObject = True
                            break

            if not FoundSimilarObject and NObjectsFound < n_points:
                NObjectsFound += 1
                FinalCenters += [Center]
                FinalVariances += [Variances]
                FinalSpeeds += [[SpeedId]]
                FinalSpeedsAverage += [self.Speeds[SpeedId]]
        
        Boxes = [[Center[0] - Variances[0], Center[1] - Variances[1], Center[0] + Variances[0], Center[1] + Variances[1]] for Center, Variances in zip(FinalCenters, FinalVariances)]

        return Boxes, FinalSpeeds

    def PlotCurrentInterestRegions(self, f, ax, n_points = 5, consistency_limit = 0.3):
        Boxes, SpeedsIDs = self.ComputeInterestCentersWithSpeeds(n_points, consistency_limit)
        n = 0
        for Box, ConcernedSpeedsIDs in zip(Boxes, SpeedsIDs):
            n += 1
            ax.plot([Box[0], Box[0]], [Box[1], Box[3]], 'grey')
            ax.plot([Box[2], Box[2]], [Box[1], Box[3]], 'grey')
            ax.plot([Box[0], Box[2]], [Box[1], Box[1]], 'grey')
            ax.plot([Box[0], Box[2]], [Box[3], Box[3]], 'grey')

            ax.text(Box[0] + 1, Box[1] + 1, "Box # {0}\n".format(n) + "\n".join(["Vx = {0:.1f}, Vy = {1:.1f}, id = {2}".format(self.Speeds[speed_id][0], self.Speeds[speed_id][1], speed_id) for speed_id in ConcernedSpeedsIDs]), color = 'grey')

    def AddSeed(self, speed, localpadding, OW, force = False):
        if force:
            for PreviousSpeed in np.array(self.Speeds)[self.ActiveSpeeds]:
                if (abs(speed - PreviousSpeed) < 0.001).all():
                    return False
        else:
            for PreviousSpeed in self.Speeds:
                if (abs(speed - PreviousSpeed) < 0.001).all():
                    return False

        speed_id = len(self.Speeds)
        self.ActiveSpeeds += [speed_id]
        self.ToInitializeSpeed += [speed_id]
        self.Speeds += [speed]
        self.LocalPaddings += [localpadding]

        self.SpeedStartTime += [0]

        self.SpeedNorms += [np.linalg.norm(speed)]
        if self.SpeedNorms[-1] != 0.:
            self.AllowedDeltaTRanges += [[(1./2)/self.SpeedNorms[-1], (2./1)/self.SpeedNorms[-1]]]
        else:
            self.AllowedDeltaTRanges += [[1./min(self.LocalPaddings[-1]), np.inf]]
        self.LandingTSsMaps += [-np.inf*self.CreateUnitaryMap(OW[2] - OW[0], OW[3] - OW[1])]
        self.StreaksMaps += [0*self.CreateUnitaryMap(OW[2] - OW[0], OW[3] - OW[1])]
        self.Masks += [None]
        self.InitializedAt += [0]
        self.TimeFilteredEvents += [0]
        self.NEventsBySpeed += [0]
        self.OWAST += [OW]

        self.Displacements += [None]
        self.Consistencies += [0]
        self.RelativeMeanStreaks += [0]

        self.NEventsMatched += [0]
        self.CumulatedAwaitingTime += [0]

        return True

    def AddExponentialSeeds(self, vx_center = 0., vy_center = 0., v_min_observed = 1., v_max_observed = 100., add_center = True): #observed are referenced to the center
        center_speed = np.array([vx_center, vy_center])

        MaxSpeedTried = v_max_observed
        self.V_seed = 2*v_min_observed
        seeds = [0.]
        i = 0
        while self.V_seed**i <= MaxSpeedTried:
            seeds += [self.V_seed*(2**i)]
            i += 1
        print "Seeds : ", seeds
        if add_center:
            self.AddSeed(center_speed, (-self.V_seed, -self.V_seed, self.V_seed, self.V_seed), OW = self.DefaultObservationWindow)
        
        for dvx in seeds:
            for dvy in seeds:
                if (dvx > 0 or dvy > 0) and dvx**2 + dvy**2 <= MaxSpeedTried**2:
                    self.AddSeed(center_speed + np.array([dvx, dvy]), (-max(dvx/2, self.V_seed), -max(dvy/2, self.V_seed), max(dvx, self.V_seed), max(dvy, self.V_seed)), OW = self.DefaultObservationWindow)
                    if dvx != 0.:
                        self.AddSeed(center_speed + np.array([-dvx, dvy]), (-max(dvx, self.V_seed), -max(dvy/2, self.V_seed), max(dvx/2, self.V_seed), max(dvy, self.V_seed)), OW = self.DefaultObservationWindow)
                    if dvy != 0:
                        self.AddSeed(center_speed + np.array([-dvx, -dvy]), (-max(dvx, self.V_seed), -max(dvy, self.V_seed), max(dvx/2, self.V_seed), max(dvy/2, self.V_seed)), OW = self.DefaultObservationWindow)
                        self.AddSeed(center_speed + np.array([dvx, -dvy]), (-max(dvx/2, self.V_seed), -max(dvy, self.V_seed), max(dvx, self.V_seed), max(dvy/2, self.V_seed)), OW = self.DefaultObservationWindow)
        print "Initialized {0} speed seeds, going from vx = {1} and vy = {2} to vx = {3} and vy = {4}.".format(len(self.Speeds), np.array(self.Speeds)[:,0].min(), np.array(self.Speeds)[:,1].min(), np.array(self.Speeds)[:,0].max(), np.array(self.Speeds)[:,1].max())

    def DynamicMinTSSet(self, ts):
        self.LastTsSave = self.GroundTS
        self.SpeedsChangesTs += [self.GroundTS]

        self.SetTS = self.GBFunction

    def GBFunction(*kwargs):
        return None

    def CreateUnitaryMap(self, SizeX, SizeY):
        return np.ones((int((SizeX + 2*self.R_Projection) * self.DensityDefinition), int((SizeY + 2*self.R_Projection) * self.DensityDefinition)))

class ProjectorV2:
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Tool to create a 2D projection of the STContext onto a plane t = cst along different speeds.
        Should allow to recove the actual speed and generator of the movement
        '''
        self._ReferencesAsked = []
        self._Name = Name
        self._Framework = Framework
        self._Type = 'Computation'
        self._CreationReferences = dict(argsCreationReferences)

        self.R_Projection = 0.5
        self.ExpansionFactor = 1.

        self.BinDt = 0.100
        self.Precision_aimed = 0.5
        self.Initial_dv_MAX = 160.

        self.DensityDefinition = 3 # In dpp
        self.MaskDensityRatio = 0.3

        self.GroundTS = 0.8

    def _Initialize(self):

        self.ActiveSpeeds = []
        self.Speeds = []

        self.ToInitializeSpeed = []
        self.SpeedStartTime = []
        self.SpeedNorms = []
        self.OWAST = [] # Stands for "Observation Window at Start Time". Means to stabilize any object present at start time, without adding new ones yet to appear. It is given at start time as it moves as time goes on.
        self.DensitiesMaps = []
        self.Masks = []
        self.PresenceMaps = []

        self.DensitiesMaxValues = []

        StreamGeometry = self._Framework.StreamsGeometries[self._Framework.StreamHistory[-1]]
        self.DefaultObservationWindow = [0., 0., float(StreamGeometry[0]+1), float(StreamGeometry[1]+1)]

        self.LocalPaddings = []
        self.InitializedAt = []
        self.NEventsBySpeed = []
        self.Displacements = []
        self.Consistencies = []
        self.RelativeMeanStreaks = []

        self.SpeedsChangesTs = []
        self.SpeedsChangesIndexes = [None]
        self.SpeedsExpanded  = [[]]

        self.AddExponentialSeeds(vx_center = 0., vy_center = 0., v_max_observed = self.Initial_dv_MAX, v_min_observed = self.Precision_aimed*2, add_center = True)

        self.LastTsSave = 0.
        self.TsHistory = []
        self.ConsistencyHistory = []
        self.RelativeMeanStreaksHistory = []

        self.SetTS = self.DynamicMinTSSet

    def _OnEvent(self, event):
        self.SetTS(event.timestamp)
        if event.timestamp > self.GroundTS:
            if len(self.ToInitializeSpeed) > 0:
                for speed_id in self.ToInitializeSpeed:
                    self.SpeedStartTime[speed_id] = event.timestamp
                self.ToInitializeSpeed = []

            for speed_id in self.ActiveSpeeds:
                if self.InitializedAt[speed_id] == 0:
                    if (event.timestamp - self.SpeedStartTime[speed_id])*max(1., abs(self.Speeds[speed_id]).min()) >= self.ExpansionFactor:
                        self.InitializedAt[speed_id] = event.timestamp
                self.ProjectEventWithSpeed(event, speed_id)

            if event.timestamp - self.LastTsSave >= self.BinDt:
                self.LastTsSave = event.timestamp
                self.TsHistory += [event.timestamp]
                
                for speed_id in self.ActiveSpeeds:
                    self.ComputeMask(speed_id)
                    self.ComputeConsistency(speed_id)
                    self.ComputeRelativeMeanStreaks(speed_id)
                
                self.ConsistencyHistory += [list(self.Consistencies)]
                self.RelativeMeanStreaksHistory += [list(self.RelativeMeanStreaks)]

        return event

    def ProjectEventWithSpeed(self, event, speed_id):
        self.UpdateDisplacement(event.timestamp, speed_id)
        x0 = event.location[0] - self.Displacements[speed_id][0]
        y0 = event.location[1] - self.Displacements[speed_id][1]
        
        OW = self.OWAST[speed_id]
        if not (OW[0] <= x0 <= OW[2] and OW[1] <= y0 <= OW[3]):
            return None

        Map = self.DensitiesMaps[speed_id]
        for x in range(int(self.DensityDefinition*((x0 - self.R_Projection) - OW[0])), int(self.DensityDefinition*((x0 + self.R_Projection) - OW[0]))):
            for y in range(int(self.DensityDefinition*((y0 - self.R_Projection) - OW[1])), int(self.DensityDefinition*((y0 + self.R_Projection) - OW[1]))):
                Map[x,y] += 1

    def UpdateDisplacement(self, t, speed_id):
        self.Displacements[speed_id] = (t - self.SpeedStartTime[speed_id])*self.Speeds[speed_id]

    def ComputeMask(self, speed_id):
        Density = self.DensitiesMaps[speed_id]
        self.PresenceMaps[speed_id] = Density > 0

        self.CurrentStreakMap = Density - self.PresenceMaps[speed_id]

        self.Masks[speed_id] = self.CurrentStreakMap > self.CurrentStreakMap.max()*self.MaskDensityRatio

    def ComputeConsistency(self, speed_id):
        Mask = self.Masks[speed_id]
        self.Consistencies[speed_id] = float((Mask[1:-1,1:-1]*Mask[2:,1:-1]*Mask[:-2,1:-1]*Mask[1:-1,:-2]*Mask[1:-1,2:]).sum())/max(1, Mask.sum())

    def ComputeRelativeMeanStreaks(self, speed_id):
        Displacement = np.linalg.norm(self.Displacements[speed_id])
        Density = self.DensitiesMaps[speed_id]
        MeanStreak = float(self.CurrentStreakMap.sum())/max(1, (self.CurrentStreakMap > 0).sum())

        self.RelativeMeanStreaks[speed_id] = MeanStreak/max(1, Displacement)

    def ComputeInterestBoxesBySpeeds(self, dSpaceRatio = 2.):
        Boxes = []
        for speed_id in self.ActiveSpeeds:
            Xs, Ys = np.where(self.Masks[speed_id])
            N_Points = Xs.shape[0]
            Xmean = Xs.mean()
            Ymean = Ys.mean()
            dX = dSpaceRatio * np.sqrt(float(((Xs - Xmean)**2).sum())/N_Points)
            dY = dSpaceRatio * np.sqrt(float(((Ys - Ymean)**2).sum())/N_Points)

            Boxes += [Xmean - dX, Ymean - dY, Xmean + dX, Ymean + dY]
        return Boxes




    def ComputeInterestCentersWithSpeeds(self, n_points, consistency_limit = 0.3, dSpaceRatio = 2.):
        PossibleIDsByConsistency = np.where(np.array(self.Consistencies) > consistency_limit)[0]

        InitialCenters = []
        InitialVariances = []
        InitialSpeedsAssociated = []

        for local_id in np.argsort(-np.array(self.RelativeMeanStreaks)[PossibleIDsByConsistency]): # We sort by -RelativeMeanStreaks to get the highest MeansStreaks first
            speed_id = PossibleIDsByConsistency[local_id]
            if self.RelativeMeanStreaks[speed_id] >= 1.:
                continue
            Xs, Ys = np.where(self.Masks[speed_id])
            Xs = np.array(Xs, dtype = float) / self.DensityDefinition + self.Displacements[speed_id][0] + self.OWAST[speed_id][0]
            Ys = np.array(Ys, dtype = float) / self.DensityDefinition + self.Displacements[speed_id][1] + self.OWAST[speed_id][1]
            N_Points = Xs.shape[0]
            Xmean = Xs.mean()
            Ymean = Ys.mean()
            dX = dSpaceRatio * np.sqrt(float(((Xs - Xmean)**2).sum())/N_Points)
            dY = dSpaceRatio * np.sqrt(float(((Ys - Ymean)**2).sum())/N_Points)
            
            InitialCenters += [np.array([Xmean, Ymean])]
            InitialVariances += [np.array([dX, dY])]
            InitialSpeedsAssociated += [speed_id]

        NObjectsFound = 0
        FinalCenters = []
        FinalVariances = []
        FinalSpeeds = []
        FinalSpeedsAverage = []

        for Center, Variances, SpeedId in zip(InitialCenters, InitialVariances, InitialSpeedsAssociated):
            FoundSimilarObject = False
            for nObject in range(NObjectsFound):
                nSpeedsInObject = len(FinalSpeeds[nObject])
                if 0.8 <= Variances[0] / FinalVariances[nObject][0] <= 1.2 and 0.8 <= Variances[1] / FinalVariances[nObject][1] <= 1.2:
                    if (abs(Center - FinalCenters[nObject]) < FinalVariances[nObject]/dSpaceRatio).all():
                        if np.linalg.norm(self.Speeds[SpeedId] - FinalSpeedsAverage[nObject])/max(1, np.linalg.norm(FinalSpeedsAverage[nObject])) <= 1.:
                            FinalCenters[nObject] = (FinalCenters[nObject]*nSpeedsInObject + Center)/(nSpeedsInObject+1)
                            FinalVariances[nObject] = (FinalVariances[nObject]*nSpeedsInObject + Variances)/(nSpeedsInObject+1)
                            FinalSpeeds[nObject] += [SpeedId]
                            FinalSpeedsAverage[nObject] = (FinalSpeedsAverage[nObject]*nSpeedsInObject + self.Speeds[SpeedId])/(nSpeedsInObject+1)

                            FoundSimilarObject = True
                            break

            if not FoundSimilarObject and NObjectsFound < n_points:
                NObjectsFound += 1
                FinalCenters += [Center]
                FinalVariances += [Variances]
                FinalSpeeds += [[SpeedId]]
                FinalSpeedsAverage += [self.Speeds[SpeedId]]
        
        Boxes = [[Center[0] - Variances[0], Center[1] - Variances[1], Center[0] + Variances[0], Center[1] + Variances[1]] for Center, Variances in zip(FinalCenters, FinalVariances)]

        return Boxes, FinalSpeeds

    def PlotCurrentInterestRegions(self, f, ax, n_points = 5, consistency_limit = 0.3):
        Boxes, SpeedsIDs = self.ComputeInterestCentersWithSpeeds(n_points, consistency_limit)
        n = 0
        for Box, ConcernedSpeedsIDs in zip(Boxes, SpeedsIDs):
            n += 1
            ax.plot([Box[0], Box[0]], [Box[1], Box[3]], 'grey')
            ax.plot([Box[2], Box[2]], [Box[1], Box[3]], 'grey')
            ax.plot([Box[0], Box[2]], [Box[1], Box[1]], 'grey')
            ax.plot([Box[0], Box[2]], [Box[3], Box[3]], 'grey')

            ax.text(Box[0] + 1, Box[1] + 1, "Box # {0}\n".format(n) + "\n".join(["Vx = {0:.1f}, Vy = {1:.1f}, id = {2}".format(self.Speeds[speed_id][0], self.Speeds[speed_id][1], speed_id) for speed_id in ConcernedSpeedsIDs]), color = 'grey')

    def AddSeed(self, speed, localpadding, OW, force = False):
        if force:
            for PreviousSpeed in np.array(self.Speeds)[self.ActiveSpeeds]:
                if (abs(speed - PreviousSpeed) < 0.001).all():
                    return False
        else:
            for PreviousSpeed in self.Speeds:
                if (abs(speed - PreviousSpeed) < 0.001).all():
                    return False

        speed_id = len(self.Speeds)
        self.ActiveSpeeds += [speed_id]
        self.ToInitializeSpeed += [speed_id]
        self.Speeds += [speed]
        self.LocalPaddings += [localpadding]

        self.SpeedStartTime += [0]

        self.SpeedNorms += [np.linalg.norm(speed)]
        self.DensitiesMaps += [self.CreateEmptyDensityMap(OW[2] - OW[0], OW[3] - OW[1])]
        self.PresenceMaps += [None]
        self.Masks += [None]
        self.InitializedAt += [0]
        self.NEventsBySpeed += [0]
        self.OWAST += [OW]

        self.Displacements += [None]
        self.Consistencies += [0]
        self.RelativeMeanStreaks += [0]

        return True

    def AddExponentialSeeds(self, vx_center = 0., vy_center = 0., v_min_observed = 1., v_max_observed = 100., add_center = True): #observed are referenced to the center
        center_speed = np.array([vx_center, vy_center])

        MaxSpeedTried = v_max_observed
        v_seed = 2.*v_min_observed
        seeds = [0.]
        i = 1
        while v_seed**i <= MaxSpeedTried:
            seeds += [v_seed**i]
            i += 1
        print "Seeds : ", seeds
        if add_center:
            self.AddSeed(center_speed, (-v_seed, -v_seed, v_seed, v_seed), OW = self.DefaultObservationWindow)
        
        for dvx in seeds:
            for dvy in seeds:
                if (dvx > 0 or dvy > 0) and dvx**2 + dvy**2 <= MaxSpeedTried**2:
                    self.AddSeed(center_speed + np.array([dvx, dvy]), (-max(dvx/2, v_seed), -max(dvy/2, v_seed), max(dvx, v_seed), max(dvy, v_seed)), OW = self.DefaultObservationWindow)
                    if dvx != 0.:
                        self.AddSeed(center_speed + np.array([-dvx, dvy]), (-max(dvx, v_seed), -max(dvy/2, v_seed), max(dvx/2, v_seed), max(dvy, v_seed)), OW = self.DefaultObservationWindow)
                    if dvy != 0:
                        self.AddSeed(center_speed + np.array([-dvx, -dvy]), (-max(dvx, v_seed), -max(dvy, v_seed), max(dvx/2, v_seed), max(dvy/2, v_seed)), OW = self.DefaultObservationWindow)
                        self.AddSeed(center_speed + np.array([dvx, -dvy]), (-max(dvx/2, v_seed), -max(dvy, v_seed), max(dvx, v_seed), max(dvy/2, v_seed)), OW = self.DefaultObservationWindow)
        print "Initialized {0} speed seeds, going from vx = {1} and vy = {2} to vx = {3} and vy = {4}.".format(len(self.Speeds), np.array(self.Speeds)[:,0].min(), np.array(self.Speeds)[:,1].min(), np.array(self.Speeds)[:,0].max(), np.array(self.Speeds)[:,1].max())

    def DynamicMinTSSet(self, ts):
        self.LastTsSave = self.GroundTS
        self.SpeedsChangesTs += [self.GroundTS]

        self.SetTS = self.GBFunction

    def GBFunction(*kwargs):
        return None

    def CreateEmptyDensityMap(self, SizeX, SizeY):
        return np.zeros((int((SizeX + 2*self.R_Projection) * self.DensityDefinition), int((SizeY + 2*self.R_Projection) * self.DensityDefinition)))

class ProjectorV1:
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Tool to create a 2D projection of the STContext onto a plane t = cst along different speeds.
        Should allow to recove the actual speed and generator of the movement
        '''
        self._ReferencesAsked = []
        self._Name = Name
        self._Framework = Framework
        self._Type = 'Computation'
        self._CreationReferences = dict(argsCreationReferences)

        self.R_Projection = 0.5
        self.ExpansionFactor = 1.

        self.BinDt = 0.002
        self.Precision_aimed = 0.5
        self.Initial_dv_MAX = 160.

        self.DensityDefinition = 5 # In dpp

    def _Initialize(self):

        self.ActiveSpeeds = []
        self.Speeds = []

        self.ToInitializeSpeed = []
        self.SpeedStartTime = []
        self.SpeedNorms = []
        self.PartsList = []
        self.IntersectingPartsList = []
        self.ZonePartsIDs = []
        self.Areas = []
        self.IntersectingAreas = []
        self.Densities = []
        self.LocalPaddings = []
        self.AssumedInitialized = []
        self.NEventsBySpeed = []

        self.SpeedsChangesTs = []
        self.SpeedsChangesIndexes = [None]
        self.SpeedsExpanded  = [[]]
        self.MinRunArea = [1.]
        
        self.AddExponentialSeeds(vx_center = 0., vy_center = 0., v_max_observed = self.Initial_dv_MAX, v_min_observed = self.Precision_aimed*2, add_center = True)

        self.LastTsSave = 0.

        self.ArgMinAreaHistory = []
        self.NormalizedAreasHistory = []
        self.NormalizedIntersectingAreasHistory = []
        self.PartsNumberHistory = []

        self.MinArea = 1.
        self.ArgMinArea = -1
        self.TsArgMin = np.inf

        self.SetTS = self.DynamicMinTSSet

    def _OnEvent(self, event):
        self.SetTS(event.timestamp)
        if event.timestamp > self.MinTS:
            if len(self.ToInitializeSpeed) > 0:
                for speed_id in self.ToInitializeSpeed:
                    self.SpeedStartTime[speed_id] = event.timestamp
                self.ToInitializeSpeed = []

            PreviousArg = self.ArgMinArea
            self.MinArea = 1.

            self.ExpandableBunch = 1
            for speed_id in self.ActiveSpeeds:
                if self.AssumedInitialized[speed_id] == 0:
                    #if (event.timestamp - self.SpeedStartTime[speed_id])*max(2., self.SpeedNorms[speed_id]) >= self.ExpansionFactor * np.sqrt(2):
                    if (event.timestamp - self.SpeedStartTime[speed_id])*max(1., abs(self.Speeds[speed_id]).min()) >= self.ExpansionFactor:
                        self.AssumedInitialized[speed_id] = event.timestamp
                    else:
                        if True or not (event.timestamp - self.SpeedStartTime[speed_id]) > 0.1:
                            self.ExpandableBunch *= 0
                self.ProjectEventWithSpeed(event, speed_id)

            if self.ArgMinArea != PreviousArg:
                print "Changed best speed to vx = {0} and vy  = {1}, with area {3:.2f} at t = {4:.2f}. Arg = {2}".format(self.Speeds[self.ArgMinArea][0], self.Speeds[self.ArgMinArea][1], self.ArgMinArea, self.MinArea, event.timestamp)
                self.TsArgMin = event.timestamp
            
            if self.ArgMinArea != -1 and event.timestamp - self.LastTsSave >= self.BinDt:
                self.LastTsSave = event.timestamp
                self.ts += [event.timestamp]
                self.ArgMinAreaHistory += [self.ArgMinArea]
                self.NormalizedAreasHistory += [(np.array(self.Areas)/np.array(self.NEventsBySpeed)).tolist()]
                self.NormalizedIntersectingAreasHistory += [(np.array(self.IntersectingAreas)/np.array(self.Areas)).tolist()]
                self.PartsNumberHistory += [[len(PartList) for PartList in self.PartsList]]
                #self.UpdateDensities()

                if True and event.timestamp - self.TsArgMin > 0.1 and self.NormalizedIntersectingAreasHistory[-1][self.ArgMinArea] > 1.:
                    self.ExpandableBunch = 1

                if self.MinArea > self.MinRunArea[-1]:
                    self.ExpandableBunch = 0

                ActiveNormalizedAreas = np.array(self.NormalizedAreasHistory[-1])[self.ActiveSpeeds]
                if ActiveNormalizedAreas.max() < 1.2*ActiveNormalizedAreas.min():
                    self.ExpandableBunch = 0

                self.ClearUselessSpeeds(event)
        return event

    def ProjectEventWithSpeed(self, event, n_speed):
        x0 = event.location[0] - (event.timestamp - self.t0) * self.Speeds[n_speed][0]
        y0 = event.location[1] - (event.timestamp - self.t0) * self.Speeds[n_speed][1]
        
        NewElement = geometry.CreateElementFromPosition(x0, y0, self.R_Projection)
        
        Center = np.array([x0, y0])
        ClosestReferencePoint = tuple(np.array((Center + np.array([self.R_Projection, self.R_Projection]))/(2*self.R_Projection), dtype = int))

        self.NEventsBySpeed[n_speed] += 1

        UsefulPartsIDsList = []
        UsefulPartsList = []
        try:
            ZoneList = self.ZonePartsIDs[n_speed][ClosestReferencePoint]
        except:
            NewParts = [NewElement]
            self.Areas[n_speed] += geometry.ComputeArea(NewParts)
            IntersectingParts = []
        else:
            for PartID in ZoneList:
                UsefulPartsList += [self.PartsList[n_speed][PartID]]
            NewParts, IntersectingParts, self.Areas[n_speed], self.IntersectingAreas[n_speed] = geometry.ComputeNewItems(UsefulPartsList, NewElement, self.Areas[n_speed], self.IntersectingAreas[n_speed])

        for NewPart in NewParts:
            ID = len(self.PartsList[n_speed])
            self.PartsList[n_speed] += [NewPart]
            Center = np.array([NewPart[2] + NewPart[0], NewPart[3] + NewPart[1]])/2
            ClosestReferencePoint = np.array((Center + np.array([self.R_Projection, self.R_Projection]))/(2*self.R_Projection), dtype = int)
            for i in range(-1,2):
                for j in range(-1,2):
                    key = (ClosestReferencePoint[0] + i, ClosestReferencePoint[1] + j)
                    try:
                        self.ZonePartsIDs[n_speed][key] += [ID]
                    except:
                        self.ZonePartsIDs[n_speed][key] = []
                        self.ZonePartsIDs[n_speed][key] += [ID]

        self.IntersectingPartsList[n_speed] += IntersectingParts

        if self.Areas[n_speed]/self.NEventsBySpeed[n_speed] < self.MinArea: # Possible check needed for this line, but seems to work better without the AssumedInitialized condition
        #if self.AssumedInitialized[n_speed] != 0 and self.Areas[n_speed]/self.NEventsBySpeed[n_speed] < self.MinArea:
            self.ArgMinArea = n_speed
            self.MinArea = self.Areas[n_speed]/self.NEventsBySpeed[n_speed]
    
    def UpdateDensities(self):
        for speed_id in self.ActiveSpeeds:
            self.Densities[speed_id] = geometry.CreateDensityProjection(self.PartsList[speed_id], self.IntersectingPartsList[speed_id], self.DensityDefinition, verbose = False)

    def ClearUselessSpeeds(self, event):
        conserve_top = max(5, int(len(self.ActiveSpeeds)/9))
        if not self.ExpandableBunch:
            return None
        
        if (abs(np.array(self.LocalPaddings[self.ArgMinArea])) <= self.Precision_aimed).all():
            return None

        self.TsArgMin = np.inf
        self.ArgMinArea = -1

        Expandables = []
        Values = []
        
        for speed_id in self.ActiveSpeeds:
            Expandables += [speed_id]
            Values += [self.NormalizedAreasHistory[-1][speed_id]]
            
        self.ActiveSpeeds = []
        self.SpeedsExpanded += [[]]

        for local_speed_id in np.argsort(Values)[:conserve_top]:
            self.SpeedsExpanded[-1] += [Expandables[local_speed_id]]
            self.AddStrictSeedsFromPadding(self.Speeds[Expandables[local_speed_id]], self.LocalPaddings[Expandables[local_speed_id]], add_center = True, force = True)
            
        self.SpeedsChangesTs += [event.timestamp]
        self.SpeedsChangesIndexes += [len(self.ts) - 2]
        self.MinRunArea += [self.MinArea]
        print "Changed speeds, now containing {0} speeds, from vx = {1} and vy = {2} to vx = {3} and vy = {4}.".format(len(self.ActiveSpeeds), np.array(self.Speeds)[self.ActiveSpeeds,0].min(), np.array(self.Speeds)[self.ActiveSpeeds,1].min(), np.array(self.Speeds)[self.ActiveSpeeds,0].max(), np.array(self.Speeds)[self.ActiveSpeeds,1].max())

    def AddSeed(self, speed, localpadding, force = False):
        if force:
            for PreviousSpeed in np.array(self.Speeds)[self.ActiveSpeeds]:
                if (abs(speed - PreviousSpeed) < 0.001).all():
                    return False
        else:
            for PreviousSpeed in self.Speeds:
                if (abs(speed - PreviousSpeed) < 0.001).all():
                    return False

        speed_id = len(self.Speeds)
        self.ActiveSpeeds += [speed_id]
        self.ToInitializeSpeed += [speed_id]
        self.Speeds += [speed]
        self.LocalPaddings += [localpadding]

        self.SpeedStartTime += [0]

        self.SpeedNorms += [np.linalg.norm(speed)]
        self.PartsList += [[]]
        self.IntersectingPartsList += [[]]
        self.Densities += [None]
        self.ZonePartsIDs += [{}]
        self.Areas += [0]
        self.IntersectingAreas += [0]
        self.AssumedInitialized += [0]
        self.NEventsBySpeed += [0]

        return True

    def AddExponentialSeeds(self, vx_center = 0., vy_center = 0., v_min_observed = 1., v_max_observed = 100., add_center = True): #observed are referenced to the center
        center_speed = np.array([vx_center, vy_center])

        MaxSpeedTried = v_max_observed
        v_seed = 2.*v_min_observed
        seeds = [0.]
        i = 1
        while v_seed**i <= MaxSpeedTried:
            seeds += [v_seed**i]
            i += 1
        print "Seeds : ", seeds
        if add_center:
            self.AddSeed(center_speed, (-v_seed, -v_seed, v_seed, v_seed))
        
        for dvx in seeds:
            for dvy in seeds:
                if (dvx > 0 or dvy > 0) and dvx**2 + dvy**2 <= MaxSpeedTried**2:
                    self.AddSeed(center_speed + np.array([dvx, dvy]), (-max(dvx/2, v_seed), -max(dvy/2, v_seed), max(dvx, v_seed), max(dvy, v_seed)))
                    if dvx != 0.:
                        self.AddSeed(center_speed + np.array([-dvx, dvy]), (-max(dvx, v_seed), -max(dvy/2, v_seed), max(dvx/2, v_seed), max(dvy, v_seed)))
                    if dvy != 0:
                        self.AddSeed(center_speed + np.array([-dvx, -dvy]), (-max(dvx, v_seed), -max(dvy, v_seed), max(dvx/2, v_seed), max(dvy/2, v_seed)))
                        self.AddSeed(center_speed + np.array([dvx, -dvy]), (-max(dvx/2, v_seed), -max(dvy, v_seed), max(dvx, v_seed), max(dvy/2, v_seed)))
        print "Initialized {0} speed seeds, going from vx = {1} and vy = {2} to vx = {3} and vy = {4}.".format(len(self.Speeds), np.array(self.Speeds)[:,0].min(), np.array(self.Speeds)[:,1].min(), np.array(self.Speeds)[:,0].max(), np.array(self.Speeds)[:,1].max())

    def AddStrictSeedsFromPadding(self, center_speed, padding, add_center = False, force = False):
        vx_seeds = [padding[0]/2, 0, padding[2]/2]
        vy_seeds = [padding[1]/2, 0, padding[3]/2]
        n_seeds_added = 0
        for n_dvx in range(len(vx_seeds)):
            xMinPaddingIndex = max(0, n_dvx-1)
            xMinPadding = vx_seeds[xMinPaddingIndex] - vx_seeds[xMinPaddingIndex+1]
            xMaxPaddingIndex = min(2, n_dvx+1)
            xMaxPadding = vx_seeds[xMaxPaddingIndex] - vx_seeds[xMaxPaddingIndex-1]
            
            for n_dvy in range(len(vy_seeds)):
                yMinPaddingIndex = max(0, n_dvy-1)
                yMinPadding = vy_seeds[yMinPaddingIndex] - vy_seeds[yMinPaddingIndex+1]
                yMaxPaddingIndex = min(2, n_dvy+1)
                yMaxPadding = vy_seeds[yMaxPaddingIndex] - vy_seeds[yMaxPaddingIndex-1]
                
                dv = np.array([vx_seeds[n_dvx], vy_seeds[n_dvy]])
                if n_dvx != 1 or n_dvy != 1 or add_center:
                    n_seeds_added += self.AddSeed(center_speed + dv, (xMinPadding, yMinPadding, xMaxPadding, yMaxPadding), force)
        return n_seeds_added

    def DynamicMinTSSet(self, ts):
        self.MinTS = 0.8
        self.ts = [self.MinTS]
        self.t0 = 0.8
        self.SpeedsChangesTs += [self.MinTS]

        self.SetTS = self.GBFunction

    def GBFunction(*kwargs):
        return None

################################               PLOT FUNCTIONS                   ################################               PLOT FUNCTIONS                   ################################

    def PlotCurrentSpeeds(self):
        f, ax = plt.subplots(1,1)
        for speed_id in self.ActiveSpeeds:
            ax.plot(self.Speeds[speed_id][0], self.Speeds[speed_id][1], 'vr')

    def PlotNormalizedAreas(self):
        f, ax = plt.subplots(1,1)
        n_start = 0
        
        for nArea in range(len(self.Speeds)):
            while len(self.NormalizedAreasHistory[n_start]) < nArea + 1:
                n_start += 1
            NormalizedAreas = []
            for n in range(n_start, len(self.NormalizedAreasHistory)):
                NormalizedAreas += [self.NormalizedAreasHistory[n][nArea]]
            ax.plot(self.ts[-len(NormalizedAreas):], NormalizedAreas)
