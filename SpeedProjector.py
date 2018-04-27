import numpy as np
import geometry
import sys
import matplotlib.pyplot as plt

class Projector:
    def __init__(self, Network):
        self.Network = Network

        self.R_Projection = 0.5
        self.ExpansionFactor = 1.

        self.Initialize()

    def Initialize(self, BinDt = 0.002, Initial_dv_MAX = 160., n_points_linear_reg = 10, LinearRegMinimumLifespan = 0.2, LinearRegThresholdValue = 0.99, Precision_aimed = 0.5):
        self.ActiveSpeeds = []
        self.Speeds = []

        self.ToInitializeSpeed = []
        self.SpeedStartTime = []
        self.SpeedNorms = []
        self.PartsList = []
        self.IntersectingPartsList = []
        self.ZonePartsIDs = []
        self.Areas = []
        self.LocalPaddings = []
        self.Expanded = []
        self.AssumedInitialized = []
        self.Slopes = []
        self.NEventsBySpeed = []

        self.Precision_aimed = Precision_aimed
        self.AddExponentialSeeds(vx_center = 0., vy_center = 0., v_max_observed = Initial_dv_MAX, v_min_observed = 1., add_center = True)
        self.InitialNumberOfSpeeds = len(self.Speeds)

        self.LastTsSave = 0.
        self.BinDt = BinDt

        self.ArgMinAreaHistory = []
        self.AreasHistory = []
        self.NormalizedAreasHistory = []

        self.MinArea = 1.
        self.ArgMinArea = -1

        self.PointsInLinearReg = n_points_linear_reg
        self.LinearRegMinimumLifespan = LinearRegMinimumLifespan
        self.LinearRegThresholdValue = LinearRegThresholdValue

        self.SetTS = self.DynamicMinTSSet

    def OnEvent(self, event):
        self.SetTS(event.timestamp)
        if event.timestamp > self.MinTS:
            if len(self.ToInitializeSpeed) > 0:
                for speed_id in self.ToInitializeSpeed:
                    self.SpeedStartTime[speed_id] = event.timestamp
                self.ToInitializeSpeed = []

            PreviousArg = self.ArgMinArea
            self.MinArea = 1.

            for speed_id in self.ActiveSpeeds:
                if self.AssumedInitialized[speed_id] == 0 and (event.timestamp - self.SpeedStartTime[speed_id])*max(2., self.SpeedNorms[speed_id]) >= self.ExpansionFactor * np.sqrt(2):
                    self.AssumedInitialized[speed_id] = event.timestamp
                self.ProjectEventWithSpeed(event, speed_id)

            if self.ArgMinArea != PreviousArg:
                print "Changed best speed to vx = {0} and vy  = {1}. Arg = {2}".format(self.Speeds[self.ArgMinArea][0], self.Speeds[self.ArgMinArea][1], self.ArgMinArea)

            if self.ArgMinArea != -1 and event.timestamp - self.LastTsSave >= self.BinDt:
                self.LastTsSave = event.timestamp
                self.ts += [event.timestamp]
                self.ArgMinAreaHistory += [self.ArgMinArea]
                self.AreasHistory += [list(self.Areas)]
                self.NormalizedAreasHistory += [(np.array(self.Areas)/np.array(self.NEventsBySpeed)).tolist()]

                self.ClearUselessSpeeds()
                    
                #if (abs(np.array(self.LocalPaddings[self.ArgMinArea])) > self.Precision_aimed).any():
                #    if len(self.ActiveSpeeds) < self.InitialNumberOfSpeeds - 8:
                #        self.ExpandSpeed(self.ArgMinArea)

    def ProjectEventWithSpeed(self, event, n_speed):
        x0 = event.location[0] - (event.timestamp - self.t0) * self.Speeds[n_speed][0]
        y0 = event.location[1] - (event.timestamp - self.t0) * self.Speeds[n_speed][1]
        
        NewElement = Geometry.CreateElementFromPosition(x0, y0, self.R_Projection)
        
        Center = np.array([x0, y0])
        ClosestReferencePoint = tuple(np.array((Center + np.array([self.R_Projection, self.R_Projection]))/(2*self.R_Projection), dtype = int))

        self.NEventsBySpeed[n_speed] += 1

        UsefulPartsIDsList = []
        UsefulPartsList = []
        try:
            ZoneList = self.ZonePartsIDs[n_speed][ClosestReferencePoint]
        except:
            NewParts = [NewElement]
            self.Areas[n_speed] += Geometry.ComputeArea(NewParts)
            IntersectingParts = []
        else:
            for PartID in ZoneList:
                UsefulPartsList += [self.PartsList[n_speed][PartID]]
            NewParts, IntersectingParts, self.Areas[n_speed] = Geometry.ComputeNewItems(UsefulPartsList, NewElement, self.Areas[n_speed])

        self.IntersectingPartsList[n_speed] += IntersectingParts

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

        if self.AssumedInitialized[n_speed] != 0 and self.Areas[n_speed]/self.NEventsBySpeed[n_speed] < self.MinArea:
            self.ArgMinArea = n_speed
            self.MinArea = self.Areas[n_speed]/self.NEventsBySpeed[n_speed]

    def ClearUselessSpeeds(self, NormalizedRatio = 0.8):
        Batch = self.NormalizedAreasHistory[-self.PointsInLinearReg:]
        if len(Batch) < self.PointsInLinearReg:
            return None # In early stages, no speed has stored enough data to carry the linear regression

        n_cleared = 0
        Ts = np.array(self.ts[-self.PointsInLinearReg:])
        TsMean = Ts.mean()
        DeltasT = Ts - TsMean

        MaxIDAvailable = len(Batch[0]) - 1 # Only the first ones initialized have enough data to arry the linear regression

        PossibleNormalizedValues = []
        PossibleIDs = []
        MinPossibleValue = 1.

        for speed_id in list(self.ActiveSpeeds):
            if self.NormalizedAreasHistory[-1][speed_id] < min(0.2, self.MinArea):
                if len(self.ActiveSpeeds) < self.InitialNumberOfSpeeds - 8:
                    self.ExpandSpeed(speed_id)
            else:
                if speed_id <= MaxIDAvailable and self.AssumedInitialized[speed_id] != 0:
                    NormalizedAreas = np.array([BatchStep[speed_id] for BatchStep in Batch])
                    NAmean = NormalizedAreas.mean()
                    DeltasNA = NormalizedAreas - NAmean
                    if (DeltasNA != 0).any():
                        slope = (DeltasNA**2).sum()/(DeltasNA*DeltasT).sum()
                        R2 = 2 - slope**2 * (DeltasT**2).sum()/(DeltasNA**2).sum()
                    else:
                        slope = 0.
                        R2 = 1.
                    if R2 > 0.95 and abs(slope) < 0.2:
                        PossibleNormalizedValues += [NormalizedAreas[-1]]
                        PossibleIDs += [speed_id]
                        MinPossibleValue = min(MinPossibleValue, NormalizedAreas[-1])

        for LocalId in range(len(PossibleIDs)):
            if PossibleNormalizedValues[LocalId] > 2*MinPossibleValue:
                n_cleared += 1
                self.ActiveSpeeds.remove(PossibleIDs[LocalId])
            elif PossibleNormalizedValues[LocalId] < 1.1*MinPossibleValue:
                if len(self.ActiveSpeeds) < self.InitialNumberOfSpeeds - 8:
                    if (abs(np.array(self.LocalPaddings[PossibleIDs[LocalId]])) > self.Precision_aimed).any():
                        self.ExpandSpeed(PossibleIDs[LocalId])

        if n_cleared > 0:
            print "Disabled {0} speeds due to excessive areas, {1} remaining".format(n_cleared, len(self.ActiveSpeeds))


    def ClearUselessSpeedsOld2(self, SlopeMargin = 0.8, verbose = False):
        Batch = self.AreasHistory[-self.PointsInLinearReg:]
        Ts = np.array(self.ts[-self.PointsInLinearReg:])
        TsMean = Ts.mean()
        DeltasT = Ts - TsMean

        MaxIDAvailable = len(Batch[0]) - 1 # Only the first ones initialized have enough data to arry the linear regression

        if len(Batch) < self.PointsInLinearReg:
            return None # In early stages, no speed has stored enough data to carry the linear regression

        Ts = np.array(self.ts[-self.PointsInLinearReg:])
        TsMean = Ts.mean()
        DeltasT = Ts - TsMean

        MaxIDAvailable = len(Batch[0]) - 1 # Only the first ones initialized have enough data to arry the linear regression

        ComputedRegressionsIDs = []
        SlopeValues = []
        SlopeSum = 0
        MinSlope = np.inf
        for speed_id in self.ActiveSpeeds:
            if speed_id <= MaxIDAvailable and Ts[-1] > self.AssumedInitialized[speed_id] + self.PointsInLinearReg*self.BinDt:
                Areas = np.array([BatchStep[speed_id] for BatchStep in Batch])
                Amean = Areas.mean()
                DeltasA = Areas - Amean
                if (DeltasA != 0).any():
                    slope = (DeltasA**2).sum()/(DeltasA*DeltasT).sum()
                else:
                    slope = 0

                R2 = 2 - slope**2 * (DeltasT**2).sum()/(DeltasA**2).sum()
                if verbose:
                    sys.stdout.write("{0:.3f}, ".format(R2))
                if R2 > self.LinearRegThresholdValue:
                    ComputedRegressionsIDs += [speed_id]
                    SlopeValues += [slope]
                    self.Slopes[speed_id] = slope
                    SlopeSum += slope
                    MinSlope = min(MinSlope, slope)
            elif speed_id > MaxIDAvailable:
                break

        if verbose:
            sys.stdout.write("\n")
        if len(SlopeValues) == 0:
            return None

        MeanSlope = SlopeSum/len(SlopeValues)
        SlopThreshold = MeanSlope + (MeanSlope - MinSlope)*SlopeMargin
        n_cleared = 0
        for local_speed_id in range(len(ComputedRegressionsIDs)):
            if SlopeValues[local_speed_id] > SlopThreshold:
                self.ActiveSpeeds.remove(ComputedRegressionsIDs[local_speed_id])
                n_cleared += 1

        if n_cleared > 0:
            print "Disabled {0} speeds due to excessive areas, {1} remaining".format(n_cleared, len(self.ActiveSpeeds))

    def ClearUselessSpeedsOld(self, AreaMargin = 0.8):
        CurrentSum = 0.
        CurrentAreas = []
        CurrentIDs = []
        for speed_id in self.ActiveSpeeds:
            if self.AssumedInitialized[speed_id] != 0:
                CurrentIDs += [speed_id]
                CurrentAreas += [self.Areas[speed_id]]
                CurrentSum += self.Areas[speed_id]

        N_Initialized = len(CurrentIDs)
        if N_Initialized == 0:
            return None
        MeanArea = CurrentSum/N_Initialized
        CurrentAreas = np.array(CurrentAreas)
        DeltaArea = MeanArea - self.MinArea

        AreaThreshold = MeanArea + DeltaArea * AreaMargin

        n_cleared = 0
        for LocalID in range(N_Initialized):
            if CurrentAreas[LocalID] > AreaThreshold:
                self.ActiveSpeeds.remove(CurrentIDs[LocalID])
                n_cleared += 1
        if n_cleared > 0:
            print "Disabled {0} speeds due to excessive areas, {1} remaining".format(n_cleared, len(self.ActiveSpeeds))

    def ExpandSpeed(self, speed_id, expansionFactor = 1.):
        if self.Expanded[speed_id]: # We first check that we didn't already expand this speed
            return None
        self.Expanded[speed_id] = True
        n_seeds_added = self.AddStrictSeedsFromPadding(self.Speeds[speed_id], self.LocalPaddings[speed_id])

        if n_seeds_added > 0:
            print "Added {0} speeds from index {1}, now {2} active speeds".format(n_seeds_added, speed_id, len(self.ActiveSpeeds))

    def AddSeed(self, speed, localpadding):
        for PreviousSpeed in self.Speeds:
            if (speed == PreviousSpeed).all():
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
        self.ZonePartsIDs += [{}]
        self.Areas += [0]
        self.Expanded += [False]
        self.AssumedInitialized += [0]
        self.Slopes += [None]
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

    def AddStrictSeedsFromPadding(self, center_speed, padding):
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
                if n_dvx != 1 or n_dvy != 1:
                    n_seeds_added += self.AddSeed(center_speed + dv, (xMinPadding, yMinPadding, xMaxPadding, yMaxPadding))
        return n_seeds_added

    def DynamicMinTSSet(self, ts):
        self.MinTS = 0.8
        self.ts = [self.MinTS]
        self.t0 = 0.8

        self.SetTS = self.GBFunction

    def GBFunction(*kwargs):
        return None

################################               PLOT FUNCTIONS                   ################################               PLOT FUNCTIONS                   ################################

    def PlotCurrentSpeeds(self):
        f, ax = plt.subplots(1,1)
        for speed_id in self.ActiveSpeeds:
            ax.plot(self.Speeds[speed_id][0], self.Speeds[speed_id][1], 'vr')


################################               OLD AND GARBAGE                  ################################               OLD AND GARBAGE                  ################################

    def HandleLinearRegressionsOld(self, t, n_speed_kept = 3, verbose = False):
        Batch = self.AreasHistory[-self.PointsInLinearReg:]
        if len(Batch) < self.PointsInLinearReg:
            return [] # In early stages, no speed has stored enough data to carry the linear regression

        Ts = np.array(self.ts[-self.PointsInLinearReg:])
        TsMean = Ts.mean()
        DeltasT = Ts - TsMean

        MaxIDAvailable = len(Batch[0]) - 1 # Only the first ones initialized have enough data to arry the linear regression

        ComputedRegressionsIDs = []
        SlopeValues = []
        for speed_id in self.ActiveSpeeds:
            if speed_id <= MaxIDAvailable and t - self.SpeedStartTime[speed_id] > self.LinearRegMinimumLifespan:
                Areas = np.array([BatchStep[speed_id] for BatchStep in Batch])
                Amean = Areas.mean()
                DeltasA = Areas - Amean
                slope = (DeltasA**2).sum()/(DeltasA*DeltasT).sum()

                R2 = 2 - slope**2 * (DeltasT**2).sum()/(DeltasA**2).sum()
                if verbose:
                    sys.stdout.write("{0:.3f}, ".format(R2))
                if R2 > self.LinearRegThresholdValue:
                    ComputedRegressionsIDs += [speed_id]
                    SlopeValues += [slope]
            else:
                break
        if verbose:
            sys.stdout.write("\n")
        if len(SlopeValues) == 0:
            return []

        AreasToConsider = [self.Areas[speed_id_computed] for speed_id_computed in ComputedRegressionsIDs]
        Sorted_IDs = np.argsort(AreasToConsider)

        if len(Sorted_IDs[n_speed_kept:]) > 0:
            print "Disabling {0} speeds due to excessive areas".format(len(Sorted_IDs[n_speed_kept:]))
            for disabled_ID in Sorted_IDs[n_speed_kept:]: #Disable the speeds with excessive areas
                self.ActiveSpeeds.remove(ComputedRegressionsIDs[disabled_ID])

        return [ComputedRegressionsIDs[LocalID] for LocalID in Sorted_IDs[:n_speed_kept]]

    def AddLinearSeeds(self, vx_center = 0., vy_center = 0., dv_MAX = 160., dvx_MAX = 10000., dvy_MAX = 10000., v_padding = 10., add_center = True):
        dvx_max = min(dv_MAX, dvx_MAX)
        if add_center:
            self.AddSeed(np.array([vx_center, vy_center]), (-v_padding, -v_padding, v_padding, v_padding))
        for dn_vx in range(1 + int(dvx_max/v_padding)):
            dvx = dn_vx*v_padding
            dvy_max = min(np.sqrt(dv_MAX ** 2 - dvx ** 2), dvy_MAX)
            for dn_vy in range(1 + int(dvy_max/v_padding)):
                if dn_vx > 0 or dn_vy > 0:
                    dvy = dn_vy*v_padding
                    self.AddSeed(np.array((vx_center + dvx, vy_center + dvy)), (-v_padding, -v_padding, v_padding, v_padding))
                    if dvy != 0.:
                        self.AddSeed(np.array((vx_center + dvx, vy_center - dvy)), (-v_padding, -v_padding, v_padding, v_padding))
                    if dvx != 0.:
                        self.AddSeed(np.array((vx_center - dvx, vy_center + dvy)), (-v_padding, -v_padding, v_padding, v_padding))
                        self.AddSeed(np.array((vx_center - dvx, vy_center - dvy)), (-v_padding, -v_padding, v_padding, v_padding))
        print "Initialized {0} speed seeds, going from vx = {1} and vy = {2} to vx = {3} and vy = {4}.".format(len(self.Speeds), np.array(self.Speeds)[:,0].min(), np.array(self.Speeds)[:,1].min(), np.array(self.Speeds)[:,0].max(), np.array(self.Speeds)[:,1].max())

    def AddSeedsFromPadding(self, vx_center, vy_center, padding):
        center_speed = np.array([vx_center, vy_center])

        vx_seeds = [padding[0], padding[0]/2, 0, padding[2]/2, padding[2]]
        vy_seeds = [padding[1], padding[1]/2, 0, padding[3]/2, padding[3]]

        for n_dvx in range(len(vx_seeds)):
            xMinPaddingIndex = max(0, n_dvx-1)
            xMinPadding = vx_seeds[xMinPaddingIndex] - vx_seeds[xMinPaddingIndex+1]
            xMaxPaddingIndex = min(4, n_dvx+1)
            xMaxPadding = vx_seeds[xMaxPaddingIndex] - vx_seeds[xMaxPaddingIndex-1]
            
            for n_dvy in range(len(vy_seeds)):
                yMinPaddingIndex = max(0, n_dvy-1)
                yMinPadding = vy_seeds[yMinPaddingIndex] - vy_seeds[yMinPaddingIndex+1]
                yMaxPaddingIndex = min(4, n_dvy+1)
                yMaxPadding = vy_seeds[yMaxPaddingIndex] - vy_seeds[yMaxPaddingIndex-1]
                
                dv = np.array([vx_seeds[n_dvx], vy_seeds[n_dvy]])

                self.AddSeed(center_speed + dv, (xMinPadding, yMinPadding, xMaxPadding, yMaxPadding))

