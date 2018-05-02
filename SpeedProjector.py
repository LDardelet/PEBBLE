import numpy as np
import geometry
import sys
import matplotlib.pyplot as plt

class Projector:
    def __init__(self, argsCreationDict):
        '''
        Tool to create a 2D projection of the STContext onto a plane t = cst along different speeds.
        Should allow to recove the actual speed and generator of the movement
        Expects:
        '''
        self.R_Projection = 0.5
        self.ExpansionFactor = 4.

        self._Type = 'Computation'

    def _Initialize(self, argsInitializationDict):
        self.BinDt = 0.002
        self.Precision_aimed = 0.5
        self.Initial_dv_MAX = 160.

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
                    if (event.timestamp - self.SpeedStartTime[speed_id])*max(2., self.SpeedNorms[speed_id]) >= self.ExpansionFactor * np.sqrt(2):
                        self.AssumedInitialized[speed_id] = event.timestamp
                    else:
                        if not (event.timestamp - self.SpeedStartTime[speed_id]) > 0.1:
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

                if event.timestamp - self.TsArgMin > 0.3 and self.NormalizedIntersectingAreasHistory[-1][self.ArgMinArea] > 1.:
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

        if self.Areas[n_speed]/self.NEventsBySpeed[n_speed] < self.MinArea: # Possible check needed for this line, but seems to work better without the AssumedInitialized condition
        #if self.AssumedInitialized[n_speed] != 0 and self.Areas[n_speed]/self.NEventsBySpeed[n_speed] < self.MinArea:
            self.ArgMinArea = n_speed
            self.MinArea = self.Areas[n_speed]/self.NEventsBySpeed[n_speed]

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
