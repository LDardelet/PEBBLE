from ModuleBase import ModuleBase
from Events import CameraEvent, TrackerEvent, TauEvent

import numpy as np

class Stabilizer(ModuleBase):
    def _OnCreation(self):
        '''
        Module stabilizing an event stream spatially from tracker events, and temporally with tau events
        '''
        self._MonitorDt = 0. # By default, a module does not stode any date over time.
        self._MonitoredVariables = [('Tau', float),
                                    ('T', np.array),
                                    ('TComp', np.array),
                                    ('Theta', float),
                                    ('ThetaComp', float),
                                    ('ThetaFromTrackers', float)]

        self._MinTrackersForStabilizing = 8
        self._MaxNormRatioOutlier = 1.2
        self._OutputEventsCameraIndex = 1
        self._NPixelsAverage = 10
        self._RatioThetaTrackers = 1.

    def _OnInitialization(self):
        self.ScreenCenter = np.array(self.Geometry)/2
        self.ScreenSize = np.array(self.Geometry)

        self.RotationCenter = np.array(self.ScreenCenter)
        self.ADis = 0.
        self.TComp = np.zeros(2, dtype = float)
        self.TSum = np.zeros(2, dtype = float)
        self.ThetaComp = 0.
        self.ThetaSum = 0.
        self.ThetaFromTrackers = 0.
        self.LastDispacementUpdate = -np.inf

        self.ActiveTrackers = set()
        self.StaticTrackerLocations = {}
        self.OnScreenTrackerLocations = {}
        self.StaticTrackerAngles = {}
        self.OnScreenTrackerAngles = {}

        self.StaticCenter = np.array(self.ScreenCenter)
        self.OnScreenCenter = np.array(self.ScreenCenter)
        self.NOutliers = 0

        self.LastTauUpdate = -np.inf
        self.ATau = 0.
        self.STau = 0.

        self.Started = False
        
        return True

    def _OnEventModule(self, event):
        if event.Has(TrackerEvent):
            self.AnalyzeTrackerData(event)
        if event.Has(TauEvent):
            self.UpdateTau(event.timestamp, event.tau)

        if not event.Has(CameraEvent):
            return

        if self.Started:
            eventStabilizedLocation = np.array(self.ComputeStaticLocationFrom(event.location)+0.5, dtype = int)
            if (eventStabilizedLocation >= 0).all() and (eventStabilizedLocation < self.ScreenSize).all():
                event.Join(CameraEvent, location = eventStabilizedLocation, polarity = event.polarity, SubStreamIndex = self._OutputEventsCameraIndex)
        return

    def AnalyzeTrackerData(self, event):
        for _ in event.Get(TrackerEvent):
            ID = event.TrackerID
            if event.TrackerColor == 'k':
                if ID in self.ActiveTrackers:
                    self.ActiveTrackers.remove(ID)
                    del self.StaticTrackerLocations[ID]
                    del self.OnScreenTrackerLocations[ID]
                    del self.StaticTrackerAngles[ID]
                    del self.OnScreenTrackerAngles[ID]
                continue
            if (event.TrackerColor != 'g' or event.TrackerMarker != 'o'):
                if ID in self.ActiveTrackers:
                    self.ActiveTrackers.remove(ID)
                continue
    
            TrackerCenteredLocation = event.TrackerLocation
            self.OnScreenTrackerLocations[ID] = np.array(TrackerCenteredLocation)
            self.OnScreenTrackerAngles[ID] = event.TrackerAngle
            if ID not in self.ActiveTrackers:
                self.ActiveTrackers.add(ID)
                if not self.Started:
                    self.StaticTrackerLocations[ID] = None
                    self.StaticTrackerAngles[ID] = None
                    if len(self.ActiveTrackers) >= self._MinTrackersForStabilizing:
                        self.Start()
                    else:
                        continue
                else:
                    self.StaticTrackerLocations[ID] = self.ComputeStaticLocationFrom(TrackerCenteredLocation)
                    self.StaticTrackerAngles[ID] = event.TrackerAngle - self.Theta
        if self.Started:
            self.UpdateDisplacement(event.timestamp)

    @property
    def T(self):
        if self.ADis == 0:
            return np.zeros(2, dtype = float)
        return self.TComp - self.TSum / self.ADis
    @property
    def Theta(self):
        if self.ADis == 0:
            return 0.
        #return -(self.ThetaComp - self.ThetaSum / self.ADis)
        return 0.

    @property
    def C(self):
        return np.cos(self.Theta)
    @property
    def S(self):
        return np.sin(self.Theta)

    @property
    def R(self): # From StaticToCurrent
        return np.array([[self.C, -self.S], [self.S, self.C]])

    def ComputeStaticLocationFrom(self, OnScreenLocation):
        return self.StaticCenter + self.R.T.dot(OnScreenLocation - self.OnScreenCenter)

    def UpdateDisplacement(self, t, Outliers = []):
        self.NOutliers = len(Outliers)
        NTrackersConsidered = len(self.ActiveTrackers) - len(Outliers)
        OnScreenLocations = np.zeros((NTrackersConsidered, 2), dtype = float)
        StaticLocations = np.zeros((NTrackersConsidered, 2), dtype = float)
        IDs = np.zeros(NTrackersConsidered, dtype = int)
        nID = 0
        for ID in self.ActiveTrackers:
            if ID in Outliers:
                continue
            OnScreenLocations[nID,:] = self.OnScreenTrackerLocations[ID]
            StaticLocations[nID,:] = self.StaticTrackerLocations[ID]
            IDs[nID] = ID
            nID += 1
        self.OnScreenCenter = OnScreenLocations.mean(axis = 0)
        self.StaticCenter = StaticLocations.mean(axis = 0)
        DeltasOnScreen = np.zeros((NTrackersConsidered, 2), dtype = float)
        DeltasOnScreen[:,0] = OnScreenLocations[:,0] - self.OnScreenCenter[0]
        DeltasOnScreen[:,1] = OnScreenLocations[:,1] - self.OnScreenCenter[1]
        DeltasStatic = np.zeros((NTrackersConsidered, 2), dtype = float)
        DeltasStatic[:,0] = StaticLocations[:,0] - self.StaticCenter[0]
        DeltasStatic[:,1] = StaticLocations[:,1] - self.StaticCenter[1]
        NStatic = np.linalg.norm(DeltasStatic, axis = 1)
        NOnScreen = np.linalg.norm(DeltasOnScreen, axis = 1)
        NDeltas = np.maximum(0.001, NStatic * NOnScreen)
        Ratios = NOnScreen / np.maximum(0.001, NStatic)
        RatiosRatios = Ratios / Ratios.mean()
        NewOutliers = (IDs[np.where(np.logical_or((RatiosRatios > self._MaxNormRatioOutlier), (RatiosRatios < 1./self._MaxNormRatioOutlier)))]).tolist()
        if NewOutliers:
            Outliers = Outliers + NewOutliers
            if len(self.ActiveTrackers) - len(Outliers) > self._MinTrackersForStabilizing:
                return self.UpdateDisplacement(t, Outliers)
        

        self.TComp = (self.OnScreenCenter - self.StaticCenter)
        CComp = ((DeltasOnScreen * DeltasStatic).sum(axis = 1) / NDeltas).mean()
        SComp = ((DeltasOnScreen[:,1] * DeltasStatic[:,0] - DeltasOnScreen[:,0] * DeltasStatic[:,1]) / NDeltas).mean()
        ThetaComp = np.arccos(CComp) * np.sign(SComp)
        nTurns = int(np.rint((self.ThetaComp - ThetaComp) / (2*np.pi)))
        self.ThetaFromTrackers = np.mean([self.OnScreenTrackerAngles[ID] - self.StaticTrackerAngles[ID] for ID in self.ActiveTrackers])
        self.ThetaComp = (1-self._RatioThetaTrackers) * (ThetaComp + 2 * np.pi * nTurns) + self._RatioThetaTrackers * self.ThetaFromTrackers

        if self._NPixelsAverage == 0:
            Decay = 0
        else:
            Decay = np.e**((self.LastDispacementUpdate - t)/(self.Tau * self._NPixelsAverage))
        self.TSum = self.TSum * Decay + self.TComp
        self.ThetaSum = self.ThetaSum * Decay + self.ThetaComp
        self.ADis = self.ADis * Decay + 1

        self.LastDispacementUpdate = t


#    def UpdateDisplacement(self):
#        Denom = (self.Sxx + self.Syy - self.Sx**2 - self.Sy**2)
#        if Denom == 0:
#            self.C, self.S, self.T = 1., 0., np.array([0., 0.])
#            return
#        self.C = (self.SxX + self.SyY - self.Sx*self.SX - self.Sy*self.SY) / Denom 
#        self.S = -(self.SxY - self.SyX + self.SX*self.Sy - self.Sx*self.SY) / Denom
#        N = np.sqrt(self.C**2 + self.S**2)
#        self.C /= N
#        self.S /= N
#        self.T = -np.array([self.SX - self.C * self.Sx + self.S * self.Sy, 
#                           self.SY - self.S * self.Sx - self.C * self.Sy])

    def Start(self):
        for ID in self.ActiveTrackers:
            self.StaticTrackerLocations[ID] = np.array(self.OnScreenTrackerLocations[ID])
            self.StaticTrackerAngles[ID] = self.OnScreenTrackerAngles[ID]
        self.Started = True
        self.LogSuccess("Started")

    def UpdateTau(self, t, tau):
        Decay = np.e**((self.LastTauUpdate - t)/self.Tau)
        self.ATau = self.ATau * Decay + 1
        self.STau = self.STau * Decay + tau
        self.LastTauUpdate = t

    @property
    def Tau(self):
        if self.ATau == 0:
            return 0.05
        else:
            return self.STau / self.ATau
