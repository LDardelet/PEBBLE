from PEBBLE import Module, Event, TrackerEvent, TauEvent

import numpy as np

class Stabilizer(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Module stabilizing an event stream spatially from tracker events, and temporally with tau events
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Computation'

        self.__ReferencesAsked__ = []
        self._MonitorDt = 0. # By default, a module does not stode any date over time.
        self._NeedsLogColumn = False
        self._MonitoredVariables = []

        self._MinTrackersForStabilizing = 8
        self._OutputEventsCameraIndex = 1
        self._NPixelsAverage = 0.2

    def _InitializeModule(self, **kwargs):
        self.ScreenCenter = np.array(self.Geometry[:2])/2
        self.ScreenSize = np.array(self.Geometry[:2])

        self.RotationCenter = np.array(self.ScreenCenter)
        self.ADis = 0.
        self.TSum = np.zeros(2, dtype = float)
        self.CSum = 0.
        self.SSum = 0.
        self.LastDispacementUpdate = -np.inf

        self.ActiveTrackers = set()
        self.StaticTrackerLocations = {}
        self.OnScreenTrackerLocations = {}



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

        if self.Started:
            eventStabilizedLocation = np.array(self.ComputeStaticLocationFrom(event.location)+0.5, dtype = int)
            if (eventStabilizedLocation >= 0).all() and (eventStabilizedLocation < self.ScreenSize).all():
                event.Attach(Event, location = eventStabilizedLocation, polarity = event.polarity, cameraIndex = self._OutputEventsCameraIndex)
        return event

    def AnalyzeTrackerData(self, event):
        ID = event.TrackerID
        if not event.TrackerColor == 'g':
            if ID in self.ActiveTrackers:
                self.ActiveTrackers.remove(ID)
                del self.StaticTrackerLocations[ID]
                del self.OnScreenTrackerLocations[ID]
            return

        TrackerCenteredLocation = event.TrackerLocation
        self.OnScreenTrackerLocations[ID] = np.array(TrackerCenteredLocation)
        if ID not in self.ActiveTrackers:
            self.ActiveTrackers.add(ID)
            if not self.Started:
                self.StaticTrackerLocations[ID] = None
                if len(self.ActiveTrackers) >= self._MinTrackersForStabilizing:
                    self.Start()
                else:
                    return
            else:
                self.StaticTrackerLocations[ID] = self.ComputeStaticLocationFrom(TrackerCenteredLocation)
        if self.Started:
            self.UpdateDisplacement(event.timestamp)

    @property
    def T(self):
        if self.ADis == 0:
            return np.zeros(2, dtype = float)
        return self.TSum / self.ADis
    @property
    def C(self):
        if self.ADis == 0:
            return 1.
        return self.CSum / self.ADis
    @property
    def S(self):
        if self.ADis == 0:
            return 0.
        return self.SSum / self.ADis

    @property
    def R(self): # From StaticToCurrent
        return np.array([[self.C, -self.S], [self.S, self.C]])

    def ComputeStaticLocationFrom(self, OnScreenLocation):
        return self.RotationCenter - self.T + self.R.T.dot(OnScreenLocation - self.RotationCenter)

    def UpdateDisplacement(self, t):
        OnScreenLocations = np.zeros((len(self.ActiveTrackers), 2), dtype = float)
        StaticLocations = np.zeros((len(self.ActiveTrackers), 2), dtype = float)
        for nID, ID in enumerate(self.ActiveTrackers):
            OnScreenLocations[nID,:] = self.OnScreenTrackerLocations[ID]
            StaticLocations[nID,:] = self.StaticTrackerLocations[ID]
        AverageOnScreenLocation = OnScreenLocations.mean(axis = 0)
        AverageStaticLocation = StaticLocations.mean(axis = 0)
        DeltasOnScreen = np.zeros((len(self.ActiveTrackers), 2), dtype = float)
        DeltasOnScreen[:,0] = OnScreenLocations[:,0] - AverageOnScreenLocation[0]
        DeltasOnScreen[:,1] = OnScreenLocations[:,1] - AverageOnScreenLocation[1]
        DeltasStatic = np.zeros((len(self.ActiveTrackers), 2), dtype = float)
        DeltasStatic[:,0] = StaticLocations[:,0] - AverageStaticLocation[0]
        DeltasStatic[:,1] = StaticLocations[:,1] - AverageStaticLocation[1]
        NDeltas = np.maximum(0.001, np.linalg.norm(DeltasOnScreen, axis = 1) * np.linalg.norm(DeltasStatic, axis = 1))

        Decay = np.e**((self.LastDispacementUpdate - t)/(self.Tau * self._NPixelsAverage))
        self.TSum = self.TSum * Decay + (AverageOnScreenLocation - AverageStaticLocation)
        self.CSum = self.CSum * Decay + ((DeltasOnScreen * DeltasStatic).sum(axis = 1) / NDeltas).mean()
        self.SSum = self.SSum * Decay + ((DeltasOnScreen[:,1] * DeltasStatic[:,0] - DeltasOnScreen[:,0] * DeltasStatic[:,1]) / NDeltas).mean()
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
