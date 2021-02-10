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

        self._OutputEventsCameraIndex = 1

    def _InitializeModule(self, **kwargs):
        self.StaticTrackerLocations = {}
        self.OnScreenTrackerLocations = {}

        self.ScreenCenter = np.array(self.Geometry[:2])/2
        self.ScreenSize = np.array(self.Geometry[:2])

        self.T = np.zeros(2, dtype = float)
        self.C = 1.
        self.S = 0.

        self.ActiveTrackers = set()
        self.Sx = 0.
        self.Sy = 0.
        self.SX = 0.
        self.SY = 0.

        self.Sxx = 0.
        self.Syy = 0.
        self.SxX = 0.
        self.SyY = 0.
        self.SxY = 0.
        self.SyX = 0.

        self.LastTauUpdate = -np.inf
        self.ATau = 0.
        self.STau = 0.
        
        return True

    def _OnEventModule(self, event):
        if event.Has(TrackerEvent):
            self.AnalyzeTrackerData(event)
        if event.Has(TauEvent):
            self.UpdateTau(event.timestamp, event.tau)

        eventStabilizedLocation = np.array(self.R.T.dot(event.location - self.ScreenCenter - self.T) + self.ScreenCenter, dtype = int)
        if (eventStabilizedLocation >= 0).all() and (eventStabilizedLocation < self.ScreenSize).all():
            event.Attach(Event, location = eventStabilizedLocation, polarity = event.polarity, cameraIndex = self._OutputEventsCameraIndex)
        return event

    def AnalyzeTrackerData(self, event):
        ID = event.TrackerID
        if not event.TrackerColor == 'g':
            if ID in self.ActiveTrackers:
                x, y = self.StaticTrackerLocations[ID]
                X, Y = self.OnScreenTrackerLocations[ID]
                self.UpdateSums(x, y, X, Y, Sign = -1)

                self.ActiveTrackers.remove(ID)
                del self.StaticTrackerLocations[ID]
                del self.OnScreenTrackerLocations[ID]
            return

        TrackerCenteredLocation = event.TrackerLocation - self.ScreenCenter
        if ID not in self.ActiveTrackers:
            self.ActiveTrackers.add(ID)
            self.StaticTrackerLocations[ID] = self.ComputeStaticLocationFrom(TrackerCenteredLocation)
            self.OnScreenTrackerLocations[ID] = np.array(TrackerCenteredLocation)
        x, y = self.StaticTrackerLocations[ID]
        X, Y = self.OnScreenTrackerLocations[ID]
        if ID in self.ActiveTrackers:
            self.UpdateSums(x, y, X, Y, Sign = -1)
            self.OnScreenTrackerLocations[ID] = np.array(TrackerCenteredLocation)
            X, Y = self.OnScreenTrackerLocations[ID]
        self.UpdateSums(x, y, X, Y)
        self.UpdateDisplacement()

    def UpdateSums(self, x, y, X, Y, Sign = +1):
        self.Sx += Sign*x
        self.Sy += Sign*y
        self.SX += Sign*X
        self.SY += Sign*Y
        self.Sxx += Sign*(x**2)
        self.Syy += Sign*(y**2)
        self.SxX += Sign*(x*X)
        self.SyY += Sign*(y*Y)
        self.SxY += Sign*(x*Y)
        self.SyX += Sign*(y*X)

    @property
    def R(self): # From StaticToCurrent
        return np.array([[self.C, -self.S], [self.S, self.C]])

    def ComputeStaticLocationFrom(self, OnScreenLocation):
        return self.R.T.dot(OnScreenLocation - self.T)

    def UpdateDisplacement(self):
        if len(self.ActiveTrackers) < 5:
            return
        Denom = (self.Sxx + self.Syy - self.Sx**2 - self.Sy**2)
        self.C = (self.SxX + self.SyY - self.Sx*self.SX - self.Sy*self.SY) / Denom 
        self.S = (self.SxY - self.SyX + self.SX*self.Sy - self.Sx*self.SY) / Denom
        self.T = np.array([self.SX - self.C * self.Sx + self.S * self.Sy, 
                           self.SY - self.S * self.Sx - self.C * self.Sy])

    def UpdateTau(self, t, tau):
        Decay = np.e**((self.LastTauUpdate - t)/self.Tau)
        self.ATau = self.ATau * Decay + 1
        self.STau = self.STau * Decay + tau

    @property
    def Tau(self):
        if self.ATau == 0:
            return 0.05
        else:
            return self.STau / self.ATau
