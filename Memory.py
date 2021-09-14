import numpy as np

from ModuleBase import ModuleBase
from Events import CameraEvent

class Memory(ModuleBase):
    def _OnCreation(self):
        '''
        Class to handle ST-context memory.
        '''
        self._DefaultTau = 0.01
        self._ExpectedDensity = 0.01
        self._EnableTauRequest = True

        self._MonitorDt = 0.01
        self._MonitoredVariables = [('STContext', np.array)]

    def _OnInitialization(self):

        self.STContext = -np.inf*np.ones(tuple(self.Geometry) + (2,))
        self.LastEvent = None

        self.AExp = np.prod(self.Geometry[:2]) * self._ExpectedDensity
        self.A = 0.
        self.LastT = -np.inf

        return True

    def _OnEventModule(self, event):
        if not event.Has(CameraEvent):
            return
        self.LastEvent = event.Copy()
        position = tuple(self.LastEvent.location.tolist() + [self.LastEvent.polarity])

        self.STContext[position] = self.LastEvent.timestamp
        self.A = self.A * np.e**((self.LastT - event.timestamp) / self._DefaultTau) + 1.
        self.LastT = event.timestamp

        return

    def _OnTauRequest(self, EventConcerned = None):
        if not self._EnableTauRequest:
            return 0
        if self.A <= 10:
            return self._DefaultTau
        return np.log(self.A/(self.A-1)) / np.log(self.AExp/(self.AExp-1)) * self._DefaultTau

    def CreateSnapshot(self):
        return np.array(self.STContext)

    def GetSquarePatch(self, xy, R):
        return self.STContext[max(0,xy[0]-R):xy[0]+R+1,max(0,xy[1]-R):xy[1]+R+1,:]

    def GetTs(self, Tau, xy = None, R = None):
        if not xy is None:
            Map = self.GetSquarePatch(xy, R)
        else:
            Map = self.STContext
        return np.e**((Map.max(axis = 2) - self.LastEvent.timestamp)/Tau)
