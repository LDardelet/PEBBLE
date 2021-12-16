import numpy as np

from ModuleBase import ModuleBase
from Events import CameraEvent

class Memory(ModuleBase):
    def _OnCreation(self):
        '''
        Class to handle ST-context memory.
        '''
        self._TauMethod = 0 # 0 is for decay over events (preferred), 1 for decay over time
        self._DefaultTau = 0.02
        self._ExpectedDensity = 0.05
        self._EnableTauRequest = True

        self._MonitorDt = 0.01
        self._MonitoredVariables = [('STContext', np.array),
                                    ('tOld', float)]

    def _OnInitialization(self):

        self.STContext = -np.inf*np.ones(tuple(self.Geometry) + (2,))
        self.LastEvent = None
        self.tOld = 0
        self.AExp = np.prod(self.Geometry[:2]) * self._ExpectedDensity
        if self._TauMethod == 0:
            self.Decay = np.e**(-1/self.AExp)
            self.Tau = 0
            self.TauUpdateMethod = self.EventDecay
            self.LastT = 0
        else:
            self.TauUpdateMethod = self.TimeDecay
            self.A = 0.
            self.LastT = -np.inf

        return True

    def _OnEventModule(self, event):
        if not event.Has(CameraEvent):
            return
        self.LastEvent = event.Copy()
        position = tuple(self.LastEvent.location.tolist() + [self.LastEvent.polarity])
        self.STContext[position] = self.LastEvent.timestamp

        self.TauUpdateMethod(event.timestamp - self.LastT)
        self.LastT = event.timestamp

        self.tOld = max(self.tOld, self.LastT - self._Tau)
        return

    def EventDecay(self, dt):
        self.Tau = self.Decay * self.Tau + dt
    def TimeDecay(self, dt):
        self.A = self.A * np.e**(-dt / self._DefaultTau) + 1.

    @property
    def _Tau(self):
        if self._TauMethod == 0:
            return self.Tau
        else:
            if self.A <= 10:
                return self._DefaultTau
            return np.log(self.A/(self.A-1)) / np.log(self.AExp/(self.AExp-1)) * self._DefaultTau

    def _OnTauRequest(self, EventConcerned = None):
        if not self._EnableTauRequest:
            return 0
        else:
            return self._Tau

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
