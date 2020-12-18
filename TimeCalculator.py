from PEBBLE import Module, Event, TauEvent
import numpy as np

class ModuleTemplate(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Module template to be filled foe specific purpose
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = ''

        self.__ReferencesAsked__ = ['Memory']
        self._MonitorDt = 0.00001 # By default, a module does not stode any date over time.
        self._NeedsLogColumn = False
        self._MonitoredVariables = [('Tau', float)]

        self._DefaultTau = 0.01
        self._StartActivity = 100
        self._TauAverageRatio = 10

    def _InitializeModule(self, **kwargs):
        self.SumTau = self._DefaultTau * self._StartActivity
        self.SumActivity = self._StartActivity
        self.Tau = self._DefaultTau
        self.LastDecay = -np.inf
        self.LinkedMemory = self.__Framework__.Tools[self.__CreationReferences__['Memory']]
        self.GridConstant = 1/0.709
        return True

    def Decay(self, t):
        if not self.Tau:
            return
        Decay = np.e**((self.LastDecay - t)/(self._TauAverageRatio * self.Tau))
        self.SumActivity *= Decay
        self.SumTau *= Decay
        self.LastDecay = t

    def _OnEventModule(self, event):
        NeighbourTs = self.LinkedMemory.GetSquarePatch(event.location, 1).max(axis=2).flatten()
        NeighbourTs.sort()
        if NeighbourTs[-4] == -np.inf:
            return event
        self.Decay(event.timestamp)
        self.SumTau += event.timestamp - NeighbourTs[-4:-1].mean()
        self.SumActivity += 1
        self.Tau = self.GridConstant * self.SumTau / self.SumActivity
        event.Attach(TauEvent, tau = self.Tau)
        return event

