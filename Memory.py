import numpy as np

from Framework import Module, Event

class Memory(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to handle ST-context memory.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Memory'

        self._MonitorDt = 0. # By default, the memory module does NOT take any shapshot, to ensure memory doesn't get filled.
        self._MonitoredVariables = [('STContext', np.array)]

    def _InitializeModule(self, **kwargs):

        self.STContext = -np.inf*np.ones(self.__Framework__._GetStreamGeometry(self))
        self.LastEvent = Event(timestamp = -np.inf, location = np.array([0,0]), polarity = 0)

        return True

    def _OnEventModule(self, event):
        self.LastEvent = event.Copy()
        position = tuple(self.LastEvent.location.tolist() + [self.LastEvent.polarity])

        self.STContext[position] = self.LastEvent.timestamp

        return event

    def CreateSnapshot(self):
        return np.array(self.STContext)

    def _Rewind(self, tNew):
        self.STContext[self.STContext >= tNew] = -np.inf

    def GetPatch(self, x, y, Rx, Ry):
        return self.STContext[max(0,x-Rx):x+Rx,max(0,y-Ry):y+Ry,:]
