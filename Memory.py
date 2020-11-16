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
        self._NeedsLogColumn = False
        self._MonitoredVariables = []

    def _InitializeModule(self, **kwargs):

        self.STContext = -np.inf*np.ones(self.Geometry)
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

    def GetSquarePatch(self, xy, R):
        return self.STContext[max(0,xy[0]-R):xy[0]+R+1,max(0,xy[1]-R):xy[1]+R+1,:]

    def GetTs(self, Tau, xy = None, R = None):
        if not xy is None:
            x, y = xy
            Map = self.GetSquarePatch(xy, R)
        else:
            Map = self.STContext
        return np.e**((Map.max(axis = 2) - self.LastEvent.timestamp)/Tau)
