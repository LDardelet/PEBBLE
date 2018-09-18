import numpy as np
from event import Event

from Framework import Module

class Memory(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to handle ST-context memory.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Memory'

    def _Initialize(self, **kwargs):
        Module._Initialize(self, **kwargs)

        self.STContext = -np.inf*np.ones(self.__Framework__.StreamsGeometries[self.__Framework__.StreamHistory[-1]])
        self.LastEvent = Event(-np.inf, [0,0], 0)

        self.Snapshots = []

        return True

    def _OnEvent(self, event):
        self.LastEvent = Event(original = event)
        position = tuple(self.LastEvent.location.tolist() + [self.LastEvent.polarity])

        self.STContext[position] = self.LastEvent.timestamp

        return event

    def GetSnapshot(self):
        self.Snapshots += [(self.LastEvent.timestamp, np.array(self.STContext))]

    def CreateSnapshot(self):
        return np.array(self.STContext)
