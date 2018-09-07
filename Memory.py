import numpy as np
from event import Event

class Memory:
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to handle ST-context memory.
        '''
        self.__ReferencesAsked__ = []
        self.__Name__ = Name
        self.__Framework__ = Framework
        self.__Type__ = 'Memory'
        self.__CreationReferences__ = dict(argsCreationReferences)

    def _Initialize(self):
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
