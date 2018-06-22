import numpy as np
from event import Event

class Memory:
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to handle ST-context memory.
        '''
        self._ReferencesAsked = []
        self._Name = Name
        self._Framework = Framework
        self._Type = 'Memory'
        self._CreationReferences = dict(argsCreationReferences)

    def _Initialize(self):
        self.STContext = -np.inf*np.ones(self._Framework.StreamsGeometries[self._Framework.StreamHistory[-1]])
        self.PreviousTsAtLocation = - np.inf
        self.LastEvent = Event(-np.inf, [0,0], 0)

        self.Snapshots = []

    def _OnEvent(self, event):
        self.LastEvent = Event(original = event)
        position = tuple(self.LastEvent.location.tolist() + [self.LastEvent.polarity])

        self.PreviousTsAtLocation = self.STContext[position[0], position[1], :].max()
        self.STContext[position] = self.LastEvent.timestamp

        return event

    def GetSnapshot(self):
        self.Snapshots += [(self.LastEvent.timestamp, np.array(self.STContext))]
