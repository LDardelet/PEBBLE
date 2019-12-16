import numpy as np

from Framework import Module, Event

class Memory(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to handle ST-context memory.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Memory'

    def _InitializeModule(self, **kwargs):

        self.STContext = -np.inf*np.ones(self.__Framework__._GetStreamGeometry(self))
        self.LastEvent = Event(-np.inf, [0,0], 0)

        self.Snapshots = []

        return True

    def _OnEventModule(self, event):
        self.LastEvent = Event(original = event)
        position = tuple(self.LastEvent.location.tolist() + [self.LastEvent.polarity])

        self.STContext[position] = self.LastEvent.timestamp

        return event

    def GetSnapshot(self):
        self.Snapshots += [(self.LastEvent.timestamp, np.array(self.STContext))]

    def CreateSnapshot(self):
        return np.array(self.STContext)

    def _Rewind(self, tNew):
        self.STContext[self.STContext >= tNew] = -np.inf
        while self.Snapshots and self.Snapshots[-1][0] >= tNew:
            self.Snapshots.pop(-1)

    def GetPatch(self, x, y, Rx, Ry):
        return self.STContext[x-Rx:x+Rx,y-Ry:y+Ry,:]
