import numpy as np

from PEBBLE import Module

class ActivityFilter(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to implement an activity filter (also known as Background Activity filter).
        Expects nothing.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)

        self.__ReferencesAsked__ = ['Memory']
        self.__Type__ = 'Filter'

        self._MinNeighbors = 3
        self._Tau = 0.01 # in seconds
        self._Radius = 2

    def _InitializeModule(self):

        self._Memory = self.__Framework__.Tools[self.__CreationReferences__['Memory']]
        self.AllowedEvents = 0
        self.FilteredEvents = 0

        self._Xmax = self._Memory.STContext.shape[0]
        self._Ymax = self._Memory.STContext.shape[1]
        print("ActivityFilter: stream geometries are {0}, {1}".format(self._Xmax, self._Ymax))

        return True

    def UpdateParameters(self, MinNeigh, Tau, Radius):
        self._MinNeighbors = MinNeigh
        self._Tau = Tau # in seconds
        self._Radius = Radius
        print("ActivityFilter: parameters updated.")

    def _OnEventModule(self, event):
        if (event.location[0] > self._Radius) and (event.location[0] < (self._Xmax - self._Radius)) and (event.location[1] > self._Radius) and (event.location[1] < (self._Ymax - self._Radius)):
            # extract neigborhood
            patch = np.copy(self._Memory.STContext[event.location[0] - self._Radius:event.location[0] + self._Radius + 1, event.location[1] - self._Radius:event.location[1] + self._Radius + 1, event.polarity])
            patch -= event.timestamp
            count = np.sum(np.abs(patch) < self._Tau)
        else:
            count = 0

        if count > self._MinNeighbors: # allow event if enough neighboors.
            self.AllowedEvents += 1
        else:
            self.FilteredEvents += 1
            event.Filter()

