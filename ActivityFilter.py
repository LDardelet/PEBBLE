import numpy as np

from Framework import Module

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

    def _Initialize(self, **kwargs):
        Module._Initialize(self, **kwargs)

        self._Memory = self.__Framework__.Tools[self.__CreationReferences__['Memory']]
        self.AllowedEvents = 0
        self.FilteredEvents = 0

        return True

    def _OnEvent(self, event):
        if (event.location[0] > self._Radius) and (event.location[0] < (640 - self._Radius)) and (event.location[1] > self._Radius) and (event.location[1] < (480 - self._Radius)):
            # extract neigborhood
            patch = np.copy(self._Memory.STContext[event.location[0] - self._Radius:event.location[0] + self._Radius + 1, event.location[1] - self._Radius:event.location[1] + self._Radius + 1, event.polarity])
            patch -= event.timestamp
            count = np.sum(np.abs(patch) < self._Tau)
        else:
            count = 0
            
        if count > self._MinNeighbors:
            self.AllowedEvents += 1
            return event
        else:
            self.FilteredEvents += 1
            return None
