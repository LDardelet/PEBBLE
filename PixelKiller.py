import numpy as np

from Framework import Module

class PixelKiller(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to filter events from spike trains.
        Expects nothing.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)

        self.__ReferencesAsked__ = []
        self.__Type__ = 'Filter'

        self._DeathRate = 0.1 # Given is seconds
        self._Active = True

    def _InitializeModule(self, **kwargs):

        self.PixelsLifeMap = -self._DeathRate * np.log(np.random.rand(self.__Framework__.StreamsGeometries[self.__Framework__.StreamHistory[-1]][0], self.__Framework__.StreamsGeometries[self.__Framework__.StreamHistory[-1]][1]))
        return True

    def _OnEventModule(self, event):
        if not self._Active or event.timestamp < self.PixelsLifeMap[event.location[0], event.location[1]]:
            return event
        else:
            return None
