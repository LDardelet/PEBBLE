import numpy as np

from PEBBLE import Module

class PixelKiller(Module):
    def __init__(self, Name, Framework, ModulesLinked):
        '''
        Randomly kills pixels from generating events.
        '''
        Module.__init__(self, Name, Framework, ModulesLinked)

        self._DeathRate = 0.1 # Given is seconds
        self._Active = True

    def _InitializeModule(self):

        Geometry = self.__Framework__._GetStreamGeometry(self)
        self.PixelsLifeMap = -self._DeathRate * np.log(np.random.rand(Geometry[0], Geometry[1]))
        return True

    def _OnEventModule(self, event):
        if self._Active and event.timestamp > self.PixelsLifeMap[event.location[0], event.location[1]]:
            event.Filter()
