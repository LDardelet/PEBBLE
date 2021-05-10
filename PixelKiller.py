import numpy as np

from PEBBLE import ModuleBase

class PixelKiller(ModuleBase):
    def _OnCreation(self):
        '''
        Randomly kills pixels from generating events.
        '''
        self._DeathRate = 0.1 # Given is seconds
        self._Active = True

    def _OnInitialization(self):
        Geometry = self.Geometry
        self.PixelsLifeMap = -self._DeathRate * np.log(np.random.rand(Geometry[0], Geometry[1]))
        return True

    def _OnEventModule(self, event):
        if self._Active and event.timestamp > self.PixelsLifeMap[event.location[0], event.location[1]]:
            event.Filter()
