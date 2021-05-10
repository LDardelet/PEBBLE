import numpy as np

from PEBBLE import ModuleBase

class Refractory(ModuleBase):
    def _OnCreation(self):
        '''
        Class to filter events from spike trains.
        Expects nothing.
        '''
        self.__ModulesLinksRequested__ = ['Memory']

        self._Period = 0.03 # Given is seconds
        self._Active = True

    def _OnInitialization(self):
        self.AllowedEvents = 0
        self.FilteredEvents = 0

        return True

    def _OnEventModule(self, event):
        if not self._Active or event.timestamp >= self.Memory.STContext[event.location[0], event.location[1], event.polarity] + self._Period:
            self.AllowedEvents += 1
        else:
            event.Filter()
