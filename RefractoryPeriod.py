import numpy as np

from PEBBLE import Module

class Refractory(Module):
    def __init__(self, Name, Framework, ModulesLinked):
        '''
        Class to filter events from spike trains.
        Expects nothing.
        '''
        Module.__init__(self, Name, Framework, ModulesLinked)

        self.__ModulesLinksRequested__ = ['Memory']

        self._NeedsLogColumn = False

        self._Period = 0.03 # Given is seconds
        self._Active = True

    def _InitializeModule(self):
        self.Memory = self.__Framework__.Tools[self.__ModulesLinked__['Memory']]
        self.AllowedEvents = 0
        self.FilteredEvents = 0

        return True

    def _OnEventModule(self, event):
        if not self._Active or event.timestamp >= self.Memory.STContext[event.location[0], event.location[1], event.polarity] + self._Period:
            self.AllowedEvents += 1
        else:
            event.Filter()
