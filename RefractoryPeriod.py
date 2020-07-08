import numpy as np

from Framework import Module

class Refractory(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to filter events from spike trains.
        Expects nothing.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)

        self.__ReferencesAsked__ = ['Memory']
        self.__Type__ = 'Filter'

        self._Period = 0.03 # Given is seconds
        self._Active = True

    def _InitializeModule(self, **kwargs):

        self._Memory = self.__Framework__.Tools[self.__CreationReferences__['Memory']]
        self.AllowedEvents = 0
        self.FilteredEvents = 0

        return True

    def _OnEventModule(self, event):
        if not self._Active or event.timestamp >= self._Memory.STContext[event.location[0], event.location[1], event.polarity] + self._Period:
            self.AllowedEvents += 1
            return event
        else:
            self.FilteredEvents += 1
            return None
