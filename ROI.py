import numpy as np

from Framework import Module

class ROI(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to implement a simple spatial ROI.
        Expects nothing.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)

        self.__ReferencesAsked__ = []
        self.__Type__ = 'Filter'

        self._Xlim = [200,400]
        self._Ylim = [150,250]

    def _InitializeModule(self, **kwargs):

        # self._Memory = self.__Framework__.Tools[self.__CreationReferences__['Memory']]
        self.AllowedEvents = 0
        self.FilteredEvents = 0

        return True

    def _OnEventModule(self, event):
        if (event.location[0] > self._Xlim[0]) and (event.location[0] < self._Xlim[1]) and (event.location[1] > self._Ylim[0]) and (event.location[1] < self._Ylim[1]):
            #print(event.location)
            self.AllowedEvents += 1
            return event
        else:
            self.FilteredEvents += 1
            return None
