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

        self._xMinOffset = 0
        self._xMaxOffset = None # Leave None for symetrical value
        self._yMinOffset = 0
        self._yMaxOffset = None # Leave None for symetrical value

    def _InitializeModule(self, **kwargs):
        if self._xMaxOffset is None:
            self._xMaxOffset = self._xMinOffset
        if self._yMaxOffset is None:
            self._yMaxOffset = self._yMinOffset
        # self._Memory = self.__Framework__.Tools[self.__CreationReferences__['Memory']]
        self.AllowedEvents = 0
        self.FilteredEvents = 0
        self.MinX = np.array([self._xMinOffset, self._yMinOffset])
        self.MaxX = self.Geometry[:2] - np.array([self._xMaxOffset, self._yMaxOffset])

        return True
    
    @property
    def OutputGeometry(self):
        return np.array([self.Geometry[0] - (self._xMinOffset + self._xMaxOffset), self.Geometry[1] - (self._yMinOffset + self._yMaxOffset), 2])

    def _OnEventModule(self, event):
        if (event.location < self.MinX).any() or (event.location >= self.MaxX).any():
            self.FilteredEvents += 1
            return None
        else:
            self.AllowedEvents += 1
            event.location[:] -= self.MinX
            return event
