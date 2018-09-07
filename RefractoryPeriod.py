import numpy as np

class Refractory:
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to filter events from spike trains.
        Expects nothing.
        '''
        self.__ReferencesAsked__ = ['Memory']
        self.__Name__ = Name
        self.__Framework__ = Framework
        self.__Type__ = 'Filter'
        self.__CreationReferences__ = dict(argsCreationReferences)

        self._Period = 0.06 # Given is seconds
        self._Active = True

    def _Initialize(self):
        self._Memory = self.__Framework__.Tools[self.__CreationReferences__['Memory']]
        self.AllowedEvents = 0
        self.FilteredEvents = 0

        return True

    def _OnEvent(self, event):
        if not self._Active or event >= self._Memory.STContext[event.location[0], event.location[1], event.polarity] + self._Period:
            self.AllowedEvents += 1
            return event
        else:
            self.FilteredEvents += 1
            return None
