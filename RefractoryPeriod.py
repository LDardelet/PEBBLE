import numpy as np

class Refractory:
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to filter events from spike trains.
        Expects nothing.
        '''
        self._ReferencesAsked = ['Memory']
        self._Name = Name
        self._Framework = Framework
        self._Type = 'Filter'
        self._CreationReferences = dict(argsCreationReferences)

        self.Period = 0.06 # Given is seconds

    def _Initialize(self):
        self.Memory = self._Framework.Tools[self._CreationReferences['Memory']]
        self.AllowedEvents = 0
        self.FilteredEvents = 0

    def _OnEvent(self, event):
        if event.timestamp - self.Memory.STContext[event.location[0], event.location[1], event.polarity] >= self.Period:
            self.AllowedEvents += 1
            return event
        else:
            self.FilteredEvents += 1
            return None
