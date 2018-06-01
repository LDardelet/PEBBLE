import numpy as np
import time

class TimeLimiter:
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to limit stream reading to real time
        Expects nothing.
        '''
        self._ReferencesAsked = []
        self._Name = Name
        self._Framework = Framework
        self._Type = 'Display'
        self._CreationReferences = dict(argsCreationReferences)

    def _Initialize(self):
        self.
        self.Memory = self._Framework.Tools[self._CreationReferences['Memory']]
        self.AllowedEvents = 0
        self.FilteredEvents = 0

    def _OnEvent(self, event):
        current_t = time.time()
