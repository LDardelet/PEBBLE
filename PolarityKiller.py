import numpy as np

from PEBBLE import Module

class PolarityKiller(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class that removes polarities (all are set to 0).
        Expects nothing.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)

        self.__ReferencesAsked__ = []
        self.__Type__ = 'Filter'

    def _InitializeModule(self, **kwargs):
        return True

    def _OnEventModule(self, event):
        event.polarity = 0
        return
