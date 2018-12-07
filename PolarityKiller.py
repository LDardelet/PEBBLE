import numpy as np

from Framework import Module

class PolarityKiller(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class that removes polarities (all are set to 0).
        Expects nothing.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)

        self.__ReferencesAsked__ = []
        self.__Type__ = 'Filter'

    def _Initialize(self, **kwargs):
        Module._Initialize(self, **kwargs)
        return True

    def _OnEvent(self, event):
        event.polarity = 0
        return event
