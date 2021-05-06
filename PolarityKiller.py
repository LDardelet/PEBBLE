import numpy as np

from PEBBLE import Module

class PolarityKiller(Module):
    def __init__(self, Name, Framework, ModulesLinked):
        '''
        Class that removes polarities (all are set to 0).
        Expects nothing.
        '''
        Module.__init__(self, Name, Framework, ModulesLinked)

    def _InitializeModule(self):
        return True

    def _OnEventModule(self, event):
        event.polarity = 0
        return
