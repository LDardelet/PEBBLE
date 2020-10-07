from Framework import Module, TrackerEvent

class Mapper(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Module that creates a stable 2D map from trackers
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Computation'

        self.__ReferencesAsked__ = []
        self._MonitorDt = 0. # By default, a module does not stode any date over time.
        self._MonitoredVariables = []

    def _InitializeModule(self, **kwargs):

        return True

    def _OnEventModule(self, event):
        return event

