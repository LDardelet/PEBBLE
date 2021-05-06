from PEBBLE import Module

class ModuleTemplate(Module):
    def __init__(self, Name, Framework, ModulesLinked):
        '''
        Module template to be filled for specific purpose
        '''
        Module.__init__(self, Name, Framework, ModulesLinked) # Do not modify this line

        self.__ModulesLinksRequested__ = []
        self._MonitorDt = 0. # By default, a module does not stode any date over time.
        self._NeedsLogColumn = False
        self._MonitoredVariables = []

    def _InitializeModule(self):
        return True

    def _OnEventModule(self, event):
        return

