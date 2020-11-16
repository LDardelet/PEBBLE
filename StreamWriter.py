from Framework import Module, Event
import atexit

class StremWriter(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Module to write an event stream (as .txt for now). Useful to re-write filtered or cut portions of streams
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Output'

        self.__ReferencesAsked__ = []
        self.__RewindForbidden__ = True
        self._MonitorDt = 0. # By default, a module does not stode any date over time.
        self._NeedsLogColumn = False
        self._MonitoredVariables = []

        self._ResetTsToZero = True
        self._Extension = '_out.txt'
        self._Separator = ' '
        self._yInvert = True
        self.CurrentFile = None

    def _InitializeModule(self, **kwargs):
        InputName = self.StreamName
        if '@' in InputName:
            InputName = InputName.split('@')[1] + '_' + InputName.split('@')[0]
        self.FileName =  InputName + self._Extension
        self.Log('Writing to file {0}'.format(self.FileName))

        self.CurrentFile = open(self.FileName, 'w')
        atexit.register(self._CloseFile)
        if not self._ResetTsToZero:
            self.Offset = 0
        else:
            self.Offset = None

        if self._yInvert:
            self.yFunc = lambda y : self.Geometry[1]-1-y
        else:
            self.yFunc = lambda y:y
        return True

    def _OnEventModule(self, event):
        if self.Offset is None:
            self.Offset = event.timestamp
        self.CurrentFile.write(self._Separator.join(["{0:.6f}".format(event.timestamp - self.Offset), str(event.location[0]), str(self.yFunc(event.location[1])), str(event.polarity)]) + '\n')
        return event

    def _CloseFile(self):
        if not self.CurrentFile is None:
            self.CurrentFile.close()
            self.Log("Closed file {0}".format(self.FileName))
            self.CurrentFile = None
