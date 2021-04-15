from PEBBLE import Module, CameraEvent, DisparityEvent, TrackerEvent
import numpy as np
import atexit

class StreamWriter(Module):
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

        self._EventType = None
        self._ResetTsToZero = True
        self._FileName = ''
        self._Separator = ' '
        self._yInvert = True
        self.CurrentFile = None

    def _InitializeModule(self, **kwargs):
        if self._EventType is None:
            self.LogWarning("No event type specified, no output data")
            self.Write = False
            return True
        if not self._FileName:
            self.LogWarning("No output file specified, no output data")
            self.Write = False
            return True
        self.CurrentFile = open(self._FileName, 'w')
        self.Write = True
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
        if not self.Write:
            return 
        if self.Offset is None:
            if self._ResetTsToZero:
                self.Offset = event.timestamp
                self.CurrentFile.write("# ts Offset = {0:.6f}\n".format(self.Offset))
                self.CurrentFile.write("# " + self._Separator.join(['timestamp'] + list(self._EventType._Fields)) + "\n")
            else:
                self.Offset = 0
            #self.CurrentFile.write("# " + self.StreamName + "\n")
        if event.Has(self._EventType):
            self.WriteEvent(event)
        return 

    def WriteEvent(self, event):
        EventList = event.AsList((self._EventType._Key,))
        Values = ["{0:.6f}".format(EventList.pop(0)-self.Offset)]
        for Extension in EventList:
            for Data in Extension[1:]:
                if type(Data) in (tuple, list, np.ndarray):
                    Values += [str(v) for v in Data]
                else:
                    Values += [str(Data)]
        self.CurrentFile.write(self._Separator.join(Values) + '\n')

    def _CloseFile(self):
        if not self.CurrentFile is None:
            self.CurrentFile.close()
            self.Log("Closed file {0}".format(self._FileName))
            self.CurrentFile = None
