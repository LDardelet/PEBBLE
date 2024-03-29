from ModuleBase import ModuleBase

import numpy as np

class StreamWriter(ModuleBase):
    def _OnCreation(self):
        '''
        Module to write an event stream (as .txt for now). Useful to re-write filtered or cut portions of streams
        '''
        self._EventType = None
        self._ResetTsToZero = True
        self._FileName = ''
        self._Separator = ' '
        self._yInvert = True
        self.CurrentFile = None

    def _OnInitialization(self):
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

    def _OnClosing(self):
        if not self.CurrentFile is None:
            self.CurrentFile.close()
            self.Log("Closed file {0}".format(self._FileName))
            self.CurrentFile = None
