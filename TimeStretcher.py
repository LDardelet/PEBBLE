import numpy as np

from Framework import Module, Event

class Stretcher(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to modify timestamps, mofifying speed norms
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Filter'

        self.__Started__ = False

        self._FuncType__ = 'sinus'
        self._FuncOptionsDict__ = {'linear':{'StrechValue':0.02, 'dT':0.2},
                                'sinus': {'Amplitude':0.8, 'Period':0.05}
                                }

    def _InitializeModule(self, **kwargs):
        self.ModFunctions = {'linear': self.LinearStretch, 'sinus':self.SinusStretch}
        
        self.SelectedFunction = self.ModFunctions[self._FuncType]
        self.TimeStart = None

        return True

    def _OnEventModule(self, event):
        if self.TimeStart is None:
            self.__Started = True
            self.TimeStart = event.timestamp
        ModEvent = Event(self.SelectedFunction(event.timestamp), event.location, event.polarity)

        return ModEvent

    def LinearStretch(self, ts):
        ts_from_origin = ts - self.TimeStart
        ModifiedTs = ts + (1./self._FuncOptionsDict__[self._FuncType__]['StrechValue'] - 1) * (1./self._FuncOptionsDict__[self._FuncType__]['dT']) * (ts_from_origin ** 2)/2

        return ModifiedTs

    def SinusStretch(self, ts):
        Pulse = 2 * np.pi / self._FuncOptionsDict__[self._FuncType__]['Period']
        ts_from_origin = ts - self.TimeStart

        ModifiedTs = ts + self._FuncOptionsDict__[self._FuncType__]['Amplitude'] / Pulse * (1 - np.cos(Pulse * ts_from_origin))

        return ModifiedTs
