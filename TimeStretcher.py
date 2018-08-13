import numpy as np
from event import Event

class Stretcher:
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to modify timestamps, mofifying speed norms
        '''
        self._ReferencesAsked = []
        self._Name = Name
        self._Framework = Framework
        self._Type = 'Filter'
        self._CreationReferences = dict(argsCreationReferences)

        self.FuncType = 'sinus'
        self.FuncOptionsDict = {'linear':{'StrechValue':0.02, 'dT':0.2},
                                'sinus': {'Amplitude':0.8, 'Period':0.05}
                                }

    def _Initialize(self):
        self.ModFunctions = {'linear': self.LinearStretch, 'sinus':self.SinusStretch}
        
        self.SelectedFunction = self.ModFunctions[self.FuncType]
        self.TimeStart = None

    def _OnEvent(self, event):
        if self.TimeStart is None:
            self.TimeStart = event.timestamp
        ModEvent = Event(self.SelectedFunction(event.timestamp), event.location, event.polarity)

        return ModEvent

    def LinearStretch(self, ts):
        ts_from_origin = ts - self.TimeStart
        ModifiedTs = ts + (1./self.FuncOptionsDict[self.FuncType]['StrechValue'] - 1) * (1./self.FuncOptionsDict[self.FuncType]['dT']) * (ts_from_origin ** 2)/2

        return ModifiedTs

    def SinusStretch(self, ts):
        Pulse = 2 * np.pi / self.FuncOptionsDict[self.FuncType]['Period']
        ts_from_origin = ts - self.TimeStart

        ModifiedTs = ts + self.FuncOptionsDict[self.FuncType]['Amplitude'] / Pulse * (1 - np.cos(Pulse * ts_from_origin))

        return ModifiedTs
