import numpy as np

class Refractory:
    def __init__(self, argsCreationDict):
        '''
        Class to filter events from spike trains.
        Expects nothing.
        '''

        self._Type = 'Filter'

        self.Period = 0.06 # Given is seconds

        for key in argsCreationDict.keys():
            self.__dict__[key.split('.')[2]] = argsCreationDict[key]

    def _Initialize(self, argsInitializationDict):
        '''
        Requires:
        'Memory.Self' -> Gets Access to a memory tool, that should contain a STContext class variable
        '''
        self.Memory = argsInitializationDict['Memory.Self']
        self.AllowedEvents = 0
        self.FilteredEvents = 0

    def _OnEvent(self, event):
        if event.timestamp - self.Memory.STContext[event.location[0], event.location[1], event.polarity] >= self.Period:
            self.AllowedEvents += 1
            return event
        else:
            self.FilteredEvents += 1
            return None
