import numpy as np
from event import Event

class Memory:
    def __init__(self, argsCreationDict):
        '''
        Class to handle ST-context memory.
        Requires :
        '''

        self._Type = 'Memory'

        for key in argsCreationDict.keys():
            self.__dict__[key.split('.')[2]] = argsCreationDict[key]

    def _Initialize(self, argInitializationDict):
        '''
        Expects:
        'Framework.StreamsGeometries' -> Gets the ST-contexts shapes to create
        'Framework.StreamHistory' -> Gets the name of the last stream created
        '''
        self.STContext = -np.inf*np.ones(argInitializationDict['Framework.StreamsGeometries'][argInitializationDict['Framework.StreamHistory'][-1]])
        self.PreviousTsAtLocation = - np.inf
        self.LastEvent = Event(-np.inf, [0,0], 0)

    def _OnEvent(self, event):
        self.LastEvent = Event(original = event)
        position = tuple(self.LastEvent.location.tolist() + [self.LastEvent.polarity])

        self.PreviousTsAtLocation = self.STContext[position[0], position[1], :].max()
        self.STContext[position] = self.LastEvent.timestamp

        return event
