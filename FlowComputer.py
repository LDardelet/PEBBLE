import numpy as np
import random
import matplotlib.pyplot as plt

from PEBBLE import Module, FlowEvent

class FlowComputer(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Tool to compute the optical flow.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__ReferencesAsked__ = ['Memory']
        self.__Type__ = 'Computation'

        self._R = 5
        self._Tau = 0.05
        self._MinDetRatio = 0.01
        self._PolaritySeparation = False
        self._NeedsLogColumn = False

        self._MaxFlowValue = np.inf

    def _InitializeModule(self, **kwargs):
        self.CurrentShape = list(self.Geometry)
        self._LinkedMemory = self.__Framework__.Tools[self.__CreationReferences__['Memory']]
        self.STContext = self._LinkedMemory.STContext

        self.N2Min = 1/self._MaxFlowValue**2

        return True

    def _OnEventModule(self, event):
        self.ComputeFullFlow(event)
        return event

    def ComputeFullFlow(self, event):
        Patch = self._LinkedMemory.GetSquarePatch(event.location, self._R)
        if self._PolaritySeparation:
            Patch = Patch[:,:,event.polarity] #for polarity separation
        else:
            Patch = Patch.max(axis = 2)
        
        Positions = np.where(Patch > event.timestamp - self._Tau)
        NEvents = Positions[0].shape[0]
        if NEvents >= 4*self._R:
            Ts = Patch[Positions]
            tMean = Ts.mean()
            
            xMean = Positions[0].mean()
            yMean = Positions[1].mean()
            
            xDeltas = Positions[0] - xMean
            yDeltas = Positions[1] - yMean
            tDeltas = Ts - tMean

            Sx2 = (xDeltas **2).sum()
            Sy2 = (yDeltas **2).sum()
            Sxy = (xDeltas*yDeltas).sum()
            Stx = (tDeltas*xDeltas).sum()
            Sty = (tDeltas*yDeltas).sum()

            Det = Sx2*Sy2 - Sxy**2
            if Det > self._MinDetRatio * (NEvents**2 * self._R ** 4):
                F = np.array([(Sy2*Stx - Sxy*Sty), (Sx2*Sty - Sxy*Stx)]) / Det

                N2 = (F**2).sum()
                if N2 > self.N2Min:
                    F /= N2
                    event.Attach(FlowEvent, location = np.array(event.location), flow = F)
