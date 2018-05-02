import numpy as np
import tools
import random
from event import Event

class FlowComputer:
    def __init__(self, argsCreationDict):
        '''
        Tool to compute the optical flow.
        '''

        print " WARNING : BUGGED"

        for key in argsCreationDict.keys():
            self.__dict__[key.split('.')[2]] = argsCreationDict[key]
        self.R = 5
        self.EventsAgeLimit = 0.3
        self.PolaritySeparation = True

        self._Type = 'Computation'

    def _Initialize(self, argsInitializationDict):
        '''
        Expects :
        'Framework.StreamsGeometries' -> Dict of Streams Geometries
        'Framework.StreamHistory' -> List of previous streams
        'Memory.STContext' -> Access to the stream ST context
        '''

        shape = list(argsInitializationDict['Framework.StreamsGeometries'][argsInitializationDict['Framework.StreamHistory'][-1]])
        self.FlowMap = np.zeros(shape + [2])
        self.NormMap = np.zeros(shape)
        self.RegValue = np.zeros(shape)
        self.STContext = argsInitializationDict['Memory.STContext']

    def _OnEvent(self, event):
        self.ComputeFullFlow(event)

        return event


    def ComputeFullFlow(self, event):
        if self.PolaritySeparation:
            Patch = (tools.PatchExtraction(self.STContext, event.location, self.R))[:,:,event.polarity] #for polarity separation
        else:
            Patch = (tools.PatchExtraction(self.STContext, event.location, self.R)).max(axis = 2)
        
        Positions = np.where(Patch > Patch.max() - self.EventsAgeLimit)
        if Positions[0].shape[0] > 2:
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
            if Det > 0:
                F = np.array([(Sy2*Stx - Sxy*Sty)/Det, (Sx2*Sty - Sxy*Stx)/Det])

                self.NormMap[event.location[0], event.location[1], event.polarity] = np.linalg.norm(F)
                St2 = (tDeltas ** 2).sum()
                if St2 != 0:
                    self.RegValue[event.location[0], event.location[1], event.polarity] = 1 - ((tDeltas - F[0]*xDeltas - F[1]*yDeltas)**2).sum()/St2
                else:
                    self.RegValue[event.location[0], event.location[1], event.polarity] = 1
                if self.NormMap[event.location[0], event.location[1], event.polarity] > 0:
                    self.FlowMap[event.location[0], event.location[1], event.polarity, :] = F / self.NormMap[event.location[0], event.location[1], event.polarity]**2
                else:
                    self.FlowMap[event.location[0], event.location[1], event.polarity, :] = 0.

            else:
                self.FlowMap[event.location[0], event.location[1], event.polarity, :] = 0.
                self.RegValue[event.location[0], event.location[1], event.polarity] = 0.
                self.NormMap[event.location[0], event.location[1], event.polarity] = 0.

        else:
            self.FlowMap[event.location[0], event.location[1], event.polarity, :] = 0.
            self.RegValue[event.location[0], event.location[1], event.polarity] = 0.
            self.NormMap[event.location[0], event.location[1], event.polarity] = 0.
