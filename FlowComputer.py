import numpy as np
import tools
import random
from event import Event
import matplotlib.pyplot as plt

class FlowComputer:
    def __init__(self, argsCreationDict):
        '''
        Tool to compute the optical flow.
        '''

        self.Self = self

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

        self.CurrentShape = list(argsInitializationDict['Framework.StreamsGeometries'][argsInitializationDict['Framework.StreamHistory'][-1]])
        self.NEventsMap = np.zeros(self.CurrentShape)
        self.DetMap = np.zeros(self.CurrentShape)
        self.FlowMap = np.zeros(self.CurrentShape + [2])
        self.NormMap = np.zeros(self.CurrentShape)
        self.RegMap = np.zeros(self.CurrentShape)
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
        self.NEventsMap[event.location[0], event.location[1], event.polarity] = Positions[0].shape[0]
        if self.NEventsMap[event.location[0], event.location[1], event.polarity] > 2:
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

            self.DetMap[event.location[0], event.location[1], event.polarity] = Sx2*Sy2 - Sxy**2
            if self.DetMap[event.location[0], event.location[1], event.polarity] > 0:
                F = np.array([(Sy2*Stx - Sxy*Sty)/self.DetMap[event.location[0], event.location[1], event.polarity], (Sx2*Sty - Sxy*Stx)/self.DetMap[event.location[0], event.location[1], event.polarity]])

                N = np.linalg.norm(F)
                if N > 0:
                    self.NormMap[event.location[0], event.location[1], event.polarity] = 1./np.linalg.norm(F)
                    St2 = (tDeltas ** 2).sum()
                    if St2 != 0:
                        self.RegMap[event.location[0], event.location[1], event.polarity] = 1 - ((tDeltas - F[0]*xDeltas - F[1]*yDeltas)**2).sum()/St2
                    else:
                        self.RegMap[event.location[0], event.location[1], event.polarity] = 1
                    self.FlowMap[event.location[0], event.location[1], event.polarity, :] = F * self.NormMap[event.location[0], event.location[1], event.polarity]**2
                else:                    
                    self.FlowMap[event.location[0], event.location[1], event.polarity, :] = 0.
                    self.RegMap[event.location[0], event.location[1], event.polarity] = 0.
                    self.NormMap[event.location[0], event.location[1], event.polarity] = 0.

            else:
                self.FlowMap[event.location[0], event.location[1], event.polarity, :] = 0.
                self.RegMap[event.location[0], event.location[1], event.polarity] = 0.
                self.NormMap[event.location[0], event.location[1], event.polarity] = 0.

        else:
            self.FlowMap[event.location[0], event.location[1], event.polarity, :] = 0.
            self.RegMap[event.location[0], event.location[1], event.polarity] = 0.
            self.NormMap[event.location[0], event.location[1], event.polarity] = 0.

###### PLOTTING TOOLS ######

    def PlotFlow(self, location, radius, timelimit = -np.inf, ax = None):
        xMin = max(0, location[0] - radius)
        yMin = max(0, location[1] - radius)
        xMax = min(self.CurrentShape[0], location[0] + radius)
        yMax = min(self.CurrentShape[1], location[0] + radius)

        if ax is None:
            fig,  ax = plt.subplots(1,1)

        for x in range(xMin, xMax + 1):
            for y  in range(yMin, yMax + 1):
                pol = self.STContext[x, y, :].argmax()
                if self.STContext[x, y, pol] > timelimit:
                    if self.NormMap[xx, y, pol] > 0:
                        F = self.FlowMap[x, y, pol, :]
                        ax.quiver(x, y, F[0], F[1])
        return fig, ax
