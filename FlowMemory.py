import numpy as np

from PEBBLE import Module, FlowEvent

class FlowMemory(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to handle ST-context memory.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Memory'

        self._MonitorDt = 0. # By default, the memory module does NOT take any shapshot, to ensure memory doesn't get filled.
        self._MonitoredVariables = []
        self._NeedsLogColumn = False

        self._DefaultTau = 0.1
        self._RTau = 7

    def _InitializeModule(self):
        self.FlowContext = np.zeros(tuple(self.Geometry) + (3,))
        self.FlowContext[:,:,2] = -np.inf
        self.LastT = -np.inf

        return True

    def _OnEventModule(self, event):
        self.LastT = event.timestamp
        if not event.Has(FlowEvent):
            return

        self.FlowContext[event.location[0], event.location[1], :] = [event.flow[0], event.flow[1], event.timestamp]
        return

    def EventTau(self, EventConcerned = None):
        if EventConcerned is None:
            Flows = self.GetFlows(self._DefaultTau)
        else:
            Flows = self.GetFlows(self._DefaultTau, event.location, self._RTau)
        if Flows.size == 0:
            return 0
        NFlows = np.linalg.norm(Flows, axis = 1)
        Taus = 1 / np.maximum(1., NFlows) # Set Maximum tau value to 1s
        return Taus.mean()

    def GetFlowPatch(self, xy, R):
        return self.FlowContext[max(0,xy[0]-R):xy[0]+R+1,max(0,xy[1]-R):xy[1]+R+1,:]

    def GetFlows(self, Tau, xy = None, R = None):
        if not xy is None:
            Map = self.GetFlowPatch(xy, R)
        else:
            Map = self.FlowContext
        xs, ys = np.where(Map[:,:,2] > self.LastT - Tau)
        return Map[xs, ys, :2]
