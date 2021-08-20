import numpy as np

from PEBBLE import ModuleBase, FlowEvent

class FlowMemory(ModuleBase):
    def _OnCreation(self):
        '''
        Class to handle optical flow memory.
        '''
        self._DefaultTau = 0.1
        self._RTau = 7
        self._FrameworkTauRatio = 4
        self._EnableEventTau = True

    def _OnInitialization(self):
        self.FlowContext = np.zeros(tuple(self.Geometry) + (3,))
        self.FlowContext[:,:,2] = -np.inf
        self.LastT = -np.inf
        self._InverseFlowNormsSum = 0.
        self._FlowActivity = 0.

        return True

    def _OnEventModule(self, event):
        self.LastT = event.timestamp
        if not event.Has(FlowEvent):
            return
        
        Tau = self.FrameworkAverageTau
        if Tau is None or Tau == 0:
            Tau = self._DefaultTau
        Tau = min(1., Tau) * self._FrameworkTauRatio
        decay = np.e**((self.LastT-event.timestamp)/Tau)
        self._InverseFlowNormsSum = self._InverseFlowNormsSum * decay + (1 / max(1., np.linalg.norm(event.flow)))
        self._FlowActivity = self._FlowActivity * decay + 1

        self.FlowContext[event.location[0], event.location[1], :] = [event.flow[0], event.flow[1], event.timestamp]
        return

    def EventTau(self, EventConcerned = None):
        if not self._EnableEventTau:
            return 0
        if EventConcerned is None:
            return self._InverseFlowNormsSum / max(0.0001, self._FlowActivity)
        Flows = self.GetFlows(EventConcerned.location, self._RTau)
        if Flows.size == 0:
            return 0
        NFlows = np.linalg.norm(Flows, axis = 1)
        Taus = 1 / np.maximum(1., NFlows) # Set Maximum tau value to 1s
        return Taus.mean()

    def GetFlowPatch(self, xy, R):
        return self.FlowContext[max(0,xy[0]-R):xy[0]+R+1,max(0,xy[1]-R):xy[1]+R+1,:]

    def GetFlows(self, xy = None, R = None):
        if not xy is None:
            Map = self.GetFlowPatch(xy, R)
        else:
            Map = self.FlowContext
        Tau = self.FrameworkAverageTau
        if Tau is None or Tau == 0:
            Tau = self._DefaultTau
        xs, ys = np.where(Map[:,:,2] > self.LastT - Tau)
        return Map[xs, ys, :2]

class ListFlowMemory(ModuleBase):
    def _OnCreation(self):
        '''
        Class to handle optical flow memory through lists.
        '''
        self._MemoryLength = 100000

    def _OnInitialization(self):
        self.Flows = np.zeros((self._MemoryLength, 5))
        self.Flows[:,0] = -np.inf
        self.Index = -1

        return True

    def _OnEventModule(self, event):
        if not event.Has(FlowEvent):
            return
        
        self.Index = (self.Index+1)%self._MemoryLength
        self.Flows[self.Index, :] = np.array([event.timestamp, event.location[0], event.location[1], event.flow[0], event.flow[1]])
        return

