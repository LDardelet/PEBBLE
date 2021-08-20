import numpy as np

from PEBBLE import ModuleBase, DisparityEvent

class DisparityMemory(ModuleBase):
    def _OnCreation(self):
        '''
        Class to handle disparity memory.
        '''
        self._MonitorDt = 0
        self._MonitoredVariables = [('DisparityContext', np.array)]

    def _OnInitialization(self):

        self.DisparityContext = -np.inf*np.ones(tuple(self.Geometry) + (2,))

        self.LastT = -np.inf

        return True

    def _OnEventModule(self, event):
        if not event.Has(DisparityEvent):
            return
        self.DisparityContext[event.location[0], event.location[1],:] = np.array([event.timestamp, event.disparity])
        self.LastT = event.timestamp

        return

    def CreateSnapshot(self):
        return np.array(self.DisparityContext)

    def GetSquarePatch(self, xy, R):
        return self.DisparityContext[max(0,xy[0]-R):xy[0]+R+1,max(0,xy[1]-R):xy[1]+R+1,:]

    def GetDisparity(self, Tau, xy = None, R = None):
        if not xy is None:
            Map = self.GetSquarePatch(xy, R)
        else:
            Map = self.DisparityContext
        xs, ys = np.where(self.LastT - Map[:,:,0] <= Tau)
        ds = Map[xs,ys,1]
        if not ds.size:
            return None
        return int(np.median(ds)+0.5)
