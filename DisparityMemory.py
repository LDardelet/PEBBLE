import numpy as np

from ModuleBase import ModuleBase
from Events import DisparityEvent

class DisparityMemory(ModuleBase):
    def _OnCreation(self):
        '''
        Class to handle disparity memory.
        '''
        self._MonitorDt = 0
        self._MonitoredVariables = [('DisparityContext', np.array)]
        self._RetreiveMethod = 'median' # among 'median', 'average', 'argmax'
        self._IntegerDisparity = True

    def _OnInitialization(self):

        self.DisparityContext = -np.inf*np.ones(tuple(self.Geometry) + (2,))

        self.LastT = -np.inf

        if self._IntegerDisparity:
            self.DispFunction = lambda x: int(x+0.5)
        else:
            self.DispFunction = lambda x:x

        if self._RetreiveMethod == 'median':
            self.GetDisparity = self._GetMedianDisparity
        elif self._RetreiveMethod == 'average':
            self.GetDisparity = self._GetAverageDisparity
        elif self._RetreiveMethod == 'argmax':
            self.GetDisparity = self._GetArgmaxDisparity

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

    def _GetMedianDisparity(self, Tau, xy = None, R = None):
        if not xy is None:
            Map = self.GetSquarePatch(xy, R)
        else:
            Map = self.DisparityContext
        xs, ys = np.where(self.LastT - Map[:,:,0] <= Tau)
        ds = Map[xs,ys,1]
        if not ds.size:
            return None
        return self.DispFunction(np.median(ds))

    def _GetAverageDisparity(self, Tau, xy = None, R = None):
        if not xy is None:
            Map = self.GetSquarePatch(xy, R)
        else:
            Map = self.DisparityContext
        xs, ys = np.where(self.LastT - Map[:,:,0] <= Tau)
        ds = Map[xs,ys,1]
        if not ds.size:
            return None
        return self.DispFunction(ds.mean())

    def _GetArgmaxDisparity(self, Tau, xy = None, R = None):
        if not xy is None:
            Map = self.GetSquarePatch(xy, R)
        else:
            Map = self.DisparityContext
        xs, ys = np.where(self.LastT - Map[:,:,0] <= Tau)
        if not xs.size:
            return None
        ds = Map[xs,ys,1]
        uds = np.unique(ds)
        ArgMax = None
        NMax = 0
        for ud in uds:
            N = (ds == ud).sum()
            if N > NMax:
                ArgMax = ud
                NMax = N
        return ArgMax
