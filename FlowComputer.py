import numpy as np
import random
import matplotlib.pyplot as plt

from PEBBLE import ModuleBase, CameraEvent, FlowEvent

class FlowComputer(ModuleBase):
    def _OnCreation(self):
        '''
        Module to compute the optical flow.
        '''
        self.__ModulesLinksRequested__ = ['Memory']

        self._R = 5
        self._Tau = np.inf
        self._TauRatio = 5
        self._MinDetRatio = 0.00000001
        self._MinNEdges = 2.5
        self._MaxNEdgesRatio = 0.8
        self._MaxNEdges = 10

        self._RecursiveFlow = True
        self._NMaxRecursions = np.inf
        self._MaxEventsRemoved = 1
        self._MaxErrorDRatio = 0.3

        self._NMaxGradientRecursions = 5
        self._MaxErrorGradientRatio = 3.

        self._PolaritySeparation = False
        self._NeedsLogColumn = False

        self._NormOrigin = "Average"
        self._DirectionOrigin = "Plan"

        self._MaxFlowValue = np.inf

    def _OnInitialization(self):
        self.STContext = self.Memory.STContext

        self.D = (2*self._R+1)
        self._MaxNEdges = min(int(self._R * self._MaxNEdgesRatio), self._MaxNEdges)
        self.NEventsMax = int(self.D*self._MaxNEdges) - 2
        self.NEventsMin = int(self.D*self._MinNEdges) - 2

        self.N2Min = 1/self._MaxFlowValue**2

        self.MaxError = self.D * self._MaxErrorDRatio

        if self._NMaxRecursions == np.inf:
            self._NMaxRecursions = int((self.NEventsMax-self.NEventsMin)/self._MaxEventsRemoved)

        self.NRecursions = 0
        self.NFlowEvents = 0
        self.ScreenSize = np.array(self.Geometry)

        #np.seterr(all='raise')
        self.Filters = [0,0,0]

        return True

    def _OnEventModule(self, event):
        if (event.location < self._R).any() or (event.location >= self.ScreenSize - self._R).any():
            return
        Flow = self.ComputeGradientFlow(event)
        if not Flow is None:
            event.Attach(FlowEvent, flow = Flow)
        return

    def ComputeGradientFlow(self, event):
        Patch = self.Memory.GetSquarePatch(event.location, self._R)
        if self._PolaritySeparation:
            Patch = Patch[:,:,event.polarity] #for polarity separation
        else:
            Patch = Patch.max(axis = 2)

        tmy = (Patch[:,1:] + Patch[:,:-1])/2
        tmx = (Patch[1:,:] + Patch[:-1,:])/2
        tm = (tmy[1:,:] + tmy[:-1,:])/2
        t = tm.max()
        if t == -np.inf:
            return None
        dtm = t - tm
        Tau = self.FrameworkAverageTau
        if Tau is None or Tau == 0:
            Tau = self._Tau
        ValidMap = dtm < Tau * self._TauRatio
        NValids = ValidMap.sum()
        if NValids < self.NEventsMin:
            return None
        Valids = np.where(ValidMap)
        Patch[np.where(Patch==-np.inf)] = 0
        tdx = Patch[1:,:] - Patch[:-1,:]
        tdy = Patch[:,1:] - Patch[:,:-1]
        tdxmy = (tdx[:,1:] + tdx[:,:-1])/2
        tdymx = (tdy[1:,:] + tdy[:-1,:])/2

        S, STD = self.ComputeTimeDisplacement(tdxmy[Valids], tdymx[Valids])

        if (abs(S) == np.inf).any():
            self.Filters[0] += 1
            return None
        for nRecurse in range(self._NMaxGradientRecursions):
            S, STD, ValidMap = self.GradientRecursion(tdxmy, tdymx, ValidMap, S, STD)
            NNewValid = ValidMap.sum()
            if NNewValid < self.NEventsMin:
                self.Filters[1] += 1
                return None
            if NNewValid == NValids:
                F = S / (S**2).sum()
                return F
            else:
                NValids = NNewValid
        self.Filters[2] += 1
        return None

    @staticmethod
    def NonZeroTimeDisplacement(S):
        S[np.where(S==0)] = 0.000001
        return S

    def ComputeTimeDisplacement(self, tdxmyUsed, tdymxUsed):
        N = tdxmyUsed.shape[0]
        #try:
        S = self.NonZeroTimeDisplacement(np.array([np.median(tdxmyUsed), np.median(tdymxUsed)]))
        STD = np.sqrt(np.array([((tdxmyUsed - S[0])**2).sum(), ((tdymxUsed - S[1])**2).sum()]) / N)
        #except:
        #    print(tdxmyUsed, tdymxUsed, N)
        #    raise KeyboardInterrupt
        return S, STD

    def GradientRecursion(self, tdxmy, tdymx, ValidMap, S, STD):
        ValidMap = (abs(tdxmy - S[0]) < self._MaxErrorGradientRatio*STD[0]) * (abs(tdymx - S[1]) < self._MaxErrorGradientRatio*STD[0]) * ValidMap
        if ValidMap.sum() < self.NEventsMin:
            return None, None, ValidMap
        Valids = np.where(ValidMap)
        S, STD = self.ComputeTimeDisplacement(tdxmy[Valids], tdymx[Valids])
        return S, STD, ValidMap

    def ComputeFlow(self, event):
        Patch = self.Memory.GetSquarePatch(event.location, self._R)
        if self._PolaritySeparation:
            Patch = Patch[:,:,event.polarity] #for polarity separation
        else:
            Patch = Patch.max(axis = 2)
        
        xs, ys = np.where(Patch > event.timestamp - self._Tau)
        NEvents = xs.shape[0]
        if NEvents >= self.NEventsMin:
            ts = Patch[xs, ys]
            SortedIndexes = ts.argsort()[-self.NEventsMax:]
            ts = ts[SortedIndexes]
            xs = xs[SortedIndexes]
            ys = ys[SortedIndexes]
            NEvents = self.NEventsMax
        else:
            return None
        if "Plan" in (self._NormOrigin, self._DirectionOrigin):
            PlanFlow = self.GetPlanFlow(ts, xs, ys)
            if PlanFlow is None:
                return None
        if "Average" in (self._NormOrigin, self._DirectionOrigin):
            AverageFlow = self.GetAverageFlow(ts, xs, ys)
            if AverageFlow is None:
                return None
        if self._DirectionOrigin == "Plan":
            N = np.linalg.norm(PlanFlow)
            if N == 0:
                return None
            Normal = PlanFlow / N
        else:
            N = np.linalg.norm(AverageFlow)
            if N == 0:
                return None
            Normal = AverageFlow / N
        if self._NormOrigin == "Plan":
            UsedFlowVector = PlanFlow
        else:
            UsedFlowVector = AverageFlow
        Nf = (UsedFlowVector * Normal).sum()
        if Nf == 0:
            return None
        return Nf * Normal

    def GetPlanFlow(self, ts, xs, ys):
        nRecursion = 0
        NEvents = ts.shape[0]
        while True:
            nRecursion += 1

            tMean = ts.mean()
            
            xMean = xs.mean()
            yMean = ys.mean()
            
            xDeltas = xs - xMean
            yDeltas = ys - yMean
            tDeltas = ts - tMean

            Sx2 = (xDeltas **2).sum()
            Sy2 = (yDeltas **2).sum()
            Sxy = (xDeltas*yDeltas).sum()
            Stx = (tDeltas*xDeltas).sum()
            Sty = (tDeltas*yDeltas).sum()

            Det = Sx2*Sy2 - Sxy**2
            if Det > self._MinDetRatio * (NEvents**2 * self._R ** 4):
                V = np.array([(Sy2*Stx - Sxy*Sty), (Sx2*Sty - Sxy*Stx)]) / Det
                n = V/np.linalg.norm(V)

                N2 = (V**2).sum()
                tExpected = xDeltas * V[0] + yDeltas * V[1]
                Errors = abs(tExpected - tDeltas) / np.sqrt(N2)
                if Errors.max() <= self.MaxError:
                    if Errors.max() <= self.MaxError and N2 > self.N2Min:
                        return V/N2
                    return
                if nRecursion == self._NMaxRecursions or NEvents <= self.NEventsMin:
                    return None
                NOutliers = (Errors > self.MaxError).sum()
                Inliers = Errors.argsort()[:-min(min(self._MaxEventsRemoved, NEvents - self.NEventsMin), NOutliers)]
                xs, ys, ts = xs[Inliers], ys[Inliers], ts[Inliers]
                NEvents = len(Inliers)
                
            else:
                return None

    def GetAverageFlow(self, ts, xs, ys):
        # We assume events are sorted 
        t, x, y = ts[-1], xs[-1], ys[-1]
        tMean = ts[:-1].mean()
        xMean = ys[:-1].mean()
        yMean = xs[:-1].mean()

        return np.array([x - xMean, y - yMean])/(t - tMean)
