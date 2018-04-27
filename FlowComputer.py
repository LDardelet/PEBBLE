import numpy as np
import tools
import random
from event import Event

class FlowManager:
    def __init__(self, Network, R, EventsAgeLimit):

        self.Network = Network
        self.R = R
        self.EventsAgeLimit = self.Network.Layers[0].TauIni
        self.OptimizationOn = False

        shape = list(self.Network.CurrentSTContext.shape[:2])
        self.FlowMap = np.zeros(shape + [2])
        self.OptimizedFlowMap = np.zeros(shape + [2])
        self.UpdatedFlowMap = np.zeros(shape + [2])
        self.NMap = np.zeros(shape)
        self.DetMap = np.zeros(shape)
        #self.DistMap = np.zeros(shape)
        self.RegValue = np.zeros(shape)

        self.NEventsMap = np.zeros(shape, dtype = int)
        self.OptimizationMap = np.zeros(shape, dtype = int)
        self.OptimizedMap = np.zeros(shape, dtype = int)

        #self.TxOnDNMap = np.zeros(shape)
        #self.TyOnDNMap = np.zeros(shape)
        self.SxxOnDMap = np.zeros(shape)
        self.SxyOnDMap = np.zeros(shape)
        self.SyyOnDMap = np.zeros(shape)
        self.xBarMap = np.zeros(shape)
        self.yBarMap = np.zeros(shape)
        self.OldestEventMap  = -10*np.ones(shape)

        self.StatsDict = {}
        self.StatsDict['Optimization computations'] = []
        self.StatsDict['Flow computations tried'] = []
        self.StatsDict['Classical flow computations'] = []
        self.StatsDict['Flow computations matches'] = []
        self.StatsDict['FClassical'] = []
        self.StatsDict['FPrevious'] = []
        self.StatsDict['FOptim'] = []
        self.StartTime = -np.inf
        self.BinTime = 0.001

        self.xPatch = np.array([[2*self.R - i for j in range(0, 2*self.R+1)] for i in range(0, 2*self.R+1)])
        self.yPatch = np.array([[2*self.R - j for j in range(0, 2*self.R+1)] for i in range(0, 2*self.R+1)])

        # Gradient Flow Variables

        self.dxt2 = np.zeros(shape)
        self.dyt2 = np.zeros(shape)
        self.dxt3 = np.zeros(shape)
        self.dyt3 = np.zeros(shape)

        self.dx2t2 = np.zeros(shape)
        self.dy2t2 = np.zeros(shape)

        self.FlowMapSol1 = np.zeros(shape + [2])
        self.FlowMapSol2 = np.zeros(shape + [2])

        self.FlowSaves = []
        self.FlowSol1Saves = []
        self.FlowSol2Saves = []
        self.dx2t2Saves = []
        self.dy2t2Saves = []
        self.tsSaves = []

        self.ComputationOrigins = np.zeros(shape)
        self.last_save_timestamp = 0.

        # Projection Variables

        self.CompleteSTContext = [[[[],[]] for y in range(shape[1])] for x in range(shape[0])]

    def CreateFlowHistogram(self, dV = 1., xRange = None, yRange = None):
        maxFlow = int(abs(self.FlowMap).max())
        Speeds = np.arange(-maxFlow, maxFlow, dV)
        self.FlowHistogram = np.zeros((2, Speeds.shape[0]))
        if xRange == None:
            xRange = [0,self.FlowMap.shape[0]]
        if yRange == None:
            yRange = [0,self.FlowMap.shape[1]]
        for x in range(xRange[0], xRange[1]):
            for y in range(yRange[0], yRange[1]):
                if np.linalg.norm(self.FlowMap[x,y,:]) != 0 and self.Network.CurrentSTContext[x,y,:].max() > self.Network.last_event.timestamp - self.EventsAgeLimit:
                    self.FlowHistogram[0, abs(Speeds - self.FlowMap[x,y,0]).argmin()] += 1
                    self.FlowHistogram[1, abs(Speeds - self.FlowMap[x,y,1]).argmin()] += 1
        print "Histograms took {0} events into account, with speeds up to {1}px/s.".format(self.FlowHistogram.sum(), maxFlow)
        return self.FlowHistogram, Speeds

    def ComputeSqueeshedFlow(self, event):
        self.CompleteSTContext[event.location[0]][event.location[1]][event.polarity] += [event.timestamp]
        #return self.ComputeSqueeshedFlowJM(event)
        #return self.ComputeSqueeshedFlowLDT(event)
        #return self.ComputeOverBasicFlow(event)
        #return self.ComputeGradientFlow(event)
        return self.ComputeFullFlow(event)

    def ComputeGradientFlow(self, event):
        Patch = np.array(tools.PatchExtraction(self.Network.CurrentSTContext, event.location, self.R))[:,:,event.polarity] #for polarity separation
        th = max(- self.EventsAgeLimit, event.timestamp - self.EventsAgeLimit)
        Mask = (Patch > th)
        Patch[np.where(1-Mask)] = 0
        dxPatch = (Patch[2:,:] - Patch[:-2,:])/2
        dx2Patch = (Patch[2:,:] + Patch[:-2,:] - 2*Patch[1:-1,:])
        dyPatch = (Patch[:,2:] - Patch[:,:-2])/2
        dy2Patch = (Patch[:,2:] - Patch[:,:-2] - 2*Patch[:,1:-1])

        GradientMaskX = Mask[2:,:]*Mask[:-2,:]
        GradientMaskY = Mask[:,2:]*Mask[:,:-2]
    
        self.dxt2[event.location[0], event.location[1]] = (np.sign(dxPatch)*GradientMaskX*dxPatch**2).sum()/max(1, GradientMaskX.sum())
        self.dyt2[event.location[0], event.location[1]] = (np.sign(dyPatch)*GradientMaskY*dyPatch**2).sum()/max(1, GradientMaskY.sum())
        self.dxt3[event.location[0], event.location[1]] = (GradientMaskX*dxPatch**3).sum()/max(1, GradientMaskX.sum())
        self.dyt3[event.location[0], event.location[1]] = (GradientMaskY*dyPatch**3).sum()/max(1, GradientMaskY.sum())

        self.dx2t2[event.location[0], event.location[1]] = (np.sign(dxPatch)*GradientMaskX*dx2Patch**2).sum()/max(1, GradientMaskX.sum())
        self.dy2t2[event.location[0], event.location[1]] = (np.sign(dyPatch)*GradientMaskY*dy2Patch**2).sum()/max(1, GradientMaskY.sum())

        #if self.dxt3[event.location[0], event.location[1]] == 0 or self.dxt2[event.location[0], event.location[1]]**2 == 0 or self.dyt3[event.location[0], event.location[1]] == 0 or self.dyt2[event.location[0], event.location[1]]**2 == 0:
        Deltax = 6 * (GradientMaskX*dxPatch**2).sum()/max(1, GradientMaskX.sum()) - 3 * ((GradientMaskX*dxPatch).sum()/max(1, GradientMaskX.sum()))**2
        Deltay = 6 * (GradientMaskY*dyPatch**2).sum()/max(1, GradientMaskY.sum()) - 3 * ((GradientMaskY*dyPatch).sum()/max(1, GradientMaskY.sum()))**2
        fx1 = (np.sqrt(Deltax) - (GradientMaskX*dxPatch).sum()/max(1, GradientMaskX.sum()))/2
        fx2 = (- np.sqrt(Deltax) - (GradientMaskX*dxPatch).sum()/max(1, GradientMaskX.sum()))/2
        fy1 = (np.sqrt(Deltay) - (GradientMaskY*dyPatch).sum()/max(1, GradientMaskY.sum()))/2
        fy2 = (- np.sqrt(Deltay) - (GradientMaskY*dyPatch).sum()/max(1, GradientMaskY.sum()))/2
        F1 = np.array([fx1, fy1])
        F2 = np.array([fx2, fy2])
        Norm1 = np.linalg.norm(F1)
        Norm2 = np.linalg.norm(F2)
        if Norm1 != 0:
            if 1./Norm1 < 300:
                F1 /= Norm1**2
            else:
                F1 *= 0
        if Norm2 != 0:
            if 1./Norm2 < 300:
                F2 /= Norm2**2
            else:
                F2 *= 0
        self.FlowMapSol1[event.location[0], event.location[1], :] = F1
        self.FlowMapSol2[event.location[0], event.location[1], :] = F2

        if event.timestamp - self.last_save_timestamp > 0.02:
            self.last_save_timestamp = event.timestamp

            self.FlowSol1Saves += [np.array(self.FlowMapSol1)]
            self.FlowSol2Saves += [np.array(self.FlowMapSol2)]
            self.tsSaves += [event.timestamp]
        return True

        if self.dxt3[event.location[0], event.location[1]] == 0 or self.dyt3[event.location[0], event.location[1]] == 0:
            a = np.array(((np.sign(dxPatch)*GradientMaskX*dxPatch**2).sum()/max(1, GradientMaskX.sum()), (np.sign(dyPatch)*GradientMaskY*dyPatch**2).sum()/max(1, GradientMaskY.sum())))
            F = np.sign(a)*np.sqrt(abs(a))
            #F = np.array(((GradientMaskX*dxPatch).sum()/max(1, GradientMaskX.sum()), (GradientMaskY*dyPatch).sum()/max(1, GradientMaskY.sum())))
            self.ComputationOrigins[event.location[0], event.location[1]] = 2
        elif False:
            Ax = np.sign(self.dxt3[event.location[0], event.location[1]])*abs(self.dxt3[event.location[0], event.location[1]])**(1./3)
            Bx = np.sign(self.dxt2[event.location[0], event.location[1]])*abs(self.dxt2[event.location[0], event.location[1]])**(1./2)
            Ay = np.sign(self.dyt3[event.location[0], event.location[1]])*abs(self.dyt3[event.location[0], event.location[1]])**(1./3)
            By = np.sign(self.dyt2[event.location[0], event.location[1]])*abs(self.dyt2[event.location[0], event.location[1]])**(1./2)
            if 2 - 4*Bx**6/Ax**6 > 0 or 2 - 4*By**6/Ay**6 > 0:
                a = np.array(((np.sign(dxPatch)*GradientMaskX*dxPatch**2).sum()/max(1, GradientMaskX.sum()), (np.sign(dyPatch)*GradientMaskY*dyPatch**2).sum()/max(1, GradientMaskY.sum())))
                F = np.sign(a)*np.sqrt(abs(a))
                self.ComputationOrigins[event.location[0], event.location[1]] = 2
            else:
                fx = Ax/np.roots([1, -3*Bx**2/Ax**2, 0, 2])[1]
                fy = Ay/np.roots([1, -3*By**2/Ay**2, 0, 2])[1]
                F = np.array([fx, fy])
            #F = np.array([self.dxt2[event.location[0], event.location[1]]**2/self.dxt3[event.location[0], event.location[1]], self.dyt2[event.location[0], event.location[1]]**2/self.dyt3[event.location[0], event.location[1]]])
                self.ComputationOrigins[event.location[0], event.location[1]] = 1
        Norm = np.linalg.norm(F)
        if Norm == 0:
            self.ComputationOrigins[event.location[0], event.location[1]] = 0
            return False
        Flow = F/Norm**2
        if np.linalg.norm(Flow) < 300:
            self.FlowMap[event.location[0], event.location[1], :] = Flow
        else:
            self.FlowMap[event.location[0], event.location[1], :] = np.array([0,0])

        if event.timestamp - self.last_save_timestamp > 0.02:
            self.last_save_timestamp = event.timestamp

            self.FlowSaves += [np.array(self.FlowMap)]
            self.dx2t2Saves += [np.array(self.dx2t2)]
            self.dy2t2Saves += [np.array(self.dy2t2)]
            self.tsSaves += [event.timestamp]


        return True

    def ComputeSqueeshedFlowLDT(self, event):
        Patch = (tools.PatchExtraction(self.Network.CurrentSTContext, event.location, self.R))[:,:,event.polarity] #for polarity separation
        #Patch = (tools.PatchExtraction(self.Network.CurrentSTContext, event.location, self.R)).max(axis = 2)
        Patch = np.maximum(Patch, 0.)
        th = max(- self.EventsAgeLimit, event.timestamp - self.EventsAgeLimit)
        Mask = (Patch > th)
        F = np.array([0., 0.])
        self.FlowMap[event.location[0], event.location[1], :] = F
        if Mask.sum() <= 1:
            return False
        MPatch = Patch*Mask
        #Patchx = Patch.mean(axis = 1)
        Patchx = MPatch.sum(axis = 1)
        MxSum = Mask.sum(axis = 1)
        PosX = np.where(MxSum > 0)[0]
        if PosX.shape[0] < 2:
            F[0] = 0.
        else:
            Patchx[PosX] = Patchx[PosX]/MxSum[PosX]
            Xm = PosX - PosX.mean()
            Tm = Patchx[PosX] - Patchx[PosX].mean()
            F[0] = ((Xm*Tm).sum()/(Xm**2).sum())
        #Patchy = Patch.mean(axis = 0)
        Patchy = MPatch.sum(axis = 0)
        MySum = Mask.sum(axis = 0)
        PosY = np.where(MySum > 0)[0]
        if PosY.shape[0] < 2:
            F[1] = 0.
        else:
            Patchy[PosY] = Patchy[PosY]/MySum[PosY]
            Ym = PosY - PosY.mean()
            Tm = Patchy[PosY] - Patchy[PosY].mean()
            F[1] = ((Ym*Tm).sum()/(Ym**2).sum())
        Norm = np.linalg.norm(F)
        if Norm == 0:
            return False
        #Flow = F*max(F[0]**2, F[1]**2)/Norm**4
        Flow = F/Norm**2
        #Flow = F/(Norm**2 + (4 * F[0]**2 * F[1]**2)/Norm**2)
        if np.linalg.norm(Flow) < 300:
            self.FlowMap[event.location[0], event.location[1], :] = Flow
        else:
            self.FlowMap[event.location[0], event.location[1], :] = np.array([0,0])
        #self.FlowMap[event.location[0], event.location[1], :] = F
        return True

    def ComputeOverBasicFlow(self, event):
        Patch = (tools.PatchExtraction(self.Network.CurrentSTContext, event.location, self.R))[:,:,event.polarity]
        Patch = np.maximum(Patch, 0.)
        th = max(0, event.timestamp - self.EventsAgeLimit)
        F = np.array([0., 0.])

        LineX = Patch[:,self.R]
        MaskX = (LineX > th)
        if (MaskX.sum()) > 1:
            MeanX = (MaskX*LineX).sum()/(MaskX.sum())
            CoordsX = np.array(range(MaskX.shape[0]), dtype = float)*MaskX
            mCoordsX = CoordsX.sum()/MaskX.sum()
            F[0] = ((LineX - MeanX)*(CoordsX - mCoordsX)*MaskX).sum() / (MaskX*(CoordsX - mCoordsX)**2).sum()
        LineY = Patch[self.R,:]
        MaskY = (LineY > th)
        if (MaskY.sum()) > 1:
            MeanY = (MaskY*LineY).sum()/(MaskY.sum())
            CoordsY = np.array(range(MaskY.shape[0]), dtype = float)*MaskY
            mCoordsY = CoordsY.sum()/MaskY.sum()
            F[1] = ((LineY - MeanY)*(CoordsY - mCoordsY)*MaskY).sum() / (MaskY*(CoordsY - mCoordsY)**2).sum()
        Norm = np.linalg.norm(F)
        if Norm == 0:
            return False
        Flow = F/Norm**2
        if np.linalg.norm(Flow) < 300:
            self.FlowMap[event.location[0], event.location[1], :] = Flow
        else:
            self.FlowMap[event.location[0], event.location[1], :] = np.array([0,0])
        return True

    def ComputeSqueeshedFlowJM(self, event):
        Patch = (tools.PatchExtraction(self.Network.CurrentSTContext, event.location, self.R))[:,:,event.polarity] #for polarity separation
        #Patch = (tools.PatchExtraction(self.Network.CurrentSTContext, event.location, self.R)).max(axis = 2)
        Patch = np.maximum(Patch, 0.)
        th = max(- self.EventsAgeLimit, event.timestamp - self.EventsAgeLimit)
        Mask = (Patch > th)
        F = np.array([0., 0.])
        if Mask.sum() <= 1:
            self.FlowMap[event.location[0], event.location[1], :] = F
            return False

        CoordsX = np.array([[i for i in range(Mask.shape[0])]], dtype = float)
        XMasked = CoordsX.T*Mask
        xMaskSums = np.maximum(1, Mask.sum(axis = 0))

        txMeans = (Mask*Patch).sum(axis = 0)/xMaskSums
        dTx = (Patch - txMeans)*Mask
        
        xMeans = XMasked.sum(axis = 0)/xMaskSums
        dX = Mask*(XMasked - xMeans)

        CoordsY = np.array([[i for i in range(Mask.shape[1])]], dtype = float)
        YMasked = Mask*CoordsY
        yMaskSums = np.maximum(1, Mask.sum(axis = 1))
        
        tyMeans = (Mask*Patch).sum(axis = 1)/yMaskSums
        dTy = (Patch - tyMeans)*Mask

        yMeans = YMasked.sum(axis = 1)/yMaskSums
        dY = Mask*(YMasked.T - yMeans).T
    
        #Fxs = (dTx*dX).sum(axis = 0) / np.maximum(1, (dX**2).sum(axis = 0))
        #F[0] = ((Fxs*Mask.sum(axis = 0)).sum()/Mask.sum())

        #Fys = (dTy*dY).sum(axis = 1) / np.maximum(1, (dY**2).sum(axis = 1))
        #F[1] = ((Fys*Mask.sum(axis = 1)).sum()/Mask.sum())

        FxDenom = (dTx*dX).sum(axis = 0)
        FxDenom[np.where(FxDenom == 0)] = 1.
        try:
            F[0] = (((dX**2).sum(axis = 0) / FxDenom)*Mask.sum(axis = 0)).sum()/Mask.sum()
        except:
            F[0] = 0.
        FyDenom = (dTy*dY).sum(axis = 1)
        FyDenom[np.where(FyDenom == 0)] = 1.
        try:
            F[1] = (((dY**2).sum(axis = 1) / FyDenom)*Mask.sum(axis = 1)).sum()/Mask.sum()
        except:
            F[1] = 0.

        Norm = np.linalg.norm(F)
        if Norm > 1000:
            self.FlowMap[event.location[0], event.location[1], :] = np.array([0., 0.])
            return False
        #F = F/Norm**2
        self.FlowMap[event.location[0], event.location[1], :] = np.array(F)


    def ComputeFullFlow(self, event, remind_opti = False):
        Patch = (tools.PatchExtraction(self.Network.CurrentSTContext, event.location, self.R))[:,:,event.polarity] #for polarity separation
        #Patch = (tools.PatchExtraction(self.Network.CurrentSTContext, event.location, self.R)).max(axis = 2)
        
        #Patch[self.R, self.R] = event.timestamp 
        
        Positions = np.where(Patch > Patch.max() - self.EventsAgeLimit)
        self.NMap[event.location[0], event.location[1]] = Positions[0].shape[0]
        if self.NMap[event.location[0], event.location[1]] > 2:
            Ts = Patch[Positions]
            
            self.xBarMap[event.location[0], event.location[1]] = Positions[0].mean() #Must save for optim
            self.yBarMap[event.location[0], event.location[1]] = Positions[1].mean() #Must save for optim
            
            Xm = Positions[0] - self.xBarMap[event.location[0], event.location[1]]
            Ym = Positions[1] - self.yBarMap[event.location[0], event.location[1]]
            Tm = Ts - Ts.mean()

            Sx2 = (Xm **2).sum()
            Sy2 = (Ym **2).sum()
            Sxy = (Xm*Ym).sum()
            Stx = (Tm*Xm).sum()
            Sty = (Tm*Ym).sum()
            self.DetMap[event.location[0], event.location[1]] = Sx2*Sy2 - Sxy**2
            if self.DetMap[event.location[0], event.location[1]] > 0:
                self.SxxOnDMap[event.location[0], event.location[1]] = Sx2/self.DetMap[event.location[0], event.location[1]] #Must save for optim
                self.SyyOnDMap[event.location[0], event.location[1]] = Sy2/self.DetMap[event.location[0], event.location[1]] #Must save for optim
                self.SxyOnDMap[event.location[0], event.location[1]] = Sxy/self.DetMap[event.location[0], event.location[1]] #Must save for optim

                #self.TxOnDNMap[event.location[0], event.location[1]] = (Sy2*Xm.sum() - Sxy*Ym.sum())/(self.DetMap[event.location[0], event.location[1]]*self.NMap[event.location[0], event.location[1]]) #Must save for optim
                #self.TyOnDNMap[event.location[0], event.location[1]] = (Sx2*Ym.sum() - Sxy*Xm.sum())/(self.DetMap[event.location[0], event.location[1]]*self.NMap[event.location[0], event.location[1]]) #Must save for optim

                self.FlowMap[event.location[0], event.location[1], :] = np.array([self.SyyOnDMap[event.location[0], event.location[1]]*Stx-self.SxyOnDMap[event.location[0], event.location[1]]*Sty, self.SxxOnDMap[event.location[0], event.location[1]]*Sty-self.SxyOnDMap[event.location[0], event.location[1]]*Stx, -1])[:2]
                N = np.linalg.norm(self.FlowMap[event.location[0], event.location[1], :])
                F = self.FlowMap[event.location[0], event.location[1], :]
                St2 = (Tm **2).sum()
                if St2 != 0:
                    R2 = 1 - ((Tm - F[0]*Xm - F[1]*Ym)**2).sum()/St2
                    self.RegValue[event.location[0], event.location[1]] = R2
                else:
                    self.RegValue[event.location[0], event.location[1]] = 1
                if N > 0:
                    #self.FlowMap[event.location[0], event.location[1], :] = self.FlowMap[event.location[0], event.location[1], :] * (self.FlowMap[event.location[0], event.location[1], :]**2).max() / N**4
                    self.FlowMap[event.location[0], event.location[1], :] = self.FlowMap[event.location[0], event.location[1], :]/ N**2
                self.OldestEventMap[event.location[0], event.location[1]] = Ts.min()

                if not remind_opti or self.OptimizationMap[event.location[0], event.location[1]] == 0:
                    self.OptimizedMap[event.location[0], event.location[1]] = 0
                    self.OptimizedFlowMap[event.location[0], event.location[1], :] = np.array(self.FlowMap[event.location[0], event.location[1], :])


                self.OptimizationMap[event.location[0], event.location[1]] = 1
                
                Points = np.array([Xm, Ym, Tm])
                #self.StatsDict['Classical flow computations'][-1] += 1
                #self.DistMap[event.location[0], event.location[1]] = (((np.transpose(Points)*np.array(self.FlowMap[event.location[0], event.location[1], :].tolist()+[-1])).sum(axis = 1))**2).sum()/np.linalg.norm()
                return True
            else:
                self.FlowMap[event.location[0], event.location[1], :] = np.array([0.,0.])
                self.DetMap[event.location[0], event.location[1]] = 0
                self.OptimizationMap[event.location[0], event.location[1]] = 0
                self.OptimizedMap[event.location[0], event.location[1]] = 0
                self.OldestEventMap[event.location[0], event.location[1]] = -10
                self.OptimizedFlowMap[event.location[0], event.location[1], :] = np.array([0.,0.])
                return False
        else:
            self.FlowMap[event.location[0], event.location[1], :] = np.array([0.,0.])
            self.OptimizedMap[event.location[0], event.location[1]] = 0
            self.OptimizationMap[event.location[0], event.location[1]] = 0
            self.OldestEventMap[event.location[0], event.location[1]] = -10
            self.OptimizedFlowMap[event.location[0], event.location[1], :] = np.array([0.,0.])
            return False

    def ComputeOptimizedFlowLocal(self, event, previous_time, return_instead_of_saving = False):
# All checks for timestamps and OptimizationMap must be done before calling this function
        DeltaFx = (- self.TxOnDNMap[event.location[0], event.location[1]] 
                    + self.SyyOnDMap[event.location[0], event.location[1]] * (self.R - self.xBarMap[event.location[0], event.location[1]]) 
                    - self.SxyOnDMap[event.location[0], event.location[1]] * (self.R - self.yBarMap[event.location[0], event.location[1]])) * (event.timestamp - previous_time)
        DeltaFy = (- self.TyOnDNMap[event.location[0], event.location[1]] 
                    + self.SxxOnDMap[event.location[0], event.location[1]] * (self.R - self.yBarMap[event.location[0], event.location[1]]) 
                    - self.SxyOnDMap[event.location[0], event.location[1]] * (self.R - self.xBarMap[event.location[0], event.location[1]])) * (event.timestamp - previous_time)
        if return_instead_of_saving:
            return DeltaFx, DeltaFy
        self.FlowMap[event.location[0], event.location[1], :] += np.array([DeltaFx, DeltaFy])

    def ComputeOptimizedFlowPatch(self, event, previous_time):
        #OptimPatch = self.OptimizationMap[event.location[0]-self.R:event.location[0]+self.R+1, event.location[1]-self.R:event.location[1]+self.R+1]
        OldTimestamps = tools.PatchExtraction(self.OldestEventMap, event.location, self.R)

        Condition1 = tools.PatchExtraction(self.OptimizationMap, event.location, self.R) # To remember the previous authorizations
        Condition2 = ((event.timestamp - OldTimestamps) < self.EventsAgeLimit) # To Ensure that no event will get out of the classical computation window
        Condition3 = (OldTimestamps < previous_time) # To ensure that the event going up is an event considered in the initial computation

        OptimPatch = Condition1*Condition2*Condition3
        self.OptimizationMap[event.location[0]-self.R:event.location[0]+self.R+1, event.location[1]-self.R:event.location[1]+self.R+1] = OptimPatch
        if not OptimPatch.any(): #If no pixel can be optimized, we dont compute anything
            #return 0, 0, 0
            return None
        xMod = self.xPatch - self.xBarMap[event.location[0]-self.R:event.location[0]+self.R+1, event.location[1]-self.R:event.location[1]+self.R+1]
        yMod = self.yPatch - self.yBarMap[event.location[0]-self.R:event.location[0]+self.R+1, event.location[1]-self.R:event.location[1]+self.R+1]
        DeltaFx =   ( self.SyyOnDMap[event.location[0]-self.R:event.location[0]+self.R+1, event.location[1]-self.R:event.location[1]+self.R+1] * xMod
                    - self.SxyOnDMap[event.location[0]-self.R:event.location[0]+self.R+1, event.location[1]-self.R:event.location[1]+self.R+1] * yMod) * (event.timestamp - previous_time)
        DeltaFy =  ( self.SxxOnDMap[event.location[0]-self.R:event.location[0]+self.R+1, event.location[1]-self.R:event.location[1]+self.R+1] * yMod
                    - self.SxyOnDMap[event.location[0]-self.R:event.location[0]+self.R+1, event.location[1]-self.R:event.location[1]+self.R+1] * xMod) * (event.timestamp - previous_time)
        #DeltaFx = (- self.TxOnDNMap[event.location[0]-self.R:event.location[0]+self.R+1, event.location[1]-self.R:event.location[1]+self.R+1]
        #            + self.SyyOnDMap[event.location[0]-self.R:event.location[0]+self.R+1, event.location[1]-self.R:event.location[1]+self.R+1] * xMod
        #            - self.SxyOnDMap[event.location[0]-self.R:event.location[0]+self.R+1, event.location[1]-self.R:event.location[1]+self.R+1] * yMod) * (event.timestamp - previous_time)
        #DeltaFy = (- self.TyOnDNMap[event.location[0]-self.R:event.location[0]+self.R+1, event.location[1]-self.R:event.location[1]+self.R+1] 
        #            + self.SxxOnDMap[event.location[0]-self.R:event.location[0]+self.R+1, event.location[1]-self.R:event.location[1]+self.R+1] * yMod
        #            - self.SxyOnDMap[event.location[0]-self.R:event.location[0]+self.R+1, event.location[1]-self.R:event.location[1]+self.R+1] * xMod) * (event.timestamp - previous_time)
        self.OptimizedMap[event.location[0]-self.R:event.location[0]+self.R+1, event.location[1]-self.R:event.location[1]+self.R+1] = OptimPatch*(self.OptimizedMap[event.location[0]-self.R:event.location[0]+self.R+1, event.location[1]-self.R:event.location[1]+self.R+1] + 1)

        self.StatsDict['Optimization computations'][-1] += OptimPatch.sum()
        self.OptimizedFlowMap[event.location[0]-self.R:event.location[0]+self.R+1, event.location[1]-self.R:event.location[1]+self.R+1, 0] += OptimPatch*DeltaFx
        self.OptimizedFlowMap[event.location[0]-self.R:event.location[0]+self.R+1, event.location[1]-self.R:event.location[1]+self.R+1, 1] += OptimPatch*DeltaFy
        #return Condition1, Condition2, Condition3

    def OnEventWithSqeechedFlow(self):
        ev = self.Network.last_event
        self.NEventsMap[ev.location[0], ev.location[1]] += 1

        return self.ComputeSqueeshedFlow(ev)

    def OnEventWithCheckFlow(self):
        Prev_time = self.Network.Previous_timestamp_at_location
        ev = self.Network.last_event

        self.NEventsMap[ev.location[0], ev.location[1]] += 1

        if ev.timestamp - self.StartTime > self.BinTime:
            self.StatsDict['Optimization computations'] += [0]
            self.StatsDict['Flow computations tried'] += [0]
            self.StatsDict['Classical flow computations'] += [0]
            self.StartTime = ev.timestamp

        self.StatsDict['Flow computations tried'][-1] += 1
        PreviousEvent = Event(original = ev)
        PreviousEvent.timestamp = Prev_time
        Path = '#'
        #if (ev.timestamp - Prev_time) < self.EventsAgeLimit/10:
        if Prev_time >= 0.:
            Path += '1'
            #C1, C2, C3 = self.ComputeOptimizedFlowPatch(ev, Prev_time)
            self.ComputeOptimizedFlowPatch(ev, Prev_time)

            if not self.OptimizationMap[ev.location[0], ev.location[1]]:
                Path += '2'
                self.StatsDict['Classical flow computations'][-1] += 1
                self.ComputeFullFlow(ev, remind_opti = False)
        else:
            Path += '3'
            self.OptimizedMap[ev.location[0]-self.R:ev.location[0]+self.R+1, ev.location[1]-self.R:ev.location[1]+self.R+1] = 0
            self.OptimizationMap[ev.location[0]-self.R:ev.location[0]+self.R+1, ev.location[1]-self.R:ev.location[1]+self.R+1] = 0
            self.StatsDict['Classical flow computations'][-1] += 1
            self.ComputeFullFlow(ev, remind_opti = False)

        if False:
            for x in range(-self.R, self.R+1):
                for y in range(-self.R, self.R+1):
                    f, n, det, d, ts_min = tools.FlowComputation(tools.PatchExtraction(self.Network.CurrentSTContext, [ev.location[0]+x, ev.location[1]+y], self.R).max(axis = 2), self.EventsAgeLimit)
                    if det != 0:
                        if x == 0 and y == 0 and self.NMap[ev.location[0], ev.location[1]] != 0 and n != self.NMap[ev.location[0], ev.location[1]] and 15 < ev.location[0] < 287 and 15 < ev.location[1] < 187:
                            print "Weird, path is {3}, NMap is {0} at {1} while n is {2}".format(self.NMap[ev.location[0], ev.location[1]], [ev.location[0], ev.location[1]], n, Path) + int(self.OptimizationMap[ev.location[0], ev.location[1]])*(", can be optimized, has been optimized {0} times".format(self.OptimizedMap[ev.location[0], ev.location[1]]))
                            print "Timings data : current ts = {0}, patch oldest ts = {1}, previous pixel ts = {2}, min ts from check = {3}".format(ev.timestamp, self.OldestEventMap[ev.location[0], ev.location[1]], Prev_time, ts_min)
                            print "History of pixel interactions :"
                        self.UpdatedFlowMap[ev.location[0]+x, ev.location[1]+y, :] = np.array(f)
                    else:
                        self.UpdatedFlowMap[ev.location[0]+x, ev.location[1]+y, :] = np.array([0.,0.])
        return True

    def ComputeUpdatedFlow(self):
        for x in range(self.Network.CurrentSTContext.shape[0]):
            for y in range(self.Network.CurrentSTContext.shape[1]):
                f, n, det, d = tools.FlowComputation(tools.PatchExtraction(self.Network.CurrentSTContext, [x, y], self.R).max(axis = 2), self.EventsAgeLimit)
                if det != 0:
                    self.UpdatedFlowMap[x, y, :] = np.array(f)

    def OnEvent(self):
        Prev_time = self.Network.Previous_timestamp_at_location
        ev = self.Network.last_event
        if ev.timestamp - self.StartTime > self.BinTime:
            self.StatsDict['Optimization computations'] += [0]
            self.StatsDict['Flow computations tried'] += [0]
            self.StatsDict['Classical flow computations'] += [0]
            self.StatsDict['Flow computations matches'] += [0]
            self.StatsDict['Flows differences'] += [[]]
            self.StartTime = ev.timestamp
        self.StatsDict['Flow computations tried'][-1] += 1
        if not self.OptimizationOn or not self.OptimizationMap[ev.location[0], ev.location[1]] or (ev.timestamp - Prev_time) > self.EventsAgeLimit:
            return self.ComputeFullFlow(ev)
        else:
            self.StatsDict['Optimization computations'][-1] += 1
            self.ComputeOptimizedFlowLocal(ev, Prev_time)
            self.StatsDict['Classical flow computations'][-1] += 1
            return True

    def OnEventWithCheckFlowLocal(self):

        Prev_time = self.Network.Previous_timestamp_at_location
        ev = self.Network.last_event

        if ev.timestamp - self.StartTime > self.BinTime:
            self.StatsDict['Optimization computations'] += [0]
            self.StatsDict['Flow computations tried'] += [0]
            self.StatsDict['Classical flow computations'] += [0]
            self.StatsDict['Flow computations matches'] += [0]
            self.StatsDict['FClassical'] += [[]]
            self.StatsDict['FPrevious'] += [[]]
            self.StatsDict['FOptim'] += [[]]
            self.StartTime = ev.timestamp
        choosen = (random.random() < 0.001)

        if choosen:
            print ""
            print "~~~~~~"
            print Prev_time, " to ", ev.timestamp
            print "Initial flow : ", self.FlowMap[ev.location[0], ev.location[1], :]
        self.StatsDict['Flow computations tried'][-1] += 1
        DFx = None
        PreviousEvent = Event(original = ev)
        PreviousEvent.timestamp = Prev_time

        Opti = self.ComputeFullFlow(PreviousEvent)
        PreviousFlow = np.array(self.FlowMap[ev.location[0], ev.location[1], :])
        if choosen:
            print "Previous flow : ", self.FlowMap[ev.location[0], ev.location[1], :]
        if self.OptimizationMap[ev.location[0], ev.location[1]] and (ev.timestamp - Prev_time) < self.EventsAgeLimit/10:
            self.StatsDict['Optimization computations'][-1] += 1
            DFx, DFy = self.ComputeOptimizedFlowLocal(ev, Prev_time, return_instead_of_saving = True)
        ComputationAchieved =  self.ComputeFullFlow(ev)
        if DFx == None:
            return False
        else:
            if not ComputationAchieved:
                print "Computed DFx, while unable to compute the flow by classical method."
                return False
            OptimizedFlow = np.array([PreviousFlow[0] + DFx, PreviousFlow[1] + DFy])
            if choosen:
                print "Actual Flow : ", self.FlowMap[ev.location[0], ev.location[1], :]
                print "Optimized flow : ", OptimizedFlow
                print "###########"

            if (OptimizedFlow == self.FlowMap[ev.location[0], ev.location[1], :]).all():
                self.StatsDict['Flow computations matches'][-1] += 1


            self.StatsDict['FPrevious'][-1] += [np.array(PreviousFlow)]
            self.StatsDict['FClassical'][-1] += [np.array(self.FlowMap[ev.location[0], ev.location[1], :])]
            self.StatsDict['FOptim'][-1] += [np.array(OptimizedFlow)]
