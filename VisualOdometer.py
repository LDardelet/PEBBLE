from PEBBLE import ModuleBase, CameraEvent, DisparityEvent, FlowEvent, TwistEvent
import numpy as np

class VisualOdometer(ModuleBase):
    def _OnCreation(self):
        '''
        Uses disparity events and optical flow events to recover a visual odometry
        '''
        self._MonitorDt = 0. # By default, a module does not stode any date over time.
        self._NeedsLogColumn = True
        self._MonitoredVariables = [('V', np.array),
                                    ('omega', np.array),
                                    ('ExpectedFlow', float),
                                    ('ReceivedFlow', float),
                                    ('ErrorFlow', float),
                                    ('ErrorFlowProbe', float),
                                    ('UsedErrorFlow', float),
                                    ('A', float),
                                    ('XMean', np.array),
                                    ('XSigma', np.array),
                                    ('d2', float),
                                    ('dSigma', float),
                                    ('AFlow', float),
                                    ('AProbe', float),
                                    ('UsedAFlow', float),
                                    ('Det', float)]

        self._EnableEventTau = True

        self._DisparityRange = [0, np.inf]
        self._DefaultK = 5e2
        self._StereoBaseVector = np.array([0.1, 0., 0.])

        self._DelayStart = 0.

        self._MinDetRatioToSolve = np.inf

        self._ValidDisparityTauRatio = 1.
        self._ValidFlowTauRatio = 1.

        self._RejectMaxErrorRatio = 1.
        self._MinActivity = 20.

        self._Tau = 0.05
        self._FrameworkTauRatio = 2

        self._FlowErrorProbeTauRatio = 10.

        self._SecurityProbeRatioOn = 3.
        self._SecurityProbeRatioOff = 1.5
        self._SecurityModeTauRatio = 10.
        self._SecurityModeSendOdometry = True

    def _OnInitialization(self):
        self.ScreenSize = np.array(self.Geometry)
        self.ScreenCenter = self.ScreenSize / 2 - 0.5

        self.FlowMap = np.zeros(tuple(self.ScreenSize) + (3,), dtype = float) # fx, fy, t
        self.DisparityMap = np.zeros(tuple(self.ScreenSize) + (2,), dtype = float) # d, t
        self.FlowMap[:,:,2] = -np.inf
        self.DisparityMap[:,:,1] = -np.inf

        self.K = self._DefaultK
        self.StereoBaseDistance = np.linalg.norm(self._StereoBaseVector)

        if self._DisparityRange[1] == np.inf:
            self._DisparityRange[1] = self.ScreenSize[0]

        self.NInput = 0.
        self.NValid = 0.

        self.LastT = -np.inf
        self.LastUsedT = -np.inf
        self.MSum = np.zeros((6,6))
        self.SigmaSum = np.zeros(6)
        self.A = 0.
        self.AW = 0.
        self.ExpectedFlowSum = 0.
        self.ReceivedFlowSum = 0.
        self.ErrorFlowSum = 0.
        self.UsedErrorFlowSum = 0.
        self.AFlow = 0.
        self.UsedAFlow = 0.
        self.FoundSolution = False

        self.XMeanSum = np.zeros(2, dtype = float)
        self.XSigma2Sum = np.zeros(2, dtype = float)
        self.d2Sum = 0.
        self.dMeanSum = 0.
        self.dSigmaSum = 0.

        self.AProbe = 0.
        self.ErrorFlowProbeSum = 0.

        self.SecurityOnLogValue = np.log10(self._SecurityProbeRatioOn)
        self.SecurityOffLogValue = np.log10(self._SecurityProbeRatioOff)

        self.SecurityMode = False

        if self._DelayStart == 0:
            self.Started = True
        else:
            self.Started = False
            self.tFirstEvents = np.array([np.inf, np.inf], dtype = float)

        self.OmegaToMotionMatrix = np.array([[0.          , 0., 0.          , -1., 0.          , 0.],
                                             [0.          , -1., 0.          , 0. , 0.          , 0.],
                                             [0.          , 0., 0.          , 0. , 0.          , -1.],
                                             [-self.StereoBaseDistance, 0., 0.          , 0. , 0.          , 0.],
                                             [0.          , 0., self.StereoBaseDistance, 0. , 0.          , 0.],
                                             [0.          , 0., 0.          , 0. , self.StereoBaseDistance, 0.]])
        self.omegaMatrix = np.array([[0., -1., 0.],
                                     [-1., 0. , 0.],
                                     [0., 0. , -1.]])

        return True

    def _OnEventModule(self, event):
        IsValidEvent = False
        if event.Has(FlowEvent):
            if not self.Started:
                if self.tFirstEvents[0] == np.inf:
                    self.tFirstEvents[0] = event.timestamp
                if (event.timestamp - self.tFirstEvents >= self._DelayStart).all():
                    self.Started = True
                    self.LogSuccess("Started")
                    del self.__dict__['tFirstEvents']
                else:
                    return
            self.FlowMap[event.location[0], event.location[1], :] = np.array([event.flow[0], event.flow[1], event.timestamp])
            if event.timestamp - self.DisparityMap[event.location[0], event.location[1], 1] < self.Tau*self._ValidDisparityTauRatio:
                IsValidEvent = True
        if event.Has(DisparityEvent): # We assume here that there is a CameraEvent, since we have a Disparity event.
            if not self.Started:
                if self.tFirstEvents[1] == np.inf:
                    self.tFirstEvents[1] = event.timestamp
                if (event.timestamp - self.tFirstEvents >= self._DelayStart).all():
                    self.Started = True
                    self.LogSuccess("Started")
                    del self.__dict__['tFirstEvents']
                else:
                    return
            self.DisparityMap[event.location[0], event.location[1], :] = np.array([event.disparity, event.timestamp])
            if event.timestamp - self.FlowMap[event.location[0], event.location[1], 2] < self.Tau*self._ValidFlowTauRatio:
                IsValidEvent = True
        
        if IsValidEvent:
            f = self.FlowMap[event.location[0], event.location[1], :2]
            disparity = self.DisparityMap[event.location[0], event.location[1], 0]
            self.Update(event.timestamp, event.location, f, disparity)
            if self.A >= self._MinActivity and (not self.SecurityMode or self._SecurityModeSendOdometry):
                event.Attach(TwistEvent, v = self.V, omega = self.omega)
        return

    def Update(self, t, location, f, disparity):
        Tau = self.Tau
        decay = np.e**((self.LastT - t)/(Tau/(1+self.SecurityMode*(self._SecurityModeTauRatio-1))))
        decayProbe = np.e**((self.LastT - t)/(Tau/self._FlowErrorProbeTauRatio))
        self.LastT = t


        Nf = np.linalg.norm(f)
        nx, ny = f / Nf
        dx, dy = (location - self.ScreenCenter) # Make it centered
        d = disparity

        P_i = self.P(dx, dy, nx, ny, d)

        ExpectedFlowNorm = abs(P_i.dot(self.Omega))
        self.ExpectedFlowSum = self.ExpectedFlowSum * decay + ExpectedFlowNorm
        self.ReceivedFlowSum = self.ReceivedFlowSum * decay + Nf
        Error = abs(ExpectedFlowNorm - Nf)
        self.ErrorFlowSum = self.ErrorFlowSum * decay + Error
        self.AFlow = self.AFlow * decay + 1.

        self.AProbe = self.AProbe * decayProbe + 1.
        self.ErrorFlowProbeSum = self.ErrorFlowProbeSum * decayProbe + Error


        if self.FoundSolution and Error > self.ErrorFlow * self._RejectMaxErrorRatio:
            self.UpdateSecurityMode()
            return

        usedDecay = np.e**((self.LastUsedT - t)/(Tau/(1+self.SecurityMode*(self._SecurityModeTauRatio-1))))
        self.LastUsedT = t
        self.UsedErrorFlowSum = self.UsedErrorFlowSum * usedDecay + Error
        self.UsedAFlow = self.UsedAFlow * usedDecay + 1.

        M_i = np.zeros((6,6))
        for k in range(6):
            M_i[:,k] = P_i*P_i[k]
        Sigma_i = P_i * Nf

        self.MSum = self.MSum * usedDecay + M_i
        self.SigmaSum = self.SigmaSum * usedDecay + Sigma_i

        self.A = self.A * usedDecay + 1.
        self.XMeanSum = self.XMeanSum * usedDecay + location
        self.XSigma2Sum = self.XSigma2Sum * usedDecay + (self.XMean - location)**2

        self.d2Sum = self.d2Sum * usedDecay + d**2
        self.dMeanSum = self.dMeanSum * usedDecay + d
        self.dSigmaSum = self.dSigmaSum * usedDecay + (self.dMean - d)**2

        self.UpdateSecurityMode()

    def P(self, dx, dy, nx, ny, d):
        return np.array([nx*d,
            nx*self.K + (nx*dx**2 + ny*dx*dy)/self.K,
            ny*d,
            ny*self.K + (ny*dy**2 + nx*dx*dy)/self.K,
            d*(nx*dx + ny*dy)/self.K,
            (nx*dy - ny*dx)])

    def Q(self, dx, dy, d):
        return np.array([[d, self.K + dx**2/self.K, 0., dx*dy/self.K, dx*d/self.K, dy],
                         [0., dx*dy/self.K, d, self.K + dy**2/self.K, dy*d/self.K, -dx]])
    
    def ExpectedVelocity(self, dx, dy, d):
        return self.Q(dx, dy, d).dot(self.Omega)

    def UpdateSecurityMode(self):
        ErrorFlow = self.ErrorFlow
        if ErrorFlow == 0:
            return 
        ErrorFlowProbe = self.ErrorFlowProbe
        if ErrorFlowProbe == 0:
            return
        V = abs(np.log10(ErrorFlow / ErrorFlowProbe)) 
        if not self.SecurityMode:
            if V > self.SecurityOnLogValue:
                self.SecurityMode = True
                self.LogWarning("Security mode is on.")
                return
        else:
            if V < self.SecurityOffLogValue:
                self.SecurityMode = False
                self.LogSuccess("Security mode is off.")
                return

    def EventTau(self, event = None):
        if not self._EnableEventTau:
            return 0
        if event is None or not event.Has(CameraEvent):
            dx, dy = self.ScreenCenter
        else:
            dx, dy = np.array(event.location) - self.ScreenCenter
        if event is None or not event.Has(DisparityEvent):
            xs, ys = np.where(self.DisparityMap[:,:,1] > self.LastT - self.Tau*self._ValidDisparityTauRatio)
            ds = self.DisparityMap[xs,ys,0]
            if ds.size != 0:
                d = ds.mean()
            else:
                d = (self._DisparityRange[0] + self._DisparityRange[1])/2
        else:
            d = event.disparity
        ExpectedVelocity = np.linalg.norm(self.ExpectedVelocity(dx, dy, d))
        if not ExpectedVelocity == 0:
            return 1./ExpectedVelocity
        else:
            return 0

    @property
    def XMean(self):
        return self.XMeanSum / max(0.01, self.A)
    @property
    def XSigma(self):
        return np.sqrt(self.XSigma2Sum / max(0.01, self.A))
    @property
    def d2(self):
        return self.d2Sum / max(0.01, self.A)
    @property
    def dMean(self):
        return self.dMeanSum / max(0.01, self.A)
    @property
    def dSigma(self):
        return np.sqrt(self.dSigmaSum / max(0.01, self.A))
    @property
    def Tau(self):
        if self._FrameworkTauRatio == 0:
            return self._Tau
        Tau = self.FrameworkAverageTau
        if Tau is None:
            return self._Tau
        else:
            return Tau * self._FrameworkTauRatio

    @property
    def M(self):
        return self.MSum / max(0.1, self.A)
    @property
    def Sigma(self):
        return self.SigmaSum / max(0.1, self.A)

    @property
    def Det(self):
        return np.linalg.det(self.M)
    @property
    def DetRatio(self): # Compute maximum det for M and change here
        return self.Det
    @property
    def Omega(self):
        Det = self.DetRatio
        if Det == 0 or (self._MinDetRatioToSolve != np.inf and Det < self._MinDetRatioToSolve):
            return np.zeros(6)
        self.FoundSolution = True
        return np.linalg.inv(self.M).dot(self.Sigma)
    @property
    def Motion(self):
        return self.OmegaToMotionMatrix.dot(self.Omega)
    @property
    def omega(self):
        return self.Motion[:3]
    @property
    def V(self):
        return self.Motion[3:]

    @property
    def ExpectedFlow(self):
        return self.ExpectedFlowSum / max(0.1, self.AFlow)
    @property
    def ReceivedFlow(self):
        return self.ReceivedFlowSum / max(0.1, self.AFlow)
    @property
    def ErrorFlow(self):
        return self.ErrorFlowSum / max(0.1, self.AFlow)
    @property
    def UsedErrorFlow(self):
        return self.UsedErrorFlowSum / max(0.1, self.UsedAFlow)
    @property
    def ErrorFlowProbe(self):
        return self.ErrorFlowProbeSum / max(0.1, self.AProbe)
