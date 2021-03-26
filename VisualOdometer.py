from PEBBLE import Module, Event, DisparityEvent, FlowEvent, OdometryEvent
import numpy as np

class VisualOdometer(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Uses disparity events and optical flow events to recover a visual odometry
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Computation'

        self.__ReferencesAsked__ = []
        self._MonitorDt = 0. # By default, a module does not stode any date over time.
        self._NeedsLogColumn = True
        self._MonitoredVariables = [('V', np.array),
                                    ('omega', np.array),
                                    ('dV', np.array),
                                    ('domega', np.array),
                                    ('PureOmega', np.array),
                                    ('ExpectedFlow', float),
                                    ('ReceivedFlow', float),
                                    ('ErrorFlow', float),
                                    ('A', float),
                                    ('AFlow', float),
                                    ('Det', float)]

        self._DisparityRange = [0, np.inf]
        self._DefaultK = 5e2
        self._Delta = 0.1

        self._MinDetRatioToSolve = np.inf

        self._DataRejectionRatio = 0
        self._MinValidRatio = 2

        self._Tau = 0.05

    def _InitializeModule(self, **kwargs):
        self.ScreenSize = np.array(self.Geometry[:2])
        self.ScreenCenter = self.ScreenSize / 2 - 0.5

        self.FlowMap = np.zeros(tuple(self.ScreenSize) + (3,), dtype = float) # fx, fy, t
        self.DisparityMap = np.zeros(tuple(self.ScreenSize) + (2,), dtype = float) # d, t
        self.FlowMap[:,:,2] = -np.inf
        self.DisparityMap[:,:,1] = -np.inf

        self.K = self._DefaultK

        if self._DisparityRange[1] == np.inf:
            self._DisparityRange[1] = self.ScreenSize[0]

        self.NInput = 0.
        self.NValid = 0.

        self.LastT = -np.inf
        self.MSum = np.zeros((6,6))
        self.dMSum = np.zeros((6,6))
        self.MWSum = np.zeros((3,3))
        self.SigmaSum = np.zeros(6)
        self.SigmaWSum = np.zeros(3)
        self.A = 0.
        self.AW = 0.
        self.ExpectedFlowSum = 0.
        self.ReceivedFlowSum = 0.
        self.ErrorFlowSum = 0.
        self.AFlow = 0.
        self.FoundSolution = False

        self.OmegaToMotionMatrix = np.array([[0.          , 0., 0.          , -1., 0.          , 0.],
                                             [0.          , 1., 0.          , 0. , 0.          , 0.],
                                             [0.          , 0., 0.          , 0. , 0.          , 1.],
                                             [-self._Delta, 0., 0.          , 0. , 0.          , 0.],
                                             [0.          , 0., -self._Delta, 0. , 0.          , 0.],
                                             [0.          , 0., 0.          , 0. , -self._Delta, 0.]])
        self.omegaMatrix = np.array([[0., -1., 0.],
                                     [1., 0. , 0.],
                                     [0., 0. , 1.]])

        return True

    def _OnEventModule(self, event):
        IsValidEvent = False
        if event.Has(FlowEvent):
            self.FlowMap[event.location[0], event.location[1], :] = np.array([event.flow[0], event.flow[1], event.timestamp])
            self.UpdateW(event.timestamp, event.location, self.FlowMap[event.location[0], event.location[1], :2])
            #event.Attach(OdometryEvent, v = self.V*0, omega = self.PureOmega)
            if event.timestamp - self.DisparityMap[event.location[0], event.location[1], 1] < self._Tau:
                IsValidEvent = True
        if event.Has(DisparityEvent):
            self.DisparityMap[event.location[0], event.location[1], :] = np.array([event.disparity, event.timestamp])
            if event.timestamp - self.FlowMap[event.location[0], event.location[1], 2] < self._Tau:
                IsValidEvent = True
        
        if IsValidEvent:
            f = self.FlowMap[event.location[0], event.location[1], :2]
            disparity = self.DisparityMap[event.location[0], event.location[1], 0]
            self.Update(event.timestamp, event.location, f, disparity)
            event.Attach(OdometryEvent, v = self.V, omega = self.omega)
        return event

    def Update(self, t, location, f, disparity):
        decay = np.e**((self.LastT - t)/self._Tau)
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
        if self.FoundSolution and Error > self.ErrorFlow/2:
            return

        dP_i = np.array([nx,
                         0,
                         ny,
                         0,
                         (nx*dx + ny*dy)/self.K,
                         0])
        M_i = np.zeros((6,6))
        dM_i = np.zeros((6,6))
        for k in range(6):
            M_i[:,k] = P_i*P_i[k]
            dM_i[:,k] = P_i*dP_i[k] + dP_i*P_i[k]
        Sigma_i = P_i * Nf

        self.MSum = self.MSum * decay + M_i
        self.dMSum = self.dMSum * decay + dM_i
        self.SigmaSum = self.SigmaSum * decay + Sigma_i
        self.A = self.A * decay + 1.

    def P(self, dx, dy, nx, ny, d):
        return np.array([nx*d,
            nx*self.K + (nx*dx**2 + ny*dx*dy)/self.K,
            ny*d,
            ny*self.K + (ny*dy**2 + nx*dx*dy)/self.K,
            d*(nx*dx + ny*dy)/self.K,
            (nx*dy - ny*dx)])

    def UpdateW(self, t, location, f):
        decay = np.e**((self.LastT - t)/self._Tau)
        self.LastT = t

        Nf = np.linalg.norm(f)
        nx, ny = f / Nf
        x, y = (location - self.ScreenCenter) # Make it centered

        Q_i = np.array([nx*self.K + (nx*x**2 + ny*x*y)/self.K,
                      ny*self.K + (ny*y**2 + nx*x*y)/self.K,
                      (nx*y - ny*x)])
        MW_i = np.zeros((3,3))
        for k in range(3):
            MW_i[:,k] = Q_i*Q_i[k]
        SigmaW_i = Q_i * Nf

        self.MWSum = self.MWSum * decay + MW_i
        self.SigmaWSum = self.SigmaWSum * decay + SigmaW_i
        self.AW = self.AW * decay + 1.

    @property
    def M(self):
        return self.MSum / max(0.1, self.A)
    @property
    def dM(self):
        return self.dMSum / max(0.1, self.A)
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
    def dOmega(self):
        Det = self.DetRatio
        if Det == 0 or (self._MinDetRatioToSolve != np.inf and Det < self._MinDetRatioToSolve):
            return np.zeros(6)
        Minv = np.linalg.inv(self.M)
        return Minv.dot(self.dM.dot(Minv.dot(self.Sigma)))
    @property
    def Motion(self):
        return self.OmegaToMotionMatrix.dot(self.Omega)
    @property
    def dMotion(self):
        return abs(self.OmegaToMotionMatrix.dot(self.dOmega))
    @property
    def omega(self):
        return self.Motion[:3]
    @property
    def V(self):
        return self.Motion[3:]
    @property
    def domega(self):
        return self.dMotion[:3]
    @property
    def dV(self):
        return self.dMotion[3:]
    @property
    def MW(self):
        return self.MWSum / max(0.1, self.AW)
    @property
    def SigmaW(self):
        return self.SigmaWSum /  max(0.1, self.AW)
    @property
    def PureOmega(self):
        M = self.MW
        if np.linalg.det(M) == 0:
            return np.zeros(3)
        return self.omegaMatrix.dot(np.linalg.inv(M).dot(self.SigmaW))

    @property
    def ExpectedFlow(self):
        return self.ExpectedFlowSum / max(0.1, self.AFlow)
    @property
    def ReceivedFlow(self):
        return self.ReceivedFlowSum / max(0.1, self.AFlow)
    @property
    def ErrorFlow(self):
        return self.ErrorFlowSum / max(0.1, self.AFlow)

class Quat:
    def __init__(self, w, x=None, y=None, z=None):
        if x is None:
            w, x, y, z = w
        elif y is None:
            x, y, z = x
        self.q = np.array([w, x, y, z])
        self.q /= np.linalg.norm(self.q)
    @property
    def conj(self):
        return self.__class__(self.q * np.array([1, -1, -1, -1]))
    @property
    def w(self):
        return self.q[0]
    @property
    def v(self):
        return self.q[1:]
    def __repr__(self):
        return str(self.q)
    def __mul__(self, rhs):
        if type(rhs) == self.__class__:
            return self.__class__(self.w*rhs.w - self.v.dot(rhs.v), self.w*rhs.v + rhs.w*self.v + np.cross(self.v, rhs.v))
        elif type(rhs) == np.ndarray:
            if rhs.shape[0] != 3:
                raise NotImplemented
            return self.__class__(- self.v.dot(rhs), self.w*rhs + np.cross(self.v, rhs))
        else:
            return self.__class__(self.q * rhs)
    def __rmul__(self, lhs):
        if type(lhs) == self.__class__:
            return self.__class__(self.w*lhs.w - self.v.dot(lhs.v), self.w*lhs.v + lhs.w*self.v - np.cross(self.v, lhs.v))
        elif type(rhs) == np.ndarray:
            if rhs.shape[0] != 3:
                raise NotImplemented
            return self.__class__(- self.v.dot(lhs), self.w*lhs - np.cross(self.v, lhs))
        else:
            return self.__class__(self.q * lhs)
