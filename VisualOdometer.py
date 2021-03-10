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
                                    ('A', float),
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
        self.SigmaSum = np.zeros(6)
        self.A = 0.
        self.FoundSolution = False

        self.OmegaToMotionMatrix = np.array([[0.          , 0., 0.          , -1., 0.          , 0.],
                                             [0.          , 1., 0.          , 0. , 0.          , 0.],
                                             [0.          , 0., 0.          , 0. , 0.          , 1.],
                                             [-self._Delta, 0., 0.          , 0. , 0.          , 0.],
                                             [0.          , 0., -self._Delta, 0. , 0.          , 0.],
                                             [0.          , 0., 0.          , 0. , -self._Delta, 0.]])

        return True

    def _OnEventModule(self, event):
        IsValidEvent = False
        if event.Has(FlowEvent):
            self.FlowMap[event.location[0], event.location[1], :] = np.array([event.flow[0], event.flow[1], event.timestamp])
            if event.timestamp - self.DisparityMap[event.location[0], event.location[1], 1] < self._Tau:
                IsValidEvent = True
        if event.Has(DisparityEvent):
            self.DisparityMap[event.location[0], event.location[1], :] = np.array([event.disparity, event.timestamp])
            if event.timestamp - self.FlowMap[event.location[0], event.location[1], 2] < self._Tau:
                IsValidEvent = True
        
        if IsValidEvent:
            self.Update(event.timestamp, event.location, self.FlowMap[event.location[0], event.location[1], :2], self.DisparityMap[event.location[0], event.location[1], 0])
            event.Attach(OdometryEvent, v = self.V, omega = self.omega)
        return event

    def Update(self, t, location, f, disparity):
        decay = np.e**((self.LastT - t)/self._Tau)
        self.LastT = t

        Nf = np.linalg.norm(f)
        nx, ny = f / Nf
        x, y = (location - self.ScreenCenter) # Make it centered
        d = disparity

        P_i = np.array([nx*d, 
                      nx*self.K + (nx*x**2 + ny*x*y)/self.K,
                      ny*d,
                      ny*self.K + (ny*y**2 + nx*x*y)/self.K,
                      d*(nx*x + ny*y)/self.K,
                      (nx*y - ny*x)])
        M_i = np.zeros((6,6))
        for k in range(6):
            M_i[:,k] = P_i*P_i[k]
        Sigma_i = P_i * Nf

        self.MSum = self.MSum * decay + M_i
        self.SigmaSum = self.SigmaSum * decay + Sigma_i
        self.A = self.A * decay + 1.

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
