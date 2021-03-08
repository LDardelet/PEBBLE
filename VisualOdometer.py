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
        self._MonitoredVariables = [('VOmega', np.array),
                                    ('omegaOmega', np.array),
                                    ('VGamma', np.array),
                                    ('omegaGamma', np.array),
                                    ('KGamma', float),
                                    ('N', float),
                                    ('DetOmega', float),
                                    ('DetGamma', float)]

        self._DisparityRange = [0, np.inf]
        self._DefaultK = 5e2
        self._Delta = 0.1
        self._SolveK = True

        self._MinDetOmegaToSolve = 5e-7
        self._MinDetGammaToSolve = 5e-7
        self._Tau = 0.05

        self._d2Correction = 0.5**2 / 3

    def _InitializeModule(self, **kwargs):
        self.ScreenSize = np.array(self.Geometry[:2])
        self.ScreenCenter = self.ScreenSize / 2

        self.FlowMap = np.zeros(tuple(self.ScreenSize) + (3,), dtype = float) # fx, fy, t
        self.DisparityMap = np.zeros(tuple(self.ScreenSize) + (2,), dtype = float) # d, t
        self.FlowMap[:,:,2] = -np.inf
        self.DisparityMap[:,:,1] = -np.inf

        self.K = self._DefaultK

        if self._DisparityRange[1] == np.inf:
            self._DisparityRange[1] = self.ScreenSize[0]

        self.N = 0.

        self.LastT = -np.inf
        self.InitComprehension()
        self.FoundSolution = False

        return True

    def _OnEventModule(self, event):
        HasValidPoint = False
        if event.Has(FlowEvent):
            self.FlowMap[event.location[0], event.location[1], :] = np.array([event.flow[0], event.flow[1], event.timestamp])
            if event.timestamp - self.DisparityMap[event.location[0], event.location[1], 1] < self._Tau:
                HasValidPoint = True
        if event.Has(DisparityEvent):
            self.DisparityMap[event.location[0], event.location[1], :] = np.array([event.disparity, event.timestamp])
            if event.timestamp - self.FlowMap[event.location[0], event.location[1], 2] < self._Tau:
                HasValidPoint = True
        
        if HasValidPoint:
            self.Update(event.timestamp, event.location, self.FlowMap[event.location[0], event.location[1], :2], self.DisparityMap[event.location[0], event.location[1], 0])
            event.Attach(OdometryEvent, v = self.VOmega, omega = self.omegaOmega)
        return event

    def Update(self, t, location, f, disparity):

        X = (location - self.ScreenCenter) + 0.5 # Make it centered
        d = disparity

        decay = np.e**((self.LastT - t)/self._Tau)
        self.LastT = t

        self.N = self.N * decay + 1
        for Sum in self.Terms.values():
            Sum.AddData(X, f, d, decay)
        if not self.FoundSolution and self.DetOmegaRatio >= self._MinDetOmegaRatioToSolve:
            self.LogSuccess("Found a motion solution")
            self.FoundSolution = True

    def IsDataPointValid(self, X, f, d, decay):

        self.NInput = self.NInput*decay + 1
        if (self._DataRejectionRatio == 0) or (self.DetOmegaRatio < self._MinDetOmegaRatioToSolve):
            self.NValid = self.NValid*decay + 1
            return True # If we dont have a solution, each data point is to be used

        Nf = np.linalg.norm(f)
        Nf_expected = self.ExpectedFlowNorm(X, d, f/Nf)
        Error = Nf - Nf_expected
        Delta2 = (Error)**2
        Valid = (self._DataRejectionRatio * abs(Error) < np.sqrt(self.AverageNormDelta2/self.NValid)) or (self.NValid * self._MinValidRatio < self.NInput-1)

        if Valid:
            self.AverageNormDelta2 = self.AverageNormDelta2*decay + Delta2
            self.NValid = self.NValid*decay + 1

        self.NInput = self.NInput*decay + 1

        return Valid

    @property
    def MOmega(self):
        M = np.zeros((6,6))
        for nLine, Line in enumerate(self.MOmegaComp):
            for nRow, Terms in enumerate(Line):
                for Name, Multiplier in Terms.items():
                    M[nLine, nRow] += Multiplier * self.Terms[Name].Value
        return M
    @property
    def DetOmega(self):
        return np.linalg.det(self.MOmega)
    @property
    def MaxDetOmega(self):
        x_max2 = (self.ScreenSize[0]/2)**2
        y_max2 = (self.ScreenSize[1]/2)**2
        return self.Terms['Rd'].Value / 576 * (self.Terms['Rd'].Value**2 - self.Terms['Rd2'].Value)**2 * (self.K + x_max2 / (3*self.K)) * (self.K + y_max2 / (3*self.K)) * (x_max2 + y_max2)**2
    @property
    def DetOmegaRatio(self):
        return self.DetOmega / max(0.1, self.MaxDetOmega)
    @property
    def MGamma(self):
        M = np.zeros((8,8))
        for nLine, Line in enumerate(self.MGammaComp):
            for nRow, Terms in enumerate(Line):
                for Name, Multiplier in Terms.items():
                    M[nLine, nRow] += Multiplier * self.Terms[Name].Value
        return M
    @property
    def DetGamma(self):
        return np.linalg.det(self.MGamma)
    @property
    def MaxDetGamma(self):
        x_max2 = (self.ScreenSize[0]/2)**2
        y_max2 = (self.ScreenSize[1]/2)**2
        return self.Terms['Rd'].Value * (self.Terms['Rd'].Value**2 - self.Terms['Rd2'].Value)**2 / 16 * x_max2**2 * y_max2**2 / 324 * (x_max2 + y_max2)**2 / 36
    @property
    def DetGammaRatio(self):
        return self.DetGamma / max(0.1, self.MaxDetGamma)
    @property
    def SigmaOmega(self):
        Sigma = np.zeros(6)
        for nRow, Name in enumerate(self.SigmaOmegaComp):
            Sigma[nRow] = self.Terms[Name].Value
        return Sigma
    @property
    def SigmaGamma(self):
        Sigma = np.zeros(8)
        for nRow, Name in enumerate(self.SigmaGammaComp):
            Sigma[nRow] = self.Terms[Name].Value
        return Sigma

    @property
    def Omega(self):
        M = self.MOmega
        if self.DetOmegaRatio < self._MinDetOmegaRatioToSolve:
            return np.zeros(6)
        return np.linalg.inv(M).dot(self.SigmaOmega)
    @property
    def MotionOmega(self):
        Omega = self.Omega
        return np.array([-Omega[3], Omega[1], Omega[5], -self._Delta*Omega[0], -self._Delta*Omega[2], -self._Delta*Omega[4]])
    @property
    def VOmega(self):
        return self.MotionOmega[:3]
    @property
    def omegaOmega(self):
        return self.MotionOmega[3:]
    @property
    def MotionGamma(self):
        M = self.MGamma
        if self.DetGammaRatio < self._MinDetGammaRatioToSolve:
            return np.zeros(6)
        Gamma = np.linalg.inv(M).dot(self.SigmaGamma)
        K2 = (Gamma[1] + Gamma[3])/(Gamma[6] + Gamma[7])
        if K2 < 0:
            return np.zeros(6)
        K = np.sqrt(K2)
        return np.array([-Gamma[7]*K, Gamma[6]*K, Gamma[5], -self._Delta*Gamma[0], -self._Delta*Gamma[2], -self._Delta*Gamma[4]])
    @property
    def VGamma(self):
        return self.MotionGamma[:3]
    @property
    def omegaGamma(self):
        return self.MotionGamma[3:]
    @property
    def KGamma(self):
        M = self.MGamma
        if self.DetGammaRatio < self._MinDetGammaRatioToSolve:
            return 0.
        Gamma = np.linalg.inv(M).dot(self.SigmaGamma)
        K2 = (Gamma[1] + Gamma[3])/(Gamma[6] + Gamma[7])
        if K2 <= 0:
            return 0
        return np.sqrt(K2)

    def ExpectedFlowNorm(self, X, d, n):
        if self.DetOmegaRatio < self._MinDetOmegaRatioToSolve:
            return None
        x, y = X - self.ScreenCenter + 0.5
        return np.array([n[0]*d, n[0]*self.K + (n[0]*x**2 + n[1]*x*y)/self.K, n[1]*d, n[1]*self.K + (n[0]*x*y + n[1]*y**2)/self.K, (n[0]*x + n[1]*y)*d/self.K, n[0]*y-n[1]*x]).dot(self.Omega)

    def InitComprehension(self):
        self.Terms = {'Rx':SummationClass('Rx'), 'Ry':SummationClass('Ry'), 'Rnx':SummationClass('Rnx'), 'Rny':SummationClass('Rny'), 'Rd':SummationClass('Rd'), 'Rd2':SummationClass('Rd2')}
        self.MGammaComp = []
        self.SigmaGammaComp = []
        self._AddGammaEquality("Sfx", [{'Snx2d':1},
            {'Snx2':1},
            {'Rnxnyd':1},
            {'Rnxny':1},
            {'Rnx2xd':1, 'Rnxnyyd':1},
            {'Rnx2y':1, 'Rnxnyx':-1},
            {'Snx2x2':1, 'Rnxnyxy':1},
            {'Rnx2xy':1, 'Rnxnyy2':1}])
        self._AddGammaEquality("Sfxd", [{'Snx2d2':1},
            {'Snx2d':1},
            {'Rnxnyd2':1},
            {'Rnxnyd':1},
            {'Rnx2xd2':1, 'Rnxnyyd2':1},
            {'Rnx2yd':1, 'Rnxnyxd':-1},
            {'Snx2x2d':1, 'Rnxnyxyd':1},
            {'Rnx2xyd':1, 'Rnxnyy2d':1}])
        self._AddGammaEquality("Sfy", [{'Rnxnyd':1},
            {'Rnxny':1},
            {'Sny2d':1},
            {'Sny2':1},
            {'Rnxnyxd':1,'Rny2yd':1},
            {'Rnxnyy':1,'Rny2x':-1},
            {'Rnxnyx2':1, 'Rny2xy':1},
            {'Rnxnyxy':1, 'Sny2y2':1}])
        self._AddGammaEquality("Sfyd", [{'Rnxnyd2':1},
            {'Rnxnyd':1},
            {'Sny2d2':1},
            {'Sny2d':1},
            {'Rnxnyxd2':1,'Rny2yd2':1},
            {'Rnxnyyd':1,'Rny2xd':-1},
            {'Rnxnyx2d':1, 'Rny2xyd':1},
            {'Rnxnyxyd':1, 'Sny2y2d':1}])
        self._AddGammaEquality("Sfyxy", [{'Rnxnyxyd':1},
            {'Rnxnyxy':1},
            {'Rny2xyd':1},
            {'Rny2xy':1},
            {'Rnxnyx2yd':1,'Rny2xy2d':1},
            {'Rnxnyxy2':1,'Rny2x2y':-1},
            {'Rnxnyx3y':1, 'Sny2x2y2':1},
            {'Rnxnyx2y2':1, 'Rny2xy3':1}])
        self._AddGammaEquality("Sfxxy", [{'Rnx2xyd':1},
            {'Rnx2xy':1},
            {'Rnxnyxyd':1},
            {'Rnxnyxy':1},
            {'Rnx2x2yd':1,'Rnxnyxy2d':1},
            {'Rnx2xy2':1,'Rnxnyx2y':-1},
            {'Rnx2x3y':1, 'Rnxnyx2y2':1},
            {'Snx2x2y2':1, 'Rnxnyxy3':1}])
        self._AddGammaEquality("Sfxx+fyy", [{"Rnx2xd":1, "Rnxnyyd":1}, 
            {"Rnx2x":1, "Rnxnyy":1},
            {"Rnxnyxd":1, "Rny2yd":1},
            {"Rnxnyx":1, "Rny2y":1},
            {"Snx2x2d":1, "Sny2y2d": 1, "Rnxnyxyd":2},
            {"Rnx2xy":1, "Rnxnyy2":1, "Rnxnyx2":-1, "Rny2xy":-1},
            {"Rnx2x3":1, "Rnxnyx2y":2, "Rny2xy2":1},
            {"Rnx2x2y":1, "Rnxnyxy2":2, "Rny2y3":1}])
        self._AddGammaEquality("Sfxy-fyx", [{'Rnx2yd':1, 'Rnxnyxd':-1},
            {'Rnx2y':1, 'Rnxnyx':-1},
            {'Rnxnyyd':1, 'Rny2xd':-1},
            {'Rnxnyy':1, 'Rny2x':-1}, 
            {'Rnx2xyd':1, 'Rnxnyy2d':1, 'Rnxnyx2d':-1, 'Rny2xyd':-1},
            {'Snx2y2':1, 'Sny2x2':1, 'Rnxnyxy':-2},
            {'Rnx2x2y':1, 'Rnxnyxy2':1, 'Rnxnyx3':-1, 'Rny2x2y':-1},
            {'Rnx2xy2':1, 'Rnxnyy3':1, 'Rnxnyx2y':-1, 'Rny2xy2':-1}])
        self.MOmegaComp = []
        self.SigmaOmegaComp = []
        self._AddOmegaEquality("Sfx", [{'Snx2d':1},
            {'Snx2':self.K, 'Snx2x2':1/self.K, 'Rnxnyxy':1/self.K},
            {'Rnxnyd':1},
            {'Rnxny':self.K, 'Rnx2xy':1/self.K, 'Rnxnyy2':1/self.K},
            {'Rnx2xd':1/self.K, 'Rnxnyyd':1/self.K},
            {'Rnx2y':1, 'Rnxnyx':-1}])
        self._AddOmegaEquality("Sfxd", [{'Snx2d2':1},
            {'Snx2d':self.K,'Snx2x2d':1/self.K,'Rnxnyxyd':1/self.K},
            {'Rnxnyd2':1},
            {'Rnxnyd':self.K,'Rnx2xyd':1/self.K,'Rnxnyy2d':1/self.K},
            {'Rnx2xd2':1/self.K,'Rnxnyyd2':1/self.K},
            {'Rnx2yd':1,'Rnxnyxd':-1}])
        self._AddOmegaEquality("Sfy", [{'Rnxnyd':1},
            {'Rnxny':self.K,'Rnxny':1/self.K,'Rny2xy':1/self.K},
            {'Sny2d':1},
            {'Sny2':self.K,'Sny2y2':1/self.K,'Rnxnyxy':1/self.K},
            {'Rnxnyxd':1/self.K,'Rny2yd':1/self.K},
            {'Rnxnyy':1,'Rny2x':-1}])
        self._AddOmegaEquality("Sfyd", [{'Rnxnyd2':1},
            {'Rnxnyd':self.K,'Rnxnyd':1/self.K,'Rny2xyd':1/self.K},
            {'Sny2d2':1},
            {'Sny2d':self.K,'Sny2y2d':1/self.K,'Rnxnyxyd':1/self.K},
            {'Rnxnyxd2':1/self.K,'Rny2yd2':1/self.K},
            {'Rnxnyyd':1,'Rny2xd':-1}])
        self._AddOmegaEquality("Sfxx+fyy", [{"Rnx2xd":1, "Rnxnyyd":1},
            {"Rnx2x":self.K, "Rnxnyy":self.K, "Rnx2x3":1/self.K, "Rnxnyx2y": 2/self.K, "Rny2xy2":1/self.K},
            {"Rnxnyxd":1, "Rny2yd":1},
            {"Rnxnyx":self.K, "Rny2y":self.K, "Rnx2x2y":1/self.K, "Rnxnyxy2":2/self.K, "Rny2y3":1/self.K},
            {"Snx2x2d":1/self.K, "Sny2y2d": 1/self.K, "Rnxnyxyd":2/self.K},
            {"Rnx2xy":1, "Rnxnyy2":1, "Rnxnyx2":-1, "Rny2xy":-1}])
        self._AddOmegaEquality("Sfxy-fyx", [{'Rnx2yd':1, 'Rnxnyxd':-1},
            {'Rnx2y':self.K, 'Rnxnyx':-self.K, 'Rnx2x2y':1/self.K, 'Rnxnyxy2':1/self.K, 'Rnxnyx3':-1/self.K, 'Rny2x2y':-1/self.K},
            {'Rnxnyyd':1, 'Rny2xd':-1},
            {'Rnxnyy':self.K, 'Rny2x':-self.K, 'Rnx2xy2':1/self.K, 'Rnxnyy3':1/self.K, 'Rnxnyx2y':-1/self.K, 'Rny2xy2':-1/self.K},
            {'Rnx2xyd':1/self.K, 'Rnxnyy2d':1/self.K, 'Rnxnyx2d':-1/self.K, 'Rny2xyd':-1/self.K},
            {'Snx2y2':1, 'Sny2x2':1, 'Rnxnyxy':-2}])

        self._d2Compensate()

    def _d2Compensate(self):
        if not self._d2Correction:
            return
        for M in [self.MOmegaComp, self.MGammaComp]:
            for Row in M:
                for Cell in Row:
                    for Term in list(Cell.keys()):
                        if 'd2' in Term:
                            NewTerm = Term.replace('d2', '')
                            if not NewTerm in self.Terms:
                                self.Terms[NewTerm] = SummationClass(NewTerm)
                            Cell[NewTerm] = -Cell[Term] * self._d2Correction
        
    def _AddOmegaEquality(self, SigmaTerm, MatrixTerm):
        self.Terms[SigmaTerm] = SummationClass(SigmaTerm)
        for Expression in MatrixTerm:
            for Term in Expression.keys():
                if not Term in self.Terms:
                    self.Terms[Term] = SummationClass(Term)
        self.SigmaOmegaComp += [SigmaTerm]
        self.MOmegaComp += [MatrixTerm]

    def _AddGammaEquality(self, SigmaTerm, MatrixTerm):
        self.Terms[SigmaTerm] = SummationClass(SigmaTerm)
        for Expression in MatrixTerm:
            for Term in Expression.keys():
                if not Term in self.Terms:
                    self.Terms[Term] = SummationClass(Term)
        self.SigmaGammaComp += [SigmaTerm]
        self.MGammaComp += [MatrixTerm]

class SummationClass:
    # Input data will be the array (x, y, d, fx, fy, nx, ny)
    def __init__(self, Name):
        return self.Value

    def AddData(self, X, f, d, Lambda):
        nf = np.linalg.norm(f)
        nx, ny = f / nf
        x, y = X
        self.S *= Lambda
        for Exponents, Sign in self.Exponents:
            self.S += Sign * (np.array([x, y, d, f[0], f[1], nx, ny])**Exponents).prod()
        self.A = self.A * Lambda + 1

    def __repr__(self):
        return str(self.Value)
