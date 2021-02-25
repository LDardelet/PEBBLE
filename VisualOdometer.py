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
                                    ('omega', np.array)]

        self._HorizontalHalfAperture = np.pi / 6
        self._DisparityRange = [0, np.inf]

        self._PixelNormalizationExponent = 1
        self._DisparityNormalizationExponent = 1
        self._CompensateDisparityInteger = True

        self._MinDetToSolve = 5e-7
        self._Tau = 0.05

    def _InitializeModule(self, **kwargs):
        self.ScreenSize = np.array(self.Geometry[:2])
        self.ScreenCenter = self.ScreenSize / 2

        self.FlowMap = np.zeros(tuple(self.ScreenSize) + (3,), dtype = float) # fx, fy, t
        self.DisparityMap = np.zeros(tuple(self.ScreenSize) + (2,), dtype = float) # d, t
        self.FlowMap[:,:,2] = -np.inf
        self.DisparityMap[:,:,1] = -np.inf

        self.F = (self.ScreenSize[0] - self.ScreenCenter[0]) / np.tan(self._HorizontalHalfAperture)

        if self._DisparityRange[1] == np.inf:
            self._DisparityRange[1] = self.ScreenSize[0]

        self.PixelNormalization = (self.ScreenSize.min() / 2) ** self._PixelNormalizationExponent
        self.DisparityNormalization = (self._DisparityRange[1]) ** self._DisparityNormalizationExponent
        #self.PixelNormalization = 1.

        if self._CompensateDisparityInteger:
            self.Gamma = -2 * ((0.5**3 / self.DisparityNormalization**3) * 2 / 3)
        else:
            self.Gamma = 0

        self.Terms, self.MComp, self.SigmaComp = _GetEquations(self.Gamma, self.PixelNormalization/self.F)
        self.N = 0.

        self.LastT = -np.inf

        self.FoundSolution = False
        self.MUpdated = False
        self.MInvUpdated = False
        self.SigmaUpdated = False

        self._MStored = np.zeros((6,6))
        self._MInvStored = np.zeros((6,6))
        self._SigmaStored = np.zeros(6)

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
            event.Attach(OdometryEvent, v = self.V, omega = self.omega)
        return event

    def Update(self, t, location, f, disparity):

        X = (location - self.ScreenCenter) / self.PixelNormalization # Make it homogeneous
        d = disparity / self.DisparityNormalization

        decay = np.e**((self.LastT - t)/self._Tau)
        self.LastT = t

        self.N += 1
        for Sum in self.Terms.values():
            Sum.AddData(X, f, d, decay)
        self.MUpdated = True
        self.SigmaUpdated = True
        if not self.FoundSolution and np.linalg.det(self.M) > self._MinDetToSolve:
            self.LogSuccess("Found a motion solution")
            self.FoundSolution = True

    @property
    def M(self):
        if self.MUpdated:
            self._MStored = np.zeros((6,6))
            for nLine, Line in enumerate(self.MComp):
                for nRow, Terms in enumerate(Line):
                    for Name, Multiplier in Terms.items():
                        self._MStored[nLine, nRow] += Multiplier * self.Terms[Name].Value
            self.MUpdated = False
            self.MInvUpdated = True
        return self._MStored
    @property
    def Sigma(self):
        if self.SigmaUpdated:
            self._SigmaStored = np.zeros(6)
            for nRow, Terms in enumerate(self.SigmaComp):
                for Name, Multiplier in Terms.items():
                    self._SigmaStored[nRow] += Multiplier * self.Terms[Name].Value
            self.SigmaUpdated = False
        return self._SigmaStored

    @property
    def MInv(self):
        if self.MInvUpdated:
            self._MInvStored = np.linalg.inv(self.M)
            self.MInvUpdated = False
        return self._MInvStored

    @property
    def Motion(self):
        M = self.M
        if abs(np.linalg.det(M)) < self._MinDetToSolve:
            return np.zeros(3), np.zeros(3)
        Omega = self.MInv.dot(self.Sigma)
        return np.array([-Omega[3]/self.F, Omega[1]/self.F, -Omega[5] / self.PixelNormalization]), np.array([Omega[0]/self.F, Omega[2]/self.F, Omega[4] / self.PixelNormalization])
    @property
    def V(self):
        return self.Motion[1]
    @property
    def omega(self):
        return self.Motion[0]

class SummationClass:
    # Input data will be the array (x, y, d, fx, fy, nx, ny)
    def __init__(self, Name):
        self.Name = Name
        print("Generating term {0}".format(Name))
        if Name[0] == 'S':
            self.Type = "Summation"
        else:
            self.Type = "Residual"
        self.Exponents = [(np.zeros(7), +1)]
        currentIndex = 0
        currentExponent = 0

        for letter in self.Name[1:]:
            if letter == 'f':
                self.Exponents[-1][0][currentIndex] += currentExponent
                currentExponent = 0
                currentIndex = 3
                continue
            elif letter == 'n':
                self.Exponents[-1][0][currentIndex] += currentExponent
                currentExponent = 0
                currentIndex = 5
                continue
            if letter in ['x', 'y', 'd']:
                if currentExponent:
                    self.Exponents[-1][0][currentIndex] += currentExponent
                    currentIndex = 0
                    currentExponent = 0
                currentIndex += int(letter == 'y') + 2 * int(letter == 'd')
                currentExponent += 1
                continue
            if letter == '+':
                self.Exponents[-1][0][currentIndex] += currentExponent
                currentExponent = 0
                currentIndex = 0
                self.Exponents += [(np.zeros(7), +1)]
                continue
            if letter == '-':
                self.Exponents[-1][0][currentIndex] += currentExponent
                currentExponent = 0
                currentIndex = 0
                self.Exponents += [(np.zeros(7), +1)]
                continue
            currentExponent = int(letter)
        self.Exponents[-1][0][currentIndex] += currentExponent

        self.S = 0.
        self.A = 0.

    @property
    def Value(self):
        return self.S / max(0.00001, self.A)

    @property
    def ExpectedValue(self):
        if self.Type == "Residual":
            return 0.
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

def _GetEquations(Gamma, G):
    Terms = {'Rx':SummationClass('Rx'), 'Ry':SummationClass('Ry')}
    MComp = _Comprehend(_MStr, Terms, G)
    _AddCompensation(Terms, 'Snx2d2', 'Snx2', MComp, Gamma)
    _AddCompensation(Terms, 'Sny2d2', 'Sny2', MComp, Gamma)
    _AddCompensation(Terms, 'Rnxnyd2', 'Rnxny', MComp, Gamma)
    _AddCompensation(Terms, 'Rnx2xd2', 'Rnx2x', MComp, Gamma)
    _AddCompensation(Terms, 'Rny2yd2', 'Rny2y', MComp, Gamma)
    _AddCompensation(Terms, 'Rnxnyxd2', 'Rnxnyx', MComp, Gamma)
    _AddCompensation(Terms, 'Rnxnyyd2', 'Rnxnyy', MComp, Gamma)
    SigmaComp = _Comprehend(_SigmaStr, Terms, G)[0]
    return Terms, MComp, SigmaComp

def _AddCompensation(Terms, InitialTerm, CompensationTerm, ComprehendedMatrix, CompensationFactor):
    if CompensationFactor == 0:
        return
    print("Compensating {0} with {1}".format(InitialTerm, CompensationTerm))
    if CompensationTerm not in Terms:
        Terms[CompensationTerm] = SummationClass(CompensationTerm)
    for Line in ComprehendedMatrix:
        for Cell in Line:
            if InitialTerm in Cell:
                InitialFactor = Cell[InitialTerm]
                if not CompensationTerm in Cell:
                    Cell[CompensationTerm] = 0.
                Cell[CompensationTerm] += CompensationFactor * InitialFactor

def _Comprehend(StrData, Terms, G):
    StorageMatrix = []
    numbersAlpha = ''.join([str(n) for n in range(1, 10)])
    Levels = []
    FinishedLevel = None
    for Line in StrData.split("#"):
        StorageMatrix += [[]]
        for Cell in Line.split("&"):
            StorageMatrix[-1] += [{}]
            Cell = Cell.strip("()")
            Multiplier = +1
            Term = ""
            for letter in Cell:
                if letter == "-":
                    if Term:
                        _AddTerm(Terms, Term, Multiplier, StorageMatrix, Levels)
                    Multiplier = -1
                    Term = ""
                    continue
                if letter == "+":
                    if Term:
                        _AddTerm(Terms, Term, Multiplier, StorageMatrix, Levels)
                    Multiplier = +1
                    Term = ""
                    continue
                if letter in numbersAlpha and not Term:
                    if abs(Multiplier) == 1:
                        Multiplier = Multiplier * int(letter)
                    else:
                        Multiplier = 10*Multiplier + int(letter)
                    continue
                if letter == "{":
                    Levels += [(Multiplier, [])]
                    Multiplier = +1
                    continue
                if letter == "}":
                    _AddTerm(Terms, Term, Multiplier, StorageMatrix, Levels)
                    Term = ""
                    Multiplier = +1
                    FinishedLevel = Levels.pop(-1)
                    continue
                if letter == "G":
                    for FinishedTerm in FinishedLevel[1]:
                        print("Correcting term {0} with 1/f^2".format(FinishedTerm))
                        StorageMatrix[-1][-1][FinishedTerm] *= G**2
                    continue
                Term += letter
            if Term:
                _AddTerm(Terms, Term, Multiplier, StorageMatrix, Levels)
    return StorageMatrix

def _AddTerm(Terms, Name, Sign, StorageMatrix, Levels):
    Sign = Sign * int(np.prod([LevelMultiplier for LevelMultiplier, Terms in Levels]))
    if Name not in Terms:
        Terms[Name] = SummationClass(Name)
    if Name in StorageMatrix[-1][-1]:
        print("Increasing multiplier of {0}".format(Name))
        StorageMatrix[-1][-1][Name] += Sign
    else:
        print("Adding multiplier for {0}".format(Name))
        StorageMatrix[-1][-1][Name] = Sign
    if Levels:
        Levels[-1][1].append(Name)

_MRaw = """S_{n_x^2 d} &
            S_{n_x^2} + \frac{S_{n_x^2 x^2} + R_{n_x n_y x y}}{f^2} &
            R_{n_x n_y d} &
            R_{n_x n_y} + \frac{R_{n_x^2 x y} + R_{n_x n_y y^2}}{f^2} &
            R_{n_x^2 x d} + R_{n_x n_y y d} &
            R_{n_x^2 y} - R_{n_x n_y x} \\
        S_{n_x^2 d^2} &
            S_{n_x^2 d} + \frac{S_{n_x^2 x^2 d} + R_{n_x n_y x y d}}{f^2} &
            R_{n_x n_y d^2} &
            R_{n_x n_y d} + \frac{R_{n_x^2 x y d} + R_{n_x n_y y^2 d}}{f^2} &
            R_{n_x^2 x d^2} + R_{n_x n_y y d^2} &
            R_{n_x^2 y d} - R_{n_x n_y x d} \\
        R_{n_x n_y d} &
            R_{n_x n_y} + \frac{R_{n_x n_y} + R_{n_y^2 x y}}{f^2} &
            S_{n_y^2 d} &
            S_{n_y^2} + \frac{S_{n_y^2 y^2} + R_{n_x n_y x y}}{f^2} &
            R_{n_x n_y x d} + R_{n_y^2 y d} &
            R_{n_x n_y y} - R_{n_y^2 x} \\
        R_{n_x n_y d^2} &
            R_{n_x n_y d} + \frac{R_{n_x n_y d} + R_{n_y^2 x y d}}{f^2} &
            S_{n_y^2 d^2} &
            S_{n_y^2 d} + \frac{S_{n_y^2 y^2 d} + R_{n_x n_y x y d}}{f^2} &
            R_{n_x n_y x d^2} + R_{n_y^2 y d^2} &
            R_{n_x n_y y d} - R_{n_y^2 x d} \\
        R_{n_x^2 x d} + R_{n_x n_y y d} &
            R_{n_x^2 x} + R_{n_x n_y y} + \frac{R_{n_x^2 x^3} + 2 R_{n_x n_y x^2 y} + R_{n_y^2 x y^2}}{f^2} &
            R_{n_x n_y x d} + R_{n_y^2 y d} &
            R_{n_x n_y x} + R_{n_y^2 y} + \frac{R_{n_x^2 x^2 y} + 2 R_{n_x n_y x y^2} + R_{n_y^2 y^3} }{f^2} & 
            S_{n_x^2 x^2 d} + S_{n_y^2 y^2 d} + 2 R_{n_x n_y x y d} &
            R_{n_x^2 x y} + R_{n_x n_y y^2} - R_{n_x n_y x^2} - R_{n_y^2 x y} \\
        R_{n_x^2 y d} - R_{n_x n_y x d} &
            R_{n_x^2 y} - R_{n_x n_y x} + \frac{R_{n_x^2 x^2 y} + R_{n_x n_y x y^2} - R_{n_x n_y x^3} - R_{n_y^2 x^2 y}}{f^2} &
            R_{n_x n_y y d} - R_{n_y^2 x d} &
            R_{n_x n_y y} - R_{n_y^2 x} + \frac{R_{n_x^2 x y^2} + R_{n_x n_y y^3} - R_{n_x n_y x^2 y} - R_{n_y^2 x y^2}}{f^2} &
            R_{n_x^2 x y d} + R_{n_x n_y y^2 d} - R_{n_x n_y x^2 d} - R_{n_y^2 x y d} &
            S_{n_x^2 y^2} + S_{n_y^2 x^2} - 2 R_{n_x n_y x y} """
_MStr = _MRaw.replace("\n", "").replace("\\", "#").replace("}}", "}").replace("}{f^2}", "}G").replace("} ", " ").replace("\frac", "").replace("_{", "").replace(" ", "").replace("_", "").replace("^", "")

_SigmaRaw = """S_{f_x} &
        S_{f_x d} &
        S_{f_y} &
        S_{f_y d} &
        S_{f_x x} + S_{f_y y} &
        S_{f_x y} - S_{f_y x} """
_SigmaStr = _SigmaRaw.replace("\n", "").replace("\\", "#").replace("}}", "}").replace("}{f^2}", "}G").replace("} ", " ").replace("\frac", "").replace("_{", "").replace(" ", "").replace("_", "").replace("^", "")


