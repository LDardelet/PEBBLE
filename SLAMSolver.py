import numpy as np

CS = np.array([320, 240])
SIZE = CS * 2
ALPHAS = np.array([320,240]) * 1.

FRAME_LW = 1
DEPTH_CONSTANT = 30.

class Solver:
    _RTRatio = 2
    _Mass = 0.001
    _Moment = 0.002
    _MuVBase = 0.1
    _MuOmegaBase = 0.2
    _MuVPerSpring = 0.1
    _MuOmegaPerSpring = 0.2

    _XTranlationIncrease = 1.

    _EnergyToMassConstant = 10.

    ScreenEdgeRepelDistance = 5.
    
    def __init__(self):
        self.t = 0.
        self.X = np.array([0., 0., 0.])
        self.Theta = np.array([np.pi, 0., 0.])
        self.V = np.array([0., 0., 0.])
        self.Omega = np.array([0., 0., 0.])

        self.A = np.zeros(3)
        self.Alpha = np.zeros(3)

        self.TotalDampening = 0.
        self.TotalSpringAdded = 0.

        self.AnchorsAndLocations = {}

        self.RStored = np.identity(3)
        self._UpToDate = False
        
        self.Map = MapClass()
        self.Camera = CameraClass(self.Map, self)
        self.Plotter = PlotterClass(self, self.Map, [self.Camera])

        self.Camera.Init()

    def AddAnchor(self, ID, Location, disparity):
        V = self.GetWorldVectorFromLocation(Location)
        WVisionAxis = np.array(self.WVisionAxis)
        N = (V * WVisionAxis).sum()

        Lambda = DEPTH_CONSTANT / (disparity * N)

        LambdaMin = DEPTH_CONSTANT / ((disparity + 0.5) * N) - Lambda
        LambdaMax = DEPTH_CONSTANT / ((disparity - 0.5) * N) - Lambda

        E = self.SpringsEnergy
        if E < 1e-10:
            Mass = np.inf
        else:
            Mass = self._EnergyToMassConstant / E

        self.AnchorsAndLocations[ID] = (AnchorClass(ID, self.X + V * Lambda, self.X, LambdaMin, LambdaMax, Mass), np.array(Location))
        d = min(Location.min(), (SIZE - 1 - Location).max())
        self.AnchorsAndLocations[ID][0].K = self.AnchorsAndLocations[ID][0].K_NonAxis * min(d / self.ScreenEdgeRepelDistance, 1)
        self._InitAnchorPlot(ID)

        self.SystemEnergyUpdate(SpringVariation = self.AnchorsAndLocations[ID][0].Energy)
        self.UpdateDampening()

    def DelAnchor(self, ID):
        self.SystemEnergyUpdate(SpringVariation = -self.AnchorsAndLocations[ID][0].Energy)
        for line in self._PlotData['Anchors'][ID].values():
            line.remove()
        del self._PlotData['Anchors'][ID]
        del self.AnchorsAndLocations[ID]
        self.Print("Removed anchor {0}".format(ID))
        self.UpdateDampening()
    
    def UpdateDampening(self):
        self.MuV = self._MuVBase
        self.MuOmega = self._MuOmegaBase

        for ID, (Anchor, _) in self.AnchorsAndLocations.items():
            self.MuV += self._MuVPerSpring
            self.MuOmega += self._MuOmegaPerSpring

    def OnTrackerData(self, t, ID = None, x = None, disparity = None):
        dt = (t - self.t) 
        self.t = t
        if not ID is None:
            Anchor, Location = self.AnchorsAndLocations[ID]
            V = self.GetWorldVectorFromLocation(Location)
            Anchor.Update(self.X, V)
            E = Anchor.Energy
            Location[:] = x
            d = min(x.min(), (SIZE - 1 - x).min())
            Anchor.K = Anchor.K_NonAxis * min(d / self.ScreenEdgeRepelDistance, 1)
            V = self.GetWorldVectorFromLocation(Location)
            Anchor.Update(self.X, V)
            self.SystemEnergyUpdate(SpringVariation = Anchor.Energy - E)
        XIni, ThetaIni = np.array(self.X), np.array(self.Theta)
        for nStep in range(self._RTRatio):
            self.Step(dt)
        if self._RTRatio > 1 and dt > 0:
            self.Speed, self.Omega = (self.X - XIni) / (dt * self._RTRatio), (self.Theta - ThetaIni) / (dt * self._RTRatio)
        self.Plotter.UpdatePlot(self.t)

    @property
    def KTotal(self):
        K = 0.
        for Anchor, _ in self.AnchorsAndLocations.values():
            K += Anchor.K
        return K

    def Step(self, dt):
        F_Total = np.zeros(3)
        Omega_Total = np.zeros(3)
        for ID, (Anchor, Location) in self.AnchorsAndLocations.items():
            V = self.GetWorldVectorFromLocation(Location)
            Anchor.Update(self.X, V)
            F, X_F = Anchor.ApplyForceAndPoint(dt)
            F_Total += F
            Omega_Total += np.cross(F, X_F)
        DampeningTranslation =  - self.MuV * self.V
        DampeningRotation =  - self.MuOmega * self.Omega
        self.SystemEnergyUpdate(DampeningVariation = ((DampeningTranslation * self.V).sum() + (DampeningRotation * self.Omega).sum()) * dt)

        F_InSolverFrame = self.R.dot(F_Total)
        F_InSolverFrame[0] *= self._XTranlationIncrease
        F_Total = self.R.T.dot(F_InSolverFrame)

        self.A = (F_Total + DampeningTranslation) / self._Mass
        self.V += dt * self.A
        self.Alpha = (Omega_Total + DampeningRotation) / self._Moment
        self.Omega += dt * self.Alpha

        self.X += self.V * dt
        #self.Theta += self.R.T.dot(self.Omega) * dt
        self.Theta += self.Omega * dt
        self._UpToDate = False

#    @property
#    def R(self):
#        # Defined as Roll, Pitch, Yaw for x, y, z respectively
#        cR, cP, cY = np.cos(self.Theta)
#        sR, sP, sY = np.sin(self.Theta)
#        R = np.array([[cY*cP, cY*sP*sR - sY*cR, cY*sP*cR + sY*sR],
#                      [sY*cP, sY*sP*sR + cY*cR, sY*sP*cR - cY*sR],
#                      [-sP  , cP*sR           , cP*cR]])
#        return R

    @property
    def R(self):
        if self._UpToDate:
            return self.RStored
        Theta = np.linalg.norm(self.Theta)
        if Theta == 0:
            self.RStored = np.identity(3)
            return self.RStored
        c, s = np.cos(Theta), np.sin(Theta)
        Ux, Uy, Uz = self.Theta / Theta
        self.RStored = np.array([[c + Ux**2*(1-c), Ux*Uy*(1-c) - Uz*s, Ux*Uz*(1-c) + Uy*s],
                      [Uy*Ux*(1-c) + Uz*s, c + Uy**2*(1-c), Uy*Uz*(1-c) - Ux*s],
                      [Uz*Ux*(1-c) - Uy*s, Uz*Uy*(1-c) + Ux*s, c + Uz**2*(1-c)]])
        return self.RStored

    @property
    def KineticEnergies(self):
        return self._Mass * (self.V**2).sum(), self._Moment * (self.Omega**2).sum()

    def SystemEnergyUpdate(self, DampeningVariation = 0., SpringVariation = 0.):
        self.TotalDampening += DampeningVariation
        self.TotalSpringAdded += SpringVariation

    @property
    def MechanicalEnergy(self):
        return np.sum(self.KineticEnergies) + self.SpringsEnergy

    @property
    def SpringsEnergy(self):
        SpringsEnergy = 0.
        for ID, (Anchor, Location) in self.AnchorsAndLocations.items():
            V = self.GetWorldVectorFromLocation(Location)
            Anchor.Update(self.X, V)
            SpringsEnergy += Anchor.Energy
        return SpringsEnergy

    def GetWorldVectorFromLocation(self, x):
        Offsets = x - CS
        U = np.ones(3)
        U[1:] = Offsets / ALPHAS
        U /= np.linalg.norm(U)
        return (self.R).T.dot(U)

    @property
    def WVisionAxis(self):
        U = np.array([1., 0., 0.])
        return self.R.T.dot(U)

    def Plot(self, ax3D = None, axEnergy = None, axPosition = None, axAngle = None, axSpeed = None, axRotation = None, Init = False):
        if Init:
            self._PlotInit(ax3D, (axEnergy, axPosition, axAngle, axSpeed, axRotation))
        EnergyLWRatio = 1.

        R = np.array(self.R)
        T = self.X
        for nDim in range(3):
            U = np.zeros(3)
            U[nDim] = 1.
            V = R.T.dot(U)
            TV = T+V
            self._PlotData['Frame'][nDim].set_data_3d([T[0], TV[0]], [T[1], TV[1]], [T[2], TV[2]])
        TA = T+self.A
        TAlpha = T+self.Alpha
        self._PlotData['Acc'].set_data_3d([T[0], TA[0]], [T[1], TA[1]], [T[2], TA[2]])
        self._PlotData['WAcc'].set_data_3d([T[0], TAlpha[0]], [T[1], TAlpha[1]], [T[2], TAlpha[2]])
        EV, EOmega = self.KineticEnergies
        ETotalNonAxis = 0
        for ID, (Anchor, Location) in self.AnchorsAndLocations.items():
            V = self.GetWorldVectorFromLocation(Location)
            Anchor.Update(self.X, V)
            Pu, Pv = Anchor.RestrictedPuPv
            TPv = T+Pv
            self._PlotData['Anchors'][ID]['CurrentVision'].set_data_3d([T[0], TPv[0]], [T[1], TPv[1]], [T[2], TPv[2]])
            ENonAxis = Anchor.Energy
            self._PlotData['Anchors'][ID]['NonAxis'].set_data_3d([Pv[0], Pu[0]], [Pv[1], Pu[1]], [Pv[2], Pu[2]])
            self._PlotData['Anchors'][ID]['NonAxis'].set_linewidth(min(4, max(1, EnergyLWRatio * ENonAxis)))
            ETotalNonAxis += ENonAxis
        if not self.t is None:
            axEnergy, axPosition, axAngle, axSpeed, axRotation = self._PlotData['axs']
            axEnergy.plot(self.t, ETotalNonAxis, 'or')
            axEnergy.plot(self.t, EV, 'xb')
            axEnergy.plot(self.t, EOmega, 'xr')
            axEnergy.plot(self.t, EV+EOmega+ETotalNonAxis, 'xk')

            for nDim, Color in enumerate(['r', 'g', 'b']):
                axPosition.plot(self.t, self.X[nDim], marker = 'x', color = Color)
                axAngle.plot(self.t, self.Theta[nDim], marker = 'x', color = Color)
                axSpeed.plot(self.t, self._RTRatio * self.V[nDim], marker = 'x', color = Color)
                axRotation.plot(self.t, self._RTRatio * self.Omega[nDim], marker = 'x', color = Color)

    def Print(self, Data):
        print("t = {0:.3f} : ".format(self.t) + Data)

    def _PlotInit(self, ax3D, axs):
        NS = [0,0]
        self._PlotData = {'Frame':[], 'Anchors':{}, 'axs': axs, 'ax3D': ax3D}
        for nDim in range(3):
            self._PlotData['Frame'] += ax3D.plot(NS, NS, NS, color = 'r', linewidth = FRAME_LW)
        self._PlotData['Acc'] = ax3D.plot(NS, NS, NS, color = 'g', linestyle = '--', linewidth = 3)[0]
        self._PlotData['WAcc'] = ax3D.plot(NS, NS, NS, color = 'g', linewidth = 3)[0]
        for ID in self.AnchorsAndLocations.keys():
            self._InitAnchorPlot(ID)

    def _InitAnchorPlot(self, ID):
        self.Print("New Anchor {0}".format(ID))
        ax3D = self._PlotData['ax3D']
        VisionLW = 1
        NS = [0,0]
        self._PlotData['Anchors'][ID] = {'CurrentVision':ax3D.plot(NS, NS, NS, 'b', linewidth = VisionLW, linestyle = '--')[0], 
                                         'Range':ax3D.plot(NS, NS, NS, 'b', linewidth = VisionLW * 2, linestyle = '-')[0], 
                                         'NonAxis':ax3D.plot(NS, NS, NS, 'k')[0]}
        Anchor = self.AnchorsAndLocations[ID][0]
        O = Anchor.X + Anchor.LambdaMin * Anchor.U
        P = Anchor.X + Anchor.LambdaMax * Anchor.U
        self._PlotData['Anchors'][ID]['Range'].set_data_3d([O[0], P[0]], [O[1], P[1]], [O[2], P[2]])

    def AnchorToMapError(self, ID):
        Anchor = self.AnchorsAndLocations[ID][0]
        Point3D = self.Map.Points[ID]
        Delta = (Point3D - Anchor.X)
        Lambda = (Delta * Anchor.U).sum()
        if Lambda < Anchor.LambdaMin:
            print("{0:.2f} < {1:.2f}".format(Lambda, Anchor.LambdaMin))
        elif Lambda > Anchor.LambdaMax:
            print("{0:.2f} > {1:.2f}".format(Lambda, Anchor.LambdaMax))
        else:
            print("{0:.2f} <= {1:.2f} <= {2:.2f}".format(Anchor.LambdaMin, Lambda, Anchor.LambdaMax))
        print("OffAxis by {0:.2f}".format(np.linalg.norm(Delta - Lambda * Anchor.U)))

class CameraClass:
    MaxDx = 0.0001
    MaxDTheta = 0.01 / 180 * np.pi
    MaxDt = 0.0001
    
    def __init__(self, Map, Solver):
        self.X = np.zeros(3, dtype = float)
        self.Theta = np.zeros(3, dtype = float)
        self.Theta[0] = np.pi
        self.V = np.zeros(3)
        self.Omega = np.zeros(3)

        self.RStored = np.identity(3)
        self._UpToDate = False

        self.Solver = Solver
        self.Map = Map
        self.OnScreen = []

        self.t = 0.

    def Init(self):
        CamVisionAxis = np.array([1., 0., 0.])
        WVisionAxis = self.R.T.dot(CamVisionAxis)
        for nPoint, Point in enumerate(self.Map.Points):
            OnScreen, Location = self.GetOnScreenLocation(Point)
            if OnScreen:
                self.OnScreen += [np.array([1, Location[0], Location[1]])]
                Depth = ((Point - self.X) * WVisionAxis).sum()
                disparity = int(DEPTH_CONSTANT/Depth + 0.5)
                self.Solver.AddAnchor(nPoint, Location, disparity)
            else:
                self.OnScreen += [np.zeros(3)]

    def Move(self, tData, XData, ThetaData, moveType = 'relative'):
        if moveType == 'relative':
            self.V = XData / tData
            self.Omega = ThetaData / tData
        elif moveType == "absolute":
            self.V = (XData - self.X) / tData
            self.Omega = (ThetaData - self.Theta) / tData

        N = max(int(tData/self.MaxDt), int(max(abs(XData / self.MaxDx).max(), abs(ThetaData / self.MaxDTheta).max())))
        dt = tData / N
        print(N)

        for n in range(N):
            MaxDisplacement = 0
            DisplacementData = (None, None)
            self.t += dt
            self.X += dt*self.V
            self.Theta += dt * self.Omega
            self._UpToDate = False
            for nPoint, Point in enumerate(self.Map.Points):
                OnScreen, Location = self.GetOnScreenLocation(Point)
                if self.OnScreen[nPoint][0]:
                    if not OnScreen:
                        self.Solver.DelAnchor(nPoint)
                        self.OnScreen[nPoint][0] = 0
                        continue
                    else:
                        N = np.linalg.norm(Location - self.OnScreen[nPoint][1:])
                        if N > MaxDisplacement:
                            MaxDisplacement = N
                            DisplacementData = (nPoint, Location)
                else:
                    if OnScreen:
                        CamVisionAxis = np.array([1., 0., 0.])
                        WVisionAxis = self.R.T.dot(CamVisionAxis)
                        Depth = ((Point - self.X) * WVisionAxis).sum()
                        disparity = int(1/(Depth/DEPTH_CONSTANT))
                        self.Solver.AddAnchor(nPoint, Location, disparity)
                        self.OnScreen[nPoint][0] = 1
                        self.OnScreen[nPoint][1:] = Location

            if MaxDisplacement > 0.05:
                self.Solver.OnTrackerData(self.t, DisplacementData[0], DisplacementData[1])
                self.OnScreen[DisplacementData[0]][1:] = DisplacementData[1]
            else:
                self.Solver.OnTrackerData(self.t)

    @property
    def R(self):
        if self._UpToDate:
            return self.RStored
        Theta = np.linalg.norm(self.Theta)
        self._UpToDate = True
        if Theta == 0:
            self.RStored = np.identity(3)
            return self.RStored
        c, s = np.cos(Theta), np.sin(Theta)
        Ux, Uy, Uz = self.Theta / Theta
        self.RStored = np.array([[c + Ux**2*(1-c), Ux*Uy*(1-c) - Uz*s, Ux*Uz*(1-c) + Uy*s],
                      [Uy*Ux*(1-c) + Uz*s, c + Uy**2*(1-c), Uy*Uz*(1-c) - Ux*s],
                      [Uz*Ux*(1-c) - Uy*s, Uz*Uy*(1-c) + Ux*s, c + Uz**2*(1-c)]])
        return self.RStored

    def GetOnScreenLocation(self, X):
        WV = X - self.X
        CV = self.R.dot(WV)
        if CV[0] <= 0.1:
            return False, None
        CV /= CV[0]
        x = CV[1:] * ALPHAS + CS
        if (x<0).any() or (x>=SIZE).any():
            return False, None
        return True, x

    def Plot(self, ax = None, ax3D = None, axPosition = None, axAngle = None, axSpeed = None, axRotation = None, Init = False):
        if Init:
            self._PlotInit(ax, ax3D, (axPosition, axAngle, axSpeed, axRotation))
        for nPoint, X in enumerate(self.Map.Points):
            OnScreen, x = self.GetOnScreenLocation(X)
            alpha = np.e**-(np.linalg.norm(X - self.X) / 3)
            if OnScreen:
                self._PlotData['Points'][nPoint].set_data(x[0], x[1])
                self._PlotData['Points'][nPoint].set_alpha(alpha)
            else:
                self._PlotData['Points'][nPoint].set_alpha(0)
        R = np.array(self.R)
        T = self.X
        for nDim in range(3):
            U = np.zeros(3)
            U[nDim] = 1.
            V = R.T.dot(U)
            TV = T+V
            self._PlotData['Frame'][nDim].set_data_3d([T[0], TV[0]], [T[1], TV[1]], [T[2], TV[2]])
        axPosition, axAngle, axSpeed, axRotation = self._PlotData['axs']
        for nDim, Color in enumerate(['r', 'g', 'b']):
            axPosition.plot(self.t, self.X[nDim], marker = 'o', color = Color)
            axAngle.plot(self.t, self.Theta[nDim], marker = 'o', color = Color)
            axSpeed.plot(self.t, self.V[nDim], marker = 'o', color = Color)
            axRotation.plot(self.t, self.Omega[nDim], marker = 'o', color = Color)

    def _PlotInit(self, ax, ax3D, axs):
        self._PlotData = {'Frame':[], 'Points':[], 'axs': axs}
        NS = [0,0]
        for nDim in range(3):
            self._PlotData['Frame'] += ax3D.plot(NS, NS, NS, color = 'g', linewidth = FRAME_LW)
        for nPoint in range(len(self.Map.Points)):
            self._PlotData['Points'] += ax.plot(0,0,0, color = 'r', marker = 'o', alpha = 0)

class AnchorClass:
    K_Axis = 0.1
    K_NonAxis = 1.
    NonAxisTolerance = 0.
    MuV = 1.

    def __init__(self, ID, X, Origin, LambdaMin = None, LambdaMax = None, Mass = None):
        self.ID = ID
        self.X = np.array(X)
        self.Origin = np.array(Origin)
        self.U = (X - Origin)
        N = np.linalg.norm(self.U)
        self.U /= N
        if LambdaMin is None:
            self.LambdaMin = N
        else:
            self.LambdaMin = LambdaMin
        if LambdaMax is None:
            self.LambdaMax = np.inf
        else:
            self.LambdaMax = LambdaMax
        if Mass is None:
            self.Mass = np.inf
        else:
            self.Mass = Mass

        self.Speed = np.zeros(3)

        self._WX = 0.
        self._W = 0.

        self.XCamera = np.array(Origin)
        self.V = np.array(self.U)

        self.K = self.K_NonAxis

    def Update(self, XCamera, V):
        self.XCamera = np.array(XCamera)
        self.V = np.array(V)
    
    @property
    def ObservedX(self):
        if self._W == 0:
            return self.X
        else:
            return self._WX / self._W

    def PlotVisionPlane(self):
        Pu, Pv = self.RestrictedPuPv
        UV = Pu - Pv
        N = np.linalg.norm(UV)
        if N == 0:
            print("Unable to plot")
            return
        UV /= np.linalg.norm(UV)
        Pu -= self.X
        Pv -= self.X
        f, ax = plt.subplots(1,1)
        Z = np.cross(UV, self.V)
        Z /= np.linalg.norm(Z)
        X = self.V - (self.V*Z).sum() * Z
        X /= np.linalg.norm(X)
        Y = np.cross(Z, X)
        LMin, LMax = self.U * self.LambdaMin, self.U * self.LambdaMax
        PLMin, PLMax = LMin - (Z*LMin).sum() * Z, LMax - (Z*LMax).sum() * Z
        ax.plot(0, 0, "ob")
        ax.plot([(PLMin*X).sum(), (PLMax*X).sum()], [(PLMin*Y).sum(), (PLMax*Y).sum()], 'b')
        for P in [Pu, Pv]:
            P = P
            PProj = P - (P*Z).sum() * Z
            Px, Py = (PProj * X).sum(), (PProj * Y).sum()
            ax.plot(Px, Py, 'xk')
        O1 = Pv - 10 * self.V
        O2 = Pv + 10 * self.V
        PO1, PO2 = O1 - (Z*O1).sum() * Z, O2 - (Z*O2).sum() * Z
        POx1, POy1 = (PO1 * X).sum(), (PO1 * Y).sum()
        POx2, POy2 = (PO2 * X).sum(), (PO2 * Y).sum()
        ax.plot([POx1, POx2], [POy1, POy2], '--k')

    @property
    def Energy(self):
        return self.K * (self.Length**2).sum()

    @property
    def Length(self):
        Pu, Pv = self.RestrictedPuPv
        return np.linalg.norm(Pu-Pv)

    @property
    def RestrictedPuPv(self):
        Delta = self.XCamera - self.X
        Alpha = (self.U * self.V).sum()
        if Alpha ** 2 == 1: # If both lines of sight are parallel
            Pu = np.array(self.X) # Application point is placed at point location
            Pv = Pu + (Delta - (Delta * self.U).sum() * self.U)
        else:
            Lambda = (Delta * (self.U - self.V * Alpha)).sum() / (1-Alpha**2)
            if Lambda < self.LambdaMin:
                #print("{0} is too close by {1:.3f}".format(self.ID, self.LambdaMin - Lambda))
                Lambda = self.LambdaMin
            elif Lambda > self.LambdaMax:
                #print("{0} is too far by {1:.3f}".format(self.ID, Lambda - self.LambdaMax))
                Lambda = self.LambdaMin
            Mu = Lambda * Alpha - (Delta * self.V).sum()
            Pu = self.X + Lambda * self.U
            Pv = self.XCamera + Mu * self.V
        return Pu, Pv

    def ApplyForceAndPoint(self, dt):
        Pu, Pv = self.RestrictedPuPv
        Delta = (Pu - Pv)
        N = np.linalg.norm(Delta)
        if N == 0:
            return Delta, Pv
        Delta *= max(0, N - self.NonAxisTolerance) / N
        F_NonAxis = Delta * self.K

        F_Anchor = (F_NonAxis + self.K_Axis * (self.X - Pu) - self.Speed * self.MuV)
        self.Speed += dt * F_Anchor / self.Mass
        self.X += dt * self.Speed

        return F_NonAxis, Pv # Forces and Application Point

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PlotterClass:
    TimeWindow = 2.
    FPS = 20.
    def __init__(self, Solver, Map, Cameras):
        self.Solver = Solver
        self.Map = Map
        self.Cameras = Cameras
        self.Init()

    def Init(self):
        NLines = 3
        self.Figure = plt.figure()
        self.GTAx = self.Figure.add_subplot(NLines, 2, 1, projection='3d')
        self.GTAx.set_title("Ground-truth Map")
        self.GTAx.set_xlabel("X")
        self.GTAx.set_ylabel("Y")
        self.GTAx.set_zlabel("Z")
        self.GTAx.set_xlim(self.Map._Min - 0.2, self.Map._Max + 0.2)
        self.GTAx.set_ylim(self.Map._Min - 0.2, self.Map._Max + 0.2)
        self.GTAx.set_zlim(self.Map._Min - 0.2, self.Map._Max + 0.2)
        self.SolverAx = self.Figure.add_subplot(NLines, 2, 2, projection='3d')
        self.SolverAx.set_title("Solver Map")
        self.SolverAx.set_xlabel("X")
        self.SolverAx.set_ylabel("Y")
        self.SolverAx.set_zlabel("Z")
        self.SolverAx.set_xlim(self.Map._Min - 0.2, self.Map._Max + 0.2)
        self.SolverAx.set_ylim(self.Map._Min - 0.2, self.Map._Max + 0.2)
        self.SolverAx.set_zlim(self.Map._Min - 0.2, self.Map._Max + 0.2)
        self.CamerasAxs = [self.Figure.add_subplot(NLines, 3, 7) for nCam in range(1)]
        self.EnergiesAx = self.Figure.add_subplot(NLines, 3, 4)
        self.EnergiesAx.set_title("Energy Data")
        self.PositionsAx = self.Figure.add_subplot(NLines, 3, 5)
        self.PositionsAx.set_title("Translation")
        self.AnglesAx = self.Figure.add_subplot(NLines, 3, 6)
        self.AnglesAx.set_title("Rotation")
        self.SpeedsAx = self.Figure.add_subplot(NLines, 3, 8)
        self.SpeedsAx.set_title("Velocity")
        self.RotationsAx = self.Figure.add_subplot(NLines, 3, 9)
        self.RotationsAx.set_title("Angular velocity")
        self.Map.Plot(self.GTAx)
        self.Solver.Plot(self.SolverAx, self.EnergiesAx, self.PositionsAx, self.AnglesAx, self.SpeedsAx, self.RotationsAx, Init = True)
        for nCam, Camera in enumerate(self.Cameras):
            Camera.Plot(self.CamerasAxs[nCam], self.GTAx, self.PositionsAx, self.AnglesAx, self.SpeedsAx, self.RotationsAx, Init = True)
            self.CamerasAxs[nCam].set_title("Camera {0}".format(nCam+1))
            self.CamerasAxs[nCam].set_xlim(0, SIZE[0])
            self.CamerasAxs[nCam].set_ylim(0, SIZE[1])
            self.CamerasAxs[nCam].set_aspect("equal")
        self.LastUpdate = 0

    def Snapshot(self):
        _TmpAxs = (self.Figure, self.GTAx, self.SolverAx, self.CamerasAxs)
        _PlotData = (self.Solver._PlotData, [Camera._PlotData for Camera in self.Cameras])
        _LastUpdate = self.LastUpdate

        self.Init()
        self.UpdatePlot(np.inf)

        self.Figure, self.GTAx, self.SolverAx, self.CamerasAxs = _TmpAxs
        self.Solver._PlotData = _PlotData[0]
        for nCam, Camera in enumerate(self.Cameras):
            Camera._PlotData = _PlotData[1][nCam]
        self.LastUpdate = _LastUpdate

    def UpdatePlot(self, t):
        if t - self.LastUpdate < 1/self.FPS:
            return
        self.LastUpdate = t

        self.Solver.Plot()
        for Camera in self.Cameras:
            Camera.Plot()
        for ax in [self.EnergiesAx, self.PositionsAx, self.AnglesAx, self.SpeedsAx, self.RotationsAx]:
            ax.set_xlim(t-self.TimeWindow, t)
        self.Figure.canvas.draw()

class MapClass:
    _NPoints = 5
    _Min = -5.4
    _Max = 4.85
    def __init__(self):
        self.Points = []
        for x in linspace(self._Min, self._Max, self._NPoints):
            for y in linspace(self._Min, self._Max, self._NPoints):
                for z in linspace(self._Min, self._Max, self._NPoints):
                    self.Points += [np.array([x, y, z])]
    def Plot(self, ax3D):
        Xs = np.array(self.Points)
        ax3D.scatter(Xs[:,0], Xs[:,1], Xs[:,2], color = 'r', marker = 'o', s=1)

