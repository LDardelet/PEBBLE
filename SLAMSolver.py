import numpy as np

CS = np.array([320, 240])
ALPHAS = np.array([320,240])

FRAME_LW = 1
DEPTH_CONSTANT = 20.

class Solver:
    _Mass = 1.
    _Moment = 1.
    _MuX = 10.
    _MuOmega = 10.
    
    def __init__(self):
        self.t = None
        self.X = np.array([0., 0., 0.])
        self.Theta = np.array([0., 0., 0.])
        self.V = np.array([0., 0., 0.])
        self.Omega = np.array([0., 0., 0.])

        self.A = np.zeros(3)
        self.Alpha = np.zeros(3)

        self.AnchorsAndLocations = {}

        self.RStored = np.identity(3)
        self._UpToDate = True
        
        self.Map = MapClass()
        self.Camera = CameraClass(self.Map, self)
        self.Plotter = PlotterClass(self, self.Map, [self.Camera])

        self.Camera.Init()



    def AddAnchor(self, ID, x, disparity):
        V = self.GetWorldVectorFromLocation(x)
        z = 1./(disparity  / DEPTH_CONSTANT)
        Delta = DEPTH_CONSTANT/(2 * disparity * (1+disparity))
        self.AnchorsAndLocations[ID] = (AnchorClass(self.X + V * z, self.X, Delta), np.array(x))
        self._InitAnchorPlot(ID)

    def DelAnchor(self, ID):
        for line in self._PlotData['Anchors'][ID].values():
            line.remove()
        del self._PlotData['Anchors'][ID]
        del self.AnchorsAndLocations[ID]

    def OnTrackerData(self, t, ID, x, disparity = None):
        if self.t is None:
            self.t = t
            dt = 0
        else:
            dt = t - self.t
            self.t = t
        self.AnchorsAndLocations[ID][1][:] = x
        self.Step(dt)
        self.Plotter.UpdatePlot(self.t)

    def Step(self, dt):
        self.X += self.V * dt
        self.Theta += self.Omega * dt
        self._UpToDate = False

        F_Total = np.zeros(3)
        Omega_Total = np.zeros(3)
        for ID, (Anchor, Location) in self.AnchorsAndLocations.items():
            V = self.GetWorldVectorFromLocation(Location)
            Anchor.Update(self.X, V)
            F, X_F = Anchor.GetForce(self.X, V)
            F_Total += F
            Omega_Total += np.cross(F, X_F)
        A = (F_Total - self._MuX * self.V) / self._Mass
        self.V += dt * (A + self.A) / 2
        self.A = A
        Alpha = (Omega_Total - self._MuOmega * self.Omega) /  self._Moment
        self.Omega += dt * (Alpha + self.Alpha) / 2
        self.Alpha = Alpha

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
    def Energies(self):
        return self._Mass * (self.V**2).sum(), self._Moment * (self.Omega**2).sum()

    def GetWorldVectorFromLocation(self, x):
        Offsets = x - CS
        U = np.ones(3)
        U[1:] = Offsets / ALPHAS
        U /= np.linalg.norm(U)
        return (self.R).T.dot(U)

    def Plot(self, ax3D = None, axGraph = None, Init = False):
        if Init:
            self._PlotInit(ax3D, axGraph)
        EnergyLWRatio = 1.

        R = np.array(self.R)
        T = self.X
        for nDim in range(3):
            U = np.zeros(3)
            U[nDim] = 1.
            V = R.T.dot(U)
            TV = T+V
            self._PlotData['Frame'][nDim].set_data_3d([T[0], TV[0]], [T[1], TV[1]], [T[2], TV[2]])

        EV, EOmega = self.Energies
        ETotalAxis = 0
        ETotalNonAxis = 0
        for ID, (Anchor, Location) in self.AnchorsAndLocations.items():
            V = self.GetWorldVectorFromLocation(Location)
            Anchor.Update(self.X, V)
            Pu, Pv = Anchor.RestrictedPuPv
            TPv = T+Pv
            self._PlotData['Anchors'][ID]['CurrentVision'].set_data_3d([T[0], TPv[0]], [T[1], TPv[1]], [T[2], TPv[2]])
            O = Anchor.Origin
            P = Anchor.X
            self._PlotData['Anchors'][ID]['InitialObservation'].set_data_3d([O[0], P[0]], [O[1], P[1]], [O[2], P[2]])
            EAxis, ENonAxis = Anchor.Energies
            self._PlotData['Anchors'][ID]['NonAxis'].set_data_3d([Pv[0], Pu[0]], [Pv[1], Pu[1]], [Pv[2], Pu[2]])
            self._PlotData['Anchors'][ID]['NonAxis'].set_linewidth(min(4, EnergyLWRatio * ENonAxis))
            self._PlotData['Anchors'][ID]['Axis'].set_data_3d([P[0], Pu[0]], [P[1], Pu[1]], [P[2], Pu[2]])
            self._PlotData['Anchors'][ID]['Axis'].set_linewidth(min(4, EnergyLWRatio * EAxis))
            ETotalAxis += EAxis
            ETotalNonAxis += ENonAxis
        if not self.t is None:
            axGraph = self._PlotData['axGraph']
            axGraph.plot(self.t, ETotalAxis, 'ob')
            axGraph.plot(self.t, ETotalNonAxis, 'or')
            axGraph.plot(self.t, EV, 'xb')
            axGraph.plot(self.t, EOmega, 'xr')
            axGraph.plot(self.t, EV+EOmega+ETotalAxis+ETotalNonAxis, 'xk')

    def _PlotInit(self, ax3D, axGraph):
        NS = [0,0]
        self._PlotData = {'Frame':[], 'Anchors':{}, 'ax3D':ax3D, 'axGraph':axGraph}
        for nDim in range(3):
            self._PlotData['Frame'] += ax3D.plot(NS, NS, NS, color = 'r', linewidth = FRAME_LW)
        for ID in self.AnchorsAndLocations.keys():
            self._InitAnchorPlot(ID)

    def _InitAnchorPlot(self, ID):
        ax3D = self._PlotData['ax3D']
        VisionLW = 1
        NS = [0,0]
        self._PlotData['Anchors'][ID] = {'CurrentVision':ax3D.plot(NS, NS, NS, 'b', linewidth = VisionLW)[0], 
                                         'InitialObservation':ax3D.plot(NS, NS, NS, 'b', linewidth = VisionLW, linestyle = '--')[0], 
                                         'NonAxis':ax3D.plot(NS, NS, NS, 'k')[0], 
                                         'Axis':ax3D.plot(NS, NS, NS, 'k')[0]}

class CameraClass:
    MaxDx = 0.0001
    MaxDTheta = 0.01 / 180 * np.pi
    
    def __init__(self, Map, Solver):
        self.X = np.zeros(3, dtype = float)
        self.Theta = np.zeros(3, dtype = float)

        self.RStored = np.identity(3)
        self._UpToDate = True

        self.Solver = Solver
        self.Map = Map
        self.OnScreen = []


    def Init(self):
        CamVisionAxis = np.array([1., 0., 0.])
        WVisionAxis = self.R.T.dot(CamVisionAxis)
        for nPoint, Point in enumerate(self.Map.Points):
            OnScreen, Location = self.GetOnScreenLocation(Point)
            if OnScreen:
                self.OnScreen += [np.array([1, Location[0], Location[1]])]
                Depth = ((Point - self.X) * WVisionAxis).sum()
                disparity = int(1/(Depth/DEPTH_CONSTANT))
                self.Solver.AddAnchor(nPoint, Location, disparity)
            else:
                self.OnScreen += [np.zeros(3)]

        self.t = 0.

    def Move(self, tData, XData, ThetaData, moveType = 'relative'):
        if moveType == 'relative':
            V = XData / tData
            Omega = ThetaData / tData
        elif moveType == "absolute":
            V = (XData - self.X) / tData
            Omega = (ThetaData - self.Theta) / tData

        N = max(100, int(max(abs(XData / self.MaxDx).max(), abs(ThetaData / self.MaxDTheta).max())))
        dt = tData / N
        print(N)

        for n in range(N):
            MaxDisplacement = 0
            DisplacementData = (None, None)
            self.t += dt
            self.X += dt*V
            self.Theta += dt * Omega
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
            if MaxDisplacement > 0.1:
                self.Solver.OnTrackerData(self.t, DisplacementData[0], DisplacementData[1])

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

    def GetOnScreenLocation(self, X):
        WV = X - self.X
        CV = self.R.dot(WV)
        if CV[0] <= 0:
            return False, None
        CV /= CV[0]
        x = CV[1:] * ALPHAS + CS
        if (x<0).any() or (x>=CS*2).any():
            return False, None
        return True, x

    def Plot(self, ax = None, ax3D = None, Init = False):
        if Init:
            self._PlotInit(ax, ax3D)
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

    def _PlotInit(self, ax, ax3D):
        self._PlotData = {'Frame':[], 'Points':[]}
        NS = [0,0]
        for nDim in range(3):
            self._PlotData['Frame'] += ax3D.plot(NS, NS, NS, color = 'g', linewidth = FRAME_LW)
        for nPoint in range(len(self.Map.Points)):
            self._PlotData['Points'] += ax.plot(0,0,0, color = 'r', marker = 'o', alpha = 0)

class AnchorClass:
    _K0_Axis = 1.
    K_NonAxis = 1.

    def __init__(self, X, Origin, Delta):
        self.X = X
        self.Origin = np.array(Origin)
        self.Delta = Delta
        self.K_Axis = self._K0_Axis * self.Delta
        self.U = (X - Origin)
        self.U /= np.linalg.norm(self.U)

        self.XCamera = np.array(X)
        self.V = np.array(self.U)

    def Update(self, XCamera, V):
        self.XCamera = XCamera
        self.V = V
    
    @property
    def Energies(self):
        Pu, Pv = self.RestrictedPuPv
        return self.K_Axis * ((Pu - self.X)**2).sum(), self.K_NonAxis * ((Pu - Pv)**2).sum()
    @property
    def RestrictedPuPv(self):
        if (self.V == self.U).all() or (self.V == -self.U).all():
            return self.X, self.XCamera + (((self.X - self.XCamera) * self.V).sum()) * self.V
        else:
            N = (self.V * self.U).sum()
            Lambda = (((self.XCamera - self.X) * ((self.V * N) - self.U)).sum()) / (1 - N**2)
            Lambda = np.sign(Lambda)*min(abs(Lambda), self.Delta) # Restriction to the anchor zone
            Pu = self.X + Lambda * self.U
            Mu = (self.V * (self.XCamera - self.X)).sum() + Lambda * N
            Pv = self.XCamera + Mu * self.V
            return Pu, Pv

    def GetForce(self, X, V):
        Pu, Pv = self.RestrictedPuPv
        F_NonAxis = (Pu - Pv) * self.K_NonAxis

        F_Offset = (Pu - self.U) * self.K_Axis
        F_Axis = F_Offset - (F_Offset * self.V).sum() * self.V

        return F_NonAxis + F_Axis, Pv # Forces and Application Point

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PlotterClass:
    def __init__(self, Solver, Map, Cameras):
        self.Solver = Solver
        self.Map = Map
        self.Cameras = Cameras
        self.Init()

    def Init(self):
        self.Figure = plt.figure()
        self.GTAx = self.Figure.add_subplot(3, 2, 1, projection='3d')
        self.GTAx.set_label("Ground-truth Map")
        self.GTAx.set_xlabel("X")
        self.GTAx.set_ylabel("Y")
        self.GTAx.set_zlabel("Z")
        self.GTAx.set_xlim(self.Map._Min - 0.2, self.Map._Max + 0.2)
        self.GTAx.set_ylim(self.Map._Min - 0.2, self.Map._Max + 0.2)
        self.GTAx.set_zlim(self.Map._Min - 0.2, self.Map._Max + 0.2)
        self.SolverAx = self.Figure.add_subplot(3, 2, 2, projection='3d')
        self.SolverAx.set_label("Solver Map")
        self.SolverAx.set_xlabel("X")
        self.SolverAx.set_ylabel("Y")
        self.SolverAx.set_zlabel("Z")
        self.SolverAx.set_xlim(self.Map._Min - 0.2, self.Map._Max + 0.2)
        self.SolverAx.set_ylim(self.Map._Min - 0.2, self.Map._Max + 0.2)
        self.SolverAx.set_zlim(self.Map._Min - 0.2, self.Map._Max + 0.2)
        self.CamerasAxs = [self.Figure.add_subplot(3, 2, 3+nCam) for nCam in range(2)]
        self.GraphAx = self.Figure.add_subplot(3, 1, 3)
        self.Map.Plot(self.GTAx)
        self.Solver.Plot(self.SolverAx, self.GraphAx, Init = True)
        for nCam, Camera in enumerate(self.Cameras):
            Camera.Plot(self.CamerasAxs[nCam], self.GTAx, Init = True)
            self.CamerasAxs[nCam].set_label("Camera {0}".format(nCam+1))
            self.CamerasAxs[nCam].set_xlim(0, CS[0]*2)
            self.CamerasAxs[nCam].set_ylim(0, CS[1]*2)
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
        if t - self.LastUpdate < 0.1:
            return
        self.LastUpdate = t

        self.Solver.Plot()
        for Camera in self.Cameras:
            Camera.Plot()
        self.Figure.canvas.draw()
        
class MapClass:
    _NPoints = 6
    _Min = -5
    _Max = 5
    def __init__(self):
        self.Points = []
        for x in linspace(self._Min, self._Max, self._NPoints):
            for y in linspace(self._Min, self._Max, self._NPoints):
                for z in linspace(self._Min, self._Max, self._NPoints):
                    self.Points += [np.array([x, y, z])]
    def Plot(self, ax3D):
        Xs = np.array(self.Points)
        ax3D.scatter(Xs[:,0], Xs[:,1], Xs[:,2], color = 'r', marker = 'o', s=1)

