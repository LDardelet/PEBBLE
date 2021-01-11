import numpy as np

class Solver:
    _Mass = 1.
    _Moment = 1.
    _MuX = 1.
    _MuOmega = 1.
    
    _Cs = np.array([320, 240])
    _Alphas = np.array([320,240])
    def __init__(self):
        self.X = np.array([0., 0., 0.])
        self.Theta = np.array([0., 0., 0.])
        self.V = np.array([0., 0., 0.])
        self.Omega = np.array([0., 0., 0.])

        self.A = np.zeros(3)
        self.Alpha = np.zeros(3)

        self.ScreenLocations = {}
        self.Anchors = {}

        self.RStored = np.identity(3)
        self._UpToDate = True
        
        self.Plotter = PlotterClass(self)

    def Step(self, Dt):
        self.X += self.V * Dt
        self.Theta += self.Omega * Dt
        self._UpToDate = False

        F_Total = np.zeros(3)
        Omega_Total = np.zeros(3)
        for ID, Anchor in self.Anchors.items():
            V = self.GetWorldVectorFromLocation(self.ScreenLocations[ID])
            Anchor.Update(self.X, V)
            F, X_F = Anchor.GetForce(self.X, V)
            F_Total += F
            Omega_Total += np.cross(F, X_F)
        A = (F_Total - self._MuX * self.V) / self._Mass
        self.V += Dt * (A + self.A) / 2
        self.A = A
        Alpha = (Omega_Total - self._MuOmega * self.Omega) /  self._Moment
        self.Omega += Dt * (Alpha + self.Alpha) / 2
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

    def GetWorldVectorFromLocation(self, x):
        Offsets = x - self._Cs
        U = np.ones(3)
        U[:2] = Offsets / self.Alphas
        U /= np.linalg.norm(U)
        return (self.R).T.dot(U)

class Anchor:
    _K0_Axis = 1.
    _K_NonAxis = 1.

    def __init__(self, X0, Delta):
        self.X0 = X0
        self.N = np.linalg.norm(Delta)
        self.K_Axis = self._K0_Axis * self.N
        self.U = Delta / self.N

        self.X = np.array(X0)
        self.V = np.array(U)

    def Update(self, X, V):
        self.X = X
        self.V = V
    
    @property
    def RestrictedPuPv(self):
        if (self.V == self.U).all() or (self.V == -self.U).all():
            return self.X0, self.X + (((self.X0 - self.X) * self.V).sum()) * self.V
        else:
            N = (self.V * self.U).sum()
            Lambda = (((self.X - self.X0) * ((self.V * N) - self.U)).sum()) / (1 - N**2)
            Lambda = max(Lambda, self.N) # Restriction to the anchor zone
            Pu = self.X0 + Lambda * self.U
            Mu = (self.V * (self.X - self.X0)).sum() + Lambda * N
            Pv = self.X + Mu * self.V
            return Pu, Pv

    def GetForce(self, X, V):
        Pu, Pv = self.RestrictedPuPv
        F_NonAxis = (Pu - Pv) * self._K_NonAxis

        F_Offset = (Pu - self.U) * self.K_Axis
        F_Axis = F_Offset - (F_Offset * self.V).sum() * self.V

        return F_NonAxis + F_Axis, Pv # Forces and Application Point

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PlotterClass:
    def __init__(self, Solver):
        self.Solver = Solver

    def Snapshot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        FrameLW = 1

        ax.plot([0, 1], [0, 0], [0, 0], color = 'k', linewidth = FrameLW)
        ax.plot([0, 0], [0, 1], [0, 0], color = 'k', linewidth = FrameLW)
        ax.plot([0, 0], [0, 0], [0, 1], color = 'k', linewidth = FrameLW)

        R = self.Solver.R
        T = self.Solver.X
        for nDim in range(3):
            U = np.zeros(3)
            U[nDim] = 1.
            V = R.T.dot(U)
            TV = T+V
            ax.plot([T[0], TV[0]], [T[1], TV[1]], [T[2], TV[2]], color = 'r', linewidth = FrameLW)
