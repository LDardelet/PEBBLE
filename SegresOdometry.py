import numpy as np

from Framework import Module, Event

class SegresOdometer(Module):
    '''
    Experimental module computing a simple odometry from tracking points on the screen using segres embeeding, from the 6 DoF 3D space to a 10 variable (almost) homogenous space.
    Requires to have attached to the events an array v = [vx, vy] from a tracker.
    Equation terms are:
        Segre's variable    | Multiplier
            v_x / \Tau          -K_2 * \Tau * \dot{dy}
            v_y / \Tau          K_2 * \Tau * \dot{dx}
            v_z / \Tau          dx * \dot{dy} - \dot{dx} * dy
            \omega_x * v_x      K_1 * K_2
            \omega_x * v_z      -K_1 * dx
            \omega_y * v_y      -K_1 * K_2
            \omega_y * v_z      K_1 * dy
            \omega_z * v_x      -K_2 * dx
            \omega_z * v_y      -K_2 * dy
            \omega_z * v_z      dx^2 + dy^2
    '''
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to handle ST-context memory.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Computation'
        self.__RewindForbidden__ = True

        #self._TypicalDistance = 1. # Typical distance of the observed scene, to normalize rotational and translational velocities
        self._TypicalTimeConstant = 1. # Typical time constant of the observed scene, to normalize rotational and translational velocities
        self._K1 = 100.
        self._K2 = 100.

    def _InitializeModule(self, **kwargs):
        self.CurrentSegrePoint = np.zeros((10))

        return True

    def _OnEventModule(self, event):
        if 'v' in event.__dict__.keys():
            PlanConstants = self._CreatePlanConstants(event)


        return event

    def _CreatePlanConstants(self, event):
        return np.array([ -self._K2 * self._TypicalTimeConstant * event.v[1],
                           self._K2 * self._TypicalTimeConstant * event.v[0],
                           event.location[0] * event.v[1] - event.location[1] * event.v[0], 
                           self._K1 * self._K2,
                          -self._K1 * event.location[0],
                          -self._K1 * self._K2,
                           self._K1 * event.location[1],
                          -self._K2 * event.location[0],
                          -self._K2 * event.location[1],
                           event.location[0]**2 + event.location[1]**2])

    def ComputeValidSolutionDistance(self):
        Eq1Distance = abs(self.CurrentSegrePoint[2] * self.CurrentSegrePoint[3] - self.CurrentSegrePoint[4] * self.CurrentSegrePoint[0])
        Eq2Distance = abs(self.CurrentSegrePoint[2] * self.CurrentSegrePoint[5] - self.CurrentSegrePoint[6] * self.CurrentSegrePoint[1])
        Eq3Distance = abs(self.CurrentSegrePoint[7] * self.CurrentSegrePoint[1] - self.CurrentSegrePoint[8] * self.CurrentSegrePoint[0])
        Eq4Distance = abs(self.CurrentSegrePoint[7] * self.CurrentSegrePoint[2] - self.CurrentSegrePoint[9] * self.CurrentSegrePoint[0])

        return Eq1Distance + Eq2Distance + Eq3Distance + Eq4Distance

    def ComputeCurrentRealSolution(self):
        V = self._TypicalTimeConstant * self.CurrentSegrePoint[:3]
        Omega = np.array([0., 0., 0.])
        Nx = 0
        Ny = 0
        Nz = 0
        if V[0] != 0:
            Nx += 1
            Nz += 1
            Omega[0] += self.CurrentSegrePoint[3] / V[0]
            Omega[2] += self.CurrentSegrePoint[7] / V[0]
        if V[1] != 0:
            Ny += 1
            Nz += 1
            Omega[1] += self.CurrentSegrePoint[5] / V[1]
            Omega[2] += self.CurrentSegrePoint[8] / V[1]
        if V[2] != 0:
            Nx += 1
            Ny += 1
            Nz += 1
            Omega[0] += self.CurrentSegrePoint[4] / V[2]
            Omega[1] += self.CurrentSegrePoint[6] / V[2]
            Omega[2] += self.CurrentSegrePoint[9] / V[2]
        if Nx:
            Omega[0] /= Nx
        if Ny:
            Omega[1] /= Ny
        if Nz:
            Omega[2] /= Nz

        return V, Omega
