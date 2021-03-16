from PEBBLE import Module, OdometryEvent
import numpy as np

import random

class OdometerMixer(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Module template to be filled foe specific purpose
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Computation'

        self.__ReferencesAsked__ = []
        self._MonitorDt = 0. # By default, a module does not stode any date over time.
        self._NeedsLogColumn = False
        self._MonitoredVariables = [('D', float),
                                    ('Omega', np.array),
                                    ('V', np.array)]

        self._ReferenceIndex = 0

        self._AddVirtualFrame = True
        self._VirtualReferenceFrameDisparity = 5.
        self._VirtualReferenceFrameLength = 1.
        self._StereoBaseDistance = 0.1
        
        self._K = 432

    def _InitializeModule(self, **kwargs):
        self.ReferenceOmega = np.zeros(3)
        self.ReferenceV = np.zeros(3)

        self.StereoOmega = np.zeros(3)
        self.StereoV = np.zeros(3)

        self.R = np.identity(3) # R denotes rotation from world to stereoBase
        self.T = np.zeros(3) # T represents the world location of the stereo base

        self.Started = False
        self.IdleCams = [True, True]
        self.LastT = -np.inf

        self.ScreenSize = np.array(self.Geometry)[:2]

        self.StereoBaseVector = np.array([self._StereoBaseDistance, 0., 0.])
        self.KMat = np.array([[self._K, 0., self.ScreenSize[0]/2], 
                              [0., self._K, self.ScreenSize[1]/2],
                              [0., 0., 1.]])
        
        self.VirtualFrameCornersLocations = [np.array([self._StereoBaseDistance/2, 0., -self._StereoBaseDistance * self._K / self._VirtualReferenceFrameDisparity, 1.]),
                                      np.array([self._StereoBaseDistance/2 + 1, 0., -self._StereoBaseDistance * self._K / self._VirtualReferenceFrameDisparity, 1.]),
                                      np.array([self._StereoBaseDistance/2, +1, -self._StereoBaseDistance * self._K / self._VirtualReferenceFrameDisparity, 1.]),
                                      np.array([self._StereoBaseDistance/2, 0., -self._StereoBaseDistance * self._K / self._VirtualReferenceFrameDisparity + 1, 1.])]

        return True

    def _OnEventModule(self, event):
        if event.Has(OdometryEvent):
            if event.cameraIndex == self._ReferenceIndex:
                self.ReferenceV = np.array(event.v)
                self.ReferenceOmega = np.array(event.omega)
            else:
                self.StereoV = np.array(event.v)
                self.StereoOmega = np.array(event.omega)
            self.IdleCams[event.cameraIndex] = False
            if not self.Started:
                self.LastT = event.timestamp
                self.IdleCams[event.cameraIndex] = False
                if not True in self.IdleCams:
                    self.Started = True

        if self.Started:
            pass
            
        return event

    @property
    def Omega(self):
        return (self.ReferenceOmega + self.StereoOmega)/2
    @property
    def V(self):
        return (self.ReferenceV + self.StereoV - np.cross(self.Omega, self.StereoBaseVector))/2

    @property
    def D(self):
        Omega = self.Omega
        if np.linalg.norm(self.Omega[1:]) == 0:
            return 0
        return -(Omega[2] * (self.ReferenceV[1] - self.StereoV[1]) - Omega[1] * (self.ReferenceV[2] - self.StereoV[2])) / (Omega[1]**2 + Omega[2]**2)

    def UpdateFrame(self, t):
        Delta = (t - self.LastT)
        InstantRotation = Delta * self.Omega
        theta = np.linalg.norm(InstantRotation)
        self.LastT = t
        self.T += self.R.T.dot(self.V * Delta)
        if theta == 0:
            return
        u = InstantRotation / theta
        u_x = np.array([[0., -u[2], u[1]],
                        [u[2], 0., -u[0]],
                        [-u[1], u[0], 0.]])
        R = np.cos(theta) * np.identity(3) + np.sin(theta) * u_x + (1-np.cos(theta))* u.reshape((3,1)).dot(u.reshape((3,1)).T)
        self.R = self.R.dot(R)

    def GenerateVirtualEvent(self, CameraIndex):
        T = self.T
        if CameraIndex != self._ReferenceIndex:
            T += self.R.T.dot(self.StereoBaseVector)

        nDim = random.randint(1,3)
        u = random.random()
        Point3D = u * self.VirtualFrameCornersLocations[0] + (1-u) * self.VirtualFrameCornersLocations[nDim]

        RT = np.concatenate((self.R.T, T.reshape((1,3)))).T
        X = RT.dot(Point3D)
        x = self.KMat.dot(X)
        if x[2] != 0:
            x = np.array(x[:2]/x[2], dtype = int)
        else:
            return None
        if (x<0).any() or (x>=self.ScreenSize).any():
            return None
        return x
