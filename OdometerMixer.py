from ModuleBase import ModuleBase
from Events import CameraEvent, TwistEvent, DisparityEvent
import numpy as np

import random

class OdometerMixer(ModuleBase):
    def _OnCreation(self):
        '''
        Computed stereo-rig odometry from stereo cameras odometry input
        '''
        self.__GeneratesSubStream__ = True
        self._MonitorDt = 0.
        self._MonitoredVariables = [('D', float),
                                    ('omega', np.array),
                                    ('V', np.array)]

        self._ReferenceIndex = 0

        self._NVirtualFrames = 2
        self._ResetVirtualFrameWhenOut = True
        self._AddStaticFrame = True
        self._FrameCenterRatioResetTrigger = 0.95
        self._StereoBaseVector = None
        self._StereoBaseRotation = None
        self._DefaultStereoBaseDistance = 0.1
        self._MaxCamToCamRotationSpeedError = 0.3
        self._ValidTimespanForStart = 0.02
        self._DisparityRange = [0, 100]

        self._Tau = 0.01

        self._VirtualFrameDefaultRotation = np.zeros(3)
        self._VirtualReferenceFrameLength = 1.

        self._KMat = None

        self._DefaultK = 450

    def _OnInitialization(self):
        self.omegaReference = np.zeros(3)
        self.VReference = np.zeros(3)

        self.omegaStereo = np.zeros(3)
        self.VStereo = np.zeros(3)

        self.R = np.identity(3) # R denotes rotation from world to stereoBase
        self.T = np.zeros(3) # T represents the world location of the stereo base

        self.Started = False
        self.IdleCams = [True, True]
        self.LastT = -np.inf

        self.StartAfter = np.inf

        self.ScreenSize = np.array(self.Geometry)
        self.ScreenCenter = self.ScreenSize / 2

        if self._StereoBaseVector is None:
            self.StereoBaseVector = np.array([self._DefaultStereoBaseDistance, 0., 0.])
            self.StereoBaseDistance = self._DefaultStereoBaseDistance
        else:
            self.StereoBaseVector = np.array(self._StereoBaseVector)
            self.StereoBaseDistance = np.linalg.norm(self.StereoBaseVector)
        
        if self._StereoBaseRotation is None:
            self.StereoBaseRotation = np.identity(3)
        else:
            self.StereoBaseRotation = np.array(self._StereoBaseRotation)

        if self._KMat is None:
            self.KMat = [np.array([[self._DefaultK, 0., self.ScreenSize[0]/2],
                                    [0., -self._DefaultK, self.ScreenSize[1]/2],
                                    [0., 0., 1.]]) for _ in range(2)]
            self.K = self._DefaultK
        else:
            self.KMat = [np.array(KMat) for KMat in self._KMat]
            self.K = self.KMat[0][0,0] / self.KMat[0][-1,-1]
        self.KInv = np.linalg.inv(self.KMat[self._ReferenceIndex])

        self.MinDepthFrameReset = self.Depth(self._DisparityRange[1])

        self.AverageDisparity = np.mean(self._DisparityRange)

        if self._NVirtualFrames:
            DX = self.ScreenSize[0] / max(1, self._NVirtualFrames)
            self.VirtualFramesStartLocations = [np.array([DX/2 + n*DX, self.ScreenCenter[1]]) for n in range(self._NVirtualFrames)]
            self.OnScreenVirtualFrames = np.ones(self._NVirtualFrames, dtype = bool)
            Dd = (self._DisparityRange[1] - self._DisparityRange[0]) / max(1, self._NVirtualFrames)
            self.VirtualFramesStartDisparities = [Dd/2 + n*Dd for n in range(self._NVirtualFrames)]

            self.VFCL = [np.array([0, 0, 0]),
                         np.array([1, 0, 0]),
                         np.array([0, 1, 0]),
                         np.array([0, 0, 1])]
            theta = np.linalg.norm(self._VirtualFrameDefaultRotation)
            if theta:
                u = self._VirtualFrameDefaultRotation / theta
                u_x = np.array([[0., -u[2], u[1]],
                                [u[2], 0., -u[0]],
                                [-u[1], u[0], 0.]])
                self.VFR = np.cos(theta) * np.identity(3) + np.sin(theta) * u_x + (1-np.cos(theta))* u.reshape((3,1)).dot(u.reshape((3,1)).T)
            else:
                self.VFR = np.identity(3)
            self.VirtualFramesCornersLocations = [self.ComputeVirtualFrameCorners(x, d) for x, d in zip(self.VirtualFramesStartLocations, self.VirtualFramesStartDisparities)]
        if self._AddStaticFrame:
            self.VirtualStaticFrameCornersLocations = self.ComputeVirtualFrameCorners(self.ScreenCenter, self.AverageDisparity)

        self.ASum = 0.
        self.DSum = 0.
        self.LastDUpdateT = -np.inf

        self.OmegaToMotionMatrix = np.array([[0.          , 0., 0.          , -1., 0.          , 0.],
                                             [0.          , -1., 0.          , 0. , 0.          , 0.],
                                             [0.          , 0., 0.          , 0. , 0.          , -1.],
                                             [-self.StereoBaseDistance, 0., 0.          , 0. , 0.          , 0.],
                                             [0.          , 0., self.StereoBaseDistance, 0. , 0.          , 0.],
                                             [0.          , 0., 0.          , 0. , self.StereoBaseDistance, 0.]])
        self.MotionToOmegaMatrix = np.linalg.inv(self.OmegaToMotionMatrix)

        return True

    def _OnEventModule(self, event):
        if event.Has(TwistEvent):
            if event.SubStreamIndex == self._ReferenceIndex:
                self.VReference = np.array(event.v)
                self.omegaReference = np.array(event.omega)
            else:
                self.VStereo = np.array(event.v)
                self.omegaStereo = np.array(event.omega)
            if not self.Started:
                self.IdleCams[event.SubStreamIndex] = False
                if not True in self.IdleCams:
                    if abs(self.omegaReference - self.omegaStereo).max() <= self._MaxCamToCamRotationSpeedError:
                        if event.timestamp > self.StartAfter:
                            self.LogSuccess("Started odometry")
                            self.LastT = event.timestamp
                            self.Started = event.timestamp
                        elif self.StartAfter == np.inf:
                            self.StartAfter = event.timestamp + self._ValidTimespanForStart
                            self.LogSuccess("Planning to start at {0:.3f}".format(self.StartAfter))
                    else:
                        if self.StartAfter != np.inf:
                            self.LogWarning("Cancelled start planned")
                            self.StartAfter = np.inf

        if self.Started:
            self.UpdateFrame(event.timestamp)
            self.UpdateD(event.timestamp)
            if self._NVirtualFrames or self._AddStaticFrame:
                VirtualFrameLocation, disparity, sign = self.GenerateVirtualEvent(event.SubStreamIndex)
                if not VirtualFrameLocation is None:
                    NewEvent = event.Join(CameraEvent, location = VirtualFrameLocation, polarity = 0)
                    if not disparity is None:
                        NewEvent.Attach(DisparityEvent, disparity = disparity, sign = 1-2*event.SubStreamIndex)
                else:
                    if self._ResetVirtualFrameWhenOut and event.SubStreamIndex == self._ReferenceIndex:
                        CloseToCenter = sign
                        if CloseToCenter:
                            nFrame = disparity
                            self.ResetFrame(nFrame)
            event.Join(TwistEvent, SubStreamIndex = self.GeneratedOdometryStreamIndex, omega = self.omega, v = self.V)
        return

    def _OnInputIndexesSet(self, Indexes):
        if len(Indexes) != 1:
            self.LogWarning("Improper number of generated streams specified")
            return False
        self.GeneratedOdometryStreamIndex = Indexes[0]
        return True

    def _OnTauRequest(self, event = None):
        if not self.Started:
            return 0
        if event is None or not event.Has(CameraEvent):
            dx, dy = self.ScreenCenter
        else:
            dx, dy = np.array(event.location) - self.ScreenCenter
        if event is None or not event.Has(DisparityEvent):
            d = self.AverageDisparity
        else:
            d = event.disparity
        omega = self.omega
        if event is None:
            V = self.V
        else:
            V = self.V + np.cross(self.omega, self.StereoBaseVector ** (-1)**event.SubStreamIndex)
        Q = np.array([[d, self.K + dx**2/self.K, 0., dx*dy/self.K, dx*d/self.K, dy],
                                     [0., dx*dy/self.K, d, self.K + dy**2/self.K, dy*d/self.K, -dx]])
        ExpectedVelocity = np.linalg.norm(Q.dot(self.MotionToOmegaMatrix.dot(np.concatenate((omega, V)))))
        if not ExpectedVelocity == 0:
            return 1./ExpectedVelocity
        else:
            return 0

    @property
    def omega(self):
        return (self.omegaReference + self.omegaStereo)/2
    @property
    def V(self):
        return (self.VReference + self.VStereo)/2
        #return self.VReference

    @property
    def D(self):
        return self.DSum / max(0.01, self.ASum)
    @property
    def DComp(self):
        omega = self.omega
        if np.linalg.norm(self.omega[1:]) == 0:
            return None
        return np.sqrt((((self.VReference[1] - self.VStereo[1]) * np.array([0,1,1]))**2).sum() / (omega[1]**2 + omega[2]**2))

    def UpdateFrame(self, t):
        Delta = (t - self.LastT)
        InstantRotation = Delta * self.omega
        theta = np.linalg.norm(InstantRotation)
        self.LastT = t
        if theta == 0:
            self.T -= self.V * Delta
            return
        u = InstantRotation / theta
        u_x = np.array([[0., -u[2], u[1]],
                        [u[2], 0., -u[0]],
                        [-u[1], u[0], 0.]])
        R = np.cos(theta) * np.identity(3) + np.sin(theta) * u_x + (1-np.cos(theta))* u.reshape((3,1)).dot(u.reshape((3,1)).T)
        self.R = self.R.dot(R)
        self.T = -self.V * Delta + R.dot(self.T)

    def UpdateD(self, t):
        DComp = self.DComp
        if DComp is None:
            return
        decay = np.e**((self.LastDUpdateT - t)/self._Tau)
        self.ASum = self.ASum * decay + 1
        self.DSum = self.DSum * decay + DComp
        self.LastDUpdateT = t

    def GenerateVirtualEvent(self, CameraIndex):
        nFrame = random.randint(0, self._NVirtualFrames-1+self._AddStaticFrame)
        nDim = random.randint(1,3)
        u = random.random()
        return self.ComputeFrameLocationOnScreen(nFrame, CameraIndex, nDim, u)

    def ComputeVirtualFrameCorners(self, x, d):
        Z = self.Depth(d)
        xH = np.array([x[0], x[1], 1])
        XH = self.KInv.dot(xH) 
        X = Z * XH / XH[-1]
        X3D = self.R.dot(X - self.T)
        
        VirtualFrameCornersLocations = []
        for nCorner, Corner in enumerate(self.VFCL):
            VirtualFrameCornersLocations += [self.VFR.dot(Corner[:3]) + X3D]
        return VirtualFrameCornersLocations

    def Disparity(self, depth):
        return int(abs(self.StereoBaseDistance * self.K / depth) + 0.5)
    def Depth(self, disparity):
        return (self.StereoBaseDistance * self.K / disparity)

    def ResetFrame(self, nFrameReset):
        self.Log("Resetting frame {0}".format(nFrameReset))
        MaxScreenDistance = np.inf*np.ones(self._NVirtualFrames)
        MaxDisparityDistance = np.inf*np.ones(self._NVirtualFrames)
        for nFrame in range(self._NVirtualFrames):
            if nFrame == nFrameReset:
                continue
            x, d, sign = self.ComputeFrameLocationOnScreen(nFrame, self._ReferenceIndex)
            if x is None:
                self.LogWarning("Two frames out of screen at once")
                continue
            for nFrameIni in range(self._NVirtualFrames):
                MaxScreenDistance[nFrameIni] = min(MaxScreenDistance[nFrameIni], np.linalg.norm(self.VirtualFramesStartLocations[nFrameIni] - x))
                MaxDisparityDistance[nFrameIni] = min(MaxDisparityDistance[nFrameIni], abs(self.VirtualFramesStartDisparities[nFrameIni] - d))
        x = self.VirtualFramesStartLocations[MaxScreenDistance.argmax()]
        d = int(self.VirtualFramesStartDisparities[MaxDisparityDistance.argmax()])
        self.VirtualFramesCornersLocations[nFrameReset] = self.ComputeVirtualFrameCorners(x, d)

    def ComputeFrameLocationOnScreen(self, nFrame, CameraIndex, nDim=0, u=1):
        if not self._AddStaticFrame or nFrame < self._NVirtualFrames:
            WorldPoint3D = u * self.VirtualFramesCornersLocations[nFrame][0] + (1-u) * self.VirtualFramesCornersLocations[nFrame][nDim]
            X = self.R.T.dot(WorldPoint3D) + self.T # In Camera Reference Frame
        else:
            X = self.VirtualStaticFrameCornersLocations[0] + (0.5-u)*self.R.T.dot(self.VirtualStaticFrameCornersLocations[nDim] - self.VirtualStaticFrameCornersLocations[0])
        if CameraIndex != self._ReferenceIndex:
            X = self.StereoBaseRotation.dot(X) + self.StereoBaseVector # If we want the stereo camera, we change the basis
        if X[2] > 0: # with these corrdinates, Z looks at the front of the camera
            xproj = self.KMat[CameraIndex].dot(X) # We project. The result should be in a Z-forward, y-upward basis
            x = np.array(xproj[:2]/xproj[2]+0.5, dtype = int)
            if (x>=0).all() and (x<self.ScreenSize).all():
                sign = 1-2*CameraIndex
                d = self.Disparity(X[2])
                if d > self._DisparityRange[1]:
                    return None, nFrame, (u > self._FrameCenterRatioResetTrigger)
                return x, d, sign 
            else:
                return None, nFrame, (u > self._FrameCenterRatioResetTrigger)
        else:
            return None, nFrame, (u > self._FrameCenterRatioResetTrigger)
