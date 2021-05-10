from PEBBLE import ModuleBase, OdometryEvent, TrackerEvent, DisparityEvent, PoseEvent
from functools import lru_cache
import numpy as np

_MAX_DEPTH = 1e3

class PoseEstimator(ModuleBase):
    def _OnCreation(self):
        '''
        Event-based pose estimator.
        Uses a simulator mechanical system that combines all the constraints, such as points tracked on screen and visual odometry
        '''

        self._MonitorDt = 0. # By default, a module does not stode any date over time.
        self._NeedsLogColumn = False
        self._MonitoredVariables = []

        self._DefaultK = 450
        self._StereoBaseVector = np.array([0.1, 0., 0.])
        self._CameraIndexToOffsetRatio = {0:0.5, 1:-0.5}

        self._InitialCameraRotationVector = np.zeros(3)
        self._InitialCameraTranslationVector = np.zeros(3)

        self._AverageLengthSpringMultiplier = 1.
        self._MuV = 1.
        self._MuOmega = 1.

        self._RightCameraSubStreamIndex = 0
        self._LeftCameraSubStreamIndex = 1
        self._MixerSubStreamIndex = 2

    def _InitializeModule(self):
        self.RigFrame = FrameClass(self._InitialCameraRotationVector, self._InitialCameraTranslationVector, np.zeros(3), np.zeros(3), 0, 0)
        self.Anchors = {}
        self.TrackersLocationsAndDisparities = {}

        self.K = self._DefaultK
        self.ScreenCenter = np.array(self.Geometry)/2
        self.StereoBaseDistance = np.linalg.norm(self._StereoBaseVector)
        self.CameraOffsetLocations = [self._CameraIndexToOffsetRatio[CameraIndex] * self._StereoBaseVector for CameraIndex in self.__SubStreamInputIndexes__]

        self.KMatInv = np.inv(np.array([[self.K, 0., self.ScreenCenter[0]],
                                        [0., -self.K, self.ScreenCenter[1]],
                                        [0., 0.,     1.]]))

        self.AverageLength = 0.
        self.SpringEnergy = 0.
        self.KineticEnergy = 0.
        self.EnvironmentV = np.zeros(3)
        self.EnvironmentOmega = np.zeros(3)

        self.LastUpdateT = 0.

        return True

    def _OnEventModule(self, event):
        if event.Has(OdometryEvent) and event.SubStreamIndex == self._MixerSubStreamIndex:
            self.EnvironmentV = event.v
            self.EnvironmentOmega = event.omega
        elif event.Has(TrackerEvent):
            if event.TrackerColor == 'k':
                self.DelAnchor((event.SubStreamIndex, event.TrackerID))
            else:
                self.UpdateFromTracker(event.SubStreamIndex, event.TrackerLocation - self.ScreenCenter, event.TrackerID)
        self.Update(event.timestamp)
        return

    def UpdateRigLocationAndLinesOfSights(self, dt):
        self.RigFrame.UpdatePosition(dt)
        self.AverageLength = 0.
        for ID, Anchor in self.Anchors.items():
            self.UpdateAnchor(ID)
            Anchor.WorldProjection, Anchor.CurrentReprojection = GetSegmentsProjections(Anchor.WorldPresenceSegment, Anchor.CurrentPresenceSegment)
            ErrorVector = Anchor.WorldProjection - Anchor.CurrentReprojection
            Anchor.Length = np.linalg.norm(ErrorVector)
            self.AverageLength += Anchor.Length

        self.AverageLength /= len(self.Anchors)
        AnchorClass.SetAverageLength(self.AverageLength * self._AverageLengthSpringMultiplier)

        self.SpringEnergy = 0.
        for ID, Anchor in self.Anchors.items():
            self.SpringEnergy += Anchor.Energy

    @property
    def ViscosityForceAndTorque(self):
        return self._MuV * (self.EnvironmentV - self.RigFrame.V), self._MuOmega * (self.EnvironmentOmega - self.RigFrame.Omega)

    @property
    def SpringsForceAndTorque(self):
        TotalForce = np.zeros(3)
        TotalTorque = np.zeros(3)
        for Anchor in self.Anchors.values():
            Force = Anchor.Force
            ApplicationPoint = Anchor.CurrentReprojection
            Torque = np.cross(ApplicationPoint - self.RigFrame.X, Force)
            TotalForce += Force
            TotalTorque += Torque
        return TotalForce, TotalTorque

    @lru_cache(max_size = 100)
    def DepthOf(self, disparity):
        if disparity <= 0:
            return _MAX_DEPTH
        return self.StereoBaseDistance * self.K / disparity

    def UpdateFromTracker(self, SubStreamIndex, ScreenLocation, TID):
        ID = (TID, SubStreamIndex)
        if ID not in self.Anchors:
            self.AddAnchor(ID, ScreenLocation, disparity)
        else:
            self.TrackersLocationsAndDisparities[ID] = (np.array(ScreenLocation), disparity)

    def AddAnchor(self, ID, ScreenLocation, disparity):
        self.Anchors[ID] = AnchorClass(self.GetWorldLocation(ScreenLocation, disparity, ID[1]), self.RigFrame.X, (self.GetWorldLocation(ScreenLocation, disparity+0.5), self.GetWorldPresenceSegment(ScreenLocation, disparity-0.5, ID[1])))
        self.TrackersLocationsAndDisparities[ID] = (np.array(ScreenLocation), disparity)

    def DelAnchor(self, ID):
        del self.Anchors[ID]
        del self.TrackersLocationsAndDisparities[ID]

    def UpdateAnchor(self, ID):
        ScreenLocation, disparity = self.TrackersLocationsAndDisparities[ID]
        self.Anchors[ID].CurrentPresenceSegment = (self.GetWorldLocation(ScreenLocation, disparity+0.5, ID[1]), self.GetWorldPresenceSegment(ScreenLocation, disparity-0.5, ID[1]))

    def UpdateTracker(self, ID, ScreenLocation, disparity):

    def GetWorldLocation(self, ScreenLocation, disparity, CameraIndex):
        return self.RigFrame.ToWorld(self.KMatInv.dot(np.array([ScreenLocation[0], ScreenLocation[1], 1]))  * self.DepthOf(disparity) + self.CameraOffsetLocations[CameraIndex])

class FrameClass:
    def __init__(self, InitialTheta, InitialT, InitialOmega, InitialV, LambdaAlpha = 0, LambdaA = 0):
        self.Theta = np.array(InitialTheta) # Defines the rotation for an object from world to this frame
        self.T = np.array(InitialT)  # Defines the origin of this frame in the world
        self.Omega = np.array(InitialOmega)
        self.V = np.array(InitialV)

        self.LambdaAlpha = LambdaAlpha
        self.LambdaA = LambdaA

        self._UpToDate = False

    @property
    def R(self):
        if self._UpToDate:
            return self.RStored
        ThetaN = np.linalg.norm(self.Theta)
        if ThetaN == 0:
            self.RStored = np.identity(3)
            return self.RStored
        c, s = np.cos(ThetaN), np.sin(ThetaN)
        Ux, Uy, Uz = self.Theta / ThetaN
        self.RStored = np.array([[c + Ux**2*(1-c), Ux*Uy*(1-c) - Uz*s, Ux*Uz*(1-c) + Uy*s],
                      [Uy*Ux*(1-c) + Uz*s, c + Uy**2*(1-c), Uy*Uz*(1-c) - Ux*s],
                      [Uz*Ux*(1-c) - Uy*s, Uz*Uy*(1-c) + Ux*s, c + Uz**2*(1-c)]])
        return self.RStored

    @staticmethod
    def dR(dTheta):
        dThetaN = np.linalg.norm(dTheta)
        if dThetaN == 0:
            return np.identity(3)
        c, s = np.cos(dThetaN), np.sin(dThetaN)
        Ux, Uy, Uz = dTheta / dThetaN
        return np.array([[c + Ux**2*(1-c), Ux*Uy*(1-c) - Uz*s, Ux*Uz*(1-c) + Uy*s],
                      [Uy*Ux*(1-c) + Uz*s, c + Uy**2*(1-c), Uy*Uz*(1-c) - Ux*s],
                      [Uz*Ux*(1-c) - Uy*s, Uz*Uy*(1-c) + Ux*s, c + Uz**2*(1-c)]])

    def FromWorld(self, WorldLocation):
        return self.R.dot(WorldLocation - self.T)

    def ToWorld(self, FrameLocation):
        return self.R.T.dot(FrameLocation) + self.T

    def UpdatePosition(self, dt):
        self.Theta += dt * self.Omega
        self.X += dt * self.V

    def UpdateDerivatives(self, dt, Alpha, A):
        self.Omega += dt * (self.LambdaAlpha * Alpha + (1 - self.LambdaAlpha) * self.Alpha)
        self.V += dt * (self.LambdaA * A + (1-self.LambdaA)*self.A)
        self.Alpha = Alpha
        self.A = A

class AnchorClass:
    K_Base = 1.
    AverageLength = np.inf

    def __init__(self, WorldLocation, Origin, WorldPresenceSegment):
        self.WorldLocation = WorldLocation
        self.WorldPresenceSegment = WorldPresenceSegment

        self.CurrentPresenceSegment = (np.array(self.WorldPresenceSegment[0]), np.array(self.WorldPresenceSegment[1]))

        self.WorldProjection = np.array(self.WorldLocation)
        self.CurrentReprojection = np.array(self.WorldLocation)
        self.Length = 0.

        self.Origin = np.array(Origin)

    @classmethod
    def SetAverageLength(self, Length):
        self.AverageLength = Length

    @property
    def ForceNorm(self):
        return self.Length * self.K_Base * np.e**(-self.Length / self.AverageLength)

    @property
    def Force(self):
        return self.ForceNorm * (self.WorldProjection - self.CurrentReprojection) / max(0.001, self.Length)

    @property
    def Energy(self):
        ExponentialValue = np.e**(-self.Length / self.AverageLength)
        return self.K_Base * (self.AverageLength**2 * (1 - ExponentialValue) - self.AverageLength * self.Length * ExponentialValue)


def GetSegmentsProjections(S1, S2):
    # Calculate denomitator
    A = S1[1] - S1[0]
    B = S2[1] - S2[0]
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)
    
    _A = A / magA
    _B = B / magB
    
    cross = np.cross(_A, _B);
    denom = np.linalg.norm(cross)**2
    
    if not denom:
        d0 = np.dot(_A,(S2[0]-S1[0]))
        d1 = np.dot(_A,(S2[1]-S1[0]))
        
        if d0 <= 0 >= d1:
            if np.absolute(d0) < np.absolute(d1):
                return S1[0],S2[0]
            return S1[0],S2[1]
            
        elif d0 >= magA <= d1:
            if np.absolute(d0) < np.absolute(d1):
                return S1[1],S2[0]
            return S1[1],S2[1]
            
        # Segments overlap, return points in the middle
        return (S1[0]+S1[1])/2,(S2[0]+S2[1])/2
    
    
    # Lines criss-cross: Calculate the projected closest points
    t = (S2[0] - S1[0]);
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA/denom;
    t1 = detB/denom;

    pA = S1[0] + (_A * t0) # Projected closest point on segment A
    pB = S2[0] + (_B * t1) # Projected closest point on segment B

    # Clamp projections
    if t0 < 0:
        pA = S1[0]
    elif t0 > magA:
        pA = S1[1]
    
    if t1 < 0:
        pB = S2[0]
    elif t1 > magB:
        pB = S2[1]
        
    # Clamp projection A
    if (t0 < 0) or (t0 > magA):
        dot = np.dot(_B,(pA-S2[0]))
        if dot < 0:
            dot = 0
        elif dot > magB:
            dot = magB
        pB = S2[0] + (_B * dot)
    
    # Clamp projection B
    if (t1 < 0) or (t1 > magB):
        dot = np.dot(_A,(pB-S1[0]))
        if dot < 0:
            dot = 0
        elif dot > magA:
            dot = magA
        pA = S1[0] + (_A * dot)

    return pA,pB
