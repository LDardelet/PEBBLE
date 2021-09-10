from PEBBLE import ModuleBase, TwistEvent, TrackerEvent, DisparityEvent, PoseEvent
from functools import lru_cache
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

_MAX_DEPTH = 1e3

class PoseEstimator(ModuleBase):
    def _OnCreation(self):
        '''
        Event-based pose estimator.
        Uses a simulator mechanical system that combines all the constraints, such as points tracked on screen and visual odometry
        '''
        self.__GeneratesSubStream__ = True
        self._MonitorDt = 0.001 # By default, a module does not stode any date over time.
        self._NeedsLogColumn = True
        self._MonitoredVariables = [('RigFrame.T', np.array),
                                    ('RigFrame.Theta', np.array),
                                    ('RigFrame.V', np.array),
                                    ('RigFrame.Omega', np.array),
                                    ('ReceivedV', np.array),
                                    ('ReceivedOmega', np.array),
                                    ('MechanicalEnergy', float),
                                    ('PotentialEnergy', float),
                                    ('KineticEnergy', float),
                                    ('TrackersEnergy', float),
                                    ('EnvironmentEnergy', float),
                                    ('BrokenSpringsEnergyLoss', float),
                                    ('MuV', float),
                                    ('MuOmega', float),
                                    ('NaturalPeriod', float),
                                    ('AverageSpringLength', float),
                                    ('Anchors@Energy', float),
                                    ('Anchors@Length', float)]

        self.__ModulesLinksRequested__ = ['RightDisparityMemory', 'LeftDisparityMemory']

        self._DefaultK = 450
        self._CameraOffsets = [np.array([-0.1, 0., 0.]), np.array([0.1, 0., 0.])]

        self._InitialCameraRotationVector = np.zeros(3)
        self._InitialCameraTranslationVector = np.zeros(3)

        self._ConstantSpringTrustDistance = 0.3 # to diable, use np.inf
        self._SpringTrustDistanceModel = "Gaussian" # str, either Constant ( 1 ), Gaussian ( e**(-delta**2) ) or Exponential ( e**(-delta) )
        self._MuV = 5.
        self._MuOmega = 5.
        self._Mass = 0.005
        self._Moment = 0.05

        self._MinTrackersPerCamera = 8
        self._NeedsStereoOdometry = True
        self._DefaultTau = 0.05
        self._TrackerDisparityRadius = 5
        self._DisparityTauRatio = 10

        self._DisparityOverlap = 0.2

        self._MaxLengthRatioBreak = np.inf
        self._MaxAbsoluteSpringLength = 2.

        self._UseAdaptiveDampening = True
        self._AdaptativeDampeningFactor = 0.5 # Should be equal to 1 mathwise

        self._UseSpeedOdometry = True # WARNING : Should be True
        self._AddLiveTrackers = True # WARNING : Should definitely be True
        self._CanDeactivateAnchor = False

    def _OnInitialization(self):
        self.RigFrame = FrameClass(self._InitialCameraRotationVector, self._InitialCameraTranslationVector, np.zeros(3), np.zeros(3), 0, 0)
        self.Anchors = {}

        self.K = self._DefaultK
        self.ScreenSize = np.array(self.Geometry)
        self.ScreenCenter = self.ScreenSize/2
        self.StereoBaseDistance = np.linalg.norm(self._CameraOffsets[0] - self._CameraOffsets[1])
        self.CameraOffsetLocations = [self._CameraOffsets[CameraIndex] for CameraIndex in self.__SubStreamInputIndexes__]

        self.KMat = np.array([[self.K, 0., self.ScreenCenter[0]],
                                        [0., -self.K, self.ScreenCenter[1]],
                                        [0., 0.,     1.]])
        self.KMatInv = np.linalg.inv(self.KMat)

        try:
            DisparityMemories = [self.LeftDisparityMemory, self.RightDisparityMemory]
            self.DisparityMemories = {}
            for SubStreamIndex in self.__SubStreamInputIndexes__:
                for DisparityMemory in DisparityMemories:
                    if SubStreamIndex in DisparityMemory.__SubStreamOutputIndexes__:
                        self.DisparityMemories[SubStreamIndex] = DisparityMemory
                        continue
            self._UseDisparityMemories = True
        except:
            self.LogWarning("No disparity modules linked, expects disparity events for each incomming tracker event")
            self._UseDisparityMemories = False

        if self._ConstantSpringTrustDistance:
            AnchorClass.SetTrustDistance(self._ConstantSpringTrustDistance)
        AnchorClass.SetTrustModel(self._SpringTrustDistanceModel)

        self.AverageSpringLength = 0.
        self.TotalSpringConstant = 0.
        self.TotalSpringTorqueBase = 0.

        self.LastUpdateT = 0.

        self.TrackersEnergy = 0.
        self.EnvironmentEnergy = 0.
        self.BrokenSpringsEnergyLoss = 0.
        self.UpdatedSpringsPreviousEnergies = {}

        self.Started = False
        self.ReceivedOdometry = np.zeros(2, dtype = bool)
        if not self._UseSpeedOdometry:
            self.ReceivedOdometry[:] = 1
        self.TrackersPerCamera = {Index:0 for Index in self.__SubStreamInputIndexes__}

        self.ReceivedV = np.zeros((3, 2))
        self.ReceivedOmega = np.zeros((3,2))

        self.DisparitySegmentWidth = 0.5 + self._DisparityOverlap / 2

        return True

    def _SetGeneratedSubStreamsIndexes(self, Indexes):
        if len(Indexes) != 1:
            self.LogWarning("Improper number of generated streams specified")
            return False
        self.StereoRigSubStream = Indexes[0]
        return True

    def _OnEventModule(self, event):
        self.UpdatedSpringsPreviousEnergies = {}
        if event.Has(TwistEvent):
            self.ReceivedOdometry[event.SubStreamIndex] = True
            self.ReceivedV[:,event.SubStreamIndex] = event.v
            self.ReceivedOmega[:,event.SubStreamIndex] = event.omega
        if event.Has(TrackerEvent):
            if event.TrackerColor == 'k':
                ID = (event.TrackerID, event.SubStreamIndex)
                if ID in self.Anchors:
                    self.RemoveAnchor(ID, "tracker disappeared")
            elif event.TrackerColor == 'g' and event.TrackerMarker == 'o': # Tracker locked
                if event.Has(DisparityEvent):
                    disparity = event.disparity
                else:
                    disparity = None
                self.UpdateFromTracker(event.SubStreamIndex, event.TrackerLocation, event.TrackerID, disparity)
        if not self.Started:
            if self.ReceivedOdometry.all() or (not self._NeedsStereoOdometry and self.ReceivedOdometry.any()):
                CanStart = True
                for NTracker in self.TrackersPerCamera.values():
                    if NTracker < self._MinTrackersPerCamera:
                        CanStart = False
                        break
                if not CanStart:
                    return
                self.Started = event.timestamp
                self.LastUpdateT = event.timestamp
                self.LogSuccess("Started")
            else:
                return
        self.Update(event.timestamp)
        return

    def RemoveAnchor(self, ID, reason):
        if self.Anchors[ID].Active:
            self.TrackersPerCamera[ID[1]] -= 1
        if ID in self.UpdatedSpringsPreviousEnergies:
            self.BrokenSpringsEnergyLoss += self.UpdatedSpringsPreviousEnergies[ID]
            del self.UpdatedSpringsPreviousEnergies[ID]
        else:
            self.BrokenSpringsEnergyLoss += self.Anchors[ID].Energy
        del self.Anchors[ID]
        self.Log("Removed anchor {0} ({1})".format(ID, reason))

    def Update(self, t):
        dt = t - self.LastUpdateT
        self.LastUpdateT = t

        self.RigFrame.UpdatePosition(dt)
        self.UpdateLinesOfSights()

        for ID, PreviousEnergy in self.UpdatedSpringsPreviousEnergies.items():
            self.TrackersEnergy += (self.Anchors[ID].Energy - PreviousEnergy)
        self.EnvironmentEnergy += dt * self.EnvironmentPower

        Force, Torque = self.ForceAndTorque
        A, Alpha = Force / self._Mass, Torque / self._Moment

        self.RigFrame.UpdateDerivatives(dt, Alpha, A)

    def UpdateFromTracker(self, SubStreamIndex, ScreenLocation, TrackerID, disparity = None):
        ID = (TrackerID, SubStreamIndex)
        Tau = self.FrameworkAverageTau
        if Tau is None or Tau == 0:
            Tau = self._DefaultTau
        if self._UseDisparityMemories:
            disparity = self.DisparityMemories[SubStreamIndex].GetDisparity(Tau*self._DisparityTauRatio, np.array(ScreenLocation, dtype = int), self._TrackerDisparityRadius)
        if ID not in self.Anchors:
            if self.Started and not self._AddLiveTrackers:
                return
            if disparity is None:
                return
            self.Anchors[ID] = AnchorClass(self.GetWorldLocation(ScreenLocation, disparity, SubStreamIndex), self.RigFrame.T, (self.GetWorldLocation(ScreenLocation, disparity+self.DisparitySegmentWidth, SubStreamIndex), self.GetWorldLocation(ScreenLocation, disparity-self.DisparitySegmentWidth, SubStreamIndex)), OnCameraDataClass(SubStreamIndex, np.array(ScreenLocation), disparity), self.GetWorldLocation)
            self.TrackersPerCamera[SubStreamIndex] += 1
            self.Log("Added anchor {0}".format(ID))
        else:
            Anchor = self.Anchors[ID]
            self.UpdatedSpringsPreviousEnergies[ID] = Anchor.Energy
            Anchor.OnCameraData.Location[:] = ScreenLocation
            if not disparity is None:
                if not self.Started:
                    Anchor.SetWorldData(self.GetWorldLocation(ScreenLocation, disparity, SubStreamIndex), (self.GetWorldLocation(ScreenLocation, disparity+self.DisparitySegmentWidth, SubStreamIndex), self.GetWorldLocation(ScreenLocation, disparity-self.DisparitySegmentWidth, SubStreamIndex)))
                Anchor.OnCameraData.Disparity = disparity
                if not Anchor.Active:
                    Anchor.Active = True
                    self.LogSuccess("Reactivated anchor {0}".format(ID))
                    self.TrackersPerCamera[SubStreamIndex] += 1
            else:
                if Anchor.Active and self._CanDeactivateAnchor:
                    Anchor.Active = False
                    self.LogWarning("Deactivated anchor {0} (no disparity)".format(ID))
                    self.TrackersPerCamera[SubStreamIndex] -= 1

    def UpdateLinesOfSights(self):
        MaxLength = 0
        MaxID = None
        self.AverageSpringLength = 0.
        self.TotalSpringConstant = 0.
        self.TotalSpringTorqueBase = 0.
        for nAnchor, (ID, Anchor) in enumerate(list(self.Anchors.items())):
            Anchor.UpdatePresenceSegment()
            if Anchor.Length > self._MaxAbsoluteSpringLength:
                self.RemoveAnchor(ID, 'excessive absolute length')
                continue
            if Anchor.Length > MaxLength:
                MaxLength = Anchor.Length
                MaxID = ID
            self.AverageSpringLength += Anchor.Length
            self.TotalSpringConstant += Anchor.K * Anchor.ExponentialValue
            self.TotalSpringTorqueBase += Anchor.K * Anchor.ExponentialValue * np.linalg.norm(Anchor.WorldLocation - self.RigFrame.T)**2

        N = len(self.Anchors)
        self.AverageSpringLength /= N
        if self._MaxLengthRatioBreak != np.inf and MaxLength > self._MaxLengthRatioBreak * self.AverageSpringLength:
            Anchor = self.Anchors[MaxID]
            self.TotalSpringConstant -= Anchor.K * Anchor.ExponentialValue
            self.TotalSpringTorqueBase -= Anchor.K * np.linalg.norm(Anchor.WorldLocation - self.RigFrame.T)**2
            self.RemoveAnchor(MaxID, 'excessive relative length')
            self.AverageSpringLength = (self.AverageSpringLength * N - MaxLength) / (N - 1)

    @property
    def PotentialEnergy(self):
        PotentialEnergy = 0.
        for ID, Anchor in self.Anchors.items():
            PotentialEnergy += Anchor.Energy
        return PotentialEnergy
    @property
    def KineticEnergy(self):
        return (self._Mass * (self.RigFrame.V**2).sum() + self._Moment * (self.RigFrame.Omega**2).sum()) / 2
    @property
    def MechanicalEnergy(self):
        return self.KineticEnergy + self.PotentialEnergy
    @property
    def ExchangedEnergy(self):
        return self.EnvironmentEnergy + self.TrackersEnergy

    @property
    def KTot(self):
        KTot = 0
        for ID, Anchor in self.Anchors.items():
            KTot += Anchor.K
        return KTot

    @property
    def NaturalPeriod(self):
        KTot = self.KTot
        if KTot == 0:
            return 0
        return 2*np.pi * np.sqrt(self._Mass / KTot)

    @property
    def EnvironmentV(self):
        if not self._UseSpeedOdometry:
            return np.zeros(3)
        return self.ReceivedV.mean(axis = 1)
    @property
    def EnvironmentOmega(self):
        if not self._UseSpeedOdometry:
            return np.zeros(3)
        return self.ReceivedOmega.mean(axis = 1)

    @property
    def ViscosityForceAndTorque(self):
        return self.MuV * (self.EnvironmentV - self.RigFrame.V), self.MuOmega * (self.EnvironmentOmega - self.RigFrame.Omega)
    @property
    def EnvironmentPower(self): # Defined as the instantaneous power transmitted to the rig frame
        Force, Torque = self.ViscosityForceAndTorque
        return ((Force*self.RigFrame.V).sum() + (Torque*self.RigFrame.Omega).sum())

    @property
    def MuV(self):
        if self._UseAdaptiveDampening:
            return 2 * np.sqrt(self._Mass * self.TotalSpringConstant) * self._AdaptativeDampeningFactor
        else:
            return self._MuV
    @property
    def MuOmega(self):
        if self._UseAdaptiveDampening:
            return 2 * np.sqrt(self._Moment * self.TotalSpringTorqueBase) * self._AdaptativeDampeningFactor
        else:
            return self._MuOmega

    @property
    def SpringsForceAndTorque(self):
        TotalForce = np.zeros(3)
        TotalTorque = np.zeros(3)
        for Anchor in self.Anchors.values():
            Force = Anchor.Force
            ApplicationPoint = Anchor.CurrentReprojection
            Torque = -np.cross(ApplicationPoint - self.RigFrame.T, Force)
            TotalForce += Force
            TotalTorque += Torque
        return TotalForce, TotalTorque

    @property
    def ForceAndTorque(self):
        TotalForce, TotalTorque = self.ViscosityForceAndTorque
        SpringsForce, SpringsTorque = self.SpringsForceAndTorque
        return TotalForce + SpringsForce, TotalTorque + SpringsTorque

    def DepthOf(self, disparity):
        if disparity <= 0:
            return _MAX_DEPTH
        return self.StereoBaseDistance * self.K / disparity

    def GetWorldLocation(self, ScreenLocation, disparity, CameraIndex):
        return self.RigFrame.ToWorld(self.KMatInv.dot(np.array([ScreenLocation[0], ScreenLocation[1], 1]))  * self.DepthOf(disparity) + self.CameraOffsetLocations[CameraIndex])

    def PlotSystem(self, Orientation = 'natural'):
        f = plt.figure()
        self.Plot3DSystem(Orientation, (f, f.add_subplot(121, projection='3d')))
        for nCamera in range(2):
            self.GenerateCameraView(nCamera, (f, f.add_subplot(2, 2, 2*(nCamera+1))))

    def PlotEnergies(self):
        f, ax = plt.subplots(1,1)
        self.PlotHistoryData("KineticEnergy", fax = (f, ax))
        self.PlotHistoryData("PotentialEnergy", fax = (f, ax))
        self.PlotHistoryData("MechanicalEnergy", fax = (f, ax))


    def Plot3DSystem(self, Orientation = 'natural', fax = None):
        if fax is None:
            f = plt.figure()
            ax = f.add_subplot(111, projection='3d')
        else:
            f, ax = fax

        if Orientation == 'natural':
            oldPlot = ax.plot
            oldScatter = ax.scatter
            oldText = ax.text
            import types
            def newPlot(ax, x, y, z, *args, **kwargs):
                oldPlot(np.array(x), np.array(z), -np.array(y), *args, **kwargs)
            ax.plot = types.MethodType(newPlot, ax)
            def newScatter(ax, x, y, z, *args, **kwargs):
                oldScatter(np.array(x), np.array(z), -np.array(y), *args, **kwargs)
            ax.scatter = types.MethodType(newScatter, ax)
            def newText(ax, x, y, z, *args, **kwargs):
                oldText(np.array(x), np.array(z), -np.array(y), *args, **kwargs)
            ax.text = types.MethodType(newText, ax)

            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('-Y')
        elif Orientation == 'initial':
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        for nDim in range(3):
            BaseVector = np.array([float(nDim == 0), float(nDim == 1), float(nDim == 2)])
            ax.plot([0, BaseVector[0]], [0, BaseVector[1]], [0, BaseVector[2]], 'k')
            ax.plot([self.RigFrame.T[0], self.RigFrame.ToWorld(BaseVector)[0]], [self.RigFrame.T[1], self.RigFrame.ToWorld(BaseVector)[1]], [self.RigFrame.T[2], self.RigFrame.ToWorld(BaseVector)[2]], 'b')
        CameraColors = ['r', 'b']
        for CameraIndex, CameraOffset in enumerate(self.CameraOffsetLocations):
            CameraWorldLocation = self.RigFrame.ToWorld(CameraOffset)
            ax.scatter(CameraWorldLocation[0], CameraWorldLocation[1], CameraWorldLocation[2], marker = 'o', color = CameraColors[CameraIndex])

        for ID, Anchor in self.Anchors.items():
            ax.plot([Anchor.WorldPresenceSegment[0][0], Anchor.WorldPresenceSegment[1][0]], [Anchor.WorldPresenceSegment[0][1], Anchor.WorldPresenceSegment[1][1]], [Anchor.WorldPresenceSegment[0][2], Anchor.WorldPresenceSegment[1][2]], color = CameraColors[ID[1]], linestyle = '--')
            ax.plot([Anchor.CurrentPresenceSegment[0][0], Anchor.CurrentPresenceSegment[1][0]], [Anchor.CurrentPresenceSegment[0][1], Anchor.CurrentPresenceSegment[1][1]], [Anchor.CurrentPresenceSegment[0][2], Anchor.CurrentPresenceSegment[1][2]], color = CameraColors[ID[1]])
            WP, CP = Anchor.WorldProjection, Anchor.CurrentReprojection
            CPpF = Anchor.CurrentReprojection + Anchor.Force
            ax.scatter(WP[0], WP[1], WP[2], color = 'k', marker = 'o')
            ax.scatter(CP[0], CP[1], CP[2], color = 'k', marker = 'o')
            ax.text(CP[0], CP[1], CP[2]+0.01, str(ID), color = 'k')
            ax.plot([CPpF[0], CP[0]], [CPpF[1], CP[1]], [CPpF[2], CP[2]], color = 'k')

        return f, ax

    def GenerateCameraView(self, nCamera, fax = None):
        if fax is None:
            f, ax = plt.subplots(1,1)
        else:
            f, ax = fax
        ax.set_aspect("equal")
        ax.set_xlim(0, self.ScreenSize[0] - 1)
        ax.set_ylim(0, self.ScreenSize[1] - 1)
        ax.plot(self.ScreenCenter[0], self.ScreenCenter[1], 'ok')
        CameraOffset = self.CameraOffsetLocations[nCamera]
        for (TID, TrackerCameraIndex), Anchor in self.Anchors.items():
            if TrackerCameraIndex != nCamera:
                continue
            for nLocation, (UsedLocation, UsedLocType) in enumerate(((Anchor.WorldProjection, 'world reprojection'), (Anchor.CurrentReprojection, 'current reprojection'))):
                CameraFrameLocation = self.RigFrame.FromWorld(UsedLocation) - CameraOffset
                DisplayFrameLocation = self.KMat.dot(CameraFrameLocation)
                if DisplayFrameLocation[-1] <= 0:
                    print("Anchor {0} {1} is behind the camera".format((TID, TrackerCameraIndex), UsedLocType))
                    continue
                OnScreenLocation = DisplayFrameLocation[:2] / DisplayFrameLocation[-1]
                if (OnScreenLocation < 0).any() or (OnScreenLocation > self.ScreenSize-1).any():
                    print("Anchor {0} is out of screen".format((TID, TrackerCameraIndex), UsedLocType))
                    continue
                ax.plot(OnScreenLocation[0], OnScreenLocation[1], marker = ['x', 'o'][nLocation], color = 'b')
            ax.plot(Anchor.OnCameraData.Location[0], Anchor.OnCameraData.Location[1], marker = 'o', color = 'g')
        return f, ax

class FrameClass:
    def __init__(self, InitialTheta, InitialT, InitialOmega, InitialV, LambdaAlpha = 0, LambdaA = 0): # Lambda is the amout of acceleration averaging with the previous value
        self.Theta = np.array(InitialTheta) # Defines the rotation for an object from world to this frame
        self.T = np.array(InitialT)  # Defines the origin of this frame in the world
        self.Omega = np.array(InitialOmega)
        self.V = np.array(InitialV)
        self.Alpha = np.zeros(3)
        self.A = np.zeros(3)

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
        self._UpToDate = True
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
        self.T += dt * self.V
        self._UpToDate = False

    def UpdateDerivatives(self, dt, Alpha, A):
        self.Omega += dt * ((1 - self.LambdaAlpha) * Alpha + self.LambdaAlpha * self.Alpha)
        self.V += dt * ((1 - self.LambdaA) * A + self.LambdaA * self.A)
        self.Alpha = Alpha
        self.A = A

class OnCameraDataClass:
    def __init__(self, CameraID, Location, Disparity):
        self.CameraID = CameraID
        self.Location = Location
        self.Disparity = Disparity

class AnchorClass:
    K_Base = 1.
    TrustDistance = np.inf
    TrustModel = 'Constant'

    def __init__(self, WorldLocation, Origin, WorldPresenceSegment, OnCameraData, GetWorldLocationFunction):
        self.SetWorldData(WorldLocation, WorldPresenceSegment)

        self.CurrentPresenceSegment = (np.array(self.WorldPresenceSegment[0]), np.array(self.WorldPresenceSegment[1]))

        self.WorldProjection = np.array(self.WorldLocation)
        self.CurrentReprojection = np.array(self.WorldLocation)
        self.Length = 0.

        self.Origin = np.array(Origin)

        self.OnCameraData = OnCameraData
        self.GetWorldLocation = GetWorldLocationFunction

        self.Active = True

    def SetWorldData(self, WorldLocation, WorldPresenceSegment):
        self.WorldLocation = WorldLocation
        self.WorldPresenceSegment = WorldPresenceSegment

    @classmethod
    def SetTrustDistance(self, Length):
        self.TrustDistance = Length
    @classmethod
    def SetTrustModel(self, Model):
        self.TrustModelIndex = {'Constant':0, 'Gaussian':1, 'Exponential':2}[Model]

    @property
    def K(self):
        return self.Active * self.K_Base * self.ExponentialValue

    @property
    def ForceNorm(self):
        return self.Length * self.K

    @property
    def Force(self):
        return self.ForceNorm * (self.WorldProjection - self.CurrentReprojection) / max(0.001, self.Length)

    @property
    def Energy(self):
        if self.TrustModelIndex != 2:
            return self.K * self.Length**2 / 2
        return self.K * self.TrustDistance * (self.TrustDistance * (1/self.ExponentialValue - 1) - self.Length)

    @property
    def ExponentialValue(self):
        if self.TrustModelIndex == 0 or self.TrustDistance == np.inf:
            return 1
        if self.TrustModelIndex == 1:
            return np.e**(-self.Length**2 / self.TrustDistance**2)
        else:
            return np.e**(-self.Length / self.TrustDistance)

    def UpdatePresenceSegment(self):
        ScreenLocation = self.OnCameraData.Location
        disparity = self.OnCameraData.Disparity
        CameraID = self.OnCameraData.CameraID
        self.CurrentPresenceSegment = (self.GetWorldLocation(ScreenLocation, disparity+0.5, CameraID), self.GetWorldLocation(ScreenLocation, disparity-0.5, CameraID))
        self.WorldProjection, self.CurrentReprojection = GetSegmentsProjections(self.WorldPresenceSegment, self.CurrentPresenceSegment)

        ErrorVector = self.WorldProjection - self.CurrentReprojection
        self.Length = np.linalg.norm(ErrorVector)

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
