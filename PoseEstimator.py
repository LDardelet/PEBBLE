from ModuleBase import ModuleBase
from Events import TwistEvent, TrackerEvent, DisparityEvent, PoseEvent
from functools import lru_cache
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

_MAX_DEPTH = 1e3

class PoseEstimator(ModuleBase):
    _TrustModelIndices = {'Constant':0, 'Gaussian':1, 'Exponential':2}
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
                                    ('SpringsForceAndTorque', np.array),
                                    ('ViscosityForceAndTorque', np.array),
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
                                    ('NTrackers', int),
                                    ('NActive', int),
                                    ('Anchors@Length', float)]

        self.__ModulesLinksRequested__ = ['RightDisparityMemory', 'LeftDisparityMemory']

        self._DefaultK = 200
        self._DefaultStereoBase = 0.2

        self._InitialCameraRotationVector = np.zeros(3)
        self._InitialCameraTranslationVector = np.zeros(3)

        self._ConstantSpringTrustDistance = 0.3 # to disable, use np.inf
        self._SpringTrustDistanceModel = "Gaussian" # str, either Constant ( 1 ), Gaussian ( e**(-delta**2) ) or Exponential ( e**(-delta) )

        self._SpringsModel = 'Orthogonal'                    # Either 'Single' or 'Orthogonal', for one direct spring or two speings for reprojection and depth

        self._UseAdaptiveParameters = True
        self._EquivalentConstraintsModel = True
        self._EquivalentConstraintsMinDetDistance = 0.05

        # Following 5 parameters only used if _UseAdaptiveParameters = False
        self._Kappa = 25.
        self._MuV = 10.
        self._MuOmega = 250.
        self._Mass = 1.
        self._Moment = 20.

        self._AdaptiveDampeningFactor = 1. # Should be equal to 1 mathwise
        self._AdaptiveKappaFactor = 1.
        self._AdaptiveBaseLength = 4.
        self._AdaptiveBaseTrackers = 1 
        self._AdaptiveBaseTau = 0.1
        self._AdaptiveMuOmegaFactor = 5.

        self._MinTrackersPerCamera = 8
        self._NeedsStereoOdometry = True
        self._DefaultTau = 0.05
        self._TrackerDisparityRadius = 5
        self._DisparityTauRatio = 10
        self._IntegerDisparity = True

        self._DisparityOverlap = 0.

        self._DisparitySegmentMargin = 0

        self._SetSpeedAtStartup = True

        self._MaxLengthRatioBreak = np.inf
        self._MaxAbsoluteSpringLength = 2.

        self._AutoReleaseOnNewTracker = False
        self._AutoReleaseTrustDistanceRatio = 0.02
        self._AutoReleaseMaxTauRatio = 2.
        self._AutoReleaseDtRatio = 10.

        self._DepthlessSpringBehaviour = "Partial" # Behaviours are either "Disable", "Remove", or in Orthogonal Model, "Partial"

        self._CanDeactivateAnchor = False

    def _OnInitialization(self):
        self.RigFrame = FrameClass(self._InitialCameraRotationVector, self._InitialCameraTranslationVector, np.zeros(3), np.zeros(3), 0, 0)
        self.NCameras = len(self.__SubStreamInputIndexes__)
        self.Anchors = {}

        if self._UseAdaptiveParameters:
            self.Mass = 1.
            if self._EquivalentConstraintsModel:
                self.Moment = self.Mass
                self.Kappa = self._AdaptiveKappaFactor * 4 * np.pi**2 * self.Mass / (self._AdaptiveBaseTau**2)
            else:
                self.Moment = self.Mass * self._AdaptiveBaseLength ** 2 / 2
                self.Kappa = self._AdaptiveKappaFactor * 4 * np.pi**2 * self.Mass / (self._AdaptiveBaseTau**2 * self._AdaptiveBaseTrackers)
        else:
            self.Mass = self._Mass
            self.Moment = self._Moment
            self.Kappa = self._Kappa

        if self._EquivalentConstraintsModel:
            self.__class__.SpringsForceAndTorque = property(lambda self: self.KinematicSpringsForceAndTorque)
            self.__class__.ReferenceSpringConstant = self.Kappa
            self.__class__.ReferenceSpringTorqueConstant = self.Kappa
        else:
            self.__class__.SpringsForceAndTorque = property(lambda self: self.MechanicSpringsForceAndTorque)
            self.__class__.ReferenceSpringConstant = property(lambda self: self.TotalSpringConstant)
            self.__class__.ReferenceSpringTorqueConstant = property(lambda self: self.TotalSpringTorqueConstant)

        if self._DepthlessSpringBehaviour == 'Partial' and self._SpringsModel == 'Single':
            self.LogError("Incompatible parameters : _DepthlessSpringBehaviour = Partial & _SpringsModel = Single")
            return False

        self.K = self._DefaultK
        self.ScreenSize = np.array(self.Geometry)
        self.ScreenCenter = self.ScreenSize/2
        self.StereoBaseDistance = self._DefaultStereoBase
        self.CameraOffsetLocations = [np.array([self.StereoBaseDistance/2, 0., 0.]), np.array([-self.StereoBaseDistance/2, 0., 0.])]

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

        if self._SpringsModel == 'Single':
            class UsedAnchorClass(SingleSpringAnchor):
                pass
            self.AnchorClass = UsedAnchorClass
        elif self._SpringsModel == 'Orthogonal':
            class UsedAnchorClass(OrthogonalSpringsAnchor):
                pass
            self.AnchorClass = UsedAnchorClass
        else:
            self.LogWarning(f"Unrecognized spring model {self._SpringsModel}")
            return False

        self.AnchorClass.Kappa = self.Kappa
        self.AnchorClass.TrustDistance = self._ConstantSpringTrustDistance
        self.AnchorClass.TrustModelIndex = self._TrustModelIndices[self._SpringTrustDistanceModel]

        self.AverageSpringLength = 0.
        self.TotalSpringConstant = 0.
        self.TotalSpringTorqueConstant = 0.

        self.LastUpdateT = 0.

        self.NUpdates = 0
        self.NActive = 0

        self._SimulationActivity = 0.
        self._SimulationDtSum = 0.

        self.TrackersEnergy = 0.
        self.EnvironmentEnergy = 0.
        self.BrokenSpringsEnergyLoss = 0.
        self.UpdatedSpringsPreviousEnergies = {}

        self.EndpointDistance = self.GetDepth(1)

        self.Started = False
        self.Relaxing = False
        self.ReceivedOdometry = np.zeros(2, dtype = bool)

        self.TrackersPerCamera = {CameraID:0 for CameraID in self.__SubStreamInputIndexes__}

        self.ReceivedV = np.zeros((3, 2))
        self.ReceivedOmega = np.zeros((3,2))

        self.DisparitySegmentHalfWidth = (float(self._IntegerDisparity) / 2) + self._DisparityOverlap / 2

        return True

    def _OnInputIndexesSet(self, Indexes):
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
            if event.TrackerColor == 'g' and event.TrackerMarker == 'o': # Tracker locked
                if event.Has(DisparityEvent):
                    disparity = event.disparity
                else:
                    if self._UseDisparityMemories:
                        Tau = self.FrameworkAverageTau
                        if Tau is None or Tau == 0:
                            Tau = self._DefaultTau
                        disparity = self.DisparityMemories[event.SubStreamIndex].GetDisparity(Tau*self._DisparityTauRatio, np.array(event.TrackerLocation, dtype = int), self._TrackerDisparityRadius)
                    else:
                        disparity = None
                self.UpdateFromTracker(event.SubStreamIndex, event.TrackerLocation, event.TrackerID, disparity)
            else:
                ID = (event.TrackerID, event.SubStreamIndex)
                if ID in self.Anchors:
                    self.RemoveAnchor(ID, "tracker disappeared")

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
                if self._SetSpeedAtStartup:
                    self.Log(f"Set speeds to current average received values")
                    self.RigFrame.V = self.ReceivedV.mean(axis = 1)
                    self.RigFrame.Omega = self.ReceivedOmega.mean(axis = 1)
                self.LogSuccess("Started")
                for Anchor in self.Anchors.values():
                    self.Fix(Anchor)
            else:
                return

        dt = event.timestamp - self.LastUpdateT
        self.LastUpdateT = event.timestamp
        self.NUpdates += 1
        self.UpdateSimulation(dt)

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

    @property
    def SimulationAverageDt(self):
        return self._SimulationDtSum / max(1., self._SimulationActivity)

    def Relax(self, TauSceneFactor = 2., NLogs = 1000, dtRatio = 5., StopLengthRatio = 0):
        if self.AverageSpringLength < self._ConstantSpringTrustDistance * StopLengthRatio:
            self.Log("Relaxation stop condition already met")
            return
        dt = self.SimulationAverageDt * dtRatio
        NUpdates = int(self._AdaptiveBaseTau * TauSceneFactor / dt)

        self.StoredRigVelocities = {'V':np.array(self.RigFrame.V), 'Omega':np.array(self.RigFrame.Omega)}
        self.Relaxing = True
        self.RigFrame.V[:] = 0.
        self.RigFrame.Omega[:] = 0.

        StartEnergy = self.PotentialEnergy

        Energies = []
        LogDnUpdate = NUpdates // (NLogs-1)

        self.Log("Relaxing over {0:.1f} seconds ({1} updates)".format(self._AdaptiveBaseTau * TauSceneFactor, NUpdates))
        for nUpdate in range(NUpdates):
            self.UpdateSimulation(dt)
            if nUpdate % LogDnUpdate == 0:
                Release = StartEnergy - self.PotentialEnergy
                print("{0:3d}% : {1}{2:.3f} J released, average length : {3:.3f} m".format(int(100. * (nUpdate+1) / NUpdates), '+'*(Release>0) + '-'*(Release<0), abs(Release), self.AverageSpringLength), end = '\r')

                Energies.append(self.PotentialEnergy)
            if self.AverageSpringLength < self._ConstantSpringTrustDistance * StopLengthRatio:
                self.LogSuccess("Relaxation condition met")
                break

        print("")
        EndEnergy = self.PotentialEnergy
        if EndEnergy < StartEnergy:
            self.Log("Released {0:.3f} / {1:.3f} J of potential energy".format(StartEnergy - EndEnergy, StartEnergy))
        else:
            self.LogWarning("Gained {0:.3f} J of energy while relaxing. This should not happen. Maybe relaxation wasn't long enough.".format(EndEnergy - StartEnergy))
        
        self.RigFrame.V = np.array(self.StoredRigVelocities['V'])
        self.RigFrame.Omega = np.array(self.StoredRigVelocities['Omega'])
        self.Relaxing = False
        del self.__dict__['StoredRigVelocities']
        return Energies

    def UpdateSimulation(self, dt):
        if not self.Relaxing:
            Decay = np.e**(-dt / self._AdaptiveBaseTau)
            self._SimulationActivity = self._SimulationActivity * Decay + 1
            self._SimulationDtSum = self._SimulationDtSum * Decay + dt

        self.RigFrame.UpdatePosition(dt)
        self.UpdateLinesOfSights()

        for ID, PreviousEnergy in self.UpdatedSpringsPreviousEnergies.items():
            self.TrackersEnergy += (self.Anchors[ID].Energy - PreviousEnergy)
        self.EnvironmentEnergy += dt * self.EnvironmentPower

        Force, Torque = self.ForceAndTorque
        A, Alpha = Force / self.Mass, Torque / self.Moment

        self.RigFrame.UpdateDerivatives(dt, Alpha, A)

    def UpdateFromTracker(self, CameraID, ScreenLocation, TrackerID, disparity = None):
        ID = (TrackerID, CameraID)
        if ID not in self.Anchors:
            if disparity is None:
                return

            if self.Started and self._AutoReleaseOnNewTracker:
                self.Relax(TauSceneFactor = self._AutoReleaseMaxTauRatio, NLogs = 100, dtRatio = self._AutoReleaseDtRatio, StopLengthRatio = self._AutoReleaseTrustDistanceRatio)
            self.Anchors[ID] = self.AnchorClass(np.array(ScreenLocation), disparity, CameraID)

            self.TrackersPerCamera[CameraID] += 1
            self.Log("Added anchor {0}".format(ID))

            if self.Started:
                self.Fix(self.Anchors[ID])
        else:
            Anchor = self.Anchors[ID]

            if disparity is None:
                if self._DepthlessSpringBehaviour == 'Remove':
                    self.RemoveAnchor(ID, "no depth")
                    return
                elif self.Started and self._DepthlessSpringBehaviour == 'Disable' and Anchor.Active:
                    Anchor.Active = False
                    self.LogWarning("Deactivated anchor {0} (no disparity)".format(ID))
                    self.TrackersPerCamera[CameraID] -= 1
                    return
            else:
                if self.Started and not Anchor.Active:
                    Anchor.Active = True
                    self.LogSuccess("Reactivated anchor {0}".format(ID))
                    self.TrackersPerCamera[CameraID] += 1

            self.UpdatedSpringsPreviousEnergies[ID] = Anchor.Energy

            Anchor.OnCameraData.Update(ScreenLocation, disparity)

    def UpdateLinesOfSights(self):
        MaxLength = 0
        MaxID = None
        self.AverageSpringLength = 0.
        self.TotalSpringConstant = 0.
        self.TotalSpringTorqueConstant = 0.
        for nAnchor, (ID, Anchor) in enumerate(list(self.Anchors.items())):
            self.Update3D(Anchor)
            if Anchor.Length > self._MaxAbsoluteSpringLength:
                self.RemoveAnchor(ID, 'excessive absolute length')
                continue
            if Anchor.Length > MaxLength:
                MaxLength = Anchor.Length
                MaxID = ID
            self.AverageSpringLength += Anchor.Length
            self.TotalSpringConstant += Anchor.K
            self.TotalSpringTorqueConstant += Anchor.K * Anchor.ArmLength**2

        N = len(self.Anchors)
        self.AverageSpringLength /= N
        if self._MaxLengthRatioBreak != np.inf and MaxLength > self._MaxLengthRatioBreak * self.AverageSpringLength:
            Anchor = self.Anchors[MaxID]
            self.TotalSpringConstant -= Anchor.K
            self.TotalSpringTorqueConstant -= Anchor.K * Anchor.ArmLength**2
            self.RemoveAnchor(MaxID, 'excessive relative length')
            self.AverageSpringLength = (self.AverageSpringLength * N - MaxLength) / (N - 1)

    @property
    def NTrackers(self):
        return len(self.Anchors)

    def Fix(self, Anchor):
        if not self.UpdateCurrentValues(Anchor): # If no depth information was available
            self.LogWarning("Could not fix an anchor, as no depth data was available")
            return
        Anchor.Fix()

    def Update3D(self, Anchor):
        if not Anchor.Active:
            return
        self.UpdateCurrentValues(Anchor)
        Anchor.UpdateProjections()

    def UpdateCurrentValues(self, Anchor):
        ScreenLocation, disparity, CameraID = Anchor.OnCameraData.Values
        Anchor.CurrentValues['Origin'] = self.GetCameraLocation(CameraID)
        Anchor.CurrentValues['LoS'] = self.GetLineOfSight(ScreenLocation)
        
        #DepthToDistanceRatio = 1 / (Anchor.CurrentValues['LoS'].dot(self.RigFrame.ToWorld(np.array([0., 0., 1.]), RelativeToOrigin = False)))
        if not disparity is None:
            DepthToDistanceRatio = self.GetDepthToDistanceRatio(ScreenLocation)
            Anchor.CurrentValues['Range'] = [Depth * DepthToDistanceRatio for Depth in self.GetDepthRange(disparity)]
            Anchor.CurrentValues['Distance'] = DepthToDistanceRatio * self.GetDepth(disparity)
            return True
        else:
            if self._DepthlessSpringBehaviour == 'Partial':
                Anchor.CurrentValues['Distance'] = None
            else:
                self.LogWarning("Conditions path should not come here")
            return False

    def GetDepth(self, disparity):
        if disparity <= 0:
            return _MAX_DEPTH
        return self.StereoBaseDistance * self.K / disparity
    def GetDepthRange(self, disparity):
        if self.DisparitySegmentHalfWidth == 0:
            return [self.GetDepth(disparity), 0]
        else:
            return [self.GetDepth(disparity+self.DisparitySegmentHalfWidth), self.GetDepth(disparity-self.DisparitySegmentHalfWidth)]

    def GetDepthToDistanceRatio(self, ScreenLocation):
        Uv = self.KMatInv.dot(np.array([ScreenLocation[0], ScreenLocation[1], 1]))
        return np.linalg.norm(Uv) / Uv[-1]

    def GetLineOfSight(self, ScreenLocation):
        u = self.RigFrame.ToWorld(self.KMatInv.dot(np.array([ScreenLocation[0], ScreenLocation[1], 1])), RelativeToOrigin = False)
        return u / np.linalg.norm(u)
    def GetCameraLocation(self, CameraID):
        return self.RigFrame.ToWorld(self.CameraOffsetLocations[CameraID])

    @property
    def PotentialEnergy(self):
        PotentialEnergy = 0.
        for ID, Anchor in self.Anchors.items():
            PotentialEnergy += Anchor.Energy
        return PotentialEnergy
    @property
    def KineticEnergy(self):
        return (self.Mass * (self.RigFrame.V**2).sum() + self.Moment * (self.RigFrame.Omega**2).sum()) / 2
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
        return 2*np.pi * np.sqrt(self.Mass / KTot)

    @property
    def EnvironmentVs(self):
        if self.Relaxing:
            return np.zeros((3, self.NCameras))
        return self.ReceivedV
    @property
    def EnvironmentOmegas(self):
        if self.Relaxing:
            return np.zeros((3, self.NCameras))
        return self.ReceivedOmega

    @property
    def ViscosityForceAndTorque(self):
        Forces = self.MuV * (self.EnvironmentVs - self.RigFrame.V.reshape((3,1)))
        VTorque = np.zeros(3)
        #for nCamera in range(self.NCameras):
        #    VTorque -= np.cross(self.GetCameraLocation(nCamera) - self.RigFrame.T, Forces[:,nCamera])
        return self.RigFrame.ToWorld(Forces.sum(axis = 1) / self.NCameras, RelativeToOrigin = False), self.RigFrame.ToWorld((self.MuOmega * (self.EnvironmentOmegas - self.RigFrame.Omega.reshape((3, 1))).sum(axis = 1) + VTorque) / self.NCameras, RelativeToOrigin = False)
    @property
    def EnvironmentPower(self): # Defined as the instantaneous power transmitted to the rig frame
        Force, Torque = self.ViscosityForceAndTorque
        return ((Force*self.RigFrame.V).sum() + (Torque*self.RigFrame.Omega).sum())

    @property
    def MuV(self):
        if self._UseAdaptiveParameters:
            return 2 * np.sqrt(self.Mass * self.ReferenceSpringConstant) * self._AdaptiveDampeningFactor
        else:
            return self._MuV
    @property
    def MuOmega(self):
        if self._UseAdaptiveParameters:
            return 2 * np.sqrt(self.Moment * self.ReferenceSpringTorqueConstant) * self._AdaptiveDampeningFactor * self._AdaptiveMuOmegaFactor
        else:
            return self._MuOmega

    @property
    def CenterApplicationPoint(self):
        Center = np.zeros(3)
        for Anchor in self.Anchors.values():
            Center += Anchor.CurrentProjection
        return Center

    @property
    def KinematicSpringsForceAndTorque(self):
        Xs = []
        Fs = []
        Cs = []
        SumForce = np.zeros(3)
        SumLocation = np.zeros(3)
        SumTrust = 0.
        self.NActive = 0
        for Anchor in self.Anchors.values():
            if Anchor.Active:
                self.NActive += 1
                Force = Anchor.RawForce
                Trust = Anchor.ExponentialValue

                SumForce += Force * Trust
                SumLocation += Anchor.CurrentProjection * Trust
                SumTrust += Trust
                Xs.append(Anchor.CurrentProjection)
                Fs.append(Force)
                Cs.append(Trust)

        if not SumTrust:
            return np.zeros(3), np.zeros(3)

        X0 = SumLocation / SumTrust
        F0 = SumForce / SumTrust
        Xs = np.array(Xs).transpose()
        Fs = np.array(Fs).transpose()
        Cs = np.array(Cs)
        
        DXs = Xs - X0.reshape((3,1))
        M = np.zeros((3,3))
        N2 = (DXs**2 * Cs).sum()
        for nDim in range(3):
            M[nDim,nDim] = N2 - (Cs * DXs[nDim,:]**2).sum()
            
        M[0,1] = M[1,0] = -(Cs * DXs[0,:] * DXs[1,:]).sum()
        M[0,2] = M[2,0] = -(Cs * DXs[0,:] * DXs[2,:]).sum()
        M[1,2] = M[2,1] = -(Cs * DXs[1,:] * DXs[2,:]).sum()

        DetM = np.linalg.det(M)
        if abs(DetM / SumTrust) < self._EquivalentConstraintsMinDetDistance ** 6:
            return np.zeros(3), np.zeros(3)

        DFs = Fs - F0.reshape((3,1))
        Gamma = np.zeros(3)
        for nPoint in range(self.NActive):
            Gamma += Cs[nPoint] * np.cross(DFs[:,nPoint], DXs[:,nPoint])
        
        Omega = (np.linalg.inv(M)).dot(Gamma)

        FEquiv = F0 + np.cross((self.RigFrame.T - X0), Omega)
        return FEquiv, Omega

    @property
    def MechanicSpringsForceAndTorque(self):
        TotalForce = np.zeros(3)
        TotalTorque = np.zeros(3)
        for Anchor in self.Anchors.values():
            Arm = Anchor.CurrentProjection - self.RigFrame.T
            Distance = np.linalg.norm(Arm)
            Force = Anchor.Force
            Torque = -np.cross(Arm, Force)
            if Distance < self.EndpointDistance:
                TotalForce += Force * (self.EndpointDistance - Distance) / self.EndpointDistance
            TotalTorque += Torque
        return TotalForce, TotalTorque

    @property
    def ForceAndTorque(self):
        TotalForce, TotalTorque = self.ViscosityForceAndTorque
        SpringsForce, SpringsTorque = self.SpringsForceAndTorque
        return TotalForce + SpringsForce, TotalTorque + SpringsTorque

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

    def PlotConstraints(self):
        f, axs = plt.subplots(2,1)
        t = np.array(self.History['t'])
        for nType, Type in enumerate(('Force', 'Torque')):
            axs[nType].set_title(Type)
            for nDim, color in enumerate('rgb'):
                SpringForce = np.array(self.History['SpringsForceAndTorque'])[:,nType,nDim]
                axs[nType].plot(t, SpringForce, color = color)
                ViscousForce = np.array(self.History['ViscosityForceAndTorque'])[:,nType,nDim]
                axs[nType].plot(t, ViscousForce, color = color, linestyle = '--')

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
            ax.plot(*[[0, BaseVector[nAxDim]] for nAxDim in range(3)], 'k')
            ax.plot(*[[self.RigFrame.T[nAxDim], self.RigFrame.ToWorld(BaseVector)[nAxDim]] for nAxDim in range(3)], 'b')
        CameraColors = ['r', 'b']
        for CameraIndex, CameraOffset in enumerate(self.CameraOffsetLocations):
            CameraWorldLocation = self.RigFrame.ToWorld(CameraOffset)
            ax.scatter(*CameraWorldLocation.tolist(), marker = 'o', color = CameraColors[CameraIndex])

        for ID, Anchor in self.Anchors.items():
            if Anchor.StaticValues['Range'][1] == 0:
                Distance = Anchor.StaticValues['Range'][0]
                ax.scatter(*[[(Anchor.StaticValues['Origin'] + Anchor.StaticValues['LoS'] * Distance)[nAxDim]] for nAxDim in range(3)], color = CameraColors[ID[1]], marker = 'x')
            else:
                ax.plot(*[[(Anchor.StaticValues['Origin'] + Anchor.StaticValues['LoS'] * Distance)[nAxDim] for Distance in Anchor.StaticValues['Range']] for nAxDim in range(3)], color = CameraColors[ID[1]], linestyle = '--')
            if Anchor.CurrentValues['Range'][1] is 0:
                Distance = Anchor.CurrentValues['Range'][0]
                ax.scatter(*[[(Anchor.CurrentValues['Origin'] + Anchor.CurrentValues['LoS'] * Distance)[nAxDim]] for nAxDim in range(3)], color = CameraColors[ID[1]])
            else:
                ax.plot(*[[(Anchor.CurrentValues['Origin'] + Anchor.CurrentValues['LoS'] * Distance)[nAxDim] for Distance in Anchor.CurrentValues['Range']] for nAxDim in range(3)], color = CameraColors[ID[1]])

            WP, CP = Anchor.StaticProjection, Anchor.CurrentProjection
            CPpF = Anchor.CurrentProjection + Anchor.Force
            ax.scatter(*WP.tolist(), color = 'k', marker = 'o')
            ax.scatter(*CP.tolist(), color = 'k', marker = 'o')
            ax.text(CP[0], CP[1], CP[2]+0.01, str(ID), color = 'k')
            #ax.plot(*[[CPpF[nAxDim], CP[nAxDim]] for nAxDim in range(3)], color = 'k')

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
            for nLocation, (UsedLocation, UsedLocType) in enumerate(((Anchor.StaticProjection, 'static projection'), (Anchor.CurrentProjection, 'current projection'))):
                CameraFrameLocation = self.RigFrame.ToRig(UsedLocation) - CameraOffset
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

    def ToRig(self, WorldLocation):
        return self.R.dot(WorldLocation - self.T)

    def ToWorld(self, FrameLocation, RelativeToOrigin = True):
        return self.R.T.dot(FrameLocation) + self.T * RelativeToOrigin

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
    def __init__(self, Location, Disparity, CameraID):
        self.Location = Location
        self.Disparity = Disparity
        self.CameraID = CameraID

    @property
    def Values(self):
        return self.Location, self.Disparity, self.CameraID

    def Update(self, Location, Disparity):
        self.Location = Location
        self.Disparity = Disparity

class TemplateAnchorClass:
    Kappa = 1.
    TrustDistance = np.inf
    TrustModelIndex = 0

    def __init__(self, ScreenLocation, disparity, CameraID):
        self.CurrentValues = {'Origin':None, 
                              'LoS': None, 
                              'Range':None}
        self.StaticValues = dict(self.CurrentValues)

        self.OnCameraData = OnCameraDataClass(ScreenLocation, disparity, CameraID)
        
        self.StaticProjection = np.zeros(3)
        self.CurrentProjection = np.zeros(3)

        self.Springs = []

        self.Active = False

    def Fix(self):
        if self.Active:
            raise Exception("Anchor already active")
        self.StaticValues = {'Origin':np.array(self.CurrentValues['Origin']),
                             'LoS': np.array(self.CurrentValues['LoS']),
                             'Range': tuple(self.CurrentValues['Range']),
                             'Distance': self.CurrentValues['Distance']}
        self.Active = True

    def UpdateProjections(self):
        SOrigin = self.StaticValues['Origin']
        SLine = self.StaticValues['LoS']
        SRange = self.StaticValues['Range']

        COrigin = self.CurrentValues['Origin']
        CLine = self.CurrentValues['LoS']
        if self.CurrentValues['Distance'] is None:
            CRange = [0, _MAX_DEPTH]
        else:
            CRange = self.CurrentValues['Range']

        if SRange[1] == 0:
            self.StaticProjection = SOrigin + SLine * SRange[0]
            if CRange[1] == 0:
                self.CurrentProjection = COrigin + CLine * CRange[0]
            else:
                CurrentDistance = (self.StaticProjection - COrigin).dot(CLine) # Compute distance from the static location to the current origin, projected on current line of sight
                CurrentDistance = max(CRange[0], min(CRange[1], CurrentDistance)) # Clamp that distance onto the allows segment
                self.CurrentProjection =  COrigin + CLine * CurrentDistance
        else:
            if CRange[1] == 0:
                self.CurrentProjection = COrigin + CLine * CRange[0]
                StaticDistance = (self.CurrentProjection - SOrigin).dot(SLine)
                StaticDistance = max(SRange[0], min(SRange[1], StaticDistance))
                self.StaticProjection = SOrigin + SLine * StaticDistance
            else:
                StaticSegment = [SOrigin + SLine * Distance for Distance in SRange]
                CurrentSegment = [COrigin + CLine * Distance for Distance in CRange]
                self.StaticProjection, self.CurrentProjection = _GetSegmentsProjections(StaticSegment, CurrentSegment)

        self.UpdateSprings()

    @property
    def StaticLocation(self):
        return self.StaticValues['Origin'] + self.StaticValues['LoS'] * self.StaticValues['Distance']
    @property
    def CurrentLocation(self):
        if self.CurrentValues['Distance'] == 0:
            raise Exception("Depth is currently unknown for this object")
        return self.CurrentValues['Origin'] + self.CurrentValues['LoS'] * self.CurrentValues['Distance']

    @property
    def ProjectionError(self): # Oriented towards the theoretical location
        return self.StaticProjection - self.CurrentProjection
    @property    
    def Length(self):
        return np.linalg.norm(self.ProjectionError)
    @property
    def ArmLength(self):
        return np.linalg.norm(self.CurrentProjection - self.CurrentValues['Origin'])

    @property
    def Energy(self):
        Energy = 0.
        for Spring in self.Springs:
            Energy += Spring.Energy
        return Energy
    @property
    def Force(self):
        Force = np.zeros(3, dtype = float)
        for Spring in self.Springs:
            Force += Spring.Force
        return Force
    @property
    def RawForce(self):
        Force = np.zeros(3, dtype = float)
        for Spring in self.Springs:
            Force += Spring.RawForce
        return Force
    @property
    def ExponentialValue(self):
        if not self.Active:
            return False
        if self.TrustModelIndex == 0 or self.TrustDistance == np.inf:
            return 1
        if self.TrustModelIndex == 1:
            return np.e**(-self.Length**2 / self.TrustDistance**2)
        else:
            return np.e**(-self.Length / self.TrustDistance)
        

class SingleSpringAnchor(TemplateAnchorClass):
    def __init__(self, *args):
        super().__init__(*args)
        self.Springs = (SpringClass(Kappa = self.Kappa,
                                     TrustDistance = self.TrustDistance,
                                     TrustModelIndex = self.TrustModelIndex), )

    def UpdateSprings(self):
        self.Springs[0].Length = self.Length
        self.Springs[0].UnitVector = _GetUnitVector(self.ProjectionError)

    @property
    def K(self):
        return self.Springs[0].K

class OrthogonalSpringsAnchor(TemplateAnchorClass):
    def __init__(self, *args):
        super().__init__(*args)
        self.Springs = (SpringClass(Kappa = self.Kappa,
                                     TrustDistance = self.TrustDistance,
                                     TrustModelIndex = self.TrustModelIndex),
                        SpringClass(Kappa = self.Kappa,
                                     TrustDistance = self.TrustDistance,
                                     TrustModelIndex = self.TrustModelIndex))

    @property
    def OrthogonalError(self):
        return self.ProjectionError - self.DepthError
    @property
    def DepthError(self):
        LoS = self.CurrentValues['LoS']
        return LoS * (LoS.dot(self.ProjectionError))

    def UpdateSprings(self):
        OrthogonalError = self.OrthogonalError
        self.Springs[0].Length = np.linalg.norm(OrthogonalError)
        self.Springs[0].UnitVector = _GetUnitVector(OrthogonalError)

        if self.CurrentValues['Distance'] is None:
            self.Springs[1].Length = 0
        else:
            DepthError = self.DepthError
            self.Springs[1].Length = np.linalg.norm(DepthError)
            self.Springs[1].UnitVector = _GetUnitVector(DepthError)

    @property
    def K(self):
        return self.Springs[0].K # We use the orthogonal spring a base K.

class SpringClass:
    def __init__(self, Kappa, TrustDistance, TrustModelIndex):
        self.Kappa = Kappa
        self.TrustDistance = TrustDistance
        self.TrustModelIndex = TrustModelIndex

        self.Length = 0
        self.UnitVector = np.zeros(3)
    @property
    def K(self):
        return self.Kappa * self.ExponentialValue
    @property
    def Force(self):
        return self.Length * self.K * self.UnitVector
    @property
    def RawForce(self):
        return self.Length * self.Kappa * self.UnitVector
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

def _GetUnitVector(V):
    N = np.linalg.norm(V)
    if N == 0:
        return np.zeros(V.shape[0])
    return V / N


def _GetSegmentsProjections(S1, S2):
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
