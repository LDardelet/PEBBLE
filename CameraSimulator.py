import numpy as np
import random

from ModuleBase import ModuleBase
from Events import CameraEvent, TrackerEvent, TwistEvent, DisparityEvent

_MAP_DIMENSION = 10. # In Meters
_MAX_T = 100.
_MAX_STEPS_NO_EVENTS = 100

_MIN_DISTANCE = 0.2

class MovementSimulatorClass(ModuleBase):
    def _OnCreation(self):
        '''
        Class to emulate stereo system moving with artificial map
        '''
        self.__IsInput__ = True
        self._MonitoredVariables = [('T', np.array),
                                    ('Theta', np.array)]

        self._MapType = 'cubic'
        self._MapDensity = 0.0006

        self._GeneratorRadius = 0.07

        self._dt = 0.00003

        self._SequenceType = 'twist'
        self._SequenceInput = None

        self._TrackerEvents = True
        self._TwistEvents = True
        self._CameraEvents = True
        self._DisparityEvents = True

        self._TranslationAcceleration = 20.
        self._RotationAcceleration = 20.

        self._KMat = None
        self._DefaultK = 200
        self._Geometry = np.array([240, 180])

        self._CameraOffsets = [np.array([-0.1, 0., 0.]), np.array([0.1, 0., 0.])] # By default, its a 2 cameras simulator

        self._TwistTau = 0.005
        self._TwistGaussianRelativeNoise = 0.1

        self._TrackersMinDisplacement = 0.1
        self._TrackersLocationGaussianNoise = 0.
        self._TrackerEdgeMargin = 10
        self._TrackersSentAtOnce = 1

        self._LambdaAveragesTrackers = 0.05
        self._TrackerOutliersSpeedFactor = 3.
        self._TrackerOutlierTau = 0.1
        self._NMaxOutliersPerCamera = 4
        self._OutliersMaxDistance = 50

    def _OnInitialization(self):
        self.NCameras = len(self._CameraOffsets)
        if self._SequenceInput is None:
            self.LogWarning("No sequence was specified")
            return False
        if type(self._SequenceInput) == str:
            with open(self._SequenceInput, 'r') as fSequence:
                self.Sequence = []
                for line in fSequence.readlines():
                    dt, wx, wy, wz, vx, vy, vz = [float(data.strip()) for data in line.strip().split(' ')]
                    self.Sequence += [(t, np.array([wx, wy, wz]), np.array([vx, vy, vz]))]
        elif type(self._SequenceInput) == list:
            self.Sequence = list(self._SequenceInput)

        if self.NCameras != 2 and self._DisparityEvents:
            self.LogWarning("Disabling disparity events as more than two cameras are specified")
            self._DisparityEvents = False
        if self._DisparityEvents:
            self.StereoBaseDistance = np.linalg.norm(self._CameraOffsets[0] - self._CameraOffsets[1])

        self.T = np.zeros(3)
        self.Theta = np.zeros(3)
        self.V = np.zeros(3)
        self.Omega = np.zeros(3)
        self.dOmega = np.zeros(3)
        self.dV = np.zeros(3)
        self.t = 0.

        self.NextTwistT = np.random.exponential(self._TwistTau, size = self.NCameras)

        self.ScreenCenter = self._Geometry / 2
        if self._KMat is None:
            self.KMat = np.array([[self._DefaultK, 0., self.ScreenCenter[0]],
                                  [0., -self._DefaultK, self.ScreenCenter[1]],
                                  [0., 0.,              1.]])
            self.K = self._DefaultK
        else:
            self.KMat = np.array(self._KMat)
            self.K = self.KMat[0,0]

        if not self.StartNextSequenceStep():
            self.LogWarning("Input sequence is empty")

        if self._TrackerOutlierTau not in [0, np.inf]:
            self.StepOutlierProba = 1-np.e**(-self._dt / self._TrackerOutlierTau)
        else:
            self.StepOutlierProba = 0

        self.GenerateMap()

        self.OnScreenTrackers = [{nCamera:np.array([0., -100, -100]) for nCamera in range(self.NCameras)} for nGenerator in range(len(self.Generators))]
        self.IsOutlier = [{nCamera:None for nCamera in range(self.NCameras)} for nGenerator in range(len(self.Generators))]
        self.ForwardGenerators = [False for nGenerator in range(len(self.Generators))]
        self._UpToDate = False
        self.TrackersGenerators = []

        self.TrackersAverageSpeed = 100
        self.NOutliers = {0:0, 1:0}

        return True

    def _OnEventModule(self, event):
        CameraEventAttached = False
        while not CameraEventAttached:
            self.t += self._dt
            if not self.UpdateSpeedAndLocation(event):
                return
            CameraEventAttached, event = self.UpdateGenerators(event)

        if self._TwistEvents and (self.t >= self.NextTwistT).any():
            nCamera = self.NextTwistT.argmin()
            omega, v = self.GetCameraTwist(nCamera)
            event.Join(TwistEvent, v = v + np.random.normal(0, np.linalg.norm(v)*self._TwistGaussianRelativeNoise, size = 3), omega = omega + np.random.normal(0, np.linalg.norm(omega)*self._TwistGaussianRelativeNoise, size = 3), SubStreamIndex = self.SubStreamIndexes[nCamera])
            self.NextTwistT[nCamera] = self.t + np.random.exponential(self._TwistTau)


    def UpdateSpeedAndLocation(self, event):
        if self.t >= self.NextEndT:
            if not self.StartNextSequenceStep():
                return False
        self.dV = np.sign(self.AimedV - self.V) * np.minimum(self._TranslationAcceleration * self._dt, abs(self.AimedV - self.V))
        self.dOmega = np.sign(self.AimedOmega - self.Omega) * np.minimum(self._RotationAcceleration * self._dt, abs(self.AimedOmega - self.Omega))
        
        self.Omega += self.dOmega
        self.V += self.dV

        self.Theta += self.Omega * self._dt
        self.T += self.V * self._dt

        return True

    def UpdateGenerators(self, event):
        UzWorld = self.R.T.dot(np.array([0., 0., 1.]))

        CameraGenerators = []

        for nGenerator, X in enumerate(self.Generators):
            Delta = (X - self.T)
            self.ForwardGenerators[nGenerator] = (Delta.dot(UzWorld) >= _MIN_DISTANCE)
            if not self.ForwardGenerators[nGenerator]:
                if len(self.TrackersGenerators) >= self._TrackersSentAtOnce:
                    continue
                for nCamera in range(self.NCameras):
                    if self.OnScreenTrackers[nGenerator][nCamera][0]:
                        self.OnScreenTrackers[nGenerator][nCamera][0] = 0
                        self.TrackersGenerators += [(self.SubStreamIndexes[nCamera], nGenerator, None, None, None)]
            else:
                for nCamera in range(self.NCameras):
                    CameraFrameLocation = self.R.dot(Delta) - self._CameraOffsets[nCamera]
                    ScreenFrameLocation = self.KMat.dot(CameraFrameLocation)
                    if self._DisparityEvents:
                        Depth = CameraFrameLocation[-1]
                        disparity, sign = self.DisparityData(Depth, nCamera)
                    else:
                        disparity = None
                        sign = None

                    OnScreenLocation = ScreenFrameLocation[:2] / ScreenFrameLocation[-1]
                    OnScreen = ((OnScreenLocation > 0).all() and (OnScreenLocation < self._Geometry-1).all())
                    if OnScreen:
                        CameraGenerators += [(nGenerator, nCamera, ScreenFrameLocation[-1])]
                        if len(self.TrackersGenerators) >= self._TrackersSentAtOnce:
                            continue
                        TrackerOnScreen = ((OnScreenLocation > self._TrackerEdgeMargin).all() and (OnScreenLocation < self._Geometry-1-self._TrackerEdgeMargin).all())
                        if TrackerOnScreen:
                            if self.OnScreenTrackers[nGenerator][nCamera][0]:
                                if self.IsOutlier[nGenerator][nCamera] is None:
                                    if self.NOutliers[nCamera] < self._NMaxOutliersPerCamera and (np.random.random() < self.StepOutlierProba):
                                        self.IsOutlier[nGenerator][nCamera] = np.random.random(size = 2) * self.TrackersAverageSpeed * self._TrackerOutliersSpeedFactor
                                        self.Log("Tracker {0} is now outlier for camera {1} (Speed = {2:.2f})".format(nGenerator, nCamera, np.linalg.norm(self.IsOutlier[nGenerator][nCamera])))
                                        self.NOutliers[nCamera] += 1
                                if not self.IsOutlier[nGenerator][nCamera] is None:
                                    if np.linalg.norm(OnScreenLocation - self.OnScreenTrackers[nGenerator][nCamera][1:]) > self._OutliersMaxDistance:
                                        self.RemoveTracker(nGenerator, nCamera)
                                        continue
                                    dx = self.IsOutlier[nGenerator][nCamera][1:] * (self.t - self.OnScreenTrackers[nGenerator][nCamera][0])
                                    if np.linalg.norm(dx) > self._TrackersMinDisplacement:
                                        OnScreenLocation = self.OnScreenTrackers[nGenerator][nCamera][1:] + dx
                                        TrackerOnScreen = ((OnScreenLocation > self._TrackerEdgeMargin).all() and (OnScreenLocation < self._Geometry-1-self._TrackerEdgeMargin).all())
                                        if not TrackerOnScreen:
                                            self.RemoveTracker(nGenerator, nCamera)
                                        else:
                                            self.OnScreenTrackers[nGenerator][nCamera] = np.array([self.t, OnScreenLocation[0], OnScreenLocation[1]])
                                            self.TrackersGenerators += [(self.SubStreamIndexes[nCamera], nGenerator, OnScreenLocation, disparity, sign)]
                                    continue
                                else:
                                    Displacement = np.linalg.norm(self.OnScreenTrackers[nGenerator][nCamera][1:] - OnScreenLocation)
                                    if Displacement >= self._TrackersMinDisplacement:
                                        self.TrackersAverageSpeed = self.TrackersAverageSpeed * (1. - self._LambdaAveragesTrackers) + self._LambdaAveragesTrackers * Displacement / (self.t - self.OnScreenTrackers[nGenerator][nCamera][0])
                                        self.OnScreenTrackers[nGenerator][nCamera] = np.array([self.t, OnScreenLocation[0], OnScreenLocation[1]])
                                        self.TrackersGenerators += [(self.SubStreamIndexes[nCamera], nGenerator, OnScreenLocation, disparity, sign)]
                                    continue
                            else:
                                self.Log("Adding tracker {0} for camera {1}".format(nGenerator, nCamera))
                                self.OnScreenTrackers[nGenerator][nCamera] = np.array([self.t, OnScreenLocation[0], OnScreenLocation[1]])
                                self.TrackersGenerators += [(self.SubStreamIndexes[nCamera], nGenerator, OnScreenLocation, disparity, sign)]
                                continue
                    if not OnScreen or not TrackerOnScreen:
                        if self.OnScreenTrackers[nGenerator][nCamera][0]:
                            self.RemoveTracker(nGenerator, nCamera)

        if not self._CameraEvents or not CameraGenerators:
            return False, event

        LocalProbas = 1/np.array(CameraGenerators)[:,-1]
        SumProbas = np.array(LocalProbas)
        for nLocalProba, LocalProba in enumerate(LocalProbas[1:]):
            SumProbas[nLocalProba+1] = LocalProba + SumProbas[nLocalProba]
        nLocalGenerator = np.argmax(np.random.random()*SumProbas[-1] < SumProbas)

        nGenerator, nCamera = CameraGenerators[nLocalGenerator][:2]
        X = self.Generators[nGenerator]

        Delta = (X - self.T)
        CameraFrameLocation = self.R.dot(Delta) - self._CameraOffsets[nCamera]
        Distance = np.linalg.norm(CameraFrameLocation)
        UX = CameraFrameLocation / Distance
        UH = np.array([1., 0., 0.]) - (np.array([1., 0., 0.]).dot(UX)) * UX
        UH /= np.linalg.norm(UH)
        UV = np.cross(UH, UX)

        ObservableCircleDistance = Distance - self._GeneratorRadius**2/Distance
        if ObservableCircleDistance > 0:
            Theta = 2*np.pi*random.random()
            ObservableRadius = (ObservableCircleDistance / Distance)**2 * self._GeneratorRadius
            XPoint = UX*ObservableCircleDistance + (UH*np.cos(Theta) + UV*np.sin(Theta)) * ObservableRadius

            for nOccluder, Occluder in enumerate(self.Generators):
                if nOccluder == nGenerator:
                    continue
                if not self.ForwardGenerators[nOccluder]:
                    continue
                DeltaOccluder = (Occluder - self.T)
                CameraFrameOccluderLocation = self.R.dot(DeltaOccluder) - self._CameraOffsets[nCamera]
                if CameraFrameOccluderLocation[-1] >= CameraFrameLocation[-1]:
                    continue
                OffLineDistance = np.linalg.norm(DeltaOccluder - (DeltaOccluder.dot(UX))*UX)
                if OffLineDistance < self._GeneratorRadius:
                    return False, event

            ScreenFrameLocation = self.KMat.dot(XPoint)
            if ScreenFrameLocation[-1] > 0:
                OnScreenLocation = np.array(ScreenFrameLocation[:2] / ScreenFrameLocation[-1], dtype = int)
                if ((OnScreenLocation < 0).any() or (OnScreenLocation > self._Geometry-1).any()):
                    return False, event
                event = event.Join(CameraEvent, timestamp = self.t, location = OnScreenLocation, polarity = 0, SubStreamIndex = self.SubStreamIndexes[nCamera])
                if self._DisparityEvents:
                    disparity, sign = self.DisparityData(CameraFrameLocation[-1], nCamera)
                    event.Attach(DisparityEvent, disparity = disparity, sign = sign) 
            else:
                return False, event
        else:
            return False, event


        if self._TrackerEvents:
            for SubStreamIndex, nGenerator, TrackerLocation, disparity, sign in self.TrackersGenerators:
                if TrackerLocation is None:
                    color = 'k'
                    TrackerLocation = np.zeros(2)
                else:
                    color = 'g'
                if self._TrackersLocationGaussianNoise and color == 'g':
                    TrackerLocation = TrackerLocation + np.random.normal(0, self._TrackersLocationGaussianNoise, size = 2)
                    if ((TrackerLocation <= self._TrackerEdgeMargin).any() or (TrackerLocation >= self._Geometry-1-self._TrackerEdgeMargin).any()):
                        continue
                JoinedTrackerEvent = event.Join(TrackerEvent, TrackerLocation = TrackerLocation, TrackerID = nGenerator, TrackerAngle = 0., TrackerScaling = 0., TrackerColor = color, TrackerMarker = 'o', SubStreamIndex = SubStreamIndex)
                if self._DisparityEvents:
                    JoinedTrackerEvent.Attach(DisparityEvent, disparity = disparity, sign = sign)
            self.TrackersGenerators = []

        return True, event

    def RemoveTracker(self, nGenerator, nCamera):
        self.OnScreenTrackers[nGenerator][nCamera][0] = 0
        if not self.IsOutlier[nGenerator][nCamera] is None:
            self.IsOutlier[nGenerator][nCamera] = None
            self.NOutliers[nCamera] -= 1
        self.Log("Removing tracker {0} for camera {1}".format(nGenerator, nCamera))
        self.TrackersGenerators += [(self.SubStreamIndexes[nCamera], nGenerator, None, None, None)]

    def DisparityData(self, Depth, nCamera):
        return int(self.StereoBaseDistance * self.K / Depth + 0.5), nCamera*2-1

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

    def StartNextSequenceStep(self):
        if not self.Sequence:
            return False
        dt, self.AimedOmega, self.AimedV = self.Sequence.pop(0)
        self.NextEndT = self.t + dt
        return True
        
    def _OnInputIndexesSet(self, Indexes):
        if len(Indexes) != len(self._CameraOffsets):
            self.LogWarning("Wrong number of SubSTreamOutput indexes compared to the number of cameras specified")
        self.SubStreamIndexes = list(Indexes)
        return True

    def GetCameraTwist(self, nCamera):
        return self.Omega, self.V + np.cross(self.Omega, self._CameraOffsets[nCamera])

    @property
    def Geometry(self):
        return self._Geometry

    def GenerateMap(self):
        self.Generators = []
        if self._MapType == 'cubic':
            Volume = _MAP_DIMENSION**3
            GeneratorVolume = 4/3*np.pi*self._GeneratorRadius**3
            NGenerators = Volume * self._MapDensity / GeneratorVolume
            NGeneratorsPerDimension = int(NGenerators**(1/3))

            Ds = np.linspace(-_MAP_DIMENSION/2, _MAP_DIMENSION/2, NGeneratorsPerDimension)
            
            for x in Ds:
                for y in Ds:
                    for z in Ds:
                        self.Generators += [np.array([x, y, z])]
