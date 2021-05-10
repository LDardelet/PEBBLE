import numpy as np
import matplotlib.pyplot as plt

from PEBBLE import ModuleBase, TrackerEvent

def _CompleteWarpBasisFrom(BasisVectors):
    # All input vectors must already by unitary and orthogonal. It is just a completion of the basis
    Dim = BasisVectors[0].shape[0]
    for n in range(Dim):
        StartVector = np.zeros((Dim), dtype = float)
        StartVector[n] = 1
        for ExistingVector in BasisVectors:
            StartVector -= ExistingVector*((StartVector*ExistingVector).sum())
        N = np.linalg.norm(StartVector)
        if N > 0.00001: # We don't put 0 here to avoid numerical issues
            BasisVectors += [StartVector/N]
            if len(BasisVectors) == Dim:
                return BasisVectors

def _CompleteProjectiveSymetricalBasisFrom(InitialLocationVector):
    x0, y0 = InitialLocationVector[:-1]/InitialLocationVector[-1]
    N2 = x0**2 + y0**2
    M = np.sqrt(1+N2)
    u2 = np.array([-x0-M*y0, -y0+M*x0, N2])
    u3 = np.array([-x0+M*y0, -y0-M*x0, N2])
    return np.array([InitialLocationVector / np.linalg.norm(InitialLocationVector), u2/np.linalg.norm(u2), u3 / np.linalg.norm(u3)])

def _GenerateSubSpaceBasisFrom(Vectors):
    # All input vectors must only be unitary here
    SubBasis = [Vectors.pop(0)]
    for Vector in Vectors:
        for BasisVector in SubBasis:
            Vector = Vector - BasisVector*(BasisVector*Vector).sum()
        N = np.linalg.norm(Vector)
        if N > 0:
            SubBasis += [Vector/N]
    return SubBasis

class BundleAdjustmentWarp(ModuleBase):
    _DISTANCE_NORMALIZATION_FACTOR = 1.

    def _OnCreation(self):
        '''
        '''
        self._ScreenCenter = [320., 0]
        self._ScreenRatio = [640., 640.]
        self._EventsConsideredRatio = 1.
        self._SplitSolving = True
        self._RatioCameraWarp = 0.7

        self._CameraWarpMethod = 'right'
        self._Point2DWarpMethod = 'bilateral'

        self._CameraFLRLimitDecay = 0.98
        self._CameraWarpDecayPerEvent = 0.01 / self._EventsConsideredRatio
        self._Point2DWarpDecayPerEvent = 0 * self._CameraWarpDecayPerEvent/100

        self._DefaultCameraVector = np.array([1., 0., 0., 1., 0., 0.])
        #self._DefaultCameraVector = np.array([1., 0., 0., 1., 0., 0., 1.])
        self._InitialCameraWarp = self._CameraFLRLimitDecay

        self._Point2DDefaultDistanceWarp = 0.99
        self._Point2DForcedDepthStretch = 0.9
        self._DefaultUnknownDepthDistance = 2.5
        self._NPointsForDistanceDefinition = 10

        self._DefaultCameraStretch = 0.9
        self._Default2DPointStretch = 0.9
        self._FirstPointWarpMethod = lambda CameraSpaceWarp: self._Default2DPointStretch
        self._UnknownPoseLimit = 1.

        self._FullyDetermined2DPointLambdaRatio = 0.1

        self.LastConsideredEventTs = 0.
        self.Determined2DPointsLocations = {}

        self.CPNM = self._CameraPoseNormalizationMethod
        self._ContinuousRotationCheck = True

        self._MonitorDt = 0.02
        self._MonitoredVariables = [('CameraSpaceWarp@Value', np.array),
                                    ('Point2DSpaceWarps@Value', np.array),
                                    ('Point2DSpaceWarps@C', np.array),
                                    ('Point2DSpaceWarps@NWarps', int),
                                    ('Point2DSpaceWarps@ScalingFacor', float),
                                    'CameraSpaceWarp@FirstLambdaRatio']
        
        self._2DPointLocationCheatEnabled = True
        self._CheatGaussianNoise = 0.01
        self._NControlPoints = 10

    def _OnInitialization(self):
        self.UsedPointsForDistanceDefinition = {}
        self.AveragedDistance = self._DefaultUnknownDepthDistance
        self.Point2DSpaceWarps = {}
        self.LastPointsReceived = {}
        self.LastTs = -np.inf
        self.CameraSpaceWarp = SpaceWarp(6, self.CPNM, self._DefaultCameraStretch, InitialValue = self._DefaultCameraVector, InitialStretch = self._InitialCameraWarp, RetreiveMethod = 'lambdas', WarpMethod = self._CameraWarpMethod)
        # P = (r_11, r_12, r_21, r_22, t_x, t_y)

        self._ScreenCenter = np.array(self._ScreenCenter)
        self._ScreenRatio = np.array(self._ScreenRatio)
        self.K = np.array([[self._ScreenRatio[0], self._ScreenCenter[0]], [0, 1]])
        self._Alpha = self._ScreenRatio[0]

        self.RemainingControlPoints = self._2DPointLocationCheatEnabled * self._NControlPoints
        self.InitialPointLocated = False

        if self._EventsConsideredRatio < 1:
            self._RandomizeEvents = True
        else:
            self._RandomizeEvents = False
        return True

    def _OnEventModule(self, event):
        if not event.Has(TrackerEvent):
            return
        if self._RandomizeEvents and np.random.rand() > self._EventsConsideredRatio:
            return

        self.OnTrackerEvent(event)
    
    def OnTrackerEvent(self, event):
        self.LastTs = event.timestamp
        self.LastPointsReceived[event.TrackerID] = (np.array(event.TrackerLocation), event.timestamp)
        PreviousPose = np.array(self.CameraSpaceWarp.Vectors[0])
        #event.TrackerLocation = 639 - event.TrackerLocation
        if event.TrackerID not in self.Point2DSpaceWarps:
            self._SystemKnowledgeDecay(event.timestamp)
            self.Log("Creating space warp for tracker {0}".format(event.TrackerID) + self._2DPointLocationCheatEnabled*" CHEATED")
            
            if self._2DPointLocationCheatEnabled:
                Noise = np.random.normal(0, self._CheatGaussianNoise * (self.RemainingControlPoints == 0), size = 2)
                if self.RemainingControlPoints:
                    Confidence = 1.
                else:
                    Confidence = max(0.1, np.e**(-np.linalg.norm(Noise) / max(self._CheatGaussianNoise*10, 0.00001)))
                HLocation = np.concatenate((self.__Framework__.Sim2D.BaseMap.EventsGenerators[event.TrackerID].Location + Noise, [1]))
                InitialBasis = _CompleteProjectiveSymetricalBasisFrom(HLocation)
                Stretch = 1 - Confidence * (1-self._FullyDetermined2DPointLambdaRatio / 2)
                self.Point2DSpaceWarps[event.TrackerID] = SpaceWarp(2+1, self._Point2DNormalizationMethod, self._Default2DPointStretch, InitialBasis = InitialBasis, InitialStretch = Stretch, WarpMethod = self._Point2DWarpMethod, AutoScale = True)
                if self.RemainingControlPoints:
                    self.RemainingControlPoints -= 1
            else:
                self.Point2DSpaceWarps[event.TrackerID] = SpaceWarp(2+1, self._Point2DNormalizationMethod, self._Default2DPointStretch, WarpMethod = self._Point2DWarpMethod, AutoScale = True)
                Vs = self._CreateDefaultDistanceOrthogonalVector()
                self.Point2DSpaceWarps[event.TrackerID].AddData(Vs, Stretch = self._Point2DDefaultDistanceWarp)
                Vs = self._Generate2DPointWarpVectors(event.TrackerLocation)
                self.Point2DSpaceWarps[event.TrackerID].AddData(Vs, Stretch = self._FirstPointWarpMethod(self.CameraSpaceWarp))
                self.Point2DSpaceWarps[event.TrackerID].RetreiveData()

                if len(self.UsedPointsForDistanceDefinition) < self._NPointsForDistanceDefinition:
                    self.UsedPointsForDistanceDefinition[event.TrackerID] = np.linalg.norm(self.Point2DSpaceWarps[event.TrackerID].Value[:-1])
                    self.AveragedDistance = np.mean(list(self.UsedPointsForDistanceDefinition.values()))

        else:
            self._SystemKnowledgeDecay(event.timestamp, event.TrackerID)

            #self.CameraSpaceWarp.AddData(Vs, DeltaVs, Certainty = 0.2 + 0.8*(1 - self.Point2DSpaceWarps[event.TrackerID].FirstLambdaRatio**2))

            if self._SplitSolving:
                RandomValue = np.random.rand()
            else:
                RandomValue = -1

            if event.TrackerID in self.Determined2DPointsLocations.keys() or RandomValue < self._RatioCameraWarp:
                Vs, DeltaVs = self._GenerateCameraWarpVectors(event.TrackerLocation, event.TrackerID)
                self.CameraSpaceWarp.AddData(Vs, DeltaVs, Stretch = min(self._DefaultCameraStretch, self.Point2DSpaceWarps[event.TrackerID].FirstLambdaRatio))
                self.CameraSpaceWarp.RetreiveData()
            if self.CameraSpaceWarp.FirstLambdaRatio < self._UnknownPoseLimit: # Meaning we know enough about the pose of the camera to correct the 2D points locations
                if self.Point2DSpaceWarps[event.TrackerID].FirstLambdaRatio > self._FullyDetermined2DPointLambdaRatio:
                    if (not self._SplitSolving or RandomValue > self._RatioCameraWarp): # Make location of 2D point more precise
                        Vs = self._Generate2DPointWarpVectors(event.TrackerLocation)
                        self.Point2DSpaceWarps[event.TrackerID].AddData(Vs, Stretch = min(self._Default2DPointStretch, self.CameraSpaceWarp.FirstLambdaRatio))
                        self.Point2DSpaceWarps[event.TrackerID].RetreiveData()

                        if event.TrackerID in self.UsedPointsForDistanceDefinition.keys():
                            self.UsedPointsForDistanceDefinition[event.TrackerID] = np.linalg.norm(self.Point2DSpaceWarps[event.TrackerID].Value[:-1])
                            self.AveragedDistance = np.mean(list(self.UsedPointsForDistanceDefinition.values()))
                elif event.TrackerID not in self.Determined2DPointsLocations.keys():
                    self.Determined2DPointsLocations[event.TrackerID] = np.array(self.Point2DSpaceWarps[event.TrackerID].Vectors[0])
                    self.Log("Added {0} to known points at {1}".format(event.TrackerID, self.Determined2DPointsLocations[event.TrackerID]), 3)

        if np.random.rand() < 0.01:
            self.Log("Point {0:3d} FLR : {1:.3f}, Pose FLR : {2:.3f}, Pose Trace : {3:.3f}".format(event.TrackerID, self.Point2DSpaceWarps[event.TrackerID].FirstLambdaRatio, self.CameraSpaceWarp.FirstLambdaRatio, self.CameraSpaceWarp.C.trace()))
            self.Log("Camera R[0,0] : {0:.3f}".format(self.CameraSpaceWarp.Value[0]))

        # The two following methods are for metric renormalization
    def _PointDistanceCorrection(self, X_i):
        return X_i * np.array([1., 1., 1. + 0 * ((self.AveragedDistance/self._DefaultUnknownDepthDistance)-1)]) # We multiply the homogeneous coodinate by the average distance, thus normalizing distances accordingly
    def _CameraDistanceCorrection(self, P): # Currently disabled in camspacewarp init
        #P[:-2] /= self.AveragedDistance # We divide the translation component by the average distance metric
        return P

    def _Point2DNormalizationMethod(self, X_i, Init = False):
        if X_i[-1] == 0:
            return X_i
        else:
            return X_i/X_i[-1]
    def _SuperDumbCameraPoseNormalizationMethod(self, P, init = False):
        N = np.sqrt((P[:4]**2).sum() / 2)
        if N:
            NormalizedPose = P / N
            return NormalizedPose*np.sign(NormalizedPose[0])
        else:
            return P
    def _DumbCameraPoseNormalizationMethod(self, P, Init = False):
        N = np.linalg.det(P[:4].reshape((2,2)))
        if N != 0:
            return P / N**(1/2)
        else:
            return P
    def _CameraPoseNormalizationMethod(self, P, Init = False):
        #if P[-1] != 0:
        #    return P[:-1] / P[-1]
        #else:
        #    return P[:-1]
        if P.shape[0] == 7:
            if P[6] != 0:
                P = P[:6]/P[6]
            else:
                P = P[:6]
        PreviousP = np.array(P)
        S = P[:4].reshape((2,2))
        if self._ContinuousRotationCheck and not Init:
            ReferToRotation = np.array(self.CameraSpaceWarp.Value[:4].reshape((2,2)))
            if (S.dot(ReferToRotation.T)).trace() < 0: # If the rotation is about a 180 degree angle
                S *= -1
                P[4:] = -P[4:]
        U, Sigmas, V = np.linalg.svd(S)
        R = U.dot(V)
        #Sign = np.sign(np.linalg.det(R))
        P[:4] = R.reshape(4)
        if ((Sigmas != 0).any()):
            #P[4:] = P[4:] / Sigmas
            SigmasInv = np.array([[1/Sigmas[0], 0.], [0., 1/Sigmas[1]]])
            P[4:] = U.dot(SigmasInv.dot(U.T.dot(P[4:])))
        return P

    def _Generate2DPointWarpVectors(self, TrackerLocation):
        P = self.CameraSpaceWarp.Value

        Vs = [None]
        delta_x_i = TrackerLocation - self._ScreenCenter
        Vs[0] = np.array([self._Alpha * P[0] - delta_x_i[0] * P[2], self._Alpha * P[1] - delta_x_i[0] * P[3], self._Alpha * P[4] - delta_x_i[0] * P[5]])
        for i in range(1):
            Vs[i] = Vs[i] / np.linalg.norm(Vs[i])
        return Vs

    def _CreateDefaultDistanceOrthogonalVector(self):
        P = self.CameraSpaceWarp.RetreiveData()
        RT = np.array([[P[0], P[1], P[4]], [P[2], P[3], P[5]]])
        Rt = RT[:,:2].T
        X = -Rt.dot(RT[:,2])
        Uv = Rt[:,1]

        V = np.array([Uv[0], Uv[1], -(self._DefaultUnknownDepthDistance + X.dot(Uv))])
        return [V/np.linalg.norm(V)]

    def _GenerateCameraWarpVectors(self, TrackerLocation, TrackerID):
        PointWarp = self.Point2DSpaceWarps[TrackerID]
        PointWarp.RetreiveData()

        delta_x_i = TrackerLocation - self._ScreenCenter

        Vs = self._GenerateNormalCameraWarpVectors(self._PointDistanceCorrection(PointWarp.Value), delta_x_i)
        Delta_Vs = []
        return Vs, Delta_Vs

    def _GenerateNormalCameraWarpVectors(self, Xi, delta_x_i):
        Vs = [None]
        #Vs[0] = np.array([self._Alpha * Xi[0], self._Alpha * Xi[1], -delta_x_i[0] * Xi[0], -delta_x_i[0] * Xi[1], self._Alpha * Xi[2], -delta_x_i[0] * Xi[2], 0.])
        Vs[0] = np.array([self._Alpha * Xi[0], self._Alpha * Xi[1], -delta_x_i[0] * Xi[0], -delta_x_i[0] * Xi[1], self._Alpha * Xi[2], -delta_x_i[0] * Xi[2]])
        for i in range(1):
            Vs[i] = Vs[i] / np.linalg.norm(Vs[i])
        return Vs

    def _SystemKnowledgeDecay(self, t, TrackerID = None):
        if self.CameraSpaceWarp.FirstLambdaRatio < self._CameraFLRLimitDecay:
            self.CameraSpaceWarp.Decay(self._CameraWarpDecayPerEvent)
        if not TrackerID is None:
            self.Point2DSpaceWarps[TrackerID].Decay(self._Point2DWarpDecayPerEvent)
        self.LastConsideredEventTs = t

    def Plot(self, fax = None, PlotPoses = 1, IDs = None, Interactive = False):
        if fax is None:
            f, ax = plt.subplots(1,1)
        else:
            f, ax = fax
        for nPose, P in enumerate(reversed(self.CameraSpaceWarp.Vectors[:PlotPoses])):
            RT = self.RT()
            Rt = RT[:,:2].T
            X = -Rt.dot(RT[:,2])
            ax.plot(X[0], X[1], 'sg')
            ax.plot([X[0], (X+Rt[:,1])[0]], [X[1], (X+Rt[:,1])[1]], '--g')
            ax.plot((X-Rt[:,0])[0], (X-Rt[:,0])[1], 'sb')
            ax.plot([(X-Rt[:,0])[0], (X+Rt[:,0])[0]], [(X-Rt[:,0])[1], (X+Rt[:,0])[1]], '--b')
        
        if IDs is None:
            IDs = list(self.Point2DSpaceWarps.keys())

        for ID in IDs:
            PW = self.Point2DSpaceWarps[ID]
            P = PW.Value

            dot = ax.plot(P[0], P[1], 'ob')[0]
            if Interactive:
                self._InteractiveMapElements['clickables'] += [{'ID':ID, 'Snap': None, 'dot':dot, 'location':np.array([P[0], P[1]])}]
            ax.text(P[0], P[1]+0.05, str(ID), color='b')
            if self.LastPointsReceived[ID][1] > self.LastTs - 0.02: #
                if ID in self.Determined2DPointsLocations.keys():
                    c = 'g'
                else:
                    c = 'b'
                ax.plot([X[0], P[0]], [X[1], P[1]], c)
        return f, ax

    def PointMapProba(self, ID, fax = None, PreviousMap = None, Definition = (200, 200), Snap = None):
        if Snap is None:
            C = self.Point2DSpaceWarps[ID].C
            SF = self.Point2DSpaceWarps[ID].ScalingFacor
        else:
            C = self.History['Point2DSpaceWarps@C'][Snap][ID]
            SF = self.History['Point2DSpaceWarps@ScalingFacor'][Snap][ID]

        if fax is None:
            f, ax = plt.subplots(1,1)
            ax.set_xlim(-2.625,  2.625)
            ax.set_ylim(-0.125,  5.125)
        else:
            f, ax = fax
        Map = np.zeros(Definition)
        xLims, yLims = ax.get_xlim(), ax.get_ylim()
        xs = np.linspace(xLims[0], xLims[1], Definition[0])
        ys = np.linspace(yLims[0], yLims[1], Definition[1])

        for nx, x in enumerate(xs):
            for ny, y in enumerate(ys):
                X = np.array([x, y, 1])
                #X = X/ np.linalg.norm(X)
                Y = X.T.dot((np.identity(3)/SF-C).dot(X))*SF
                Map[nx, ny] = np.e**Y
        if PreviousMap is None:
            PreviousMap = ax.imshow(np.transpose(Map), origin = 'lower', extent = xLims + yLims, vmin = 0., vmax = 1.)
        else:
            PreviousMap.set_data(np.transpose(Map))
        return (f, ax), PreviousMap

    def DrawPastOf(self, fax, ID, Interactive = False):
        if fax is None:
            f, ax = plt.subplots(1,1)
        else:
            f, ax = fax
        for nSnap, Snap in enumerate(self.History['Point2DSpaceWarps@Value']):
            if not ID in Snap.keys():
                continue
            P = Snap[ID][:2]
            dot = ax.plot(P[0], P[1], 'vb')[0]
            if Interactive:
                self._InteractiveMapElements['clickables'] += [{'ID':ID, 'Snap': nSnap, 'dot':dot, 'location':np.array([P[0], P[1]])}]

    def InteractiveMap(self, SimPlotMethod):
        self._InteractiveMapElements = {'fax': plt.subplots(1,1), 'clickables':[], 'currPoint': {'ID':None, 'Snap':None}, 'map':None}
        self._InteractiveMapElements['fax'][1].set_title("t = {0:.3f}, no ID".format(self.LastTs))
        SimPlotMethod(self._InteractiveMapElements['fax'])
        self.Plot(self._InteractiveMapElements['fax'], Interactive = True)
        self._InteractiveMapElements['fax'][0].canvas.mpl_connect('button_press_event', lambda event: self._OnInteractiveClick(event, self._InteractiveMapElements))
    def _OnInteractiveClick(self, event, InteractiveMapElements):
        loc = np.array([event.xdata, event.ydata])
        Distances = [np.linalg.norm(loc - Point['location']) for Point in InteractiveMapElements['clickables']]
        if np.min(Distances) > 0.2:
            return
        NewCurrPoint = InteractiveMapElements['clickables'][np.argmin(Distances)]
        if NewCurrPoint['ID'] == InteractiveMapElements['currPoint']['ID'] and NewCurrPoint['Snap'] == InteractiveMapElements['currPoint']['Snap']:
            return
        if NewCurrPoint['Snap'] is None:
            t = self.LastTs
        else:
            t = self.History['t'][NewCurrPoint['Snap']]
        InteractiveMapElements['fax'][1].set_title("t = {0:.3f}, ID = {1}".format(t, NewCurrPoint['ID']))

        if NewCurrPoint['ID'] != InteractiveMapElements['currPoint']['ID']: # We have to remove all points from the past of the currPoint point
            NewList = []
            for Point in InteractiveMapElements['clickables']:
                if Point['ID'] == InteractiveMapElements['currPoint']['ID'] and not Point['Snap'] is None:
                    Point['dot'].remove()
                else:
                    NewList += [Point]
            InteractiveMapElements['clickables'] = NewList
            self.DrawPastOf(InteractiveMapElements['fax'], NewCurrPoint['ID'], Interactive = True)
        self.PointMapProba(NewCurrPoint['ID'], InteractiveMapElements['fax'], InteractiveMapElements['map'], Snap = NewCurrPoint['Snap'])
        InteractiveMapElements['currPoint'] = NewCurrPoint
        InteractiveMapElements['fax'][0].show()

    def Reconstruction(self, Tolerance = 2):
        Errors = {}
        for ID, LastPoint in self.LastPointsReceived.items():
            ApparentScreenLoc = self.GetCurrentPointLocOnScreen(ID)
            Errors[ID] = (ApparentScreenLoc, ApparentScreenLoc - LastPoint[0][0], ((LastPoint[0][0] < Tolerance) or (LastPoint[0][0] > (self._ScreenRatio[0] - Tolerance - 1))))
        return Errors

    def RT(self):
        P = self.CameraSpaceWarp.Value
        return np.array([[P[0], P[1], P[4]], [P[2], P[3], P[5]]]) # Why  is there a "-" ? Else, reconstruction is behind and Plot is reversed
    def KRT(self):
        return self.K.dot(self.RT())

    def GetCurrentPointLocOnScreen(self, ID):
        PW = self.Point2DSpaceWarps[ID]
        ApparentScreenLoc = self.KRT().dot(PW.Value)
        if ApparentScreenLoc[1] < 0:
            Bonus = 1000
        else:
            Bonus = 0
        ApparentScreenLoc = ApparentScreenLoc[0] / ApparentScreenLoc[1]
        ApparentScreenLoc = max(0, min(self._ScreenRatio[0]-1, ApparentScreenLoc))
        return ApparentScreenLoc + Bonus

class SpaceWarp:
    _RemoveOthogonalVectors = False
    _WARP_METHOD = {'left': 2, 'bilateral': 3, 'right': 1}
    def __init__(self, Dim, NormalizationMethod, DefaultStretch, InitialValue = None, InitialBasis = None, InitialStretch = None, RetreiveMethod = 'lambdas', WarpMethod = 'right', AutoScale = False):
        self.Dim = Dim
        self.DefaultStretch = DefaultStretch
        self._WarpMethod = self._WARP_METHOD[WarpMethod]
        self._AutoScale = AutoScale
        
        self.RetreiveMethod = {'lambdas':0, 'pointwarp':1}[RetreiveMethod] # 0 for Lambdas, 1 for warp of previous point

        self.ScalingFacor = 1.
        self.NWarps = 0
        self._UpToDate = False

        self.Lambdas = []
        self.ProbaSum = []
        self.Vectors = []
        self.FirstLambdaRatio = 1.
        self._NormalizationMethod = NormalizationMethod

        self.NormalVectorsHistory = []

        self.T = np.identity(self.Dim)
        self.M = np.identity(self.Dim)

        if not InitialValue is None or not InitialBasis is None:
            self._BuildCFromInitial(InitialValue = InitialValue, InitialBasis = InitialBasis, InitialStretch = InitialStretch)
        else:
            self.C = np.identity(Dim)

    def Decay(self, DecayValue):
        self.C = DecayValue*np.identity(self.Dim) + (1-DecayValue)*self.C

    def AddData(self, NormalUnitaryVectors, UntouchedUnitaryVectors = [], Certainty = 1., Stretch = None):
        ConservedVectors = []
        if self._RemoveOthogonalVectors and self.NormalVectorsHistory:
            for V in NormalUnitaryVectors:
                Add = True
                for PreviousV in self.NormalVectorsHistory[-1]:
                    if (V*self.NormalVectorsHistory[-1]).sum() > 0.999:
                        Add = False
                        break
            if Add:
                ConservedVectors += [V]
            if not ConservedVectors:
                return
            else:
                NormalUnitaryVectors = ConservedVectors
        if Stretch is None:
            Stretch = self.DefaultStretch
        self.NormalVectorsHistory += [[np.array(Vector) for Vector in NormalUnitaryVectors]]
        self.GenerateWarpMatrix(NormalUnitaryVectors, (Stretch ** (1/(1 + int(self._WarpMethod == 3))))**Certainty , UntouchedUnitaryVectors)
        if self._WarpMethod & 0b1:
            self.C = self.C.dot(self.M)
        if self._WarpMethod & 0b10:
            self.C = self.M.dot(self.C)
        self.NWarps += 1
        self._UpToDate = False
        #self.RetreiveData()

    def GenerateWarpMatrix(self, NormalUnitaryVectors, AssociatedStretches, UntouchedUnitaryVectors = []):
        # So far, we assume a maximum of 2 Normal vectors and 2 untouched vectors
        if not type(AssociatedStretches) == list:
            AssociatedStretches = [AssociatedStretches for nVector in range(len(NormalUnitaryVectors))]

        # First we remove all untouched components, and change the scaling accordingly
        FullyWarpedVectors = []
        FullyWarpedStretches = []
        if UntouchedUnitaryVectors:
            UntouchedVectorsBasis = _GenerateSubSpaceBasisFrom(UntouchedUnitaryVectors)
            for nV, V in enumerate(NormalUnitaryVectors):
                for O in UntouchedVectorsBasis:
                    NormalUnitaryVectors[nV] -= O*(V*O).sum()
            for V in NormalUnitaryVectors:
                N = np.linalg.norm(V)
                if N > 0:
                    FullyWarpedVectors += [V/N]
                    FullyWarpedStretches += [AssociatedStretches[nV]**N]
        else:
            FullyWarpedVectors = list(NormalUnitaryVectors)
            FullyWarpedStretches = list(AssociatedStretches)

        if len(FullyWarpedVectors) == 2: # We orthogonalize them
            s1, s2, v12 = FullyWarpedStretches[0], FullyWarpedStretches[1], (FullyWarpedVectors[0]*FullyWarpedVectors[1]).sum()
            M = np.array([[s1, s1*v12], [s2*v12, s2]])
            LocalStretches, LocalVectors = [np.real(data) for data in np.linalg.eig(M)]

            FinalWarpedVectors = [FullyWarpedVectors[0] * LocalVectors[0,0] + FullyWarpedVectors[1] * LocalVectors[1,0], FullyWarpedVectors[0] * LocalVectors[0,1] + FullyWarpedVectors[1] * LocalVectors[1,1]]
            FinalWarpedVectors = [FinalWarpedVectors[0]/np.linalg.norm(FinalWarpedVectors[0]), FinalWarpedVectors[1]/np.linalg.norm(FinalWarpedVectors[1])]
            FinalStretches = (np.array(AssociatedStretches)**LocalStretches).tolist()
        else:
            FinalWarpedVectors = list(FullyWarpedVectors)
            FinalStretches = AssociatedStretches + [1] # For simplicity

        self.T = np.array(_CompleteWarpBasisFrom(FinalWarpedVectors))
        self.S = np.identity(self.Dim)
        self.S[0,0] = 1./FinalStretches[0]
        self.S[1,1] = 1./FinalStretches[1]

        self.M = self.T.T.dot(self.S.dot(self.T))

    def _BuildCFromInitial(self, InitialValue = None, InitialBasis = None, InitialStretch = None):
        if InitialStretch is None:
            InitialStretch = self.DefaultStretch
        if InitialBasis is None:
            InitialBasis = np.array(_CompleteWarpBasisFrom([InitialValue / np.linalg.norm(InitialValue)]))
        InitialStretchMatrix = np.identity(self.Dim)
        for i in range(1, self.Dim):
            if hasattr(InitialStretch, '__iter__'):
                InitialStretchMatrix[i,i] = 1./InitialStretch[i]
            else:
                InitialStretchMatrix[i,i] = 1./InitialStretch
        self.C = InitialBasis.T.dot(InitialStretchMatrix.dot(InitialBasis))
        self.Vectors = [InitialValue]
        self.RetreiveData(Init = True)

    def RetreiveData(self, Init = False):
        if self._UpToDate:
            return self.Value
        if self._WarpMethod == 3: # We force the use of symmetrical properties of the matrix
            Lambdas, Vectors = [np.real(data) for data in np.linalg.eigh(self.C)]
        else:
            Lambdas, Vectors = [np.real(data) for data in np.linalg.eig(self.C)]
        if self.RetreiveMethod == 0:
            if False and Lambdas.max() > 1.001:
                print("Excessive lambda, dimension is {0}".format(Vectors.shape[0]))
            Indexes = np.argsort(Lambdas)
            
            if self._AutoScale:
                LambdaScaling = Lambdas.max()
                self.ScalingFacor *= LambdaScaling
                self.C /= LambdaScaling
            else:
                LambdaScaling = 1
            
            self.FirstLambdaRatio = Lambdas[Indexes[1]] / Lambdas[Indexes[0]]
            self.Lambdas = []
            self.ProbaSum = []
            self.Vectors = []
            ProbaSum = 0
            for nResult, Index in enumerate(Indexes):
                Lambda = Lambdas[Index]/LambdaScaling
                ProbaSum += (Lambda > 0.8)*Lambda**2
                self.Lambdas += [Lambda]
                self.ProbaSum += [ProbaSum]
                self.Vectors += [self._NormalizationMethod(Vectors[:,Index], Init)]
        elif self.RetreiveMethod == 1:
            self.Lambdas = list(reversed(np.sort(Lambdas)))
            self.ProbaSum = [self.Lambdas[0]]
            for Lambda in self.Lambdas[1:]:
                self.ProbaSum += [self.ProbaSum[-1] + (Lambda > 0.9)*Lambda**2]
            self.Vectors = [self._NormalizationMethod(self.M.dot(self.Vectors[0]), Init)]
            self.FirstLambdaRatio = self.Lambdas[1] / self.Lambdas[0]

        self.Value = self.Vectors[0]
        self._UpToDate = True
        return self.Value

class BundleAdjustment(BundleAdjustmentWarp):
    pass

