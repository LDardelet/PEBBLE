import numpy as np
import matplotlib.pyplot as plt

from Framework import Module, TrackerEvent

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

class BundleAdjustmentWarp(Module):
    _DISTANCE_NORMALIZATION_FACTOR = 1.

    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to handle ST-context memory.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Computation'
        
        self._ScreenCenter = [320., 0]
        self._ScreenRatio = [640., 640.]
        self._EventsConsideredRatio = 1.
        self._SplitSolving = False
        self._RatioCameraWarp = 0.7

        self._CameraWarpMethod = 'right'
        self._Point2DWarpMethod = 'right'

        self._InitialCameraWarp = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        self._DefaultCameraVector = np.array([1., 0., 0., 1., 0., 0., 1.])

        self._Point2DDefaultDistanceWarp = 0.99
        self._Point2DForcedDepthStretch = 0.9
        self._DefaultUnknownDepthDistance = 2.5
        self._NPointsForDistanceDefinition = 10
        self.UsedPointsForDistanceDefinition = {}
        self.AveragedDistance = self._DefaultUnknownDepthDistance

        self._DefaultCameraStretch = 0.95
        self._Default2DPointStretch = 0.1
        self._FirstPointWarpMethod = lambda CameraSpaceWarp: self._Default2DPointStretch
        self._UnknownPoseLimit = 1.

        self._FullyDetermined2DPointLambdaRatio = 0.1

        self._CameraFLRLimitDecay = 0.98
        self._CameraWarpDecayPerEvent = 0.001 / self._EventsConsideredRatio
        self._Point2DWarpDecayPerEvent = 0 * self._CameraWarpDecayPerEvent/100

        self.LastConsideredEventTs = 0.
        self.Determined2DPointsLocations = {}

        self._MonitorDt = 0.01
        self._MonitoredVariables = [('CameraSpaceWarp@Value', np.array),
                                    ('Point2DSpaceWarps@Value', np.array),
                                    ('Point2DSpaceWarps@C', np.array),
                                    ('Point2DSpaceWarps@NWarps', int),
                                    'CameraSpaceWarp@FirstLambdaRatio']

    def _InitializeModule(self, **kwargs):
        self.Point2DSpaceWarps = {}
        self.LastPointsReceived = {}
        self.LastTs = -np.inf
        self.CameraSpaceWarp = SpaceWarp(6+1, lambda P: self._CameraDistanceCorrection(self._CameraPoseNormalizationMethod(P)), self._DefaultCameraStretch, self._DefaultCameraVector, self._InitialCameraWarp, RetreiveMethod = 'lambdas', WarpMethod = self._CameraWarpMethod)
        # P = (r_11, r_12, r_21, r_22, t_x, t_y)

        self._ScreenCenter = np.array(self._ScreenCenter)
        self._ScreenRatio = np.array(self._ScreenRatio)
        self._Alpha = self._ScreenRatio[0]

        self.InitialPointLocated = False

        if self._EventsConsideredRatio < 1:
            self._RandomizeEvents = True
        else:
            self._RandomizeEvents = False
        return True

    def _OnEventModule(self, event):
        if event.__class__ != TrackerEvent:
            return event
        if self._RandomizeEvents and np.random.rand() > self._EventsConsideredRatio:
            return event
        
        self.LastTs = event.timestamp
        self.LastPointsReceived[event.TrackerID] = (np.array(event.TrackerLocation), event.timestamp)
        PreviousPose = np.array(self.CameraSpaceWarp.Vectors[0])
        #event.TrackerLocation = 639 - event.TrackerLocation
        if event.TrackerID not in self.Point2DSpaceWarps:
            self._SystemKnowledgeDecay(event.timestamp)
            self.Log("Creating space warp for tracker {0}".format(event.TrackerID))
            self.Point2DSpaceWarps[event.TrackerID] = SpaceWarp(3, lambda X_i: self._Point2DNormalizationMethod(X_i), self._Default2DPointStretch, WarpMethod = self._Point2DWarpMethod)

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

            Vs, DeltaVs = self._GenerateCameraWarpVectors(event.TrackerLocation, event.TrackerID)
            #self.CameraSpaceWarp.AddData(Vs, DeltaVs, Certainty = 0.2 + 0.8*(1 - self.Point2DSpaceWarps[event.TrackerID].FirstLambdaRatio**2))
            if False and event.timestamp < 10:
                for nV, V in enumerate(Vs):
                    V[-1] = V[0]+V[3]
                    V[:4] = 0.
                    Vs[nV] = V / np.linalg.norm(V)

            if self._SplitSolving:
                RandomValue = np.random.rand()
            else:
                RandomValue = -1

            if event.TrackerID in self.Determined2DPointsLocations.keys() or RandomValue < self._RatioCameraWarp:
                self.CameraSpaceWarp.AddData(Vs, DeltaVs, Stretch = min(self._DefaultCameraStretch, self.Point2DSpaceWarps[event.TrackerID].FirstLambdaRatio))
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
            self.CameraSpaceWarp.RetreiveData()
            if False and event.timestamp < 10:
                self.CameraSpaceWarp.Value[:4] = self._DefaultCameraVector[:4]

        if np.random.rand() < 0.01:
            self.Log("Point {0:3d} FLR : {1:.3f}, Pose FLR : {2:.3f}, Pose Trace : {3:.3f}".format(event.TrackerID, self.Point2DSpaceWarps[event.TrackerID].FirstLambdaRatio, self.CameraSpaceWarp.FirstLambdaRatio, self.CameraSpaceWarp.C.trace()))
        return event

        # The two following methods are for metric renormalization
    def _PointDistanceCorrection(self, X_i):
        return X_i * np.array([1., 1., 1. + 0 * ((self.AveragedDistance/self._DefaultUnknownDepthDistance)-1)]) # We multiply the homogeneous coodinate by the average distance, thus normalizing distances accordingly
    def _CameraDistanceCorrection(self, P):
        #P[:-2] /= self.AveragedDistance # We divide the translation component by the average distance metric
        return P

    def _Point2DNormalizationMethod(self, X_i):
        if X_i[-1] == 0:
            return X_i
        else:
            return X_i/X_i[-1]
    def _SuperDumbCameraPoseNormalizationMethod(self, P):
        N = np.sqrt((P[:4]**2).sum() / 2)
        if N:
            NormalizedPose = P / N
            return NormalizedPose*np.sign(NormalizedPose[0])
        else:
            return P
    def _DumbCameraPoseNormalizationMethod(self, P):
        N = np.linalg.det(P[:4].reshape((2,2)))
        if N != 0:
            return P / N**(1/2)
        else:
            return P
    def _CameraPoseNormalizationMethod(self, P):
        #if P[-1] != 0:
        #    return P[:-1] / P[-1]
        #else:
        #    return P[:-1]
        if P[-1] != 0:
            P = P[:-1]/P[-1]
        PreviousP = np.array(P)
        U, Sigmas, V = np.linalg.svd(P[:4].reshape((2,2)))
        R = U.dot(V)
        Sign = np.sign(np.linalg.det(R))
        P[:4] = R.reshape(4)
        #if ((Sigmas != 0).any()):
        #    P[4:] = P[4:] / Sigmas
        #P = P*Sign
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
        Vs[0] = np.array([self._Alpha * Xi[0], self._Alpha * Xi[1], -delta_x_i[0] * Xi[0], -delta_x_i[0] * Xi[1], self._Alpha * Xi[2], -delta_x_i[0] * Xi[2], 0.])
        for i in range(1):
            Vs[i] = Vs[i] / np.linalg.norm(Vs[i])
        return Vs

    def _SystemKnowledgeDecay(self, t, TrackerID = None):
        if self.CameraSpaceWarp.FirstLambdaRatio < self._CameraFLRLimitDecay:
            self.CameraSpaceWarp.Decay(self._CameraWarpDecayPerEvent)
        if not TrackerID is None:
            self.Point2DSpaceWarps[TrackerID].Decay(self._Point2DWarpDecayPerEvent)
        self.LastConsideredEventTs = t

    def Plot(self, fax = None):
        if fax is None:
            f, ax = plt.subplots(1,1)
        else:
            f, ax = fax
        P = self.CameraSpaceWarp.Value
        RT = np.array([[P[0], P[1], P[4]], [P[2], P[3], P[5]]])
        Rt = RT[:,:2].T
        X = -Rt.dot(RT[:,2])
        ax.plot(X[0], X[1], 'sg')
        ax.plot([X[0], (X+Rt[:,1])[0]], [X[1], (X+Rt[:,1])[1]], '--g')
        ax.plot((X-Rt[:,0])[0], (X-Rt[:,0])[1], 'sb')
        ax.plot([(X-Rt[:,0])[0], (X+Rt[:,0])[0]], [(X-Rt[:,0])[1], (X+Rt[:,0])[1]], '--b')

        for ID, PW in self.Point2DSpaceWarps.items():
            P = PW.Value
            ax.plot(P[0], P[1], 'ob')
            ax.text(P[0], P[1]+0.05, str(ID), color='b')
            if self.LastPointsReceived[ID][1] > self.LastTs - 0.02: #
                ax.plot([X[0], P[0]], [X[1], P[1]], 'b')
        return f, ax

class SpaceWarp:
    _AutoScale = False
    _RemoveOthogonalVectors = False
    _WARP_METHOD = {'left': 2, 'bilateral': 3, 'right': 1}
    def __init__(self, Dim, NormalizationMethod, DefaultStretch, InitialValue = None, InitialStretch = None, RetreiveMethod = 'lambdas', WarpMethod = 'right'):
        self.Dim = Dim
        self.DefaultStretch = DefaultStretch
        self._WarpMethod = self._WARP_METHOD[WarpMethod]
        
        self.RetreiveMethod = {'lambdas':0, 'pointwarp':1}[RetreiveMethod] # 0 for Lambdas, 1 for warp of previous point

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

        if not InitialValue is None:
            self._BuildCFromInitial(InitialValue, InitialStretch)
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
        self.S[0,0] = FinalStretches[0]
        self.S[1,1] = FinalStretches[1]

        self.M = self.T.T.dot(self.S.dot(self.T))

    def _BuildCFromInitial(self, InitialValue, InitialStretch):
        if InitialStretch is None:
            InitialStretch = self.DefaultStretch
        InitialBasis = np.array(_CompleteWarpBasisFrom([InitialValue / np.linalg.norm(InitialValue)]))
        InitialStretchMatrix = np.identity(self.Dim)
        for i in range(1, self.Dim):
            if type(InitialStretch) == list:
                InitialStretchMatrix[i,i] = InitialStretch[i]
            else:
                InitialStretchMatrix[i,i] = InitialStretch
        self.C = InitialBasis.T.dot(InitialStretchMatrix.dot(InitialBasis))
        self.Vectors = [InitialValue]
        self.RetreiveData()

    def RetreiveData(self):
        if self._UpToDate:
            return self.Value
        if self._WarpMethod == 3: # We force the use of symmetrical properties of the matrix
            Lambdas, Vectors = [np.real(data) for data in np.linalg.eigh(self.C)]
        else:
            Lambdas, Vectors = [np.real(data) for data in np.linalg.eig(self.C)]
        if self.RetreiveMethod == 0:
            if Lambdas.max() > 1.001:
                print("Excessive lambda, dimension is {0}".format(Vectors.shape[0]))
            Indexes = np.argsort(-Lambdas)
            
            if self._AutoScale:
                LambdaScaling = Lambdas.max()
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
                self.Vectors += [self._NormalizationMethod(Vectors[:,Index])]
        elif self.RetreiveMethod == 1:
            self.Lambdas = list(reversed(np.sort(Lambdas)))
            self.ProbaSum = [self.Lambdas[0]]
            for Lambda in self.Lambdas[1:]:
                self.ProbaSum += [self.ProbaSum[-1] + (Lambda > 0.9)*Lambda**2]
            self.Vectors = [self._NormalizationMethod(self.M.dot(self.Vectors[0]))]
            self.FirstLambdaRatio = self.Lambdas[1] / self.Lambdas[0]

        self.Value = self.Vectors[0]
        self._UpToDate = True
        return self.Value

class BundleAdjustmentPhysical(Module):
    _DISTANCE_NORMALIZATION_FACTOR = 1.

    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to handle ST-context memory.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Computation'
        
        self._ScreenCenter = [320., 0]
        self._ScreenRatio = [640., 640.]
        self._EventsConsideredRatio = 1.

        self._Point2DWarpMethod = 'bilateral'

        self._Point2DDefaultDistanceWarp = 1
        self._DefaultUnknownDepthDistance = 2.
        self._NPointsForDistanceDefinition = 10
        self.UsedPointsForDistanceDefinition = {}
        self.AveragedDistance = self._DefaultUnknownDepthDistance

        self._Default2DPointStretch = 0.5

        self._FullyDetermined2DPointLambdaRatio = 0.1

        self.LastConsideredEventTs = 0.
        self.Determined2DPointsLocations = {}

        self._MonitorDt = 0.01
        self._MonitoredVariables = [('Point2DSpaceWarps@Value', np.array),
                                    ('Point2DSpaceWarps@C', np.array),
                                    ('Point2DSpaceWarps@NWarps', int)]

    def _InitializeModule(self, **kwargs):
        self.Point2DSpaceWarps = {}
        self.LastScreenLocation = {}
        self.Camera = PhysicalCameraClass()

        self._ScreenCenter = np.array(self._ScreenCenter)
        self._ScreenRatio = np.array(self._ScreenRatio)
        self._Alpha = self._ScreenRatio[0]

        if self._EventsConsideredRatio < 1:
            self._RandomizeEvents = True
        else:
            self._RandomizeEvents = False
        return True

    def _OnEventModule(self, event):
        if event.__class__ != TrackerEvent:
            return event
        if self._RandomizeEvents and np.random.rand() > self._EventsConsideredRatio:
            return event
        
        if event.TrackerID not in self.Point2DSpaceWarps:
            self.Log("Creating space warp for tracker {0}".format(event.TrackerID))
            self.Point2DSpaceWarps[event.TrackerID] = SpaceWarp(3, lambda X_i: self._Point2DNormalizationMethod(X_i), self._Default2DPointStretch, WarpMethod = self._Point2DWarpMethod)

            Vs = self._CreateDefaultDistanceOrthogonalVector()
            self.Point2DSpaceWarps[event.TrackerID].AddData(Vs, Stretch = self._Point2DDefaultDistanceWarp)

            Vs = self._Generate2DPointWarpVectors(event.TrackerLocation)
            self.Point2DSpaceWarps[event.TrackerID].AddData(Vs, Stretch = 0.9)
            self.Point2DSpaceWarps[event.TrackerID].RetreiveData()
            #Xi = self._Create2DPointAtDefaultDistance(event.TrackerID)

            #InitialWarp = np.array(self.Point2DSpaceWarps[event.TrackerID].C)
            #self.Point2DSpaceWarps[event.TrackerID]._BuildCFromInitial(Xi, self._Point2DDefaultDistanceWarp)
            #self.Point2DSpaceWarps[event.TrackerID].C = self.Point2DSpaceWarps[event.TrackerID].C.dot(InitialWarp)
            #self.Point2DSpaceWarps[event.TrackerID]._UpToDate = False

            if len(self.UsedPointsForDistanceDefinition) < self._NPointsForDistanceDefinition:

                self.UsedPointsForDistanceDefinition[event.TrackerID] = np.linalg.norm(self.Point2DSpaceWarps[event.TrackerID].Value[:-1])
                self.AveragedDistance = np.mean(list(self.UsedPointsForDistanceDefinition.values()))
            self.Camera.OnTrackerEvent(event.timestamp) 
            return event

        self.Camera.OnTrackerEvent(event.timestamp, event.TrackerID, self.Point2DSpaceWarps[event.TrackerID].Value[:-1] / self.Point2DSpaceWarps[event.TrackerID].Value[-1], -(event.TrackerLocation[0] - self._ScreenCenter[0]) / (self._ScreenRatio[0] / 2))
        Vs = self._Generate2DPointWarpVectors(event.TrackerLocation)
        self.Point2DSpaceWarps[event.TrackerID].AddData(Vs, Stretch = self._Default2DPointStretch)
        self.Point2DSpaceWarps[event.TrackerID].RetreiveData()

        if self.Point2DSpaceWarps[event.TrackerID].FirstLambdaRatio > self._FullyDetermined2DPointLambdaRatio:
            if event.TrackerID in self.UsedPointsForDistanceDefinition.keys():
                self.UsedPointsForDistanceDefinition[event.TrackerID] = np.linalg.norm(self.Point2DSpaceWarps[event.TrackerID].Value[:-1])
                self.AveragedDistance = np.mean(list(self.UsedPointsForDistanceDefinition.values()))
        elif event.TrackerID not in self.Determined2DPointsLocations.keys():
            self.Determined2DPointsLocations[event.TrackerID] = np.array(self.Point2DSpaceWarps[event.TrackerID].Vectors[0])
            self.Log("Added {0} to known points at {1}".format(event.TrackerID, self.Determined2DPointsLocations[event.TrackerID]), 3)

        if np.random.rand() < 0.01:
            self.Log("Point {0:3d} FLR : {1:.3f}, SLR : {2:.3f}".format(event.TrackerID, self.Point2DSpaceWarps[event.TrackerID].FirstLambdaRatio, self.Point2DSpaceWarps[event.TrackerID].Lambdas[2] / self.Point2DSpaceWarps[event.TrackerID].Lambdas[0]))
            self.Log("CL : ({0:3f}, {1:.3f}), {2:.3f}, E : {3:.3f}".format(self.Camera.Location[0], self.Camera.Location[1], self.Camera.Angle, self.Camera.kEnergy()))
        return event

        # The two following methods are for metric renormalization
    def _PointDistanceCorrection(self, X_i):
        return X_i * np.array([1., 1., (self.AveragedDistance/self._DefaultUnknownDepthDistance)]) # We multiply the homogeneous coodinate by the average distance, thus normalizing distances accordingly
    def _CameraDistanceCorrection(self, P):
        #P[:-2] /= self.AveragedDistance # We divide the translation component by the average distance metric
        return P

    def _Point2DNormalizationMethod(self, X_i):
        if X_i[-1] == 0:
            return X_i
        else:
            return X_i/X_i[-1]

    def _Generate2DPointWarpVectors(self, TrackerLocation):
        c, s = np.sin(self.Camera.Angle), np.cos(self.Camera.Angle)
        P = np.array([c, s, -s, c, self.Camera.Location[0], self.Camera.Location[1]])

        Vs = [None]
        delta_x_i = (TrackerLocation - self._ScreenCenter)
        Vs[0] = np.array([self._Alpha * P[0] - delta_x_i[0] * P[2], self._Alpha * P[1] - delta_x_i[0] * P[3], self._Alpha * P[4] - delta_x_i[0] * P[5]])
        for i in range(1):
            Vs[i] = Vs[i] / np.linalg.norm(Vs[i])
        return Vs

    def _CreateDefaultDistanceOrthogonalVector(self):
        xv, yv = self.Camera.GetUvUx()[0]
        x, y = self.Camera.Location
        V = np.array([xv, yv, -(self._DefaultUnknownDepthDistance + x*xv + y*yv)])
        return [V/np.linalg.norm(V)]


    def _Create2DPointAtDefaultDistance(self, PointID):
        V1, V2 = self.Point2DSpaceWarps[PointID].Vectors[:2]
        T = self.Camera.Location
        if V1[-1] != 0:
            if V2[-1] != 0:
                D1 = V1 / V1[-1]
                D2 = V2 / V2[-1]
                O, Delta = V1 / V1[-1], (D1 - D2)/np.linalg.norm(D1 - D2)
            else:
                O, Delta = V1 / V1[-1], V2 / np.linalg.norm(V2)
        else:
            if V2[-1] != 0:
                O, Delta = V2 / V2[-1], V1 / np.linalg.norm(V1)
            else:
                raise Exception("Both vectors describe non finite points")
        k = (Delta[:-1] * (O[:-1] - T)).sum()

        Xi = O + (k + np.sign(Delta[1])*self._DefaultUnknownDepthDistance)*Delta

        return Xi

class PhysicalCameraClass:
    _M = 0.01
    _J = 0.01
    _Mu = 0.01
    _JMu = 0.01
    _k = 10.
    _FPD = 0.1 # FocalPlanDistance
    _Aperture = 26.56 * np.pi / 180

    _SideLocationTolerance = 0.95

    def __init__(self):
        self.Location = np.array([0., 0.])
        self.Angle = 0.

        self.TSpeed = np.array([0., 0.])
        self.RSpeed = 0.

        self.LastTs = 0.

        self.kForces = {}

        self._tAperture = np.tan(self._Aperture)

    def _Update(self, t):
        DeltaT = t - self.LastTs
        self.LastTs = t
        if self.kForces:
            Forces = np.array(list(self.kForces.values()))
            Acceleration = Forces[:,:2].sum(axis = 0)
            Torque = (Forces[:,2]*Forces[:,1] - self.Location[1]*Forces[:,0]).sum()
        else:
            Acceleration = 0.
            Torque = 0.

        self.TSpeed += DeltaT * (self._k * Acceleration - self._Mu * self.TSpeed) / self._M
        self.RSpeed += DeltaT * (self._k * Torque - self._JMu * self.RSpeed * abs(self.RSpeed)) / self._J

        self.Location += DeltaT * self.TSpeed
        self.Angle += DeltaT * self.RSpeed

    def GetForceAndApplicationPoint(self, Point2DLocation, RelativeScreenLocation):
        Uv, Ux = self.GetUvUx()

        ApplicationPoint = np.array([self._FPD, RelativeScreenLocation * self._tAperture * self._FPD]) # In (Uv, Ux)
        PointVector = Point2DLocation - self.Location
        PointVector /= np.linalg.norm(PointVector)
        RelativePointVector = np.array([(PointVector*Uv).sum(), (PointVector*Ux).sum()])
        RelativePointVector /= np.linalg.norm(RelativePointVector)

        ForceValue = ApplicationPoint[0]*RelativePointVector[1] - ApplicationPoint[1]*RelativePointVector[0]
        ForceVector = ForceValue / np.linalg.norm(ApplicationPoint) * np.array([-ApplicationPoint[1], ApplicationPoint[0]])

        return ForceVector, ApplicationPoint


    def GetUvUx(self):
        c, s = np.cos(self.Angle), np.sin(self.Angle)
        return np.array([c, s]), np.array([-s, c])

    def AddPoint(self, ID):
        self.kForces[ID] = np.zeros(4)

    def OnTrackerEvent(self, t, ID = None, Point2DLocation = None, RelativeScreenLocation = None):
        if not ID is None:
            if ID not in self.kForces.keys():
                self.AddPoint(ID)
            if abs(RelativeScreenLocation) > self._SideLocationTolerance:
                del self.kForces[ID]
                self._Update(t)
                return
            Force, ApplicationPoint = self.GetForceAndApplicationPoint(Point2DLocation, RelativeScreenLocation)
            self.kForces[ID][:2] = Force
            self.kForces[ID][2:] = ApplicationPoint

        self._Update(t)

    def PlotForces(self):
        f, ax = plt.subplots(1,1)
        ax.plot(0,0, 'ok')
        ax.plot([self._FPD, self._FPD], [-self._tAperture * self._FPD, self._tAperture * self._FPD], 'k')
        FRatio = 0.5*self._FPD / np.linalg.norm(np.array(list(self.kForces.values()))[:,:2], axis = 1).max()
        for ID, kForce in self.kForces.items():
            ax.plot(kForce[2], kForce[3], 'or')
            ax.plot([kForce[2], kForce[2]+FRatio*kForce[0]], [kForce[3], kForce[3]+FRatio*kForce[1]], 'r')
            ax.text(kForce[2]+FRatio*kForce[0], kForce[3]+FRatio*kForce[1], str(ID), color = 'r')
    def kEnergy(self):
        return ((np.array(list(self.kForces.values())))[:,:2]**2).sum()

class BundleAdjustment(BundleAdjustmentWarp):
    pass

