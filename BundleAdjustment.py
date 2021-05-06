import numpy as np

from PEBBLE import Module, Event

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

class BundleAdjustment(Module):
    _DISTANCE_NORMALIZATION_FACTOR = 1.

    def __init__(self, Name, Framework, ModulesLinked):
        '''
        '''
        Module.__init__(self, Name, Framework, ModulesLinked)
        
        self._ScreenCenter = [320., 240.]
        self._ScreenRatio = [640., 640.]
        self._EventsConsideredRatio = 1.

        self._CameraWarpMethod = 'right'
        self._Point3DWarpMethod = 'right'

        self._PartiallyDetermined3DPointLambdaRatio = 1.
        self._FullyDetermined3DPointLambdaRatio = 0.1

        self._DefaultCameraStretch = 0.5
        self._Default3DPointStretch = 0.90
        self._DefaultCameraVector = np.array([1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.])
        self._Point3DDefaultDistanceWarp = 0.999
        self._Point3DForcedLocationWarp = 0.05

        self._DefaultUnknownDepthDistance = 1.
        self._TryAlternativePoseFLRMinValue = 1.
        self._UnknownPoseLimit = 0.99

        self._CameraWarpDecayPerEvent = 0.0001 / self._EventsConsideredRatio
        self._Point3DWarpDecayPerEvent = 0 * self._CameraWarpDecayPerEvent/100
        self.LastConsideredEventTs = 0.

        self.Determined3DPointsLocations = {}

        self._MonitorDt = 0.01
        self._MonitoredVariables = [('CameraSpaceWarp@Value', np.array),
                                    ('Point3DSpaceWarps@Value', np.array),
                                    ('Point3DSpaceWarps@NWarps', int),
                                    'CameraSpaceWarp@FirstLambdaRatio']

    def _InitializeModule(self):
        self.Point3DSpaceWarps = {}
        self.CameraSpaceWarp = SpaceWarp(12, self._CameraPoseNormalizationMethod, self._DefaultCameraStretch, self._DefaultCameraVector, 0.5, RetreiveMethod = 'lambdas', WarpMethod = self._CameraWarpMethod)
        # P = (r_11, r_12, r_13, r_21, r_22, r_23, r_31, r_32, r_33, t_x, t_y, t_z)

        self._ScreenCenter = np.array(self._ScreenCenter)
        self._ScreenRatio = np.array(self._ScreenRatio)
        self._Alpha = self._ScreenRatio[0]
        self._Beta = self._ScreenRatio[1]

        self.InitialPointLocated = False

        if self._EventsConsideredRatio < 1:
            self._RandomizeEvents = True
        else:
            self._RandomizeEvents = False
        return True

    def _OnEventModule(self, event):
        if event.Has(TrackerEvent):
            return
        if self._RandomizeEvents and np.random.rand() > self._EventsConsideredRatio:
            return
        
        self.OnTrackerEvent(event)

    def OnTrackerEvent(self, event):
        PreviousPose = np.array(self.CameraSpaceWarp.Vectors[0])

        if event.TrackerID not in self.Point3DSpaceWarps:
            self._SystemKnowledgeDecay(event.timestamp)
            self.Log("Creating space warp for tracker {0}".format(event.TrackerID))
            self.Point3DSpaceWarps[event.TrackerID] = SpaceWarp(4, self._Point3DNormalizationMethod, self._Default3DPointStretch, WarpMethod = self._Point3DWarpMethod)

            Vs = self._Generate3DPointWarpVectors(event.TrackerLocation)
            self.Point3DSpaceWarps[event.TrackerID].AddData(Vs, Stretch = self.CameraSpaceWarp.FirstLambdaRatio)
            self.Point3DSpaceWarps[event.TrackerID].RetreiveData()
            Xi = self._Create3DPointAtDefaultDistance(event.TrackerID)

            InitialWarp = np.array(self.Point3DSpaceWarps[event.TrackerID].C)
            if not self.InitialPointLocated and (np.abs(event.TrackerLocation - self._ScreenCenter) / (self._ScreenRatio / 2)).max() < 0.3: # Here we force the system's distance. For that, we use one of the point closer to the screen center and set its location.
                self.Point3DSpaceWarps[event.TrackerID]._BuildCFromInitial(Xi, self._Point3DForcedLocationWarp)
                self.InitialPointLocated = True
            else:
                self.Point3DSpaceWarps[event.TrackerID]._BuildCFromInitial(Xi, self._Point3DDefaultDistanceWarp)
            self.Point3DSpaceWarps[event.TrackerID].C = self.Point3DSpaceWarps[event.TrackerID].C.dot(InitialWarp)

        else:
            self._SystemKnowledgeDecay(event.timestamp, event.TrackerID)
                
            Vs, DeltaVs = self._GenerateCameraWarpVectors(event.TrackerLocation, event.TrackerID)
            #self.CameraSpaceWarp.AddData(Vs, DeltaVs, Certainty = 0.2 + 0.8*(1 - self.Point3DSpaceWarps[event.TrackerID].FirstLambdaRatio**2))
            self.CameraSpaceWarp.AddData(Vs, DeltaVs, Stretch = min(self._DefaultCameraStretch, self.Point3DSpaceWarps[event.TrackerID].FirstLambdaRatio))
            if self.CameraSpaceWarp.FirstLambdaRatio < self._UnknownPoseLimit: # Meaning we know enough about the pose of the camera to correct the 3D points locations
                if self.Point3DSpaceWarps[event.TrackerID].FirstLambdaRatio > self._FullyDetermined3DPointLambdaRatio: # Make location of 3D point more precise
                    Vs = self._Generate3DPointWarpVectors(event.TrackerLocation)
                    self.Point3DSpaceWarps[event.TrackerID].AddData(Vs)
                    self.Point3DSpaceWarps[event.TrackerID].RetreiveData()
                elif event.TrackerID not in self.Determined3DPointsLocations.keys():
                    self.Determined3DPointsLocations[event.TrackerID] = np.array(self.Point3DSpaceWarps[event.TrackerID].Vectors[0])
                    self.Log("Added {0} to known points at {1}".format(event.TrackerID, self.Determined3DPointsLocations[event.TrackerID]), 3)
            self.CameraSpaceWarp.RetreiveData()

        if np.random.rand() < 0.01:
            self.Log("Point {0:3d} FLR : {1:.3f}, Pose FLR : {2:.3f}, Pose Trace : {3:.3f}".format(event.TrackerID, self.Point3DSpaceWarps[event.TrackerID].FirstLambdaRatio, self.CameraSpaceWarp.FirstLambdaRatio, self.CameraSpaceWarp.C.trace()))

    def _Point3DNormalizationMethod(self, X_i):
        if X_i[-1] == 0:
            return X_i
        else:
            return X_i/X_i[-1]
    def _SuperDumbCameraPoseNormalizationMethod(self, P):
        N = np.sqrt((P[:9]**2).sum() / 3)
        if N:
            NormalizedPose = P / N
            return NormalizedPose * np.sign(NormalizedPose[0] + NormalizedPose[3] + NormalizedPose[6])
        else:
            return P
    def _DumbCameraPoseNormalizationMethod(self, P):
        N = np.linalg.det(P[:9].reshape((3,3)))
        if N != 0:
            return P / N**(1/3)
        else:
            return P
    def _CameraPoseNormalizationMethod(self, P):
        U, Sigmas, V = np.linalg.svd(P[:9].reshape((3,3)))
        R = U.dot(V)
        Sign = np.sign(np.linalg.det(R))
        P[:9] = R.reshape(9)
        if ((Sigmas != 0).all()):
            P[9:] = P[9:] / Sigmas
        P = P*Sign
        return P

    def _Generate3DPointWarpVectors(self, TrackerLocation):
        
        if self.CameraSpaceWarp.FirstLambdaRatio < self._TryAlternativePoseFLRMinValue:
            P = self.CameraSpaceWarp.Vectors[0]
        else:
            RandomValue = self.CameraSpaceWarp.ProbaSum[-1] * np.random.rand()
            for nVectorUsed, ProbaSum in enumerate(self.CameraSpaceWarp.ProbaSum):
                if RandomValue <= ProbaSum:
                    break
            P = self.CameraSpaceWarp.Vectors[nVectorUsed]
        Vs = [None, None]
        delta_x_i = TrackerLocation - self._ScreenCenter
        Vs[0] = np.array([self._Alpha * P[0] - delta_x_i[0] * P[6], self._Alpha * P[1] - delta_x_i[0] * P[7], self._Alpha * P[2] - delta_x_i[0] * P[8], self._Alpha * P[9] - delta_x_i[0] * P[11]])
        Vs[1] = np.array([self._Beta  * P[3] - delta_x_i[1] * P[6], self._Beta  * P[4] - delta_x_i[1] * P[7], self._Beta  * P[5] - delta_x_i[1] * P[8], self._Beta  * P[10] - delta_x_i[1] * P[11]])
        for i in range(2):
            Vs[i] = Vs[i] / np.linalg.norm(Vs[i])
        return Vs

    def _Create3DPointAtDefaultDistance(self, PointID):
        V1, V2 = self.Point3DSpaceWarps[PointID].Vectors[:2]
        T = self.CameraSpaceWarp.RetreiveData()[-3:]
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

        Xi = O + (k + np.sign(Delta[2])*self._DefaultUnknownDepthDistance)*Delta

        return Xi

    def _GenerateCameraWarpVectors(self, TrackerLocation, TrackerID):
        PointWarp = self.Point3DSpaceWarps[TrackerID]
        PointWarp.RetreiveData()

        delta_x_i = TrackerLocation - self._ScreenCenter

        if PointWarp.FirstLambdaRatio > self._PartiallyDetermined3DPointLambdaRatio: # Undetermined depth
            Xi = self._Create3DPointAtDefaultDistance(TrackerID)
        else:
            Xi = PointWarp.Vectors[0]
        Vs = self._GenerateNormalCameraWarpVectors(Xi, delta_x_i)
        Delta_Vs = []
        return Vs, Delta_Vs

    def _GenerateNormalCameraWarpVectors(self, Xi, delta_x_i):
        Vs = [None, None]
        Vs[0] = np.array([self._Alpha * Xi[0], self._Alpha * Xi[1], self._Alpha * Xi[2], 0, 0, 0, -delta_x_i[0] * Xi[0],  -delta_x_i[0] * Xi[1], -delta_x_i[0] * Xi[2], self._Alpha, 0, -delta_x_i[0]])
        Vs[1] = np.array([0, 0, 0, self._Beta  * Xi[0], self._Beta  * Xi[1], self._Beta  * Xi[2], -delta_x_i[1] * Xi[0],  -delta_x_i[1] * Xi[1], -delta_x_i[1] * Xi[2], 0, self._Beta , -delta_x_i[1]])
        for i in range(2):
            Vs[i] = Vs[i] / np.linalg.norm(Vs[i])
        return Vs
    def _GenerateForbiddenWarpVectors(self, Delta_i, delta_x_i): # Unused for now as we used assumed depth instead of unidimensional warp
        Delta_Vs = [None, None]
        Delta_Vs[0] = np.array([self._Alpha * Delta_i[0], self._Alpha * Delta_i[1], self._Alpha * Delta_i[2], 0, 0, 0, -delta_x_i[0] * Delta_i[0],  -delta_x_i[0] * Delta_i[1], -delta_x_i[0] * Delta_i[2], 0, 0, 0])
        Delta_Vs[1] = np.array([0, 0, 0, self._Beta  * Delta_i[0], self._Beta  * Delta_i[1], self._Beta  * Delta_i[2], -delta_x_i[1] * Delta_i[0],  -delta_x_i[1] * Delta_i[1], -delta_x_i[1] * Delta_i[2], 0, 0, 0])
        for i in range(2):
            Delta_Vs[i] = Delta_Vs[i] / np.linalg.norm(Delta_Vs[i])
        return Delta_Vs

    def _SystemKnowledgeDecay(self, t, TrackerID = None):
        self.CameraSpaceWarp.Decay(self._CameraWarpDecayPerEvent)
        if not TrackerID is None:
            self.Point3DSpaceWarps[TrackerID].Decay(self._Point3DWarpDecayPerEvent)
        self.LastConsideredEventTs = t

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
            return self.Vectors[0]
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
