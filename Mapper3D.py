from Framework import Module, DisparityEvent, CameraPoseEvent

class Mapper3D(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Module that takes disparity events and CameraPoseEvents to compute a dense 3D map of the environment
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Computation'

        self.__ReferencesAsked__ = []
        self._MonitorDt = 0. # By default, a module does not stode any date over time.
        self._NeedsLogColumn = True
        self._MonitoredVariables = []

    def _InitializeModule(self, **kwargs):
        self.KeyPoints = []
        self.MapsTriangles = []
        self.BorderPoints = set()
        return True

    def AddKeyPoint(self, X, VisionLine):
        PointID = len(self.KeyPoints)
        self.KeyPoints += [KeyPointClass(PointID, X)]
        if len(self.MapsTriangles) == 0:
            self.BorderPoints.add(PointID)
            if len(self.KeyPoints) == 3:
                ConnectedPointsIDs = (0,1,2)
                self.MapsTriangles += [TriangleClass(ConnectedPointsIDs, self.TriangleMeanLocation(ConnectedPointsIDs))]
                for KeyPoint in self.KeyPoints:
                    KeyPoint.ConnectToIDs(ConnectedPointsIDs)
            return

        # Adds a 3D points X to the set of keypoints, and creates

    def _OnEventModule(self, event):
        return event

    def TriangleMeanLocation(self, ConnectedPointsIDs):
        return np.mean([self.KeyPoints[nPoint].X for nPoint in ConnectedPointsIDs], axis = 0)

class KeyPointClass:
    def __init__(self, ID, X):
        self.ID = ID
        self.X = np.array(X)
        self.Connexions = set()

    def ConnectToID(self, ID):
        if ID == self.ID:
            return
        self.Connexions.add(ID)

    def ConnectToIDs(self, IDs):
        self.Connexions = set([ID for ID in IDs if ID != self.ID])

    @property
    def IsBorder(self):
        return len(self.Connexions) < 3

class TriangleClass:
    def __init__(self, CornersIDs, X):
        self.CornersIDs = CornersIDs
        self.X = X
