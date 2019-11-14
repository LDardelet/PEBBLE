import numpy as np
from scipy import misc 

class GTMakerClass:
    def __init__(self, SpeedTracker, LookupRadius = 2):
        self.SpeedTracker = SpeedTracker
        self.WindowSize = int(SpeedTracker._DetectorDefaultWindowsLength * 1.5)
        self.WindowSize -= self.WindowSize%2
        self.LookupRadius = LookupRadius

        self.DataFolder = None
        self.ImagesLookupTable = None
        self.ImagesRefDict = {}
        self.GTPositions = {}
        self.CurrentDataIndex = None

        self.SearchVectors = None

        self.DistMatch = 9e-2

    def CreateSearchVectors(self):
        xs = np.arange(0, 2*self.LookupRadius + 1) - self.LookupRadius
        ys = np.arange(0, 2*self.LookupRadius + 1) - self.LookupRadius
        Vectors = []
        for x in xs:
            for y in ys:
                Vectors += [np.array([x,y])]
        Norms = np.linalg.norm(np.array(Vectors), axis = 1)
        self.SearchVectors = []
        for i in np.argsort(Norms):
            self.SearchVectors += [Vectors[i]]

    def _OpenLookupTableFile(self, filename):
        self.ImagesLookupTable = {'t': [], 'filename': []}
        with open(filename, 'rb') as LookupTableFile:
            for line in LookupTableFile.readlines():
                t, ImageName = line.strip().split(' ')
                self.ImagesLookupTable['t'] += [float(t)]
                self.ImagesLookupTable['filename'] += [ImageName]
        print "Lookup table loaded from file"

    def _CreateLookupTableSnaps(self):
        self.ImagesLookupTable = {'t': [], 'filename': []}
        for nt, Duo in enumerate(self.SpeedTracker._LinkedMemory.Snapshots):
            self.ImagesLookupTable['t'] += [Duo[0]]
            self.ImagesLookupTable['filename'] += [nt]
        print "Lookup table created from snapshots"

    def GetTrackerPositionAt(self, t, TrackerID):
        self.GetCurrentFrameIndex(t)
        if not TrackerID in self.GTPositions[self.ImagesLookupTable['filename'][self.CurrentFrameIndex]].keys():
            return None
        if self.CurrentFrameIndex + 1 >= len(self.ImagesLookupTable['filename']) or not self.ImagesLookupTable['filename'][self.CurrentFrameIndex + 1] in self.GTPositions.keys() or not TrackerID in self.GTPositions[self.ImagesLookupTable['filename'][self.CurrentFrameIndex + 1]].keys():
            return self.GTPositions[self.ImagesLookupTable['filename'][self.CurrentFrameIndex]][TrackerID]
        t1 = self.ImagesLookupTable['t'][self.CurrentFrameIndex]
        t2 = self.ImagesLookupTable['t'][self.CurrentFrameIndex+1]
        p1 = self.GTPositions[self.ImagesLookupTable['filename'][self.CurrentFrameIndex]][TrackerID]
        p2 = self.GTPositions[self.ImagesLookupTable['filename'][self.CurrentFrameIndex+1]][TrackerID]
        return p1 + (p2-p1) * (t-t1) / (t2-t1)

    def GetTrackerPositions(self, TrackerID):
        ts = []
        Xs = []
        for t, ImageFile in zip(self.ImagesLookupTable['t'], self.ImagesLookupTable['filename']):
            if ImageFile in self.GTPositions.keys():
                if TrackerID in self.GTPositions[ImageFile].keys():
                    ts += [t]
                    Xs += [self.GTPositions[ImageFile][TrackerID]]
        return ts, Xs

    def _FindMatch(self, TrackerID, NewPosition):
        RefFeature = self.ImagesRefDict[TrackerID]
        BestMatchDist = 1
        BestMatchPosition = None
        if (NewPosition < self.WindowSize/2).any() or (NewPosition > np.array(self.CurrentImage.shape) + self.WindowSize/2).any():
            return None

        for Vector in self.SearchVectors:
            CurrentPosition = np.array(np.rint(NewPosition) + Vector, dtype = int)
            CurrentFeature = np.array(self.CurrentImage[CurrentPosition[0] - self.WindowSize/2:CurrentPosition[0] + self.WindowSize/2, CurrentPosition[1] - self.WindowSize/2:CurrentPosition[1] + self.WindowSize/2], dtype = float)
            if CurrentFeature.shape != (self.WindowSize, self.WindowSize):
                continue
            CurrentFeature = CurrentFeature / CurrentFeature.max()
            #d = ((CurrentFeature - RefFeature)**2).sum() / (self.WindowSize**2)
            d = ((CurrentFeature - RefFeature)**2).sum() / ((CurrentFeature**2).sum() * (RefFeature**2).sum())
            if d < BestMatchDist:
                BestMatchDist = d
                BestMatchPosition = np.array(CurrentPosition)

        self.MatchingDistance += BestMatchDist
        if BestMatchDist < self.DistMatch:
            self.MatchSuccess += 1
            return BestMatchPosition
        else:
            self.MatchFailures += 1
            return None

    def AddActiveTracker(self, TrackerID, Position):
        CurrentPosition = np.array(np.rint(Position), dtype = int)
        CurrentFeature = np.array(self.CurrentImage[CurrentPosition[0] - self.WindowSize/2:CurrentPosition[0] + self.WindowSize/2, CurrentPosition[1] - self.WindowSize/2:CurrentPosition[1] + self.WindowSize/2], dtype = float)
        if CurrentFeature.shape != (self.WindowSize, self.WindowSize):
            return False
        CurrentFeature = CurrentFeature / CurrentFeature.max()
        self.ImagesRefDict[TrackerID] = CurrentFeature
        return True

    def GetCurrentDataIndex(self, t):
        self.CurrentDataIndex = abs(np.array(self.SpeedTracker.TrackersPositionsHistoryTs) - t).argmin()

    def GetCurrentFrameIndex(self, t):
        self.CurrentFrameIndex = abs(np.array(self.ImagesLookupTable['t']) - t).argmin()

    def LoadFileImage(self, filename):
        self.CurrentImage = np.flip(np.transpose(misc.imread(self.DataFolder + filename)), axis = 1)

    def LoadSTImage(self, ID, BinDt = 0.05):
        Map = self.SpeedTracker._LinkedMemory.Snapshots[ID][1]
        Map = Map.max(axis = 2)
        N = np.linalg.norm(self.SpeedTracker.FeatureManager.SpeedTranslationHistory[ID])
        if N > 0:
            BinDt = min(BinDt, 3. / N)
        t = self.SpeedTracker._LinkedMemory.Snapshots[ID][0]
        self.CurrentImage = np.e**((Map - t)/BinDt) * 255.
        #self.CurrentImage = np.array(Map > t - BinDt, dtype = float) * 255

    def ComputeErrorToGT(self, TrackerID, LimitError = np.inf, tMax = np.inf):
        tst, Xst = self.GetTrackerPositions(TrackerID)
        ts, Xs = self.SpeedTracker.GetTrackerPositions(TrackerID)

        CommonIndexes = [(i, ts.index(tst[i])) for i in range(len(tst)) if tst[i] in ts and tst[i] <= tMax]
        if CommonIndexes:
            Diffs = []
            for it, i in CommonIndexes:
                Err = np.linalg.norm(Xs[i] - Xst[it])
                if Err <= LimitError:
                    Diffs += [Err]
                else:
                    break
            return Diffs, [tst[it] for it, i in CommonIndexes]
        else:
            return [], []

    def GenerateGroundTruth(self, DataFolder = None, tMax = None, BinDt = 0.1):
        self.CreateSearchVectors()
        self.ImagesRefDict = {}
        self.GTPositions = {}
        self.CurrentDataIndex = None
        self.CurrentImage = None
        self.MinActiveTrackerID = 0
        self.UsedVectors = []
        self.Deltas = {}
        self.MatchingDistance = 0

        self.MatchSuccess = 0
        self.MatchFailures = 0

        if tMax is None:
            tMax = self.SpeedTracker._LinkedMemory.LastEvent.timestamp

        if not DataFolder is None:
            if DataFolder[-1] != '/':
                DataFolder = DataFolder + '/'
            self.DataFolder = DataFolder
            self._OpenLookupTableFile(self.DataFolder + 'images.txt')
        else:
            self.DataFolder = ''
            self._CreateLookupTableSnaps()

        PreviousFrame = None
        for t, ImageFile in zip(self.ImagesLookupTable['t'], self.ImagesLookupTable['filename']):
            if t > tMax:
                return None
            self.GTPositions[ImageFile] = {}
            if self.DataFolder:
                self.LoadFileImage(ImageFile)
            else:
                self.LoadSTImage(ImageFile)
            self.GetCurrentDataIndex(t)

            FoundAlive = False
            MinActiveOffset = 0
            for LocalID, StatusTuple in enumerate(self.SpeedTracker.TrackersStatuses[self.CurrentDataIndex][self.MinActiveTrackerID:]):
                if not FoundAlive and StatusTuple[0] == self.SpeedTracker._STATUS_DEAD:
                    MinActiveOffset += 1
                    continue
                elif StatusTuple[0] != self.SpeedTracker._STATUS_DEAD:
                    FoundAlive = True
                TrackerID = LocalID + self.MinActiveTrackerID
                if StatusTuple[1] == self.SpeedTracker._PROPERTY_LOCKED or TrackerID in self.ImagesRefDict.keys():
                    if TrackerID in self.ImagesRefDict.keys():
                        NewPosition = self._FindMatch(TrackerID, self.GTPositions[PreviousFrame][TrackerID] + self.Deltas[TrackerID])
                        if NewPosition is None:
                            del self.ImagesRefDict[TrackerID]
                        else:
                            self.Deltas[TrackerID] = NewPosition - self.GTPositions[PreviousFrame][TrackerID]
                            self.GTPositions[ImageFile][TrackerID] = np.array(NewPosition)
                    else:
                        if TrackerID not in self.UsedVectors:
                            if self.AddActiveTracker(TrackerID, self.SpeedTracker.TrackersPositionsHistory[self.CurrentDataIndex][TrackerID]):
                                self.GTPositions[ImageFile][TrackerID] = np.rint(self.SpeedTracker.TrackersPositionsHistory[self.CurrentDataIndex][TrackerID])
                                self.UsedVectors += [TrackerID] # Avoids reinitalize a tracker that wasn't found at one frame
                                self.Deltas[TrackerID] = np.array([0, 0])
                        else:
                            None
                else:
                    if TrackerID in self.ImagesRefDict.keys():
                        del self.ImagesRefDict[TrackerID]
            self.MinActiveTrackerID += MinActiveOffset
            PreviousFrame = ImageFile

