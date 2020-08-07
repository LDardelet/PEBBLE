from pycpd import rigid_registration as registration
import matplotlib.pyplot as plt

import numpy as np
import datetime
import os
import inspect
import pickle
from sys import stdout
from Framework import Module, TrackerEvent

import imageio

import pathos.multiprocessing as mp
from functools import partial

class FeatureManagerClass:
    def __init__(self, TrackerManager):
        self.TrackerManager = TrackerManager
        self.CenterPoint = np.array(self.TrackerManager._LinkedMemory.STContext.shape[:2], dtype = float) / 2
        self.SpeedsLockVectorsDict = {}
        self.PositionsLockVectorsDict = {}

        self.SpeedTranslationValue = np.array([0., 0.])
        self.SpeedScalingValue = 0.
        self.SpeedRotationValue = 0.

        self.TranslationValue = np.array([0., 0.])
        self.RotationValue = 0.
        self.ScalingValue = 0.

        self.TranslationHistory = []
        self.RotationHistory = []
        self.ScalingHistory = []
        self.SpeedTranslationHistory = []
        self.SpeedRotationHistory = []
        self.SpeedScalingHistory = []

        self.LastSnap = None

        self.Features = []
        self.AssociationDistance = 0.5

    def AddLock(self, Tracker):
        self.SpeedsLockVectorsDict[Tracker.ID] = Tracker.Speed
        self.PositionsLockVectorsDict[Tracker.ID] = Tracker.Position

    def RemoveLock(self, Tracker):
        del self.SpeedsLockVectorsDict[Tracker.ID]
        del self.PositionsLockVectorsDict[Tracker.ID]

    def FindAssociatedFeatureWith(self, Tracker):
        self._ComputePoseInformation()
        MinDistance = np.inf
        RelativeDistance = Tracker.Position - self.CenterPoint
        CurrentPoints = Tracker.Lock.Events[:,1:]
        CurrentPoints = CurrentPoints - CurrentPoints.mean(axis = 0)
        for Feature in self.Features:
            if (abs(Feature.CenterRelativePosition - RelativeDistance)).max() <= self.TrackerManager._TrackerDiameter / 2:
                for Tracker, LockIndex in Feature.Trackers:
                    ReferencePoints = Tracker.LockSaves[LockIndex].Events[:,1:]
                    ReferencePoints = ReferencePoints - ReferencePoints.mean(axis = 0)
                    ExpectedRotation, ExpectedScaling = Feature.Rotation - self.RotationValue, Feature.Scaling - self.Scaling
        # TODO

    def EventCameraDatasetIMU(self):
        filename = '/'.join(self.TrackerManager.__Framework__._GetStreamFormattedName(self.TrackerManager).split('/')[:-1])+'/imu.txt'
        f = open(filename, 'r')
        Data = [[] for i in range(7)]
        for line in f.readlines():
            for i, data in enumerate(line.strip().split(' ')):
                Data[i] += [float(data)]
        f.close()
        return Data

    def GenerateIMUData(self, NPointsMin = 5, fillVoid = False, IMUGTFunction = None, AutoScaleToGT = True, SigmaError = 2.):
        self.IMUData = {'t':[], 'x':[], 'y':[], 'z':[]}
        if SigmaError:
            self.IMUError = {'t':[], 'x':[], 'y':[], 'z':[]}
        self.ComputationData = {'Delta': [], 'S':[], 'R2':[], 'Omega':[], 'N': []}
        PreviousIDsAndPositions = {}
        PreviousT = None
        ScreenCenter = self.TrackerManager.__Framework__._GetStreamGeometry(self.TrackerManager)[:2]/2
        def AddSnap(t, x, y, z):
            self.IMUData['t'] += [t]
            self.IMUData['x'] += [x]
            self.IMUData['y'] += [y]
            self.IMUData['z'] += [z]
        def AddSnapError(t, Sx, Sy, Sz):
            self.IMUError['t'] += [t]
            self.IMUError['x'] += [Sx]
            self.IMUError['y'] += [Sy]
            self.IMUError['z'] += [Sz]
        for nSnap, tSnap in enumerate(self.TrackerManager.History['t']):
            AvailableIDs = self.TrackerManager.History['RecordedTrackers@ID'][nSnap]
            Statuses = self.TrackerManager.History['RecordedTrackers@State.Value'][nSnap]
            Positions = self.TrackerManager.History['RecordedTrackers@Position'][nSnap]

            CurrentIDsAndPositions = {}
            MatchingIDs = set()
            for nID, ID_Status in enumerate(zip(AvailableIDs, Statuses)):
                ID, Status = ID_Status
                if Status != (self.TrackerManager._StateClass._STATUS_LOCKED, 0): # If its either unlocked or is disengaged
                    continue
                if ID in PreviousIDsAndPositions.keys():
                    MatchingIDs.add(ID)
                CurrentIDsAndPositions[ID] = Positions[nID][:2] - ScreenCenter
            if len(MatchingIDs) >= NPointsMin:
                MeanPositions = np.array([(CurrentIDsAndPositions[ID] + PreviousIDsAndPositions[ID])/2 for ID in MatchingIDs])
                Displacements = np.array([(CurrentIDsAndPositions[ID] - PreviousIDsAndPositions[ID]) for ID in MatchingIDs])

                N = len(MatchingIDs)
                S = MeanPositions.sum(axis = 0)
                Delta = Displacements.sum(axis = 0)
                Omega = (Displacements[:,0]*MeanPositions[:,1] - Displacements[:,1]*MeanPositions[:,0]).sum()
                R2 = (MeanPositions**2).sum()

                self.ComputationData['N'].append(N)
                self.ComputationData['Delta'].append(Delta)
                self.ComputationData['S'].append(S)
                self.ComputationData['R2'].append(R2)
                self.ComputationData['Omega'].append(Omega)
                
                DeltaT = tSnap - PreviousT
                dwz = -(N*Omega + Delta[0]*S[1] - Delta[1]*S[0]) / (N*R2 - (S**2).sum())
                #dwy = -(Delta[0] - S[1]*dwz)/N
                #dwx = -(Delta[1] + S[0]*dwz)/N
                dwy = -(Delta[0]*(R2*N - S[0]**2) + (N*Omega - S[0]*Delta[1])*S[1])/ (N * (R2*N - (S**2).sum()))
                dwx = -(Delta[1]*(R2*N - S[1]**2) - (N*Omega + S[1]*Delta[0])*S[0])/ (N * (R2*N - (S**2).sum()))
                
                AddSnap((tSnap + PreviousT)/2, dwx / DeltaT, dwy / DeltaT, dwz / DeltaT)
                if SigmaError:
                    ddwz = 2*N*SigmaError*( abs(Delta.sum()) + np.sqrt(2)*abs(S.sum()) + 2*abs(dwz * S.sum())) / (R2*N - (S**2).sum() )
                    ddwx = 0
                    ddwy = 0
                    AddSnapError((tSnap + PreviousT)/2, ddwx / DeltaT, ddwy / DeltaT, ddwz / DeltaT)
            else:
                if fillVoid:
                    if PreviousT is None:
                        PreviousT = tSnap
                    AddSnap((tSnap + PreviousT)/2, 0, 0, 0)
                    if SigmaError:
                        AddSnapError((tSnap + PreviousT)/2, 0, 0, 0)
            PreviousT = tSnap
            PreviousIDsAndPositions = CurrentIDsAndPositions
        if not IMUGTFunction is None:
            Data = IMUGTFunction()
            MaxIndex = np.where(np.array(Data[0]) > PreviousT)[0][0]
            for nData, Name in enumerate(['tTh', 'axTh', 'ayTh', 'azTh', 'xTh', 'yTh', 'zTh']):
                self.IMUData[Name] = np.array(Data[nData][:MaxIndex])
            if AutoScaleToGT:
                for Axis in ['x', 'y', 'z']:
                    self.IMUData[Axis] = np.array(self.IMUData[Axis])
                    Scaling = np.sqrt((self.IMUData[Axis+'Th']**2 - self.IMUData[Axis+'Th'].mean()**2).mean() / (self.IMUData[Axis]**2 - self.IMUData[Axis].mean()**2).mean())
                    self.IMUData[Axis] *= Scaling
                    if SigmaError:
                        self.IMUError[Axis] = np.array(self.IMUError[Axis]) * Scaling
                    print("Axis {0} scaling : {1:.5f}".format(Axis, Scaling))

    def GetSnapshot(self, t):
        self._ComputePoseInformation()

        if self.LastSnap is None:
            self.LastSnap = t
        else:
            Delta = t - self.LastSnap
            self.LastSnap = t
            self.TranslationValue = self.TranslationValue + self.SpeedTranslationValue * Delta
            self.RotationValue += self.SpeedRotationValue * Delta
            self.ScalingValue += self.SpeedScalingValue * Delta

        self.TranslationHistory += [np.array(self.TranslationValue)]
        self.RotationHistory += [self.RotationValue]
        self.ScalingHistory += [self.ScalingValue]
        self.SpeedTranslationHistory += [np.array(self.SpeedTranslationValue)]
        self.SpeedRotationHistory += [self.SpeedRotationValue]
        self.SpeedScalingHistory += [self.SpeedScalingValue]

    def _ComputePoseInformation(self):
        if len(self.SpeedsLockVectorsDict) <= 1:
            return None
        TSum = 0.
        RSum = 0.
        SSum = 0.

        NDuos = 0
        for i, ID_i in enumerate(self.SpeedsLockVectorsDict.keys()):
            for j, ID_j in enumerate(list(self.SpeedsLockVectorsDict.keys())[i+1:]):
                vi = self.SpeedsLockVectorsDict[ID_i]
                ri = self.PositionsLockVectorsDict[ID_i] - self.CenterPoint
                vj = self.SpeedsLockVectorsDict[ID_j]
                rj = self.PositionsLockVectorsDict[ID_j] - self.CenterPoint
                dv = vi - vj
                dr = ri - rj
                
                NR2 = (dr * dr).sum()
                if np.sqrt(NR2) < 30:
                    continue
                s = (dv * dr).sum() / NR2
                
                r = float(np.cross(dr, dv - dr * (dv*dr).sum() / NR2) / NR2)

                ti = vi - np.cross(np.array([0., 0., r]), ri)[:2] - s * ri
                tj = vj - np.cross(np.array([0., 0., r]), rj)[:2] - s * rj

                TSum += ti + tj
                RSum += r
                SSum += s

                NDuos += 1

        if NDuos > 0:
            self.SpeedTranslationValue = TSum / (2*NDuos)
            self.SpeedRotationValue = RSum / NDuos
            self.SpeedScalingValue = SSum / NDuos

class FeatureClass:
    def __init__(self, InitialCenterRelativePosition, RegisteredRotation, RegisteredScaling, InitialTracker, ID):
        self.ID = ID
        self.RegisteredRotation = RegisteredRotation
        self.RegisteredScaling = RegisteredScaling
        self.CenterRelativePosition = np.array(InitialCenterRelativePosition)
        self.Trackers = [(InitialTracker, len(InitialTracker.LocksSaves)-1)]

import cv2
import imutils

class GTMakerClass:
    def __init__(self, TrackerManager, DiameterRatio = 1):
        self.TrackerManager = TrackerManager
        self.WindowSize = int(self.TrackerManager._TrackerDiameter*DiameterRatio)
        self.WidthHeigth = (int(self.TrackerManager._TrackerDiameter*DiameterRatio), int(self.TrackerManager._TrackerDiameter*DiameterRatio))
        self.RadiusOffset = -DiameterRatio * np.array([self.TrackerManager._TrackerDiameter, self.TrackerManager._TrackerDiameter]) / 2

        self.ImagesLookupTable = None
        self.GTData = {'DataFolder':None, 'Type':None, 'BinDt':None}
        self.GTTrackersDict = {}
        self.GTPositions = {}
        self.GTImagesIDs = []
        self.CurrentDataIndex = None
        self.CurrentImage = None

    def GetGTErrorVectors(self, TrackerID):
        if TrackerID not in self.GTPositions.keys():
            raise Exception("Tracker missing from ground-truth database")
        ts, xys = self.TrackerManager.RetreiveHistoryData('RecordedTrackers@Position', TrackerID)
        ts = np.array(ts)
        xys = xys[:,:2]
        tths_xyths = self.GTPositions[TrackerID]
        Errors = []
        if self.GTData['Type'] == 0: # Assume images are given so GT is not extracted from snapshot. Must interpolate snaps to images
            for tth, xth, yth in tths_xyths:
                SnapsIndexes = (abs(ts - tth)).argsort()[:2]
                xy = xys[Indexes[0],:] + (tth - ts[Indexes[0]]) * (xys[Indexes[0],:] - xys[Indexes[1],:]) / (ts[Indexes[0]] - ts[Indexes[1]])
                Errors += [np.linalg.norm(xy - np.array([xth, yth]))]
        else:
            for tth, xth, yth in tths_xyths:
                SnapsIndex = abs(ts - tth).argmin()
                xy = xys[SnapsIndex,:]
                Errors += [xy - np.array([xth, yth])]
        return np.array(Errors)

    def GetGTErrorNorms(self, TrackerID, RemoveOffsetFromBelow = 0, RemoveStaticOffset = True):
        ErrorVectors = self.GetGTErrorVectors(TrackerID)
        if RemoveStaticOffset:
            ErrorVectors[:,0] -= 0.5
            ErrorVectors[:,1] -= 0.5
        if RemoveOffsetFromBelow:
            ValidPoints = (np.linalg.norm(ErrorVectors, axis = 1) < RemoveOffsetFromBelow)
            if ValidPoints.any():
                Offset = ErrorVectors[np.where(ValidPoints),:].mean(axis = 0)
            else:
                Offset = np.array([0.,0.])
        else:
            Offset = np.array([0.,0.])
        return norm(ErrorVectors - Offset, axis = 1)

    def GetValidTrackingDuration(self, TrackerID, RemoveOffset = True, ValidDistance = 5):
        ErrorNorms = self.GetGTErrorNorms(TrackerID, RemoveOffset)
        return (ErrorNorms < ValidDistance).sum() * self.GTDt

    def DrawGTImage(self, ImageIndex = None, delay = 1, ID = None):
        if not ImageIndex is None:
            self.LoadImage(self.ImagesLookupTable['filename'][ImageIndex])
            self.t = self.ImagesLookupTable['t'][ImageIndex]
            self.ImageIndex = ImageIndex
        else:
            pass

        frame = self.CurrentImage.transpose(1,0,2)
        frame = np.ascontiguousarray(frame)
        NTrackersDisplayed = 0
        for GTID in self.GTImagesIDs[self.ImageIndex]:
            txys = self.GTPositions[GTID]
            if not ID is None and ID != GTID:
                continue
            if len(txys) > 0:
                txys = np.array(txys)
                if (abs(txys[:,0] - self.t) < 1e-6).any():
                    Index = abs(txys[:,0] - self.t).argmin()
                    xy = txys[Index, 1:]
                    y, x, h, w = self.TrackerToBox(xy)
                    cv2.rectangle(np.ascontiguousarray(frame), (x, y), (x+w, y+h), (0, 255, 0), 1)
                    NTrackersDisplayed += 1

                    ts, xs = self.TrackerManager.RetreiveHistoryData('RecordedTrackers@Position', GTID)
                    xs = xs[:,:2]
                    Indexes = (abs(np.array(ts) - self.t)).argsort()[:2]
                    x = xs[Indexes[0],:] + (self.t - ts[Indexes[0]]) * (xs[Indexes[0],:] - xs[Indexes[1],:]) / (ts[Indexes[0]] - ts[Indexes[1]])
                    x = np.rint(x)
                    cv2.circle(np.ascontiguousarray(frame), tuple(int(v) for v in x), int(self.TrackerManager._TrackerDiameter/2), (0, 0, 255), 1)

        cv2.imshow("Frame", imutils.resize(np.flip(frame, axis = 0), width = 600))
        key = cv2.waitKey(delay)

    def GenerateGroundTruth(self, DataFolder = None, tMax = None, DiameterRatio = 1., BinDt = 0.02, trackerType = 'csrt', MaxAssumedDisplacement = 50):
        self.__init__(self.TrackerManager, DiameterRatio) # Re-init all variables
        self.TrackerCreateFunction = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
    	}[trackerType]
        if tMax is None:
            self.tMax = self.TrackerManager._LinkedMemory.LastEvent.timestamp
        else:
            self.tMax = tMax

        if not DataFolder is None:
            self.GTData['Type'] = 0
            if DataFolder[-1] != '/':
                DataFolder = DataFolder + '/'
            self.GTData['DataFolder'] = DataFolder
            self._OpenLookupTableFile(self.GTData['DataFolder'] + 'images.txt')
        else:
            self.GTData['Type'] = 1
            self.GTData['DataFolder'] = ''
            self.GTData['BinDt'] = BinDt
            self._CreateLookupTableSnaps()

        self.ImageIndex = 0
        self.SnapIndex = 0
        self.t = 0.

        self.MaxAssumedDisplacement = MaxAssumedDisplacement

        while self.t <= self.tMax and self.ImageIndex+1 < len(self.ImagesLookupTable['t']) and self.SnapIndex+1 < len(self.TrackerManager.History['t']): # We will loop over images for GT and Snaps at the same time. Snaps will indicate which trackers to consider, and where to initialize them. When we change the frame, we update all GT trackers.
            if self.ImagesLookupTable['t'][self.ImageIndex+1] < self.TrackerManager.History['t'][self.SnapIndex+1]:
                self._NextImage()
            else:
                self._NextSnap()

    def _NextImage(self):
        self.ImageIndex += 1
        self.t = self.ImagesLookupTable['t'][self.ImageIndex]
        self.LoadImage(self.ImagesLookupTable['filename'][self.ImageIndex])

        self.GTImagesIDs += [[]]

        NSuccess = 0
        for ID, GTTracker_MASCOTLocation in self.GTTrackersDict.items():
            GTTracker, box = GTTracker_MASCOTLocation
            if not GTTracker is None:
                success, box = GTTracker.update(self.CurrentImage)
                if success:
                    Displacement = int(10*np.linalg.norm(np.array(self.BoxToTracker(box)) - np.array(self.GTPositions[ID][-1][1:]))) / 10
                else:
                    Displacement = None
                if success and (self.MaxAssumedDisplacement == 0 or Displacement < self.MaxAssumedDisplacement):
                    NSuccess += 1
                    self.GTPositions[ID] += [(self.t, ) + self.BoxToTracker(box)]
                    self.GTImagesIDs[-1] += [ID]
                    if (np.array(box[2:]) < 0).any():
                        print("Negative box size")
                else:
                    self.TrackerManager.LogWarning("Failure to match tracker {0} at t = {1:.3f} (ImageIndex = {2}), displacement = {3}.".format(ID, self.t, self.ImageIndex, Displacement))
            else:
                if (np.array(box[:2]) < 0).any() or (np.array(box[:2]) + np.array(box[2:]) >= np.flip(np.array(self.CurrentImage.shape[:2]))).any():
                    continue
                self.GTPositions[ID] += [(self.t, ) + self.BoxToTracker(box)]
                self.GTTrackersDict[ID][0] = self.TrackerCreateFunction()
                self.GTTrackersDict[ID][0].init(self.CurrentImage, box)
                NSuccess += 1
                self.GTImagesIDs[-1] += [ID]
        self.TrackerManager.Log("t = {0:.2f}s / {1:.2f}s, {2}/{3} active trackers".format(self.t, self.tMax, NSuccess, len(self.GTTrackersDict)))

    def LoadImage(self, Key):
        return [self.LoadFileImage, self.LoadSTImage][self.GTData['Type']](Key)

    def LoadFileImage(self, filename):
        CurrentBinaryImage = imageio.imread(self.GTData['DataFolder'] + filename)
        self.CurrentImage = np.zeros(CurrentBinaryImage.shape + (3,), dtype = np.uint8)
        self.CurrentImage[:,:,0] = CurrentBinaryImage
        self.CurrentImage[:,:,1] = CurrentBinaryImage
        self.CurrentImage[:,:,2] = CurrentBinaryImage

    def LoadSTImage(self, ID):
        Map = self.TrackerManager._LinkedMemory.History['STContext'][ID]
        Map = Map.max(axis = 2)
        t = self.TrackerManager._LinkedMemory.History['t'][ID]
        CurrentBinaryImage = np.rint(np.e**((Map - t)/self.GTData['BinDt']) * 255.)
        self.CurrentImage = np.zeros(CurrentBinaryImage.shape + (3,), dtype = np.uint8)
        self.CurrentImage[:,:,0] = CurrentBinaryImage
        self.CurrentImage[:,:,1] = CurrentBinaryImage
        self.CurrentImage[:,:,2] = CurrentBinaryImage

    def _NextSnap(self):
        self.SnapIndex += 1
        self.t = self.TrackerManager.History['t'][self.SnapIndex]
        for ID, Status in zip(self.TrackerManager.History['RecordedTrackers@ID'][self.SnapIndex], self.TrackerManager.History['RecordedTrackers@State.Value'][self.SnapIndex]):
            # Here we manage which tracker appears, which disappear, and so on
            if Status == (self.TrackerManager._StateClass._STATUS_LOCKED, 0):
                xy = self.TrackerManager.RetreiveHistoryData('RecordedTrackers@Position', ID, Snap = self.SnapIndex)[1][:2]
                box = self.TrackerToBox(xy)
                if ((np.array(box[:2]) < 0).any() or (np.array(box[:2]) + np.array(box[2:]) >= np.flip(np.array(self.CurrentImage.shape[:2]))).any()):
                    if ID in self.GTTrackersDict.keys():
                        del self.GTTrackersDict[ID]
                    else:
                        pass
                    continue
                if ID in self.GTTrackersDict.keys():
                    self.GTTrackersDict[ID][1] = box
                else:
                    self.GTTrackersDict[ID] = [None, box]
                    if ID not in self.GTPositions.keys():
                        self.GTPositions[ID] = []
                continue
            else:
                if ID in self.GTTrackersDict.keys():
                    del self.GTTrackersDict[ID]
                else:
                    pass

    def _OpenLookupTableFile(self, filename):
        self.ImagesLookupTable = {'t': [], 'filename': []}
        with open(filename, 'r') as LookupTableFile:
            for line in LookupTableFile.readlines():
                t, ImageName = str(line).strip().split(' ')
                self.ImagesLookupTable['t'] += [float(t)]
                self.ImagesLookupTable['filename'] += [ImageName]
        self.GTDt = (self.ImagesLookupTable['t'][-1] - self.ImagesLookupTable['t'][0]) / (len(self.ImagesLookupTable['t']) - 1)
        self.TrackerManager.LogSuccess("Lookup table loaded from file")

    def _CreateLookupTableSnaps(self):
        self.ImagesLookupTable = {'t': [], 'filename': []}
        
        self.ImagesLookupTable['t'] = list(self.TrackerManager._LinkedMemory.History['t'])
        self.ImagesLookupTable['filename'] = list(range(len(self.ImagesLookupTable['t'])))

        self.GTDt = (self.ImagesLookupTable['t'][-1] - self.ImagesLookupTable['t'][0]) / (len(self.ImagesLookupTable['t']) - 1)
        self.TrackerManager.LogSuccess("Lookup table created from snapshots")

    def TrackerToBox(self, xy):
        return (int(xy[1]-self.WindowSize/2), int(xy[0]-self.WindowSize/2), self.WindowSize, self.WindowSize)
    def BoxToTracker(self, box):
        return (box[1] + box[3]/2, box[0] + box[2]/2)

    def GetCurrentFrameIndex(self, t):
        self.CurrentFrameIndex = abs(np.array(self.ImagesLookupTable['t']) - t).argmin()

    def _SaveDataToDict(self):
        DataDict = {}
        DataDict['GTData'] = self.GTData
        DataDict['GTPositions'] = self.GTPositions
        DataDict['GTImagesIDs'] = self.GTImagesIDs
        DataDict['ImagesLookupTable'] = self.ImagesLookupTable
        return DataDict
    def _RecoverDataFromDict(self, DataDict):
        self.GTData = DataDict['GTData']
        self.GTPositions = DataDict['GTPositions']
        self.GTImagesIDs = DataDict['GTImagesIDs']
        self.ImagesLookupTable = DataDict['ImagesLookupTable']

class PlotterClass:
    _TrackersScalingSize = 20
    def __init__(self, TM):
        self.TrackerManager = TM

    def Reload(self):
        FileLocation = inspect.getfile(self.__class__)
        FileLoaded = __import__(FileLocation.split('/')[-1].split('.py')[0])
        for Key, Value in self.TrackerManager.__dict__.items():
            if Value == self:
                self.TrackerManager.__dict__[Key] = getattr(FileLoaded, self.__class__.__name__)(self.TrackerManager)
                break

    def CreateTrackingShot(self, TrackerIDs = None, IgnoreTrackerIDs = [], SnapshotNumber = 0, BinDt = 0.005, ax_given = None, cmap = None, addTrackersIDsFontSize = 0, removeTicks = True, add_ts = True, DisplayedStatuses = ['Stabilizing', 'Converged', 'Locked'], DisplayedProperties = ['Aperture issue', 'OffCentered', 'Disengaged'], RemoveNullSpeedTrackers = 0, VirtualPoint = None, GT = None, TrailDt = 0, TrailWidth = 2):
        S = self.TrackerManager
        if TrackerIDs is None:
            TrackerIDs = [ID for ID in S.History['RecordedTrackers@ID'][SnapshotNumber] if ID not in IgnoreTrackerIDs]
        else:
            for ID in IgnoreTrackerIDs:
                if ID in TrackerIDs:
                    raise Exception("Asking to ignore tracker {0} while also asking for it. Aborting".format(ID))
    
        if ax_given is None:
            f, ax = plt.subplots(1,1)
        else:
            ax = ax_given
        Map = S._LinkedMemory.History['STContext'][SnapshotNumber]
        t = S._LinkedMemory.History['t'][SnapshotNumber]
        Mask = (Map.max(axis = 2) > t - BinDt) * Map.max(axis = 2)
        FinalMap = (Map[:,:,0] == Mask) - 1 * (Map[:,:,1] == Mask)
        if cmap is None:
            ax.imshow(np.transpose(FinalMap), origin = 'lower') 
        else:
            ax.imshow(np.transpose(FinalMap), origin = 'lower', cmap = plt.get_cmap(cmap))
    
        def TrackerValidityCheck(TrackerID, SnapshotNumber):
            NullAnswer = (None, None)
            try:
                StatusValue, PropertyValue = S.RetreiveHistoryData('RecordedTrackers@State.Value', TrackerID, SnapshotNumber)[1]
            except TypeError:
                return NullAnswer
            if S._StateClass._StatusesNames[StatusValue] not in DisplayedStatuses:
                return NullAnswer
            else:
                TrackerColor = S._StateClass._COLORS[StatusValue]
            for Property, PropertyName in S._StateClass._PropertiesNames.items():
                if Property and (not PropertyName in DisplayedProperties) and (Property & PropertyValue):# First condition removes 'None' property. Second checks that we dont want to show this property. Third checks that tracker has this property.
                    return NullAnswer
            TrackerMarker = S._StateClass._MARKERS[PropertyValue]
            if RemoveNullSpeedTrackers:
                t, Speed = S.RetreiveHistoryData('RecordedTrackers@Speed', TrackerID, SnapshotNumber)
                if not Speed is None and np.linalg.norm(Speed[:2]) < RemoveNullSpeedTrackers:
                    return NullAnswer
            return (TrackerColor, TrackerMarker)

        if add_ts:
            ax.set_title("t = {0:.3f}".format(t))
        for n_tracker, TrackerID in enumerate(TrackerIDs):
            TrackerColor, TrackerMarker = TrackerValidityCheck(TrackerID, SnapshotNumber)
            if not TrackerColor is None:
                x, y, theta, s = S.RetreiveHistoryData('RecordedTrackers@Position', TrackerID, SnapshotNumber)[1]
                dx, dy = self._TrackersScalingSize * s * np.cos(theta), self._TrackersScalingSize * s * np.sin(theta)
                ax.plot(x, y, color = TrackerColor, marker = TrackerMarker)
                ax.plot([x, x+dx], [y, y+dy], color = TrackerColor)

                if addTrackersIDsFontSize:
                    ax.text(x + 5, y + 2, str(TrackerID), color = TrackerColor, fontsize = addTrackersIDsFontSize)

            if TrailDt > 0:
                nSnap = SnapshotNumber 
                CurrentTrackerLineProps = (None, None)
                TrackerLocationsPerStyle = {CurrentTrackerLineProps: [[]]}
                for nSnap in range(0, SnapshotNumber + 1):
                    if S.TrackersPositionsHistoryTs[SnapshotNumber] - S.TrackersPositionsHistoryTs[nSnap] > TrailDt:
                        continue
                    TrackerColor, TrackerLineStyle, Box = TrackerValidityCheck(TrackerID, nSnap)
                    TrackerLineProps = TrackerColor, TrackerLineStyle
                    if TrackerLineProps != CurrentTrackerLineProps:
                        if TrackerLineProps not in TrackerLocationsPerStyle.keys():
                            TrackerLocationsPerStyle[TrackerLineProps] = []
                        TrackerLocationsPerStyle[TrackerLineProps] += [[]]
                        if TrackerLocationsPerStyle[CurrentTrackerLineProps][-1]:
                            TrackerLocationsPerStyle[TrackerLineProps][-1] += [TrackerLocationsPerStyle[CurrentTrackerLineProps][-1][-1]]
                        TrackerLocationsPerStyle[TrackerLineProps][-1] += [np.array(S.TrackersPositionsHistory[nSnap][TrackerID])]
                        CurrentTrackerLineProps = TrackerLineProps
                    elif not TrackerColor is None:
                        TrackerLocationsPerStyle[TrackerLineProps][-1] += [np.array(S.TrackersPositionsHistory[nSnap][TrackerID])]
                for LineProps, Data in TrackerLocationsPerStyle.items():
                    TrackerColor, TrackerLineStyle = LineProps
                    if TrackerColor is None:
                        continue
                    for DataSerie in Data:
                        ax.plot(np.array(DataSerie)[:,0], np.array(DataSerie)[:,1], TrackerColor, ls = TrackerLineStyle, lw = TrailWidth)


        if not VirtualPoint is None:
            
            Center = np.array(S._LinkedMemory.STContext.shape[:2], dtype = float) / 2
            VirtualPoint = [Center + np.array(S.FeatureManager.TranslationHistory[SnapshotNumber]), np.array([0., 0.])]

            if 0 < VirtualPoint[0][0] - VirtualPoint[1][0] and VirtualPoint[0][0] + VirtualPoint[1][0] < self.TrackerManager._LinkedMemory.STContext.shape[0] and 0 < VirtualPoint[0][1] - VirtualPoint[1][1] and VirtualPoint[0][1] + VirtualPoint[1][1]< self.TrackerManager._LinkedMemory.STContext.shape[1]:
                ax.plot(VirtualPoint[0][0], VirtualPoint[0][1], 'or')
                #ax.plot([VirtualPoint[0][0], VirtualPoint[0][0]], [VirtualPoint[0][1] - VirtualPoint[1][1], VirtualPoint[0][1] + VirtualPoint[1][1]], 'r')
                #ax.plot([VirtualPoint[0][0] - VirtualPoint[1][0], VirtualPoint[0][0] + VirtualPoint[1][0]], [VirtualPoint[0][1], VirtualPoint[0][1]], 'r')
                ax.arrow(VirtualPoint[0][0], VirtualPoint[0][1], - 10 * (1 + S.FeatureManager.ScalingHistory[SnapshotNumber]) * np.sin(S.FeatureManager.RotationHistory[SnapshotNumber]), 10 * (1 + S.FeatureManager.ScalingHistory[SnapshotNumber]) * np.cos(S.FeatureManager.RotationHistory[SnapshotNumber]), color = 'r')

        ax.set_xlim(0, FinalMap.shape[0]-1)
        ax.set_ylim(0, FinalMap.shape[1]-1)

        if removeTicks:
            ax.tick_params(bottom = 'off', left = 'off', labelbottom = 'off', labelleft = 'off')
        if ax_given is None:
            return f, ax
    
    def GenerateTrackingGif(self, TrackerIDs = None, IgnoreTrackerIDs = [], AddVirtualPoint = False, SnapRatio = 1, tMin = 0., tMax = np.inf, Folder = '/home/dardelet/Pictures/GIFs/AutoGeneratedTracking/', BinDt = 0.005, add_ts = True, cmap = None, DoGif = True, addTrackersIDsFontSize = 0, DisplayedStatuses = ['Stabilizing', 'Converged', 'Locked'], DisplayedProperties = ['Aperture issue', 'OffCentered'], RemoveNullSpeedTrackers = 0, NoRemoval = False, GT = None, TrailDt = 0, TrailWidth = 2):
        if BinDt is None:
            BinDt = self.TrackerManager._MonitorDt

        if AddVirtualPoint:
            VirtualPoint = [np.array(self.TrackerManager._LinkedMemory.STContext.shape[:2], dtype = float) / 2, np.array([0., 0.])]
            LastVPUpdate = None
            Offset_Start = 0
            NUpdates = 0
        else:
            VirtualPoint = None
    
        Snaps_IDs = [snap_id for snap_id in range(len(self.TrackerManager.History['t'])) if (snap_id % SnapRatio == 0 and tMin <= self.TrackerManager.History['t'][snap_id] and self.TrackerManager.History['t'][snap_id] <= tMax)]
        if not NoRemoval:
            os.system('rm '+ Folder + '*.png')
        self.TrackerManager.Log("Generating {0} png frames on {1} possible ones.".format(len(Snaps_IDs), len(self.TrackerManager.History['t'])))
        f = plt.figure(figsize = (16,9), dpi = 100)
        ax = f.add_subplot(1,1,1)
        for nSnap, snap_id in enumerate(Snaps_IDs):
            self.TrackerManager.Log(" > {0}/{1}".format(int(snap_id/SnapRatio + 1), len(Snaps_IDs)))

            if AddVirtualPoint:
                if LastVPUpdate is None:
                    LastVPUpdate = self.TrackerManager.History['t'][snap_id]
                else:
                    Delta = self.TrackerManager.History['t'][snap_id] - LastVPUpdate
                    LastVPUpdate = self.TrackerManager.History['t'][snap_id]
                    AddOffset, Vx, Vy, SVx, SVy = self._GetSnapSceneSpeed(snap_id, Offset_Start, ['Converged'], ['Locked'])
                    if AddOffset:
                        Offset_Start += AddOffset
                    if not Vx is None:
                        NUpdates += 1
                        VirtualPoint[0] += np.array([Vx, Vy]) * Delta
                        VirtualPoint[1] += np.array([SVx, SVy]) / (2 * np.sqrt(NUpdates)) * Delta

            if self.TrackerManager.History['t'][snap_id] > tMax:
                break
    
            self.CreateTrackingShot(TrackerIDs, IgnoreTrackerIDs, snap_id, BinDt, ax, cmap, addTrackersIDsFontSize, True, add_ts, DisplayedStatuses = DisplayedStatuses, DisplayedProperties = DisplayedProperties, RemoveNullSpeedTrackers = RemoveNullSpeedTrackers, VirtualPoint = VirtualPoint, GT = GT, TrailDt = TrailDt, TrailWidth = TrailWidth)
    
            f.savefig(Folder + 't_{0:05d}.png'.format(nSnap))
            ax.cla()
        self.TrackerManager.Log(" > Done.          ")
        plt.close(f.number)
    
        if DoGif:
            File = 'tracking_'+datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')+'.mp4'
            command = 'ffmpeg -framerate {0} -i {1}t_'.format(int(0.5/(SnapRatio * self.TrackerManager._MonitorDt)), Folder)+'0'*(5-int(np.log10(len(Snaps_IDs))+1)) + '%0{0}d.png -c:v libx264 -r 30 '.format(int(np.log10(len(Snaps_IDs))+1)) + Folder + File
            #command = 'convert -delay {0} -loop 0 '.format(int(100*self.TrackerManager._MonitorDt))+Folder+'*.png ' + Folder + File
            print("Generating gif. (command : " + command + " )")
            os.system(command)
            print("GIF generated : {0}".format(File))
            os.system('kde-open ' + Folder + File)
    
            ans = input('Rate this result ((n)ice/(b)ad/(d)elete) : ')
            if '->' in ans:
                ans, name = ans.split('->')
                ans = ans.strip()
                name = name.strip()
                if '.gif' in name:
                    name = name.split('.gif')[0]
                NewFile = name + '_' + File
            else:
                NewFile = File
            if len(ans) == 0:
                os.system('mv ' + Folder + File + ' ' + Folder + 'Meh/' + NewFile)
                print("Moving gif file to Meh folder.")
                return
            if ans.lower()[0] == 'n' or 'nice' in ans.lower():
                os.system('mv ' + Folder + File + ' ' + Folder + 'Nice/' + NewFile)
                print("Moving gif file to Nice folder.")
            elif ans.lower()[0] == 'b' or 'bad' in ans.lower():
                os.system('mv ' + Folder + File + ' ' + Folder + 'Bad/' + NewFile)
                print("Moving gif file to Bad folder.")
            elif ans.lower()[0] == 'd' or 'delete' in ans.lower():
                os.system('rm ' + Folder + File)
                print("Deleted Gif file. RIP.")
    
    def SceneSpeedAnalysis(self, tMin = 0., tMax = np.inf, UsedStatuses = ['Converged'], UsedProperties = ['Locked']):
        f, axs = plt.subplots(2,1)
        axs[0].set_title('Speed Vx')
        axs[1].set_title('Speed Vy')

        S = self.TrackerManager
        Vxs = []
        Vys = []
        Ts = []
        Offset_Start = 0
        for SnapID, t in enumerate(S.TrackersPositionsHistoryTs):
            if t < tMin or t > tMax:
                continue
            AddOffset, Vx, Vy, SVx, SVy = self._GetSnapSceneSpeed(SnapID, Offset_Start, UsedStatuses, UsedProperties)
            if AddOffset:
                Offset_Start += AddOffset
            if not Vx is None:
                Ts += [t]
                Vxs += [Vx]
                Vys += [Vy]
                axs[0].plot([t, t], [Vx - SVx, Vx + SVx], 'k')
                axs[1].plot([t, t], [Vy - SVy, Vy + SVy], 'k')
        axs[0].plot(Ts, Vxs, '--r')
        axs[1].plot(Ts, Vys, '--r')

    def _GetSnapSceneSpeed(self, SnapID, Offset_Start = 0, UsedStatuses = ['Converged'], UsedProperties = ['Locked']):
        S = self.TrackerManager
        N_Considered = 0
        Vx_Sum = 0
        Vx2_Sum = 0
        Vy_Sum = 0
        Vy2_Sum = 0
        AddOffset = None
        for nTracker, SpeedValue in enumerate(S.TrackersSpeedsHistory[SnapID][Offset_Start:]):
            if S.TrackersStatuses[SnapID][Offset_Start + nTracker][0] == S._STATUS_DEAD:
                continue
            if AddOffset is None:
                AddOffset = nTracker
            if S._StateClass._StatusesNames[S.TrackersStatuses[SnapID][Offset_Start + nTracker][0]] not in UsedStatuses:
                continue
            if S._StateClass._PropertiesNames[S.TrackersStatuses[SnapID][Offset_Start + nTracker][1]] not in UsedProperties:
                continue
            N_Considered += 1
            Vx_Sum += SpeedValue[0]
            Vx2_Sum += SpeedValue[0] ** 2
            Vy_Sum += SpeedValue[1]
            Vy2_Sum += SpeedValue[1] ** 2
        if N_Considered > 0:
            Vx = Vx_Sum / N_Considered
            Vy = Vy_Sum / N_Considered

            VxSigma = np.sqrt(Vx2_Sum / N_Considered - Vx ** 2)
            VySigma = np.sqrt(Vy2_Sum / N_Considered - Vy ** 2)

            return AddOffset, Vx, Vy, VxSigma, VySigma
        else:
            return AddOffset, None, None, None, None

    def FullAnalysisOfTracker(self, TrackerID, SpeedLogValue = 2):
        S = self.TrackerManager
        try:
            if not self.TS._FullEventsHistory:
                raise Exception
        except:
            print("Unable to perform full analysis, as \"_FullEventsHistory\" isn't defined or set to False")
            return

        self._TrackerMeanPositionRelativeTC = 2.
        Tracker = S.Trackers[TrackerID]
        tStart = S.StartTimes[Tracker.ID]
        if tStart is None:
            ans = input("Tracker never started, proceed anyway ? (y/N)")
            if ans.lower() != 'y':
                return None, None
        tDeath = S.DeathTimes[TrackerID]

        plt.ion()
        f, axs = plt.subplots(4, 2)
        ScreenAx = axs[0,0]
        ProjectedMapAx = axs[1,0]
        ProjectedMapAx.set_xlim(-Tracker.HalfSize, Tracker.HalfSize)
        ProjectedMapAx.set_ylim(-Tracker.HalfSize, Tracker.HalfSize)
        ProjectedMapAx.set_aspect('equal')
        SpeedOrientations = axs[2,0]
        SpeedOrientations.set_aspect('equal')
        SpeedOrientations.set_title('Speeds orientations')
        PAx = axs[0,1]
        PAx.set_title('Tracker {0} : Position - '.format(Tracker.ID))
        PAx.set_xlim(0, S._LinkedMemory.STContext.shape[0])
        PAx.set_ylim(0, S._LinkedMemory.STContext.shape[1])

        ActivityAx = axs[1,1]
        ActivityAx.set_title('Absolute activity (g) and correction activity (b)')
        ScalarRatiosAx = axs[2,1]
        ScalarRatiosAx.set_title('Speed error, vectorial and aperture relative scalars in delta (x) and speed (v)')
        ScalarRatiosAx.plot([0., Tracker.PEH[-1][0]], [0, 0], '--k')
        ScalarRatiosAx.plot([0., Tracker.PEH[-1][0]], [100, 100], '--k')
        DeathValue = S._TrackerDiameter * S._TrackerMinDeathActivity
        StartValue = S._DetectorMinActivityForStart
        ActivityAx.plot([0., S.DeathTimes[Tracker.ID]], [DeathValue, DeathValue], '--k')
        ActivityAx.plot([0., S.DeathTimes[Tracker.ID]], [StartValue, StartValue], '--k')
        FeatureOrientationAx = axs[3,0]
        FeatureOrientationAx.set_xlim(-1.1, 1.1)
        FeatureOrientationAx.set_aspect('equal')
        FeatureOrientationAx.set_ylim(-1.1, 1.1)
        FeatureOrientationAx.set_title('Feature Orientation')
        ScalarAx = axs[3,1]
        ScalarAx.set_xlim(-Tracker.HalfSize, Tracker.HalfSize)
        ScalarAx.set_ylim(-Tracker.HalfSize, Tracker.HalfSize)
        ScalarAx.set_aspect('equal')
        ScalarAx.set_title('V.dV')

        TC = S._TrackerDefaultTimeConstant
        nEvent = 0
        nSnap = 0
        while S.TrackersPositionsHistoryTs[nSnap] < tStart:
            nSnap += 1
        PAx.plot(S.TrackersPositionsHistory[max(0, nSnap-1)][Tracker.ID][0], S.TrackersPositionsHistory[max(0, nSnap-1)][Tracker.ID][1], 'xr')

        ActivityAx.plot(S.TrackersPositionsHistoryTs[:nSnap], [TrackersSnapActivities[Tracker.ID] for TrackersSnapActivities in S.TrackersActivitiesHistory[:nSnap]], '--r')

        while tStart - Tracker.PEH[nEvent][0] > S._EdgeBinRatio * TC:
            nEvent += 1
        DrawnPoints = {}
        SavedPoints = {}
        DrawnScalars = {}
        last_t = 0.
        print("Starting ProjectedMap at nEvent = {0}, nSnap = {1}".format(nEvent, nSnap))
        while tStart - Tracker.PEH[nEvent][0] > 0:
            if S.TrackersPositionsHistoryTs[nSnap] < Tracker.PEH[nEvent][0]:
                PAx.plot(S.TrackersPositionsHistory[nSnap][Tracker.ID][0], S.TrackersPositionsHistory[nSnap][Tracker.ID][1], 'xg')
                ActivityAx.plot(S.TrackersPositionsHistoryTs[nSnap], S.TrackersActivitiesHistory[nSnap][Tracker.ID], 'xg')
                nSnap += 1

            DrawnPoints[tuple(Tracker.PEH[nEvent])] = ProjectedMapAx.plot(Tracker.PEH[nEvent][1], Tracker.PEH[nEvent][2], marker = 'o', color = 'k', alpha = 1. - max(0., min(1., (tStart - Tracker.PEH[nEvent][0]) / (S._EdgeBinRatio * TC))))[0]

            nEvent += 1
            last_t = Tracker.PEH[nEvent][0]
        print("Finished initialization at nEvent = {0}, nSnap = {1}".format(nEvent, nSnap))
        SnapShown = max(0, nSnap-1)
        x, y = int(S.TrackersPositionsHistory[SnapShown][Tracker.ID][0]), int(S.TrackersPositionsHistory[SnapShown][Tracker.ID][1])
        Map = S._LinkedMemory.History['STContext'][SnapShown][int(x - Tracker.HalfSize):int(x + Tracker.HalfSize + 1), int(y - Tracker.HalfSize):int(y + Tracker.HalfSize + 1)]
        t = S._LinkedMemory.History['t'][SnapShown]
        FinalMap = np.e**(-(t - Map.max(axis = 2))/(TC * S._EdgeBinRatio)) * (2*Map.argmax(axis = 2) - 1)

        Image = ScreenAx.imshow(np.transpose(FinalMap), origin = 'lower', cmap = 'binary', vmin = -1, vmax = 1)
        ScreenAx.set_title("Snap {0}, t = {1:.3f}".format(SnapShown, t))
        CurrentSpeedArrow = SpeedOrientations.arrow(0., 0., 0., 0., color = 'k')
        ModSpeedArrow = SpeedOrientations.arrow(0., 0., 0., 0., color = 'r')
        ModPosSpeedArrow = SpeedOrientations.arrow(0., 0., 0., 0., color = 'b')
        CurrentFeatureOriantationArrow = FeatureOrientationAx.arrow(0., 0., 0., 0., color = 'k')
        Speed = [0., 0.]
        SpeedError = [0., 0.]
        ans = ''
        ScalarAverage = 0.
        ffValue = None

        Locked = False
        LockedTime = None

        while (not ffValue is None or ans.lower() != 'q') and nEvent < len(Tracker.PEH):
            try:
                Changed = False
                while nSnap < len(S.TrackersPositionsHistoryTs) and S.TrackersPositionsHistoryTs[nSnap] < Tracker.PEH[nEvent][0]:
                    PAx.plot(S.TrackersPositionsHistory[nSnap][Tracker.ID][0], S.TrackersPositionsHistory[nSnap][Tracker.ID][1], 'x', color = self.StatusesColors[S._StateClass._StatusesNames[S.TrackersStatuses[nSnap][Tracker.ID][0]]])

                    ActivityAx.plot(S.TrackersPositionsHistoryTs[nSnap], S.TrackersActivitiesHistory[nSnap][Tracker.ID], 'xg')
                    ActivityAx.plot(S.TrackersPositionsHistoryTs[nSnap], S.TrackersScalarCorrectionActivities[nSnap][Tracker.ID], 'xb')
                    
                    ScalarRatiosAx.plot(S.TrackersPositionsHistoryTs[nSnap], 100 * S.TrackersScalarCorrectionValues[nSnap][Tracker.ID] / max(1., S.TrackersScalarCorrectionActivities[nSnap][Tracker.ID]), 'ob')
                    ScalarRatiosAx.plot(S.TrackersPositionsHistoryTs[nSnap], 100 * S.TrackersVectorialCorrectionValues[nSnap][Tracker.ID] / max(1., S.TrackersScalarCorrectionActivities[nSnap][Tracker.ID]), 'or')
                    ScalarRatiosAx.plot(S.TrackersPositionsHistoryTs[nSnap], 100 * S.TrackersApertureScalarEstimationByDeltas[nSnap][Tracker.ID] / max(1., S.TrackersScalarCorrectionActivities[nSnap][Tracker.ID]), 'xm')
                    ScalarRatiosAx.plot(S.TrackersPositionsHistoryTs[nSnap], 100 * S.TrackersApertureScalarEstimationBySpeeds[nSnap][Tracker.ID] / max(1., S.TrackersScalarCorrectionActivities[nSnap][Tracker.ID]), 'vm')
    
                    Speed = S.TrackersSpeedsHistory[nSnap][Tracker.ID]
                    print("New speed reference at {0:.3f}s: {1}".format(S.TrackersPositionsHistoryTs[nSnap], Speed))
                    N = np.linalg.norm(Speed)
                    if N > 0:
                        TC = 1. / N
                    else:
                        TC = S._TrackerDefaultTimeConstant
                    nSnap += 1
                    Changed = True
                if Changed:
                    SnapShown = max(0, nSnap-1)
                    print("Current average position : {0}".format(S.TrackersAverageInternalPositions[SnapShown][Tracker.ID]))
                    CurrentSpeedArrow.remove()
                    ModPosSpeedArrow.remove()
                    CurrentSpeedArrow = SpeedOrientations.arrow(0., 0., Speed[0],  Speed[1])
                    PositionSpeedError = S.TrackersPositionErrorsHistory[SnapShown][Tracker.ID] / (S.TrackersActivitiesHistory[SnapShown][Tracker.ID] * self._TrackerMeanPositionRelativeTC * TC)
                    ModPosSpeedArrow = SpeedOrientations.arrow(0., 0., PositionSpeedError[0], PositionSpeedError[1], color = 'b')

                    CurrentFeatureOriantationArrow.remove()
                    CurrentFeatureOriantationArrow = FeatureOrientationAx.arrow(0., 0., S.TrackersAverageOrientations[SnapShown][Tracker.ID][0], S.TrackersAverageOrientations[SnapShown][Tracker.ID][1], color = 'k')

                    x, y = int(S.TrackersPositionsHistory[SnapShown][Tracker.ID][0]), int(S.TrackersPositionsHistory[SnapShown][Tracker.ID][1])
                    Map = S._LinkedMemory.History['STContext'][SnapShown][x - Tracker.HalfSize:x + Tracker.HalfSize, y - Tracker.HalfSize:y + Tracker.HalfSize]
                    t = S._LinkedMemory.History['t'][SnapShown]
                    FinalMap = np.e**(-(t - Map.max(axis = 2))/(TC * S._EdgeBinRatio)) * (2*Map.argmax(axis = 2) - 1)
    
                    Image.set_data(np.transpose(FinalMap))
                    ScreenAx.set_title("Snap {0}, t = {1:.3f}".format(SnapShown, t))
                    PAx.set_title('Tracker {1} : Position - {0}'.format(S._StateClass._StatusesNames[S.TrackersStatuses[SnapShown][Tracker.ID][0]], Tracker.ID) + (S.TrackersStatuses[SnapShown][Tracker.ID][1] != 0) * ' - {0}'.format(S._PropertiesNames[S.TrackersStatuses[SnapShown][Tracker.ID][1]]))

                    if not Locked and S.TrackersStatuses[SnapShown][Tracker.ID][1] == S._PROPERTY_LOCKED:
                        Locked = True
                        LockedTime = S.TrackersPositionsHistoryTs[SnapShown]
                        SavedPoints = {}
                        for Event, Point in list(DrawnPoints.items()):
                            SavedPoints[Event] = Point
                            Point.set_color('b')
                        for Event, Point in list(SavedPoints.items()):
                            if Event in list(DrawnScalars.keys()):
                                #DrawnVectorials[Event].remove()
                                DrawnScalars[Event].remove()
                                #del DrawnVectorials[Event]
                                del DrawnScalars[Event]
                        DrawnPoints = {}
                    elif Locked and S.TrackersStatuses[SnapShown][Tracker.ID][1] != S._PROPERTY_LOCKED:
                        Locked = False
                        LockedTime = None
                        for Event, Point in list(SavedPoints.items()):
                            Point.remove()
                        SavedPoints = {}
                
                t = Tracker.PEH[nEvent][0]
                Delta_t = t - last_t
                Decay = np.e**(-Delta_t/TC)
                last_t = t
                ScalarAverage *= Decay
                ProjectedMapAx.set_title('Projected Map, nEvent = {0}, t = {1:.3f}'.format(nEvent, t) + (not Tracker.IgnoredComputationReasons[nEvent] is None) * ', ignored due to {0}'.format(Tracker.IgnoredComputationReasons[nEvent]))
                ToDel = []
                Associations = [tuple(Event) for Event in Tracker.AssociationsHistory[nEvent]]
                ModSpeedArrow.remove()
                if len(Associations) >= S._MinConsideredNeighbours:
                    ConsideredNeighbours = np.array(Associations)
    
                    if Locked:
                        CurrentProjectedEvent = [LockedTime + TC] + list(Tracker.PEH[nEvent])[1:]
                    else:
                        CurrentProjectedEvent = list(Tracker.PEH[nEvent])
                    SpeedError, ProjError = Tracker._ComputeSpeedError(Tracker, CurrentProjectedEvent, ConsideredNeighbours)
    
                    Vectorial = float(Speed[0]*SpeedError[1] - Speed[1]*SpeedError[0])
                    NSpeed = np.linalg.norm(Speed)
                    NSpeedError = np.linalg.norm(SpeedError)
                    if NSpeed > 0 and NSpeedError > 0:
                        Scalar = (Speed*SpeedError).sum() / (NSpeed * NSpeedError)
                    else:
                        Scalar = 0
                    Scalar = float(Scalar)
    
                    ScalarAverage += Scalar
                    VectorialColor = 'r'*(Vectorial < 0) + 'g'*(Vectorial > 0) + 'k'*(Vectorial == 0)
                    ScalarColor = 'r'*(Scalar < 0) + 'g'*(Scalar > 0) + 'k'*(Scalar == 0)
                    #DrawnVectorials[tuple(Tracker.PEH[nEvent])] = VectorialAx.plot(Tracker.PEH[nEvent][1], Tracker.PEH[nEvent][2], marker = 'o', color = VectorialColor, alpha = 1.)[0]
                    DrawnScalars[tuple(Tracker.PEH[nEvent])] = ScalarAx.plot(Tracker.PEH[nEvent][1], Tracker.PEH[nEvent][2], marker = 'o', color = ScalarColor, alpha = 1.)[0]
                    
                    ModSpeedArrow = SpeedOrientations.arrow(0., 0., SpeedError[0], SpeedError[1], color = 'r')
                else:
                    ModSpeedArrow = SpeedOrientations.arrow(0., 0., 0., 0., color = 'r')
                    SpeedError = [0., 0.]
                VMax = max(abs(np.array(Speed)).max(), abs(np.array(SpeedError)).max())
                if VMax == 0:
                    VMax = 1
                else:
                    VMax = max(SpeedLogValue, SpeedLogValue**(int(np.log(VMax)/np.log(SpeedLogValue)+1)))
                SpeedOrientations.set_xlim(-VMax, VMax)
                SpeedOrientations.set_ylim(-VMax, VMax)
    
                #print Associations
                if Locked:
                    for Event, Point in list(SavedPoints.items()):
                        SavedPoints[Event].set_color('b')
                        if len(Associations):
                            for EventAssociated in Associations:
                                if EventAssociated == Event:
                                    SavedPoints[Event].set_color('r')
                                    Associations.remove(Event)
                                    break
                for Event, Point in list(DrawnPoints.items()):
                    if Event[0] < t - TC * S._EdgeBinRatio:
                        try:
                            Point.remove()
                        except:
                            None
                        if Event in list(DrawnScalars.keys()):
                            #DrawnVectorials[Event].remove()
                            DrawnScalars[Event].remove()
                            #del DrawnVectorials[Event]
                            del DrawnScalars[Event]
                        ToDel += [Event]
                        continue
                    DrawnPoints[Event].set_color('k')
                    if len(Associations):
                        for EventAssociated in Associations:
                            if EventAssociated == Event:
                                DrawnPoints[Event].set_color('r')
                                Associations.remove(Event)
                                break
                    Alpha = 1. - max(0., min(1., (t - Event[0]) / (S._EdgeBinRatio * TC)))
                    DrawnPoints[Event].set_alpha(Alpha)
                    if Event in list(DrawnScalars.keys()):
                        #DrawnVectorials[Event].set_alpha(Alpha)
                        DrawnScalars[Event].set_alpha(Alpha)
                for Event in ToDel:
                    del DrawnPoints[Event]
    
                DrawnPoints[tuple(Tracker.PEH[nEvent])] = ProjectedMapAx.plot(Tracker.PEH[nEvent][1], Tracker.PEH[nEvent][2], marker = 'o', color = 'g', alpha = 1.)[0]
    
                nEvent += 1

            except KeyboardInterrupt:
                ffValue = None
            
            try:
                if ffValue is None:
                    plt.show()
                    ans = input("")
                    if 'ff' in ans:
                        ans = ans.split('ff')[1]
                        if 's' in ans or '.' in ans:
                            if '@' in ans:
                                ffValue = float(ans.split('s')[0].split('@')[1])
                                print("Fast forward up to {0:.3}s".format(ffValue))
                            else:
                                ffValue = float(t) + float(ans.split('s')[0])
                                print("Fast forward {0:.3f}s up to {1:.3}s".format(ffValue - t, ffValue))
                        else:
                            ffValue = int(ans)
                            print("Fast forward {0} events".format(ffValue))
                else:
                    if type(ffValue) == float and t > ffValue:
                        ffValue = None
                    elif type(ffValue) == int:
                        ffValue -= 1
                        if ffValue <= 0:
                            ffValue = None
            except:
                ffValue = ""
        return axs, nEvent, DrawnPoints, SavedPoints

# Shape Analysis section. WIP
    
    def ShapePointsReduction(self, TrackerID, NFinal):
        Events = list(self.TrackerManager.Trackers[TrackerID].LocksSaves[0].Events)
        for Event in Events:
            Event[0] = 1
        M = 100000 * np.ones((len(Events), len(Events)))
        for i in range(len(Events)):
            for j in range(i+1, len(Events)):
                M[i,j] = np.linalg.norm(Events[i][1:] - Events[j][1:])
        if NFinal is None:
            return Events, M
        while len(Events) > NFinal:
            i, j = np.unravel_index(M.argmin(), M.shape)
            Events += [(np.array(Events[i])*Events[i][0] + np.array(Events[j])*Events[j][0])/(Events[i][0]+Events[j][0])]
            Events[-1][0] = (Events[i][0]+Events[j][0])
            Events.pop(max(i,j))
            Events.pop(min(i,j))
            M[max(i,j):-1,:] = M[max(i,j)+1:,:]
            M[min(i,j):-1,:] = M[min(i,j)+1:,:]
            M[:,max(i,j):-1] = M[:,max(i,j)+1:]
            M[:,min(i,j):-1] = M[:,min(i,j)+1:]
            M[len(Events),:] = 100000
            M[:,len(Events)] = 100000
            for k_i in range(len(Events)-1):
                M[k_i, len(Events)-1] = np.linalg.norm(Events[k_i][1:] - Events[len(Events)-1][1:])
        return Events, M 

    def AnalyzeTrackerShape(self, TrackerID, nReducedPoints = 50, nBins = 20, ax_given = None):
        ReducedEvents, M = self.ShapePointsReduction(TrackerID, nReducedPoints)
        if ax_given is None:
            f, axs = plt.subplots(1,3)
            axs[0].set_aspect('equal')
        else:
            axs = ax_given
        axs[0].set_title('Tracker {0}'.format(TrackerID))
        axs[0].plot(np.array(self.TrackerManager.Trackers[TrackerID].LocksSaves[0].Events)[:,1], np.array(self.TrackerManager.Trackers[TrackerID].LocksSaves[0].Events)[:,2], 'or')
        axs[0].plot(np.array(ReducedEvents)[:,1], np.array(ReducedEvents)[:,2], 'ob')
        Angles = CreateCenteredAnglesHistogram(ReducedEvents)
        Distances = CreateRelativeDistancesHistogram(ReducedEvents, M)
        AnglesOccurences, AnglesValues, ax = axs[1].hist(Angles, bins=nBins, range=(0, np.pi*2))
        DistancesOccurences, DistancesValues, ax = axs[2].hist(Distances, bins=nBins)
        return AnglesOccurences, AnglesValues, DistancesOccurences, DistancesValues, axs

    def CompareTrackersShapesHistograms(self, ShowBests = (0, 10), IDRange = [0, None], nReducedPoints = 50, nBins = 20, PlotScenes = False, PlotDuos = False, Unique = True):
        if IDRange[1] is None:
            IDRange[1] = len(self.TrackerManager.Trackers)

        TrackersAnglesHistDict = {} 
        TrackersDistancesHistDict = {} 
        f, axs = plt.subplots(3,1)
        for TrackerID in range(IDRange[0], IDRange[1]):
            Tracker = self.TrackerManager.Trackers[TrackerID] 
            if not len(Tracker.LocksSaves):
                continue
            AnglesHist, AnglesValues, DistancesHist, DistancesValues, axs = self.AnalyzeTrackerShape(TrackerID, nReducedPoints=nReducedPoints, ax_given=axs)
            TrackersAnglesHistDict[TrackerID] = AnglesHist / float(AnglesHist.sum())
            TrackersDistancesHistDict[TrackerID] = DistancesHist / float(DistancesHist.sum())
        plt.close(f.number)

        N = len(TrackersAnglesHistDict)

        M_Angles = np.ones((N, N))
        M_Distances = np.ones((N, N))
        for i, key_i in enumerate(TrackersAnglesHistDict.keys()):
            for j, key_j in enumerate(TrackersAnglesHistDict.keys()):
                if j > i:
                    M_Angles[i,j] = Battacharia(TrackersAnglesHistDict[key_i], TrackersAnglesHistDict[key_j]) 
                    M_Distances[i,j] = Battacharia(TrackersDistancesHistDict[key_i], TrackersDistancesHistDict[key_j])
                    #M[i,j] = 1 - (TrackersHistDict[key_i]*TrackersHistDict[key_j]).sum()

        M = M_Angles + M_Distances
        ListDuos = np.dstack(np.unravel_index(np.argsort(M.ravel()), M.shape))[0,:,:].tolist()
        OutDuos = []
        OutIDs = []
        if PlotScenes:
            f_scenes, axs_scenes = plt.subplots(2, ShowBests[1] - ShowBests[0] + 1)
        BestDuoID = 0
        while BestDuoID <= ShowBests[1]:
            if BestDuoID < ShowBests[0]:
                BestDuoID += 1
                continue
            i, j = list(TrackersAnglesHistDict.keys())[ListDuos[BestDuoID][0]], list(TrackersAnglesHistDict.keys())[ListDuos[BestDuoID][1]]
            if Unique and (i in OutIDs or j in OutIDs):
                BestDuoID += 1
                continue
            OutDuos += [(i, j)]
            if i not in OutIDs:
                OutIDs += [i]
            if j not in OutIDs:
                OutIDs += [j]
            if PlotDuos:
                f, axs = plt.subplots(2,3)
                print(axs.shape)
                _, _, _, _, _ = self.AnalyzeTrackerShape(i, nReducedPoints=nReducedPoints, ax_given=axs[0,:])
                _, _, _, _, _ = self.AnalyzeTrackerShape(j, nReducedPoints=nReducedPoints, ax_given=axs[1,:])
                axs[0,1].set_title('Angles Variations : {0:.2f}'.format(M_Angles[ListDuos[BestDuoID][0], ListDuos[BestDuoID][1]]))
                axs[0,2].set_title('Distances variations : {0:.2f}'.format(M_Distances[ListDuos[BestDuoID][0], ListDuos[BestDuoID][1]]))

            if PlotScenes:
                SnapID_i = ((np.array(self.TrackerManager.TrackersPositionsHistoryTs) - self.TrackerManager.Trackers[i].LocksSaves[0].Time) > 0).tolist().index(True)
                self.CreateTrackingShot(TrackerIDs = [i], SnapshotNumber = SnapID_i, BinDt = 0.005, ax_given = axs_scenes[0, BestDuoID-ShowBests[0]], cmap = 'binary', addTrackersIDsFontSize = 10, removeTicks = True, add_ts = True, DisplayedStatuses = ['Stabilizing', 'Converged'], DisplayedProperties = ['Aperture issue', 'Locked', 'None'], RemoveNullSpeedTrackers = 0, VirtualPoint = None)
                SnapID_j = ((np.array(self.TrackerManager.TrackersPositionsHistoryTs) - self.TrackerManager.Trackers[j].LocksSaves[0].Time) > 0).tolist().index(True)
                self.CreateTrackingShot(TrackerIDs = [j], SnapshotNumber = SnapID_j, BinDt = 0.005, ax_given = axs_scenes[1, BestDuoID-ShowBests[0]], cmap = 'binary', addTrackersIDsFontSize = 10, removeTicks = True, add_ts = True, DisplayedStatuses = ['Stabilizing', 'Converged'], DisplayedProperties = ['Aperture issue', 'Locked', 'None'], RemoveNullSpeedTrackers = 0, VirtualPoint = None)
            BestDuoID += 1

        return OutDuos

    def CompareTrackersShapesAngularVariations(self, ShowBestsFrom = 0, nShowed = 5, IDRange = [0, None], nReducedPoints = None, NLinesHoriz = 10, PlotScenes = False, PlotDuos = False, Unique = True, MaxShift = 0, AddLines = False):
        if IDRange[1] is None:
            IDRange[1] = len(self.TrackerManager.Trackers)
        TrackersAngularVariancesDict = {} 
        LinesDict = {}

        for TrackerID in range(IDRange[0], IDRange[1]):
            Tracker = self.TrackerManager.Trackers[TrackerID] 
            if not len(Tracker.LocksSaves):
                continue
            Angles, Values, dTheta, dLine, Lines = self.GetAngularRepartition(TrackerID, NLinesHoriz, nReducedPoints)
            ValuesArray = []
            for ValueSet in Values:
                ValuesArray += ValueSet
            Variances = np.array([((np.array(LineSet) - np.array(LineSet).mean()) ** 2).sum() for LineSet in Values])
            TrackersAngularVariancesDict[TrackerID] = Variances / Variances.sum()
            LinesDict[TrackerID] = []
            for ValueSet, LineSet in zip(Values, Lines):
                for Value, Line in zip(ValueSet, LineSet):
                    if (Value <= np.array(ValuesArray)).sum() <= 3:
                        LinesDict[TrackerID] += [Line]
            print(LinesDict[TrackerID])
        N = len(TrackersAngularVariancesDict)
        VariancesDistanceMatrix = 1000 * np.ones((N, N))
        AssociatedShifts = np.zeros((N, N))
        for i, key_i in enumerate(TrackersAngularVariancesDict.keys()):
            for j, key_j in enumerate(TrackersAngularVariancesDict.keys()):
                if j > i:
                    MinDist = 1000
                    MinShift = 0
                    for iShift, Shift in enumerate(Angles):
                        VShift_j = np.roll(TrackersAngularVariancesDict[key_j], iShift)
                        Vshift_i = TrackersAngularVariancesDict[key_i]
                        #Scalar = 1 - (VShift_j * TrackersAngularVariancesDict[key_i]).sum()
                        #Scalar = Battacharia(VShift_j, TrackersAngularVariancesDict[key_i])
                        Scalar = np.linalg.norm(Vshift_i - VShift_j) / sqrt(2)
                        if Scalar < MinDist and Shift<= MaxShift:
                            MinDist = Scalar
                            MinShift = Shift
                    VariancesDistanceMatrix[i,j] = MinDist
                    AssociatedShifts[i,j] = MinShift

        M = VariancesDistanceMatrix
        ListDuos = np.dstack(np.unravel_index(np.argsort(M.ravel()), M.shape))[0,:,:].tolist()
        OutDuos = []
        OutIDs = []
        if PlotScenes:
            f_scenes, axs_scenes = plt.subplots(2, nShowed)
        BestDuoID = ShowBestsFrom
        nBestFound = 0
        while nBestFound < nShowed:
            TrackerID_i = list(TrackersAngularVariancesDict.keys())[ListDuos[BestDuoID][0]]
            TrackerID_j = list(TrackersAngularVariancesDict.keys())[ListDuos[BestDuoID][1]] 
            
            BestDuoID += 1
            if Unique and (TrackerID_i in OutIDs or TrackerID_j in OutIDs):
                continue
            OutDuos += [(TrackerID_i, TrackerID_j)]
            if i not in OutIDs:
                OutIDs += [TrackerID_i]
            if j not in OutIDs:
                OutIDs += [TrackerID_j]
            if PlotDuos:
                f, axs = plt.subplots(2,2)
                ReducedEvents, _ = self.ShapePointsReduction(TrackerID_i, nReducedPoints)
                axs[0,0].set_title('Tracker {0}'.format(TrackerID_i))
                axs[0,0].plot(np.array(self.TrackerManager.Trackers[TrackerID_i].LocksSaves[0].Events)[:,1], np.array(self.TrackerManager.Trackers[TrackerID_i].LocksSaves[0].Events)[:,2], 'or')
                axs[0,0].plot(np.array(ReducedEvents)[:,1], np.array(ReducedEvents)[:,2], 'ob')
                try:
                    axs[0,1].set_title('Angles Variations : {0:.2f}'.format(VariancesDistanceMatrix[ListDuos[BestDuoID][0], ListDuos[BestDuoID][1]]))
                    axs[1,1].set_title('Shift : {0:.2f}deg'.format(180 / np.pi *AssociatedShifts[ListDuos[BestDuoID][0], ListDuos[BestDuoID][1]]))
                except:
                    print(ListDuos[BestDuoID][0], ListDuos[BestDuoID][1], M.shape)
                axs[0,1].plot(Angles, TrackersAngularVariancesDict[TrackerID_i])

                ReducedEvents, _ = self.ShapePointsReduction(TrackerID_j, nReducedPoints)
                axs[1,0].set_title('Tracker {0}'.format(TrackerID_j))
                axs[1,0].plot(np.array(self.TrackerManager.Trackers[TrackerID_j].LocksSaves[0].Events)[:,1], np.array(self.TrackerManager.Trackers[TrackerID_j].LocksSaves[0].Events)[:,2], 'or')
                axs[1,0].plot(np.array(ReducedEvents)[:,1], np.array(ReducedEvents)[:,2], 'ob')
                axs[1,1].plot(Angles, TrackersAngularVariancesDict[TrackerID_j])

                if AddLines:
                    PlotLines(LinesDict[TrackerID_i], self.TrackerManager._TrackerDiameter, axs[0,0])
                    PlotLines(LinesDict[TrackerID_j], self.TrackerManager._TrackerDiameter, axs[1,0])

            if PlotScenes:
                SnapID_i = ((np.array(self.TrackerManager.TrackersPositionsHistoryTs) - self.TrackerManager.Trackers[TrackerID_i].LocksSaves[0].Time) > 0).tolist().index(True)
                self.CreateTrackingShot(TrackerIDs = [TrackerID_i], SnapshotNumber = SnapID_i, BinDt = 0.005, ax_given = axs_scenes[0, nBestFound], cmap = 'binary', addTrackersIDsFontSize = 10, removeTicks = True, add_ts = True, DisplayedStatuses = ['Stabilizing', 'Converged'], DisplayedProperties = ['Aperture issue', 'Locked', 'None'], RemoveNullSpeedTrackers = 0, VirtualPoint = None)
                SnapID_j = ((np.array(self.TrackerManager.TrackersPositionsHistoryTs) - self.TrackerManager.Trackers[TrackerID_j].LocksSaves[0].Time) > 0).tolist().index(True)
                self.CreateTrackingShot(TrackerIDs = [TrackerID_j], SnapshotNumber = SnapID_j, BinDt = 0.005, ax_given = axs_scenes[1, nBestFound], cmap = 'binary', addTrackersIDsFontSize = 10, removeTicks = True, add_ts = True, DisplayedStatuses = ['Stabilizing', 'Converged'], DisplayedProperties = ['Aperture issue', 'Locked', 'None'], RemoveNullSpeedTrackers = 0, VirtualPoint = None)
            nBestFound += 1

        return OutDuos, VariancesDistanceMatrix, TrackersAngularVariancesDict


    def GetAngularRepartition(self, TrackerID, NLinesHoriz = 10, nReducedPoints = None):
        dLine = self.TrackerManager._TrackerDiameter / float(NLinesHoriz - 1)
        NAngles = int(np.pi / (2 * np.arctan(1. / NLinesHoriz)))
        dTheta = np.pi / NAngles
        Angles = [i*dTheta for i in range(NAngles)]
        Lines = []
        XF = np.array([self.TrackerManager._TrackerDiameter/2., self.TrackerManager._TrackerDiameter/2.])
        for Angle in Angles:
            if Angle < np.pi/2:
                X0 = np.array([self.TrackerManager._TrackerDiameter/2., -self.TrackerManager._TrackerDiameter/2.])
                XF = np.array([-self.TrackerManager._TrackerDiameter/2., self.TrackerManager._TrackerDiameter/2.])
            else:
                X0 = np.array([self.TrackerManager._TrackerDiameter/2., self.TrackerManager._TrackerDiameter/2.])
                XF = np.array([-self.TrackerManager._TrackerDiameter/2., -self.TrackerManager._TrackerDiameter/2.])
            X = np.array(X0)
            Lines += [[]]
            n = np.array([-np.sin(Angle), np.cos(Angle)])
            c = -(n*X0).sum()
            Lines[-1] += [np.array(n.tolist() + [c])]
            while (c + (XF*n).sum() >= -dLine):
                X = X + dLine * n
                c = -(n*X).sum()
                Lines[-1] += [np.array(n.tolist() + [c])]
        
        RE, M = self.ShapePointsReduction(TrackerID, nReducedPoints)
        Points = []
        for Set in Lines:
            Points += [[]]
            for Line in Set:
                Points[-1] += [0]
                for Event in RE:
                    Points[-1][-1] += max(0, 1. - abs((Event[1:] * Line[:-1]).sum() + Line[-1]) / dLine)
        return Angles, Points, dLine, dTheta, Lines

def PlotLines(Lines, WindowSize, ax = None):
    print(Lines)
    if ax is None:
        f, ax = plt.subplots(1,1)
    for Line in Lines:
        if Line[0] == 0:
            x0 = -WindowSize/2
            x1 = WindowSize/2
            y0 = -Line[2] / Line[1]
            y1 = y0
        elif Line[1] == 0:
            y0 = -WindowSize/2
            y1 = WindowSize/2
            x0 = -Line[2] / Line[0]
            x1 = y0
        else:
            x0 = -WindowSize/2
            x1 = WindowSize/2
            y0 = -(x0 * Line[0] + Line[2]) / Line[1]
            y1 = -(x1 * Line[0] + Line[2]) / Line[1]
            #if y0 > y1:
            #    x0, y0, x1, y1 = x1, y1, x0, y0
            #if y0 < -WindowSize/2:
            #    y0 = -WindowSize/2
            #    x0 = -(y0 * Line[1] + Line[2]) / Line[0]
            #if y1 > WindowSize/2:
            #    y0 = WindowSize/2
            #    x0 = -(y0 * Line[1] + Line[2]) / Line[0]
        ax.plot([x0, x1], [y0, y1], 'k')
        ax.set_xlim(-WindowSize/2, WindowSize/2)
        ax.set_ylim(-WindowSize/2, WindowSize/2)

def Battacharia(H1, H2):
    return - np.log((sqrt(H1*H2)).sum())

def CreateCenteredAnglesHistogram(ReducedEvents):                                 
    RE = np.array(ReducedEvents)[:,1:]          
    CRE = RE - RE.mean(axis = 0)                           
    Angles = []            
    for i in range(len(ReducedEvents)):             
        if CRE[i,0] == 0:         
            Angles += [np.sign(CRE[i,1]) * np.pi / 2]
        else:                                                      
            Angles += [np.arctan(CRE[i,1] / CRE[i,0]) + np.pi * (np.sign(CRE[i,0]) == -1)]
    return Angles

def CreateCenteredDistancesHistogram(ReducedEvents, M):                                 
    RE = np.array(ReducedEvents)[:,1:]          
    CRE = RE - RE.mean(axis = 0)                           
    Distances = np.linalg.norm(CRE, axis = 1).tolist()
    return Distances

def CreateRelativeDistancesHistogram(ReducedEvents, M):
    M = M.flatten()
    Distances = []
    for i in range(len(M.tolist())):
        if M[i] < 100000:
            Distances += [M[i]]
    return Distances
