import numpy as np
from matplotlib import pyplot as plt
import matplotlib

import datetime
import os
import pickle
from sys import stdout
from Framework import Module, TrackerEvent

import pathos.multiprocessing as mp
from functools import partial

class TrackerTRS(Module):
    _STATUS_DEAD = 0
    _STATUS_IDLE = 1
    _STATUS_STABILIZING = 2
    _STATUS_CONVERGED = 3

    _PROPERTY_NONE = 0
    _PROPERTY_APERTURE = 1
    _PROPERTY_LOCKED = 2

    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to handle ST-context memory.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__ReferencesAsked__ = ['Memory']
        self.__Type__ = 'Computation'
        self.__RewindForbidden__ = True                                             # Obviously, no rewind is allowed for this module, as each event can change number of parameters.

        self._TrackerStopIfAllDead = True                                           # Bool. Stops stream if all trackers die. Overrided if _DetectorAutoRestart is TRUE. Default : True

        self._DetectorPointsGridMax = (10, 8)                                       # (Int, Int). Reference grid to place trackers, capping the discrete grid made by dividing the screen with _TrackerDiameter. Default : (10,8), i.e 80 trackers
        self._DetectorMinRelativeStartActivity = 3.                                 # Float. Activity of a tracker needed to activate, divided by _TrackerDiameter. Low values are noise sensible, while high value leave tracker in IDLE state. Default : 3.
        self._DetectorAutoRestart = True                                            # Bool. Restarts or not a tracker when another dissapears. If True, overrides _TrackerStopIfAllDead to False. Default : True.

        self._TrackerSingleTrackerPoint = False                                     # Bool. If TRUE, limits an event to be used only by one tracker. This may allow for tracker not to stack on the same spot. In practise, makes trackers unstables. Default : False

        self._TrackerDiameter = 20                                                  # Int. Size of a tracker's side lenght. Usually changed to 30 for 640x480 camera definitions. Default : 20
        self._TrackerTimeConstantRatioRemoval = 4.                                  # Float. Number of successive edges pixels lines kept. More pixels increases stability, but also inertia and thus lower reactivity. Default : 4.
        self._DetectorDefaultSpeedTrigger = 5.                                      # Float. Default speed trigger, giving a lower bound for speeds in the scene, and higher boundary in time constants.
        self._TrackerMinDeathActivity = 0.7                                         # Float. Low boundary for a tracker to survive divided by _TrackerDiameter. Must be lower than _DetectorMinRelativeStartActivity. Default : 0.7

        self._TrackerIgnoreBorderEvents = False                                     # Bool. Allows to ignore events on the border of the tracker's window for correction, as they are most of the time biased toward the interior of it. Default : False
        self._TrackerAccelerationFactor = 1.2                                       # Float. Acceleration factor for speed correction. Default : 1.2
        self._TrackerUnlockedActivityPower = 1.                                     # Float. Exponent of the activity dividing the speed correction in unlocked mode. May allow to lower inertia in highly textured scenes. Default : 1.
        self._ObjectEdgePropagationTC = 1e-4                                        # Float. Security time window for events considered to be previous edge propagation, and not neighbours later spiking. Default : 1e-4
        self._ClosestEventProximity = 2                                             # Float. Maximum distance for an event to be considered in the simple flow computation. Default : 2.
        self._MaxConsideredNeighbours = 20                                          # Int. Maximum number of considered events in the simple flow computation. Higher values slighty increase flow quality, but highly increase computation time. Default : 20
        self._MinConsideredNeighbours = 4                                           # Int. Minimum number of considered events in the simple flow computation. Lower values allow for more events to be used in the correction, but lower the correction quality. Default : 4

        self._TrackerUsePositionMean = True                                         # Bool. Activate partial recentering of features in the center of the window. Can incrase stability. Default : True
        self._TrackerMeanPositionRelativeRadiusTolerance = 0.3                      # Float. Minimum distance of recentering relative to _TrackerDiameter. Default : 0.3
        self._TrackerMeanPositionAccelerationFactor = 1.2                           # Float. Amount of correction due to recentering process. Default : 1.2

        self._TrackerLockedRotModFactor = 1.                                        # Float. Displacement factor for rotation correction when locked. Default : 1.
        self._TrackerLockedRotModDisplacementActivityPower = 1.                     # Float. Exponent of the activity power of the activity used for the locked tracker rotation correction. Default : 0.2
        self._TrackerLockedRotActivityPower = 1.                                    # Float. Exponent of the activity power of the activity used for tracker speed correction. Default : 0.1

        self._TrackerConvergenceThreshold = 0.2                                     # Float. Amount of correction relative to activity allowing for a converged feature. Default : 0.2
        self._TrackerConvergenceHysteresis = 0.1                                    # Float. Hysteresis value of _TrackerConvergenceThreshold. Default : 0.1

        self._TrackerApertureIssueByDeltaThreshold = 0.8                            # Float. Amount of aperture scalar value by shape relative to correction activity. Default : 0.8
        self._TrackerApertureIssueByDeltaHysteresis = 0.05                          # Float. Hysteresis value of _TrackerApertureIssueByDeltaThreshold. Default : 0.05

        self._TrackerApertureIssueBySpeedThreshold = 0.25                           # Float. Amount of aperture scalar value by speed relative to correction activity. Default : 0.25
        self._TrackerApertureIssueBySpeedHysteresis = 0.05                          # Float. Hysteresis value of _TrackerApertureIssueBySpeedThreshold. Default : 0.05

        self._TrackerAllowShapeLock = True                                          # Bool. Shape locking feature. Should never be disabled. Default : True
        self._TrackerLockingMaxRecoveryBuffer = 100                                 # Int. Size of the buffer storing events while locked, used in case of feature loss. Greater values increase computationnal cost. Default : 100
        self._TrackerLockMaxRelativeActivity = np.inf                               # Float. Maximum activity allowing for a lock, divided by _TrackerDiameter. Can forbid locking on highly textured parts of the scene. np.inf inhibits this feature. Default : np.inf
        self._TrackerLockedPosModFactor = 1.                                        # Float. Displacement factor for position correction when locked. Default : 1.
        self._TrackerLockedRelativeCorrectionsFailures = 0.6                        # Float. Minimum correction activity relative to tracker activity for a lock to remain. Assesses for shape loss. Default : 0.6
        self._TrackerLockedPosModDisplacementActivityPower = 0.2                    # Float. Exponent of the activity power of the activity used for the locked tracker displacement correction. Default : 0.2
        self._TrackerLockedActivityPower = 0.1                                      # Float. Exponent of the activity power of the activity used for tracker speed correction. Default : 0.1
        self._TrackerLockedCanHardCorrect = True                                    # Bool. Allows trackers to remain locked even with high speed correction values. Default : True

        self._ComputeSpeedErrorMethod = 'LinearMean'                                # String. Speed error computation method. Can be 'LinearMean', 'ExponentialMean' or 'TwoPointsProjection'. See associated functions for details. Default : 'LinearMean'
        # Monitoring Variables

        self._SnapDt = 0.01                                                         # Float. Time difference between two snapshots of the system.

        self._StatusesNames = {self._STATUS_DEAD: 'Dead', self._STATUS_IDLE:'Idle', self._STATUS_STABILIZING:'Stabilizing', self._STATUS_CONVERGED:'Converged'} # Statuses names with associated int values.
        self._PropertiesNames = {0: 'None', 1: 'Aperture issue', 2:'Locked'}                                                                                    # Properties names with associated int values.

    def _InitializeModule(self, **kwargs):
        self._LinkedMemory = self.__Framework__.Tools[self.__CreationReferences__['Memory']]

        self.FeatureManager = FeatureManagerClass(self)
        self.GTMaker = GTMakerClass(self)
        if self._DetectorAutoRestart: #  One cannot expect all trackers to die if they auto restart
            self._TrackerStopIfAllDead = False

        L_X, L_Y = self._LinkedMemory.STContext.shape[:2]
        self.Trackers = []
        self.IsAlive = []
        self.AliveIDs = []
        self.StartTimes = []
        self.DeathTimes = []

        d_x = self._TrackerDiameter
        d_y = self._TrackerDiameter
        N_X = L_X / d_x
        N_Y = L_Y / d_y

        if N_X > self._DetectorPointsGridMax[0]:
            N_X = self._DetectorPointsGridMax[0]
            d_x = L_X / N_X
        if N_Y > self._DetectorPointsGridMax[1]:
            N_Y = self._DetectorPointsGridMax[1]
            d_y = L_Y / N_Y

        r_X = (L_X - (N_X - 1) * d_x)/2
        r_Y = (L_Y - (N_Y - 1) * d_y)/2

        self._TrackerDefaultTimeConstant = 1. / self._DetectorDefaultSpeedTrigger
        
        self._TrackerMinMaxNeighboursAutoValues = {'TwoPointsProjection': [2, 2]}
        if self._ComputeSpeedErrorMethod in list(self._TrackerMinMaxNeighboursAutoValues.keys()):
            self._MinConsideredNeighbours = self._TrackerMinMaxNeighboursAutoValues[self._ComputeSpeedErrorMethod][0]
            self._MaxConsideredNeighbours = self._TrackerMinMaxNeighboursAutoValues[self._ComputeSpeedErrorMethod][1]

        self.DetectorInitialPositions = []
        for nX in range(N_X):
            for nY in range(N_Y):
                self.DetectorInitialPositions += [np.array([r_X + nX * d_x, r_Y + nY * d_y])]

        for ID, InitialPosition in enumerate(self.DetectorInitialPositions):
            self._AddTracker(InitialPosition)

        self.LastTsSnap = -np.inf

        self._DetectorMinActivityForStart = self._TrackerDiameter * self._DetectorMinRelativeStartActivity

        self.Plotter = Plotter(self)
        self.LocksSubscribers = [self.FeatureManager.AddLock]

        return True

    def _OnEventModule(self, event):
        UpdateEvent = False
        if event.timestamp - self.LastTsSnap >= self._SnapDt:
            UpdateEvent = True

        self.NewTrackersAsked = 0
        Associated = False
        for TrackerID in self.AliveIDs:
            if not Associated or not self._TrackerSingleTrackerPoint:
                Associated = self.Trackers[TrackerID].RunEvent(event)
                if Associated and self.Trackers[TrackerID].Lock:
                    event = TrackerEvent(original = event, TrackerLocation = np.array(self.Trackers[TrackerID].Position), TrackerID = TrackerID)
            else:
                if UpdateEvent:
                    self.Trackers[TrackerID].UpdateWithEvent(event)
                    if self.Trackers[TrackerID].Lock:
                        event = TrackerEvent(original = event, TrackerLocation = np.array(self.Trackers[TrackerID].Position), TrackerID = TrackerID)

        if UpdateEvent:
            self.LastTsSnap = event.timestamp
            self._LinkedMemory.GetSnapshot()

        if self.NewTrackersAsked:
            self._PlaceNewTrackers()
            self.NewTrackersAsked = 0

        if self._TrackerStopIfAllDead and not self.AliveIDs:
            self.__Framework__.Paused = self.__Name__
            
        return event

    def _OnTrackerLock(self, Tracker):
        for SubscriberMethod in self.LocksSubscribers:
            SubscriberMethod(Tracker)

    def AddLocksSubscriber(self, ListenerMethod):
        self.LocksSubscribers += [ListenerMethod]

    def _KillTracker(self, Tracker, event):
        Tracker.Status = self._STATUS_DEAD
        self.IsAlive[Tracker.ID] = False
        self.DeathTimes[Tracker.ID] = event.timestamp
        self.AliveIDs.remove(Tracker.ID)

        if self._DetectorAutoRestart:
            self.NewTrackersAsked += 1

    def _PlaceNewTrackers(self):
        for NewID in range(self.NewTrackersAsked):
            SelectedPositionID = None
            MaxDistance = 0
            
            for InitialPositionID, InitialPosition in enumerate(self.DetectorInitialPositions):
                InitialPositionMinDistance = np.inf
                for AliveID in self.AliveIDs:
                    InitialPositionMinDistance = min(InitialPositionMinDistance, np.linalg.norm(InitialPosition - self.Trackers[AliveID].Position))
                if InitialPositionMinDistance >= MaxDistance:
                    MaxDistance = InitialPositionMinDistance
                    SelectedPositionID = InitialPositionID
            self._AddTracker(self.DetectorInitialPositions[SelectedPositionID])

    def _AddTracker(self, InitialPosition):
        ID = len(self.Trackers)
        self.Trackers += [TrackerClass(self, ID, InitialPosition)]
        self.IsAlive += [True]
        self.AliveIDs += [ID]
        self.StartTimes += [None]
        self.DeathTimes += [None]

    def GetTrackerPositions(self, TrackerID, Statuses = ['Converged'], Properties = ['Locked']):
        ts = []
        Xs = []
        for nBatch, Batch in enumerate(self.TrackersStatuses):
            if TrackerID >= len(Batch):
                continue
            if self._StatusesNames[Batch[TrackerID][0]] in Statuses and self._PropertiesNames[Batch[TrackerID][1]] in Properties:
                ts += [self.TrackersPositionsHistoryTs[nBatch]]
                Xs += [self.TrackersPositionsHistory[nBatch][TrackerID]]
        return ts, Xs

class LockClass:
    def __init__(self, Time, Activity, Events):
        self.Time = Time
        self.Activity = Activity
        self.Events = Events
        self.ReleaseTime = None

class FlowEstimator:
    def __init__(self):
        # Hidden variables are the sums. Displayed variables are the weights sum, or the averaged variables

        self.LastUpdate = -np.inf
        self._UpToDateMatrix = False
        self.W = 0.
        self._M = np.zeros((4,4))
        self._InvM = np.zeros((4,4))

        self._Es = np.array([0., 0., 0., 0.]) # Estimator in speed
        self._Ed = np.array([0., 0., 0., 0.]) # Estimator in displacement

        self._x = self._y = 0.
        self._r = 0 # This aditionnal variable is useful to recover the time constant. We put it here for simplicity

        self._Sc  = 0. # use W for _Ss
        self._Sxc = self._Sxs = 0.
        self._Syc = self._Sys = 0.

        self._Rcs = 0.
        self._Rxcs = self._Rycs = 0.
        self._Rxcc = 0. # use _x for _Rxss
        self._Rycc = 0. # use _y for _Ryss

        self._Rxxcs = self._Ryycs = 0.
        self._Rxycs = self._Rxycc = self._Rxyss = 0.

    def _RecoverGeneralData(self):
        self.x = self._x / self.W
        self.y = self._y / self.W
        self.r = self._r / self.W

    def AddData(self, Event, Flow, Displacement, Tau):
        F2 = Flow**2
        N2 = F2.sum()
        if N2 == 0:
            return

        self._Decay(Event.timestamp, Tau)

        x, y = Event.location
        xx, yy = x**2, y**2
        xy = x*y

        N = np.sqrt(N2)
        c , s  = Flow/N
        cc, ss = F2/N2
        cs = c*s

        self.W += 1
        self._Es += np.array([Flow[0], Flow[1], Flow[1]*x-Flow[0]*y, Flow[0]*x+Flow[1]*y])
        self._Ed += np.array([Displacement[0], Displacement[1], Displacement[1]*x-Displacement[0]*y, Displacement[0]*x+Displacement[1]*y])

        self._x += x
        self._y += y
        self._r += np.sqrt(xx+yy)

        self._Sc += cc
        self._Sxc += xx*cc
        self._Sxs += xx*ss
        self._Syc += yy*cc
        self._Sys += yy*ss

        self._Rcs  += cs
        self._Rxcs += x*cs
        self._Rycs += y*cs
        self._Rxcc += x*cc
        self._Rycc += y*cc

        self._Rxxcs += xx*cs
        self._Ryycs += yy*cs
        self._Rxycs += xy*cs
        self._Rxycc += xy*cc
        self._Rxyss += xy*ss

        self.UpToDateMatrix = False

    def _Decay(self, newT, Tau):
        DeltaUpdate = newT - self.LastUpdate
        self.LastUpdate = newT
        Decay = np.e**(-DeltaUpdate/Tau)

        self.W *= Decay

        self._Es *= Decay
        self._Ed *= Decay

        self._x *= Decay
        self._y *= Decay
        self._r *= Decay

        self._Sc  *= Decay
        self._Sxc *= Decay
        self._Sxs *= Decay
        self._Syc *= Decay
        self._Sys *= Decay

        self._Rcs  *= Decay
        self._Rxcs *= Decay
        self._Rycs *= Decay
        self._Rxcc *= Decay
        self._Rycc *= Decay

        self._Rxxcs *= Decay
        self._Ryycs *= Decay
        self._Rxycs *= Decay
        self._Rxycc *= Decay
        self._Rxyss *= Decay

    def _GetInverseMatrix(self):
        if self._UpToDateMatrix:
            return
        self._M[0,0] = self._Sc
        self._M[1,1] = self.W - self._Sc
        self._M[2,2] = self._Sxs + self._Syc - 2*self._Rxycs
        self._M[3,3] = self._Sxc + self._Sys + 2*self._Rxycs

        self._M[1,0] = self._M[0,1] = self._Rcs
        self._M[2,0] = self._M[0,2] = self._Rxcs - self._Rycc
        self._M[3,0] = self._M[0,3] = self._Rxcc + self._Rycs

        self._M[2,1] = self._M[1,2] = (self._x - self._Rxcc) - self._Rycs
        self._M[3,1] = self._M[1,3] = self._Rxcs + (self._y - self._Rycc)
        self._M[3,2] = self._M[2,3] = self._Rxxcs + self._Rxyss - self._Rxycc - self._Ryycs
        
        self._InvM = np.linalg.inv(self._M)
        self._UpToDateMatrix = True

    def GetSpeed(self):
        self._GetInverseMatrix()
        return self._InvM.dot(self._Es)
    def GetDisplacement(self):
        self._GetInverseMatrix()
        return self._InvM.dot(self._Ed)

class TrackerClass:
    def __init__(self, TrackerManager, ID, InitialPosition):
        self.TM = TrackerManager
        self.ID = ID

        self.Estimator = FlowEstimator()

        self.Status = self.TM._STATUS_IDLE
        # Non status properties :
        self.ApertureIssue = False
        self.Lock = None
        self.LocksSaves = []

        self.Position = np.array(list(InitialPosition) + [0., 1.], dtype = float) # We now include all 4 parameters of the tracker in a single variable. Easier resolution and code compacity
        self.Speed    = np.array([0., 0., 0., 0.], dtype = float)
        # self.Position = [tx, ty, theta, s]
        self.Radius = self.TM._TrackerDiameter / 2
        self.SquaredRadius = self.HalfSize**2

        self.TimeConstant = self.TM._TrackerDefaultTimeConstant
        
        self.ProjectedEvents = []
        self.LastUpdate = 0.

        self.Activity = 0.
        self.ScalarCorrectionActivity = 0.
        self.ScalarCorrectionValue = 0.
        self.VectorialCorrectionValue = 0.

        self.AverageOrientation = np.array([0.,0.])
        self.AverageSpeedCorrectionOrientation = np.array([0., 0.])
        self.ApertureScalarEstimationByDelta = 0.
        self.ApertureScalarEstimationBySpeed = 0.
        self.ApertureVectorEstimationBySpeed = np.array([0., 0.])

        self.WeightedRotationalActivity = 0.

        self.SpeedError = np.array([0., 0.])
        self.RotSpeedError = 0.
        #self.PositionError = np.array([0., 0.])
        self._ComputeSpeedError = self.__class__.__dict__['_ComputeSpeedError' + self.TM._ComputeSpeedErrorMethod]

    def UpdateWithEvent(self, event):
        DeltaUpdate = event.timestamp - self.LastUpdate
        self.LastUpdate = event.timestamp
        Decay = np.e**(-DeltaUpdate / (self.TimeConstant * self.TM._TrackerTimeConstantRatioRemoval))

        self.Position += self.Speed * DeltaUpdate
        self.Rotation += self.RotSpeed * DeltaUpdate

        self.Activity *= Decay
        self.SpeedError *= Decay
        self.RotSpeedError *= Decay
        #self.PositionError *= Decay
        self.MeanPositionSummed *= Decay
        self.ScalarCorrectionActivity *= Decay
        self.ScalarCorrectionValue *= Decay
        self.VectorialCorrectionValue *= Decay
        self.ApertureScalarEstimationByDelta *= Decay
        self.ApertureScalarEstimationBySpeed *= Decay
        self.ApertureVectorEstimationBySpeed *= Decay
        self.AverageOrientation *= Decay
        self.AverageSpeedCorrectionOrientation *= Decay
        self.WeightedRotationalActivity *= Decay

        if self.Status == self.TM._STATUS_IDLE:
            return True
        if (self.Position < self.HalfSize).any() or (self.Position >= np.array(self.TM._LinkedMemory.STContext.shape[:2]) - self.HalfSize).any(): # out of bounds
            if self.Lock:
                self.Unlock('out of bounds')
            self.TM._KillTracker(self, event)
            return False
        if self.Activity < self.TM._TrackerDiameter * self.TM._TrackerMinDeathActivity: # Activity remaining too low
            if self.Lock:
                self.Unlock('low activity')
            self.TM._KillTracker(self, event)
            return False

        return True

    def MeanPosition(self):
        if self.Activity:
            return self.MeanPositionSummed / self.Activity
        else:
            return np.array([0., 0.])

    def RunEvent(self, event):
        if not self.UpdateWithEvent(event): # We update the position and activity. They do not depend on the event location. False return means the tracker has died : out of screen or activity too low (we should potentially add 1 for current event but shoud be marginal)
            return False # We also consider that even if the event may have matched, the tracker still dies. The activity increase of 1 is not enough

        RC, RS = np.cos(-self.Rotation), np.sin(-self.Rotation)
        Dx, Dy = event.location - self.Position
        Dx2, Dy2 = Dx**2, Dy**2
        if Dx2 + Dy2 > self.HS2:
            return False

        #RelativeLocation = CenteredLocation - self.MeanPosition()

        CurrentProjectedEvent = np.array([float(event.timestamp), Dx*RC + Dy*RS, Dy*RC - Dx*RS])
        SavedProjectedEvent = np.array(CurrentProjectedEvent) # Possibly error here, as reference to event.timestamp is lost

        self.Activity += 1
        self.MeanPositionSummed += CurrentProjectedEvent[1:]
        if self.TM._TrackerIgnoreBorderEvents and (abs(CurrentProjectedEvent[1:]) > (self.HalfSize - self.TM._ClosestEventProximity)).any(): # MOVED HERE (Limit event)
            if not self.Lock:
                self.ProjectedEvents += [SavedProjectedEvent]
            return True

        if self.Status == self.TM._STATUS_IDLE:
            self.ProjectedEvents += [SavedProjectedEvent]
            if self.Activity > self.TM._DetectorMinActivityForStart: # This StatusUpdate is put here, since it would be costly in computation to have it somewhere else and always be checked 
                self.Status = self.TM._STATUS_STABILIZING
                self.TM.StartTimes[self.ID] = event.timestamp
            return True

        # FROM HERE (Limit event)

        if not self.Lock:
            while len(self.ProjectedEvents) > 0 and self.LastUpdate - self.ProjectedEvents[0][0] >= self.TimeConstant * self.TM._TrackerTimeConstantRatioRemoval: # We look for the first event recent enough
                self.ProjectedEvents.pop(0)
        
        if self.Lock:
            CurrentProjectedEvent[0] = self.Lock.Time + self.TimeConstant
            UsedEventsList = self.Lock.Events
            self.ProjectedEvents = self.ProjectedEvents[:self.TM._TrackerLockingMaxRecoveryBuffer-1] + [SavedProjectedEvent]
        else:
            UsedEventsList = self.ProjectedEvents
            self.ProjectedEvents += [SavedProjectedEvent]

        ConsideredNeighbours = []
        for nEventReversed in range(len(UsedEventsList)):
            ArrayedEvent = UsedEventsList[-(nEventReversed+1)]
            if np.linalg.norm(CurrentProjectedEvent[1:] - ArrayedEvent[1:]) < self.TM._ClosestEventProximity:
                if self.Lock or CurrentProjectedEvent[0] - ArrayedEvent[0] > self.TM._ObjectEdgePropagationTC:
                    ConsideredNeighbours += [np.array(ArrayedEvent)]
                    if len(ConsideredNeighbours) == self.TM._MaxConsideredNeighbours:
                        break

        if len(ConsideredNeighbours) < self.TM._MinConsideredNeighbours:
            return True

        ConsideredNeighbours = np.array(ConsideredNeighbours)
        SpeedError, ProjectionError = self._ComputeSpeedError(self, CurrentProjectedEvent, ConsideredNeighbours)
        if self.Lock:
            SpeedMod = SpeedError / self.Lock.Activity ** self.TM._TrackerLockedActivityPower
            self.Position += self.TM._TrackerLockedPosModFactor * ProjectionError / self.Lock.Activity ** self.TM._TrackerLockedPosModDisplacementActivityPower
        else:
            SpeedMod = SpeedError / self.Activity ** self.TM._TrackerUnlockedActivityPower

        SpeedErrorNorm = np.linalg.norm(SpeedError)
        self.ScalarCorrectionActivity += 1

        self.SpeedError += SpeedError
        self.Speed += SpeedMod * self.TM._TrackerAccelerationFactor

        if self.TM._CorrectAxialRotation and Dx2 + Dy2 > self.HS2 * self.TM._MinRotCorrectionSizeRatio2 and SpeedErrorNorm > 0:
            n = SpeedError / SpeedErrorNorm
            EventLocalError = SpeedError - (n*(n*(self.SpeedError / self.ScalarCorrectionActivity)).sum())
            #RotSpeedError = -(CurrentProjectedEvent[0]*EventLocalError[1] - CurrentProjectedEvent[1]*EventLocalError[0]) / (Dx2 + Dy2)
            RotSpeedError = -(CurrentProjectedEvent[0]*EventLocalError[1] - CurrentProjectedEvent[1]*EventLocalError[0]) / self.HalfSize

            #Weight = np.e**(-self.HalfSize / (Dx2 + Dy2))
            Weight = 1.
            self.WeightedRotationalActivity += Weight
            if self.Lock:
                RotSpeedMod = RotSpeedError / self.Lock.Activity ** self.TM._TrackerLockedRotActivityPower
                #InducedRotation = (CurrentProjectedEvent[0]*ProjectionError[1] - CurrentProjectedEvent[1]*ProjectionError[0]) / (Dx2 + Dy2)
                InducedRotation = (CurrentProjectedEvent[0]*ProjectionError[1] - CurrentProjectedEvent[1]*ProjectionError[0]) / self.HalfSize
                self.Rotation += Weight * self.TM._TrackerLockedRotModFactor * InducedRotation / (self.WeightedRotationalActivity ** self.TM._TrackerLockedRotModDisplacementActivityPower)
            else:
                RotSpeedMod = Weight * RotSpeedError / (self.WeightedRotationalActivity ** self.TM._TrackerUnlockedActivityPower)
            self.RotSpeedError += RotSpeedError
            self.RotSpeed += Weight * RotSpeedMod * self.TM._TrackerAccelerationFactor

        Xm = (self.MeanPositionSummed / self.Activity)
        if self.TM._TrackerUsePositionMean and not self.Lock:
            DiffaXm = abs(Xm) - self.TM._TrackerMeanPositionRelativeRadiusTolerance * self.TM._TrackerDiameter
            if (DiffaXm > 0).any():
                self.Position += self.TM._TrackerMeanPositionAccelerationFactor * DiffaXm * np.sign(Xm) / self.Activity

        DeltaPos = CurrentProjectedEvent[1:] - Xm
        if DeltaPos[0] < 0:
            DeltaPos *= -1
        nDelta = np.linalg.norm(DeltaPos)
        if nDelta > 0:
            DeltaPos = DeltaPos / nDelta
            self.AverageOrientation += DeltaPos
            nOrientation = np.linalg.norm(self.AverageOrientation)
            
            self.ApertureScalarEstimationByDelta += ((DeltaPos * self.AverageOrientation).sum() / (nOrientation))**2

        SpeedNorm = np.linalg.norm(self.Speed)
        if SpeedNorm > 0:
            self.TimeConstant = min(1./SpeedNorm, self.TM._TrackerDefaultTimeConstant)
            if SpeedErrorNorm > 0:
                ScalarValue = (SpeedError * self.Speed).sum() / (SpeedErrorNorm * SpeedNorm)
                self.ScalarCorrectionValue += ScalarValue
                VectorialValue = (self.Speed[0]*SpeedError[1] - self.Speed[1]*SpeedError[0]) / (SpeedErrorNorm * SpeedNorm)
                self.VectorialCorrectionValue += VectorialValue

        else:
            self.TimeConstant = self.TM._TrackerDefaultTimeConstant

        if SpeedErrorNorm > 0:
            SpeedModOrientation = SpeedError / SpeedErrorNorm
            RotatedSpeedModOrientation = np.array([SpeedModOrientation[0]**2 - SpeedModOrientation[1]**2, 2*SpeedModOrientation[0]*SpeedModOrientation[1]])
            if np.linalg.norm(RotatedSpeedModOrientation) > 1.05:
                self.TM.LogWarning("{0}, {1}".format(SpeedError, RotatedSpeedModOrientation))
            if SpeedMod[0] < 0:
                SpeedModOrientation = - SpeedModOrientation
            self.AverageSpeedCorrectionOrientation += SpeedModOrientation
            self.ApertureVectorEstimationBySpeed += RotatedSpeedModOrientation

            self.ApertureScalarEstimationBySpeed += ((self.AverageSpeedCorrectionOrientation * SpeedModOrientation).sum() / np.linalg.norm(self.AverageSpeedCorrectionOrientation))**2


        self.IgnoredComputationReasons += [None]
        self.ComputeCurrentStatus(event.timestamp)
        return True

    def ComputeCurrentStatus(self, t): # Cannot be called when DEAD or IDLE. Thus, we first check evolutions for aperture issue and lock properties, then update the status
        if not self.ApertureIssue and (self.ApertureScalarEstimationByDelta > self.ScalarCorrectionActivity * (self.TM._TrackerApertureIssueByDeltaThreshold + self.TM._TrackerApertureIssueByDeltaHysteresis) or np.linalg.norm(self.ApertureVectorEstimationBySpeed) > self.ScalarCorrectionActivity * (self.TM._TrackerApertureIssueBySpeedThreshold + self.TM._TrackerApertureIssueBySpeedHysteresis)):
            self.ApertureIssue = True
            Reason = "aperture issue"
            if self.ApertureScalarEstimationByDelta > self.ScalarCorrectionActivity * (self.TM._TrackerApertureIssueByDeltaThreshold + self.TM._TrackerApertureIssueByDeltaHysteresis):
                Reason = Reason + ' (Delta = {0:.3f})'.format(self.ApertureScalarEstimationByDelta/self.ScalarCorrectionActivity)
            else:
                Reason = Reason + ' (Speed = {0:.3f})'.format(np.linalg.norm(self.ApertureVectorEstimationBySpeed)/self.ScalarCorrectionActivity)
        elif self.ApertureIssue and (self.ApertureScalarEstimationByDelta < self.ScalarCorrectionActivity * (self.TM._TrackerApertureIssueByDeltaThreshold - self.TM._TrackerApertureIssueByDeltaHysteresis) and np.linalg.norm(self.ApertureVectorEstimationBySpeed) < self.ScalarCorrectionActivity * (self.TM._TrackerApertureIssueBySpeedThreshold - self.TM._TrackerApertureIssueBySpeedHysteresis)):
            self.ApertureIssue = False

        CanBeLocked = True
        if self.ApertureIssue:
            CanBeLocked = False
        elif self.Activity > self.TM._TrackerLockMaxRelativeActivity * self.TM._TrackerDiameter:
            CanBeLocked = False
            Reason = "excessive absolute activity"
        elif self.ScalarCorrectionActivity < self.TM._TrackerLockedRelativeCorrectionsFailures * self.Activity:
            CanBeLocked = False
            Reason = "unsufficient number of points matching"

        if self.Lock and not CanBeLocked:
            self.Unlock(Reason)
            return None

        if self.Status == self.TM._STATUS_CONVERGED:
            if abs(self.ScalarCorrectionValue) > self.ScalarCorrectionActivity * (self.TM._TrackerConvergenceThreshold + self.TM._TrackerConvergenceHysteresis): # Part where we downgrade to stabilizing, when the corrections are too great
                if self.Lock:
                    if not self.TM._TrackerLockedCanHardCorrect:
                        self.Unlock("excessive correction")
                        self.Status = self.TM._STATUS_STABILIZING
                else:
                    self.Status = self.TM._STATUS_STABILIZING
                return None
            if not self.Lock and CanBeLocked and self.TM._TrackerAllowShapeLock:
                self.Lock = LockClass(t, self.Activity, list(self.ProjectedEvents))
                self.LocksSaves += [LockClass(t, self.Activity, list(self.ProjectedEvents))]
                
                self.TM._OnTrackerLock(self)

                self.TM.Log("Tracker {0} has locked".format(self.ID))
                return None # We assume we have checked everything here
        elif self.Status == self.TM._STATUS_STABILIZING:
            if abs(self.ScalarCorrectionValue) < self.ScalarCorrectionActivity * (self.TM._TrackerConvergenceThreshold - self.TM._TrackerConvergenceHysteresis):
                if self.ScalarCorrectionActivity >= self.TM._TrackerLockedRelativeCorrectionsFailures * self.Activity:
                    self.Status = self.TM._STATUS_CONVERGED
                    return None

    def Unlock(self, Reason):
        self.LocksSaves[-1].ReleaseTime = self.TM._LinkedMemory.LastEvent.timestamp
        self.Lock = None
        self.TM.FeatureManager.RemoveLock(self)
        self.TM.LogWarning("Tracker {0} was released ({1})".format(self.ID, Reason))

    def _ComputeSpeedErrorLinearMean(self, CurrentProjectedEvent, ConsideredNeighbours):
        MeanPoint = np.array([(ConsideredNeighbours[:,1]*ConsideredNeighbours[:,0]).sum(), (ConsideredNeighbours[:,2]*ConsideredNeighbours[:,0]).sum()]) / ConsideredNeighbours[:,0].sum()
        MeanTS = ConsideredNeighbours[:,0].mean()

        DeltaPos = (CurrentProjectedEvent[1:] - MeanPoint)
        SpeedError = DeltaPos / (CurrentProjectedEvent[0] - MeanTS)
        return SpeedError, DeltaPos

    def _ComputeSpeedErrorExponentialMean(self, CurrentProjectedEvent, ConsideredNeighbours):
        Weights = np.e**((ConsideredNeighbours[:,0] - CurrentProjectedEvent[0]) / (self.TimeConstant * self.TM._TrackerTimeConstantRatioRemoval))
        MeanPoint = np.array([(ConsideredNeighbours[:,1] * Weights).sum(), (ConsideredNeighbours[:,2] * Weights).sum()]) / Weights.sum()
        MeanTS = (ConsideredNeighbours[:,0] * Weights).sum() / Weights.sum()

        DeltaPos = (CurrentProjectedEvent[1:] - MeanPoint)
        SpeedError = DeltaPos / (CurrentProjectedEvent[0] - MeanTS)
        return SpeedError, DeltaPos

    def _ComputeSpeedErrorTwoPointsProjection(self, CurrentProjectedEvent, ConsideredNeighbours):

        Epsilon12 = ConsideredNeighbours[0,1:] - ConsideredNeighbours[1,1:]
        N2 = (Epsilon12 * Epsilon12).sum()
        if N2 <= 1.1:
            return np.array([0., 0.])
        Epsilon02 = CurrentProjectedEvent[1:] - ConsideredNeighbours[1,1:]

        Lambda = (Epsilon02 * Epsilon12).sum() / (Epsilon12 * Epsilon12).sum()
        #if Lambda < 0 or Lambda > 1: # Here, we check that the point newly arrived is inside the considered segment.
        #    return np.array([0., 0.])
        DeltaP = Epsilon02 - Epsilon12 * Lambda
        MeanTS = ConsideredNeighbours[:,0].mean()

        SpeedError = DeltaP / (CurrentProjectedEvent[0] - MeanTS)
        return SpeedError, DeltaP

from pycpd import rigid_registration as registration

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

from scipy import misc 

class GTMakerClass:
    def __init__(self, SpeedTracker, LookupRadius = 2):
        self.SpeedTracker = SpeedTracker
        self.WindowSize = int(SpeedTracker._TrackerDiameter * 1.5)
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
        print("Lookup table loaded from file")

    def _CreateLookupTableSnaps(self):
        self.ImagesLookupTable = {'t': [], 'filename': []}
        for nt, Duo in enumerate(self.SpeedTracker._LinkedMemory.Snapshots):
            self.ImagesLookupTable['t'] += [Duo[0]]
            self.ImagesLookupTable['filename'] += [nt]
        print("Lookup table created from snapshots")

    def GetTrackerPositionAt(self, t, TrackerID):
        self.GetCurrentFrameIndex(t)
        if not TrackerID in list(self.GTPositions[self.ImagesLookupTable['filename'][self.CurrentFrameIndex]].keys()):
            return None
        if self.CurrentFrameIndex + 1 >= len(self.ImagesLookupTable['filename']) or not self.ImagesLookupTable['filename'][self.CurrentFrameIndex + 1] in list(self.GTPositions.keys()) or not TrackerID in list(self.GTPositions[self.ImagesLookupTable['filename'][self.CurrentFrameIndex + 1]].keys()):
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
            if ImageFile in list(self.GTPositions.keys()):
                if TrackerID in list(self.GTPositions[ImageFile].keys()):
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
                    None
                    #break
            return Diffs, [tst[it] for it, i in CommonIndexes]
        else:
            return [], []

    def GenerateGroundTruth(self, DataFolder = None, tMax = None, BinDt = 0.1, CheckWithGlobalSpeed = False):
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
        PreviousT = None
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
                if StatusTuple[1] == self.SpeedTracker._PROPERTY_LOCKED or TrackerID in list(self.ImagesRefDict.keys()):
                    if TrackerID in list(self.ImagesRefDict.keys()):
                        NewPosition = self._FindMatch(TrackerID, self.GTPositions[PreviousFrame][TrackerID] + self.Deltas[TrackerID])
                        if NewPosition is None:
                            del self.ImagesRefDict[TrackerID]
                        else:
                            if self.DataFolder or not CheckWithGlobalSpeed:
                                self.Deltas[TrackerID] = NewPosition - self.GTPositions[PreviousFrame][TrackerID]
                                self.GTPositions[ImageFile][TrackerID] = np.array(NewPosition)
                            else:
                                if not PreviousT is None and self.SpeedTracker.FeatureManager.SpeedTranslationHistory:
                                    V = (NewPosition - self.GTPositions[PreviousFrame][TrackerID]) / (t - PreviousT)
                                    Vth = (self.SpeedTracker.FeatureManager.SpeedTranslationHistory[ImageFile])
                                    if np.linalg.norm(V - Vth) < np.linalg.norm(Vth) * 10. and np.linalg.norm(V - Vth) > np.linalg.norm(Vth) * 0.1:
                                        self.Deltas[TrackerID] = NewPosition - self.GTPositions[PreviousFrame][TrackerID]
                                        self.GTPositions[ImageFile][TrackerID] = np.array(NewPosition)
                                    else:
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
                    if TrackerID in list(self.ImagesRefDict.keys()):
                        del self.ImagesRefDict[TrackerID]
            self.MinActiveTrackerID += MinActiveOffset
            PreviousFrame = ImageFile
            PreviousT = t

class Plotter:
    def __init__(self, TM):
        self.TM = TM
        self.StatusesColors = {'Dead' : 'k', 'Converged': 'g', 'Idle': 'r', 'Stabilizing': 'b'}
        self.PropertiesLineStyles = {'None': '-.', 'Aperture issue': '--', 'Locked' : '-'}

    def CreateTrackingShot(self, TrackerIDs = None, IgnoreTrackerIDs = [], SnapshotNumber = 0, BinDt = 0.005, ax_given = None, cmap = None, addTrackersIDsFontSize = 0, removeTicks = True, add_ts = True, DisplayedStatuses = ['Stabilizing', 'Converged'], DisplayedProperties = ['Aperture issue', 'Locked', 'None'], RemoveNullSpeedTrackers = 0, VirtualPoint = None, GT = None, TrailDt = 0, TrailWidth = 2, ForcedColors = {}):
        if type(ForcedColors) == str and ForcedColors == 'cycle':
            Colors = ['#14607A', '#1D81A2', '#04BB9F', '#09A785', '#17A1CD', '#38F3BC', '#FFB560', '#FF4839', '#C71D1D', '#9E1F63', '#FFD603', '#F05929']
            ForcedColors = {ID : Colors[ID%len(Colors)] for ID in range(len(self.TM.Trackers))}

        S = self.TM
        if TrackerIDs is None:
            TrackerIDs = [ID for ID in range(len(S.Trackers)) if ID not in IgnoreTrackerIDs]
        else:
            for ID in IgnoreTrackerIDs:
                if ID in TrackerIDs:
                    raise Exception("Asking to ignore tracker {0} while also asking for it. Aborting".format(ID))
    
        if ax_given is None:
            f, ax = plt.subplots(1,1)
        else:
            ax = ax_given
        Map = S._LinkedMemory.Snapshots[SnapshotNumber][1]
        t = S._LinkedMemory.Snapshots[SnapshotNumber][0]
        Mask = (Map.max(axis = 2) > t - BinDt) * Map.max(axis = 2)
        FinalMap = (Map[:,:,0] == Mask) - 1 * (Map[:,:,1] == Mask)
        if cmap is None:
            ax.imshow(np.transpose(FinalMap), origin = 'lower') 
        else:
            ax.imshow(np.transpose(FinalMap), origin = 'lower', cmap = plt.get_cmap(cmap))
    
        def TrackerValidityCheck(TrackerID, SnapshotNumber):
            NullAnswer = (None, None, None)
            try:
                StatusValue = S.TrackersStatuses[SnapshotNumber][TrackerID][0]
            except:
                return NullAnswer
            PropertyValue = S.TrackersStatuses[SnapshotNumber][TrackerID][1]
            if S._StatusesNames[StatusValue] not in DisplayedStatuses:
                return NullAnswer
            else:
                TrackerColor = self.StatusesColors[S._StatusesNames[StatusValue]]
            if S._PropertiesNames[PropertyValue] not in DisplayedProperties:
                return NullAnswer
            else:
                TrackerLineStyle = self.PropertiesLineStyles[S._PropertiesNames[PropertyValue]]
            try:
                HalfSize = self.TM._TrackerDiameter / 2
                Location = np.array(self.TM.TrackersPositionsHistory[SnapshotNumber][TrackerID])
                Orientation = self.TM.TrackersRotationsHistory[SnapshotNumber][TrackerID]
                M = np.array([[np.cos(Orientation), np.sin(Orientation)], [-np.sin(Orientation), np.cos(Orientation)]])
                Box = [Location + HalfSize * M.dot(np.array([1, 1])), 
                       Location + HalfSize * M.dot(np.array([1, -1])),
                       Location + HalfSize * M.dot(np.array([-1, -1])),
                       Location + HalfSize * M.dot(np.array([-1, 1]))]
            except:
                return NullAnswer
            
            if RemoveNullSpeedTrackers:
                if np.linalg.norm(S.TrackersSpeedsHistory[SnapshotNumber][TrackerID]) < RemoveNullSpeedTrackers:
                    return NullAnswer
            return (TrackerColor, TrackerLineStyle, Box)

        if add_ts:
            ax.set_title("t = {0:.3f}".format(t))
        for n_tracker, TrackerID in enumerate(TrackerIDs):
            TrackerColor, TrackerLineStyle, Box = TrackerValidityCheck(TrackerID, SnapshotNumber)
            if not TrackerColor is None:
                if TrackerID in ForcedColors.keys():
                    TrackerColor = ForcedColors[TrackerID]
                ax.plot([Box[0][0], Box[1][0]], [Box[0][1], Box[1][1]], TrackerColor, ls = TrackerLineStyle)
                ax.plot([Box[1][0], Box[2][0]], [Box[1][1], Box[2][1]], TrackerColor, ls = TrackerLineStyle)
                ax.plot([Box[2][0], Box[3][0]], [Box[2][1], Box[3][1]], TrackerColor, ls = TrackerLineStyle)
                ax.plot([Box[3][0], Box[0][0]], [Box[3][1], Box[0][1]], TrackerColor, ls = TrackerLineStyle)

                if not GT is None:
                    GTPos = GT.GetTrackerPositionAt(t, TrackerID)
                    if not GTPos is None:
                        ax.plot(GTPos[0], GTPos[1], color = TrackerColor, marker = 'v')
                        ax.plot([GTPos[0], Box[0]], [GTPos[1], Box[1]], color = TrackerColor, lw = 0.3)
                        ax.plot([GTPos[0], Box[2]], [GTPos[1], Box[3]], color = TrackerColor, lw = 0.3)

                if addTrackersIDsFontSize:
                    ax.text(Box[0][0] + 5, (Box[0][1] + Box[1][1])/2, '{0}'.format(TrackerID), color = TrackerColor, fontsize = addTrackersIDsFontSize)

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
                    if TrackerID in ForcedColors.keys():
                        TrackerColor = ForcedColors[TrackerID]
                    for DataSerie in Data:
                        ax.plot(np.array(DataSerie)[:,0], np.array(DataSerie)[:,1], TrackerColor, ls = TrackerLineStyle, lw = TrailWidth)


        if not VirtualPoint is None:
            
            Center = np.array(S._LinkedMemory.STContext.shape[:2], dtype = float) / 2
            VirtualPoint = [Center + np.array(S.FeatureManager.TranslationHistory[SnapshotNumber]), np.array([0., 0.])]

            if 0 < VirtualPoint[0][0] - VirtualPoint[1][0] and VirtualPoint[0][0] + VirtualPoint[1][0] < self.TM._LinkedMemory.STContext.shape[0] and 0 < VirtualPoint[0][1] - VirtualPoint[1][1] and VirtualPoint[0][1] + VirtualPoint[1][1]< self.TM._LinkedMemory.STContext.shape[1]:
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
    
    def GenerateTrackingGif(self, TrackerIDs = None, IgnoreTrackerIDs = [], AddVirtualPoint = False, SnapRatio = 1, tMin = 0., tMax = np.inf, Folder = '/home/dardelet/Pictures/GIFs/AutoGeneratedTracking/', BinDt = 0.005, add_ts = True, cmap = None, DoGif = True, addTrackersIDsFontSize = 0, DisplayedStatuses = ['Stabilizing', 'Converged'], DisplayedProperties = ['Aperture issue', 'Locked', 'None'], RemoveNullSpeedTrackers = 0, NoRemoval = False, GT = None, TrailDt = 0, TrailWidth = 2, ForcedColors = {}):
        if BinDt is None:
            BinDt = self.TM._SnapDt

        if AddVirtualPoint:
            VirtualPoint = [np.array(self.TM._LinkedMemory.STContext.shape[:2], dtype = float) / 2, np.array([0., 0.])]
            LastVPUpdate = None
            Offset_Start = 0
            NUpdates = 0
        else:
            VirtualPoint = None
    
        Snaps_IDs = [snap_id for snap_id in range(len(self.TM.TrackersPositionsHistoryTs)) if (snap_id % SnapRatio == 0 and tMin <= self.TM.TrackersPositionsHistoryTs[snap_id] and self.TM.TrackersPositionsHistoryTs[snap_id] <= tMax)]
        if not NoRemoval:
            os.system('rm '+ Folder + '*.png')
        self.TM.Log("Generating {0} png frames on {1} possible ones.".format(len(Snaps_IDs), len(self.TM.TrackersPositionsHistoryTs)))
        f = plt.figure(figsize = (16,9), dpi = 100)
        ax = f.add_subplot(1,1,1)
        for snap_id in Snaps_IDs:
            self.TM.Log(" > {0}/{1}".format(snap_id/SnapRatio + 1, len(Snaps_IDs)))

            if AddVirtualPoint:
                if LastVPUpdate is None:
                    LastVPUpdate = self.TM.TrackersPositionsHistoryTs[snap_id]
                else:
                    Delta = self.TM.TrackersPositionsHistoryTs[snap_id] - LastVPUpdate
                    LastVPUpdate = self.TM.TrackersPositionsHistoryTs[snap_id]
                    AddOffset, Vx, Vy, SVx, SVy = self._GetSnapSceneSpeed(snap_id, Offset_Start, ['Converged'], ['Locked'])
                    if AddOffset:
                        Offset_Start += AddOffset
                    if not Vx is None:
                        NUpdates += 1
                        VirtualPoint[0] += np.array([Vx, Vy]) * Delta
                        VirtualPoint[1] += np.array([SVx, SVy]) / (2 * np.sqrt(NUpdates)) * Delta

            if self.TM.TrackersPositionsHistoryTs[snap_id] > tMax:
                break
    
            self.CreateTrackingShot(TrackerIDs, IgnoreTrackerIDs, snap_id, BinDt, ax, cmap, addTrackersIDsFontSize, True, add_ts, DisplayedStatuses = DisplayedStatuses, DisplayedProperties = DisplayedProperties, RemoveNullSpeedTrackers = RemoveNullSpeedTrackers, VirtualPoint = VirtualPoint, GT = GT, TrailDt = TrailDt, TrailWidth = TrailWidth, ForcedColors = ForcedColors)
    
            f.savefig(Folder + 't_{0:05d}.png'.format(snap_id))
            ax.cla()
        self.TM.Log(" > Done.          ")
        plt.close(f.number)
    
        if DoGif:
            GifFile = 'tracking_'+datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')+'.gif'
            command = 'convert -delay {0} -loop 0 '.format(int(1000*self.TM._SnapDt))+Folder+'*.png ' + Folder + GifFile
            print("Generating gif. (command : " + command + " )")
            os.system(command)
            print("GIF generated : {0}".format(GifFile))
            os.system('kde-open ' + Folder + GifFile)
    
            ans = input('Rate this result (1-nice/0-bad/(d)elete) : ')
            if '->' in ans:
                ans, name = ans.split('->')
                ans = ans.strip()
                name = name.strip()
                if '.gif' in name:
                    name = name.split('.gif')[0]
                NewGifFile = name + '_' + GifFile
            else:
                NewGifFile = GifFile
            if '1' in ans.lower() or 'nice' in ans.lower():
                os.system('mv ' + Folder + GifFile + ' ' + Folder + 'Nice/' + NewGifFile)
                print("Moving gif file to Nice folder.")
            elif '0' in ans.lower() or 'bad' in ans.lower():
                os.system('mv ' + Folder + GifFile + ' ' + Folder + 'Bad/' + NewGifFile)
                print("Moving gif file to Bad folder.")
            elif len(ans) > 0 and ans.lower()[0] == 'd' or 'delete' in ans.lower():
                os.system('rm ' + Folder + GifFile)
                print("Deleted Gif file. RIP.")
            else:
                os.system('mv ' + Folder + GifFile + ' ' + Folder + 'Meh/' + NewGifFile)
                print("Moving gif file to Meh folder.")
    
    def SceneSpeedAnalysis(self, tMin = 0., tMax = np.inf, UsedStatuses = ['Converged'], UsedProperties = ['Locked']):
        f, axs = plt.subplots(2,1)
        axs[0].set_title('Speed Vx')
        axs[1].set_title('Speed Vy')

        S = self.TM
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
        S = self.TM
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
            if S._StatusesNames[S.TrackersStatuses[SnapID][Offset_Start + nTracker][0]] not in UsedStatuses:
                continue
            if S._PropertiesNames[S.TrackersStatuses[SnapID][Offset_Start + nTracker][1]] not in UsedProperties:
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
        S = self.TM
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

        while tStart - Tracker.PEH[nEvent][0] > S._TrackerTimeConstantRatioRemoval * TC:
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

            DrawnPoints[tuple(Tracker.PEH[nEvent])] = ProjectedMapAx.plot(Tracker.PEH[nEvent][1], Tracker.PEH[nEvent][2], marker = 'o', color = 'k', alpha = 1. - max(0., min(1., (tStart - Tracker.PEH[nEvent][0]) / (S._TrackerTimeConstantRatioRemoval * TC))))[0]

            nEvent += 1
            last_t = Tracker.PEH[nEvent][0]
        print("Finished initialization at nEvent = {0}, nSnap = {1}".format(nEvent, nSnap))
        SnapShown = max(0, nSnap-1)
        x, y = int(S.TrackersPositionsHistory[SnapShown][Tracker.ID][0]), int(S.TrackersPositionsHistory[SnapShown][Tracker.ID][1])
        Map = S._LinkedMemory.Snapshots[SnapShown][1][int(x - Tracker.HalfSize):int(x + Tracker.HalfSize + 1), int(y - Tracker.HalfSize):int(y + Tracker.HalfSize + 1)]
        t = S._LinkedMemory.Snapshots[SnapShown][0]
        FinalMap = np.e**(-(t - Map.max(axis = 2))/(TC * S._TrackerTimeConstantRatioRemoval)) * (2*Map.argmax(axis = 2) - 1)

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
                    PAx.plot(S.TrackersPositionsHistory[nSnap][Tracker.ID][0], S.TrackersPositionsHistory[nSnap][Tracker.ID][1], 'x', color = self.StatusesColors[S._StatusesNames[S.TrackersStatuses[nSnap][Tracker.ID][0]]])

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
                    Map = S._LinkedMemory.Snapshots[SnapShown][1][x - Tracker.HalfSize:x + Tracker.HalfSize, y - Tracker.HalfSize:y + Tracker.HalfSize]
                    t = S._LinkedMemory.Snapshots[SnapShown][0]
                    FinalMap = np.e**(-(t - Map.max(axis = 2))/(TC * S._TrackerTimeConstantRatioRemoval)) * (2*Map.argmax(axis = 2) - 1)
    
                    Image.set_data(np.transpose(FinalMap))
                    ScreenAx.set_title("Snap {0}, t = {1:.3f}".format(SnapShown, t))
                    PAx.set_title('Tracker {1} : Position - {0}'.format(S._StatusesNames[S.TrackersStatuses[SnapShown][Tracker.ID][0]], Tracker.ID) + (S.TrackersStatuses[SnapShown][Tracker.ID][1] != S._PROPERTY_NONE) * ' - {0}'.format(S._PropertiesNames[S.TrackersStatuses[SnapShown][Tracker.ID][1]]))

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
                    if Event[0] < t - TC * S._TrackerTimeConstantRatioRemoval:
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
                    Alpha = 1. - max(0., min(1., (t - Event[0]) / (S._TrackerTimeConstantRatioRemoval * TC)))
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

    def _CreateTrackingPicture(ax, snap_id, F, S, SpeedDuos, BoxColors, BinDt, cmap, add_timestamp_title, add_Feature_label_Fontsize = 0, titleSize = 15, lw = 1, BorderMargin = 5, FeatureNumerotation = 'computer', FeatureInitialOriginKept = True, IncludeSpeedError = 0, MinDMValue = 0, PolaritySeparation = True, Trail = 0, TrailWidth = 2, TrailPointList = [], DrawBox = True, AddedInfos = [], CurrentPositionMarker = None):
        Map = np.array(F.Mem.Snapshots[snap_id][1])
        Mask = (Map.max(axis = 2) > Map.max()-BinDt) * Map.max(axis = 2)
        FinalMap = (Map[:,:,0] == Mask) + (1 - 2*int(PolaritySeparation)) * (Map[:,:,1] == Mask)
        if cmap is None:
            ax.imshow(np.transpose(FinalMap), origin = 'lower') 
        else:
            ax.imshow(np.transpose(FinalMap), origin = 'lower', cmap = plt.get_cmap(cmap))
    
        for n_speed, duo in enumerate(SpeedDuos):
            speed_id, zone_id = duo
            try:
                Box = [S.OWAPT[speed_id][0] + S.DisplacementSnaps[snap_id][speed_id][0], S.OWAPT[speed_id][1] + S.DisplacementSnaps[snap_id][speed_id][1], S.OWAPT[speed_id][2] + S.DisplacementSnaps[snap_id][speed_id][0], S.OWAPT[speed_id][3] + S.DisplacementSnaps[snap_id][speed_id][1]]
            except:
                continue
            if (np.array(Box) < BorderMargin).any() or Box[2] >= F.Mem.Snapshots[snap_id][1].shape[0] - BorderMargin or Box[3] >= F.Mem.Snapshots[snap_id][1].shape[1] - BorderMargin:
                continue
            color = BoxColors[zone_id]
            if Trail > 0:
                while len(TrailPointList[n_speed]) and S.TsSnaps[snap_id] - TrailPointList[n_speed][0][0] > Trail:
                    TrailPointList[n_speed].pop(0)
                TrailPointList[n_speed] += [[S.TsSnaps[snap_id], (Box[0] + Box[2])/2, (Box[1] + Box[3])/2]]
            if IncludeSpeedError:
                Error = S.SpeedErrorSnaps[snap_id][speed_id]
                t = S.TsSnaps[snap_id]
                Speed = np.array([0., 0.])
                for SpeedChange in S.SpeedsChangesHistory[speed_id]:
                    if t < SpeedChange[0]:
                        break
                    Speed = SpeedChange[1]
                AlphaValue = max(0., 1. - IncludeSpeedError * np.linalg.norm(Error)/np.linalg.norm(Speed))
                if S.DMReferences[speed_id] is None:
                    AlphaValue = min(AlphaValue,  max(0., S.DMSnaps[snap_id][speed_id].sum()/MinDMValue))
                else:
                    AlphaValue = min(AlphaValue,  max(0., S.DMSnaps[snap_id][speed_id].sum()/S.DMReferences[speed_id].sum()))
            else:
                AlphaValue = 1
            if DrawBox:
                ax.plot([Box[0], Box[0]], [Box[1], Box[3]], c = color, lw = lw, alpha = AlphaValue)
                ax.plot([Box[0], Box[2]], [Box[3], Box[3]], c = color, lw = lw, alpha = AlphaValue)
                ax.plot([Box[2], Box[2]], [Box[3], Box[1]], c = color, lw = lw, alpha = AlphaValue)
                ax.plot([Box[2], Box[0]], [Box[1], Box[1]], c = color, lw = lw, alpha = AlphaValue)
            if not CurrentPositionMarker is None:
                ax.plot((Box[0] + Box[2])/2, (Box[1] + Box[3])/2, c = color, alpha = AlphaValue, marker = CurrentPositionMarker[0], markersize = CurrentPositionMarker[1])
            if Trail > 0:
                TrailPointList[n_speed][-1] += [AlphaValue]
                for nSegment in range(len(TrailPointList[n_speed])-1):
                    ax.plot([TrailPointList[n_speed][nSegment][1], TrailPointList[n_speed][nSegment+1][1]], [TrailPointList[n_speed][nSegment][2], TrailPointList[n_speed][nSegment+1][2]], c = color, alpha = TrailPointList[n_speed][nSegment][3], lw = TrailWidth)
            if add_Feature_label_Fontsize:
                ax.text(Box[2] + 5, Box[1] + (Box[3] - Box[1])*0.4, ('zone_id' in AddedInfos)*'Feature {0}'.format((FeatureInitialOriginKept * zone_id + (not FeatureInitialOriginKept) * n_speed) + (FeatureNumerotation == 'human')) + ', '*('zone_id' in AddedInfos and 'speed_id' in AddedInfos) + ('speed_id' in AddedInfos)*'ID = {0}'.format(speed_id), color = BoxColors[zone_id], fontsize = add_Feature_label_Fontsize)
        if add_timestamp_title:
            ax.set_title("t = {0:.2f}s".format(S.TsSnaps[snap_id]), fontsize = titleSize)

# Shape Analysis section. WIP
    
    def ShapePointsReduction(self, TrackerID, NFinal):
        Events = list(self.TM.Trackers[TrackerID].LocksSaves[0].Events)
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
        axs[0].plot(np.array(self.TM.Trackers[TrackerID].LocksSaves[0].Events)[:,1], np.array(self.TM.Trackers[TrackerID].LocksSaves[0].Events)[:,2], 'or')
        axs[0].plot(np.array(ReducedEvents)[:,1], np.array(ReducedEvents)[:,2], 'ob')
        Angles = CreateCenteredAnglesHistogram(ReducedEvents)
        Distances = CreateRelativeDistancesHistogram(ReducedEvents, M)
        AnglesOccurences, AnglesValues, ax = axs[1].hist(Angles, bins=nBins, range=(0, np.pi*2))
        DistancesOccurences, DistancesValues, ax = axs[2].hist(Distances, bins=nBins)
        return AnglesOccurences, AnglesValues, DistancesOccurences, DistancesValues, axs

    def CompareTrackersShapesHistograms(self, ShowBests = (0, 10), IDRange = [0, None], nReducedPoints = 50, nBins = 20, PlotScenes = False, PlotDuos = False, Unique = True):
        if IDRange[1] is None:
            IDRange[1] = len(self.TM.Trackers)

        TrackersAnglesHistDict = {} 
        TrackersDistancesHistDict = {} 
        f, axs = plt.subplots(3,1)
        for TrackerID in range(IDRange[0], IDRange[1]):
            Tracker = self.TM.Trackers[TrackerID] 
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
                SnapID_i = ((np.array(self.TM.TrackersPositionsHistoryTs) - self.TM.Trackers[i].LocksSaves[0].Time) > 0).tolist().index(True)
                self.CreateTrackingShot(TrackerIDs = [i], SnapshotNumber = SnapID_i, BinDt = 0.005, ax_given = axs_scenes[0, BestDuoID-ShowBests[0]], cmap = 'binary', addTrackersIDsFontSize = 10, removeTicks = True, add_ts = True, DisplayedStatuses = ['Stabilizing', 'Converged'], DisplayedProperties = ['Aperture issue', 'Locked', 'None'], RemoveNullSpeedTrackers = 0, VirtualPoint = None)
                SnapID_j = ((np.array(self.TM.TrackersPositionsHistoryTs) - self.TM.Trackers[j].LocksSaves[0].Time) > 0).tolist().index(True)
                self.CreateTrackingShot(TrackerIDs = [j], SnapshotNumber = SnapID_j, BinDt = 0.005, ax_given = axs_scenes[1, BestDuoID-ShowBests[0]], cmap = 'binary', addTrackersIDsFontSize = 10, removeTicks = True, add_ts = True, DisplayedStatuses = ['Stabilizing', 'Converged'], DisplayedProperties = ['Aperture issue', 'Locked', 'None'], RemoveNullSpeedTrackers = 0, VirtualPoint = None)
            BestDuoID += 1

        return OutDuos

    def CompareTrackersShapesAngularVariations(self, ShowBestsFrom = 0, nShowed = 5, IDRange = [0, None], nReducedPoints = None, NLinesHoriz = 10, PlotScenes = False, PlotDuos = False, Unique = True, MaxShift = 0, AddLines = False):
        if IDRange[1] is None:
            IDRange[1] = len(self.TM.Trackers)
        TrackersAngularVariancesDict = {} 
        LinesDict = {}

        for TrackerID in range(IDRange[0], IDRange[1]):
            Tracker = self.TM.Trackers[TrackerID] 
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
                axs[0,0].plot(np.array(self.TM.Trackers[TrackerID_i].LocksSaves[0].Events)[:,1], np.array(self.TM.Trackers[TrackerID_i].LocksSaves[0].Events)[:,2], 'or')
                axs[0,0].plot(np.array(ReducedEvents)[:,1], np.array(ReducedEvents)[:,2], 'ob')
                try:
                    axs[0,1].set_title('Angles Variations : {0:.2f}'.format(VariancesDistanceMatrix[ListDuos[BestDuoID][0], ListDuos[BestDuoID][1]]))
                    axs[1,1].set_title('Shift : {0:.2f}deg'.format(180 / np.pi *AssociatedShifts[ListDuos[BestDuoID][0], ListDuos[BestDuoID][1]]))
                except:
                    print(ListDuos[BestDuoID][0], ListDuos[BestDuoID][1], M.shape)
                axs[0,1].plot(Angles, TrackersAngularVariancesDict[TrackerID_i])

                ReducedEvents, _ = self.ShapePointsReduction(TrackerID_j, nReducedPoints)
                axs[1,0].set_title('Tracker {0}'.format(TrackerID_j))
                axs[1,0].plot(np.array(self.TM.Trackers[TrackerID_j].LocksSaves[0].Events)[:,1], np.array(self.TM.Trackers[TrackerID_j].LocksSaves[0].Events)[:,2], 'or')
                axs[1,0].plot(np.array(ReducedEvents)[:,1], np.array(ReducedEvents)[:,2], 'ob')
                axs[1,1].plot(Angles, TrackersAngularVariancesDict[TrackerID_j])

                if AddLines:
                    PlotLines(LinesDict[TrackerID_i], self.TM._TrackerDiameter, axs[0,0])
                    PlotLines(LinesDict[TrackerID_j], self.TM._TrackerDiameter, axs[1,0])

            if PlotScenes:
                SnapID_i = ((np.array(self.TM.TrackersPositionsHistoryTs) - self.TM.Trackers[TrackerID_i].LocksSaves[0].Time) > 0).tolist().index(True)
                self.CreateTrackingShot(TrackerIDs = [TrackerID_i], SnapshotNumber = SnapID_i, BinDt = 0.005, ax_given = axs_scenes[0, nBestFound], cmap = 'binary', addTrackersIDsFontSize = 10, removeTicks = True, add_ts = True, DisplayedStatuses = ['Stabilizing', 'Converged'], DisplayedProperties = ['Aperture issue', 'Locked', 'None'], RemoveNullSpeedTrackers = 0, VirtualPoint = None)
                SnapID_j = ((np.array(self.TM.TrackersPositionsHistoryTs) - self.TM.Trackers[TrackerID_j].LocksSaves[0].Time) > 0).tolist().index(True)
                self.CreateTrackingShot(TrackerIDs = [TrackerID_j], SnapshotNumber = SnapID_j, BinDt = 0.005, ax_given = axs_scenes[1, nBestFound], cmap = 'binary', addTrackersIDsFontSize = 10, removeTicks = True, add_ts = True, DisplayedStatuses = ['Stabilizing', 'Converged'], DisplayedProperties = ['Aperture issue', 'Locked', 'None'], RemoveNullSpeedTrackers = 0, VirtualPoint = None)
            nBestFound += 1

        return OutDuos, VariancesDistanceMatrix, TrackersAngularVariancesDict


    def GetAngularRepartition(self, TrackerID, NLinesHoriz = 10, nReducedPoints = None):
        dLine = self.TM._TrackerDiameter / float(NLinesHoriz - 1)
        NAngles = int(np.pi / (2 * np.arctan(1. / NLinesHoriz)))
        dTheta = np.pi / NAngles
        Angles = [i*dTheta for i in range(NAngles)]
        Lines = []
        XF = np.array([self.TM._TrackerDiameter/2., self.TM._TrackerDiameter/2.])
        for Angle in Angles:
            if Angle < np.pi/2:
                X0 = np.array([self.TM._TrackerDiameter/2., -self.TM._TrackerDiameter/2.])
                XF = np.array([-self.TM._TrackerDiameter/2., self.TM._TrackerDiameter/2.])
            else:
                X0 = np.array([self.TM._TrackerDiameter/2., self.TM._TrackerDiameter/2.])
                XF = np.array([-self.TM._TrackerDiameter/2., -self.TM._TrackerDiameter/2.])
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
