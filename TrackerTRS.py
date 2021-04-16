import numpy as np
from matplotlib import pyplot as plt
import matplotlib

import datetime
import os
import inspect
import pickle
from sys import stdout
from PEBBLE import Module, TrackerEvent, TauEvent

from functools import partial

from TrackerExtensions import *

def RotateVector(Vector):
    return np.array([Vector[0]**2 - Vector[1]**2, 2*Vector[0]*Vector[1]])
def RotateVectors(Vectors):
    RotatedVectors = np.zeros(Vectors.shape)
    RotatedVectors[:,0] = Vectors[:,0]**2 - Vectors[:,1]**2
    RotatedVectors[:,1] = 2*Vectors[:,0]*Vectors[:,1]
    return RotatedVectors

def NonZeroNumber(x):
    return x + int(x == 0)
def NonZeroArray(A):
    return A + np.array(A == 0, dtype = int)
def NonZero(x):
    if type(x) == n.ndarray:
        return NonZeroArray(x)
    else:
        return NonZeroNumber(x)

class TrackerTRS(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to handle ST-context memory.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__ReferencesAsked__ = ['Memory']
        self.__Type__ = 'Computation'

        self._SendTrackerEventForStatuses = [StateClass._STATUS_STABILIZING,
                                              StateClass._STATUS_CONVERGED,
                                              StateClass._STATUS_LOCKED]
        self._SendTrackerEventIfApertureIssue = True

        self._TrackerStopIfAllDead = True                                           # Bool. Stops stream if all trackers die. Overrided if _DetectorAutoRestart is TRUE. Default : True

        self._RSLockModel = True                                                    # MModel for which unlock trackers are only corrected through flow error, w/o estimators.

        self._DetectorPointsGridMax = (10, 8)                                       # (Int, Int). Reference grid to place trackers, capping the discrete grid made by dividing the screen with _TrackerDiameter. Default : (10,8), i.e 80 trackers
        self._DetectorMinRelativeStartActivity = 2.                                 # Float. Activity of a tracker needed to activate, divided by _TrackerDiameter. Low values are noise sensible, while high value leave tracker in IDLE state. Default : 3.
        self._DetectorAutoRestart = True                                            # Bool. Restarts or not a tracker when another dissapears. If True, overrides _TrackerStopIfAllDead to False. Default : True.

        self._NPixelsEdge = 1.5
        self._TrackerDiameter = 25.                                                 # Int. Size of a tracker's side lenght. Usually changed to 30 for 640x480 camera definitions. Default : 20
        self._RelativeActivityCap = 2.0                                             # Float. Boundary for maximum activity a tracker can reach, taking into account _TrackerDiameter and _NPixelsEdge. Default : 2.0
        self._EdgeBinRatio = 4.                                                     # Float. Number of successive _NPixelsEdge lines kept. Garbage remover purpose obly
        self._DetectorDefaultSpeedTrigger = 5.                                      # Float. Default speed trigger, giving a lower bound for speeds in the scene, and higher boundary in time constants.
        self._TrackerMinDeathActivity = 0.1                                         # Float. Low boundary for a tracker to survive divided by _TrackerDiameter. Must be lower than _DetectorMinRelativeStartActivity. Default : 0.7
        self._OutOfBondsDistance = 0.                                               # Float. Distance to the screen edge to kill a tracker. Usually 0 or _TrackerDiameter/2
        self._DynamicsEstimatorTMRatio = 0.5                                        # Float. Factor for DynamicsEstimator time constant. Modifies inertia for correction.
        self._ProjectFlow = True                                                    # Bool. Is the optical flow computed projected onto the local edge orthogonal vector

        self._ShapeMaxOccupancy = 0.3                                               # Float. Maximum occupancy of the tracker ROI
        self._TrackerMaxEventsBuffer = np.pi*(self._TrackerDiameter/2)**2 * self._EdgeBinRatio * self._ShapeMaxOccupancy
        self._TrackerAccelerationFactor = 2.                                        # Float. Acceleration factor for speed correction. Default : 1.
        self._TrackerDisplacementFactor = 2.                                        # Float. Displacement factor for position correction with the relative flow. Default : 1.
        self._TrackerUnlockedSpeedModActivityPower = 1.0                                     # Float. Exponent of the activity dividing the speed correction in unlocked mode. May allow to lower inertia in highly textured scenes. Default : 1.
        self._TrackerUnlockedDisplacementModActivityPower = 1.0                                     
        self._ObjectEdgePropagationTC = 1e-5                                        # Float. Security time window for events considered to be previous edge propagation, and not neighbours later spiking. Default : 1e-4
        self._ClosestEventProximity = 2.5                                             # Float. Maximum distance for an event to be considered in the simple flow computation. Default : 2.
        self._MaxConsideredNeighbours = 20                                          # Int. Maximum number of considered events in the simple flow computation. Higher values slighty increase flow quality, but highly increase computation time. Default : 20
        self._MinConsideredNeighbours = int(self._ClosestEventProximity * 2 * 1.5)  # Int. Minimum number of considered events in the simple flow computation. Lower values allow for more events to be used in the correction, but lower the correction quality. Default : 4

        self._CenteringProcess = 'Step'                                             # Str. Method used for centering. 'Step' for single recentering at convergence. 'Continuous' for permanent recentering. Second one is dampened by optical flow. 'None' for no recentering
        self._TrackerOffCenterThreshold = 0.4                                       # Float. Minimum distance of to consider a tracker off-centered relative to _TrackerDiameter/2.
        self._TrackerSpringThreshold = self._TrackerOffCenterThreshold / 2          # Float. Minimum distance of recentering relative to _TrackerDiameter/2. Default : 0.3
        self._TrackerMeanPositionDisplacementFactor = {'Step':0.8, 'Continuous':0.2, 'None':0}[self._CenteringProcess]                            
        # Float. Amount of correction due to recentering process. Default : 1.
        self._LockOnlyCentered = True                                               # Bool. Forces to wait for the mean position of projected events to be within self._TrackerOffCenterThreshold

        self._ConvergenceEstimatorType = 'PD'                                       # Str. Defines the way we compute the convergence estimator. 'PD' (ProjectedDistance) for the average distance of an event to the surrounding events.
                                                                                    # 'CF' (CompensatedFlow) for a flow that is on average 0
        self._TrackerConvergenceThreshold = {'CF':0.30, 'PD':0.6}[self._ConvergenceEstimatorType]                                     
        # Float. Amount of correction relative to activity allowing for a converged feature. Default : 0.2
        self._TrackerConvergenceHysteresis = {'CF':0.05, 'PD':0.1}[self._ConvergenceEstimatorType]                  
        # Float. Hysteresis value of _TrackerConvergenceThreshold. Default : 0.05

        self._LocalEdgeRadius = self._ClosestEventProximity * 1.1
        self._LocalEdgeNumberOfEvents = int(4*self._LocalEdgeRadius)
        self._TrackerApertureIssueThreshold = 0.55                           # Float. Amount of aperture scalar value by speed relative to correction activity. Default : 0.25
        self._TrackerApertureIssueHysteresis = 0.05                          # Float. Hysteresis value of _TrackerApertureIssueThreshold. Default : 0.05

        self._TrackerAllowShapeLock = True                                          # Bool. Shape locking feature. Should never be disabled. Default : True
        self._LockupMaximumSpread = 0.7                                             # Float. Maximum occupancy of the shape defined by the  projected events at lockup. This protects against highly textured ROIs and stabilization failures. >=1 disables this filter. Default : 0.8
        self._TrackerLockingMaxRecoveryBuffer = 100                                 # Int. Size of the buffer storing events while locked, used in case of feature loss. Greater values increase computationnal cost. Default : 100
        self._TrackerLockMaxRelativeActivity = np.inf                               # Float. Maximum activity allowing for a lock, divided by _TrackerDiameter. Can forbid locking on highly textured parts of the scene. np.inf inhibits this feature. Default : np.inf
        self._TrackerLockedRelativeCorrectionsFailures = 0.7                        # Float. Minimum correction activity relative to tracker activity for a lock to remain. Assesses for shape loss. Default : 0.6
        self._TrackerLockedRelativeCorrectionsHysteresis = 0.05                     # Float. Hysteresis value of _TrackerLockedRelativeCorrectionsFailures. Default : 0.05
        self._TrackerLockedSpeedModActivityPower = 1.                               # Float. Exponent of the activity power of the activity used for tracker speed correction. Default : 1.
        self._TrackerLockedDisplacementModActivityPower = 1.                        # Float. Exponent of the activity power of the activity used for the locked tracker displacement correction. Default : 0.2
        self._LockedSpeedModReduction = 1.
        self._LockedPosModReduction = 1.
        self._TrackerLockedCanHardCorrect = True                                    # Bool. Allows trackers to remain locked even with high speed correction values. Default : True
        self._AllowDisengage = True                                                 # Save process when a tracker is locked but has too low activity. Typically used when sudden deceleration appears
        self._TrackerDisengageActivityThreshold = 0.5
        self._TrackerDisengageActivityHysteresis = 0.1
        self._TrackerUnlockedSurvives = False                                       # Bool. Behaviour of unlocked trackers. If True, the trackers goes as converged, if False it is killed and restarted somewhere else.

        self._MeanPositionDampeningFactor = 0.

        self._ComputeSpeedErrorMethod = 'BareMean'                                # String. Speed error computation method. Can be 'LinearMean', 'ExponentialMean' or 'PlanFit'. See associated functions for details. Default : 'LinearMean'
        # Monitoring Variables

        self._MonitorDt = 0.1                                                         # Float. Time difference between two snapshots of the system.
        self._MonitoredVariables = [('RecordedTrackers@ID', int),
                                    ('RecordedTrackers@TrackerActivity', float),
                                    ('RecordedTrackers@Position', np.array),
                                     ('RecordedTrackers@Speed', np.array),
                                     ('RecordedTrackers@State.Value', tuple),
                                     #('RecordedTrackers@ProjectedEvents', np.array),
                                     ('RecordedTrackers@TimeConstant', float),
                                     #('RecordedTrackers@DynamicsEstimator.W', float),
                                     #('RecordedTrackers@DynamicsEstimator.MDet', float),
                                     ('RecordedTrackers@DynamicsEstimator.Speed', np.array),
                                     #('RecordedTrackers@DynamicsEstimator.Displacement', np.array),
                                     #('RecordedTrackers@DynamicsEstimator._Es', np.array),
                                     #('RecordedTrackers@DynamicsEstimator.X', np.array),
                                     #('RecordedTrackers@ApertureIssue', bool),
                                     #('RecordedTrackers@OffCentered', bool)
                                     ('RecordedTrackers@ApertureEstimator.Value', float),
                                     ('RecordedTrackers@SpeedConvergenceEstimator.Value', np.array)
                                     ]
        self._StateClass = StateClass

    def _InitializeModule(self):
        self.FeatureManager = FeatureManagerClass(self)
        self.GTMaker = GTMakerClass(self)
        if self._DetectorAutoRestart: #  One cannot expect all trackers to die if they auto restart
            self._TrackerStopIfAllDead = False

        L_X, L_Y = self.Geometry[:2]
        self.Trackers = []
        self.AliveTrackers = []
        self.JustDeadTrackers = [] # Used to store trackers that just died, in order to have their last values recorded.
        self.RecordedTrackers = []
        self.StartTimes = []
        self.DeathTimes = []

        d_x = self._TrackerDiameter
        d_y = self._TrackerDiameter
        N_X = int(L_X / d_x)
        N_Y = int(L_Y / d_y)

        if N_X > self._DetectorPointsGridMax[0]:
            N_X = self._DetectorPointsGridMax[0]
            d_x = L_X / N_X
        if N_Y > self._DetectorPointsGridMax[1]:
            N_Y = self._DetectorPointsGridMax[1]
            d_y = L_Y / N_Y

        r_X = (L_X - (N_X - 1) * d_x)/2
        r_Y = (L_Y - (N_Y - 1) * d_y)/2

        self._TrackerDefaultTimeConstant = self._NPixelsEdge / self._DetectorDefaultSpeedTrigger
        
        self._TrackerMinMaxNeighboursAutoValues = {'TwoPointsProjection': [2, 2]}
        if self._ComputeSpeedErrorMethod in list(self._TrackerMinMaxNeighboursAutoValues.keys()):
            self._MinConsideredNeighbours = self._TrackerMinMaxNeighboursAutoValues[self._ComputeSpeedErrorMethod][0]
            self._MaxConsideredNeighbours = self._TrackerMinMaxNeighboursAutoValues[self._ComputeSpeedErrorMethod][1]

        #self._BoolSpeedCorrection = np.array([self._CorrectTranslationSpeed, self._CorrectTranslationSpeed, self._CorrectRotationSpeed, self._CorrectScalingSpeed], dtype = float)
        #self._BoolPositionCorrection = np.array([self._CorrectTranslationPosition, self._CorrectTranslationPosition, self._CorrectRotationPosition, self._CorrectScalingPosition], dtype = float)

        self.DetectorInitialPositions = []
        for nX in range(N_X):
            for nY in range(N_Y):
                self.DetectorInitialPositions += [np.array([r_X + nX * d_x, r_Y + nY * d_y])]

        for ID, InitialPosition in enumerate(self.DetectorInitialPositions):
            self._AddTracker(InitialPosition)

        self.LastTsSnap = -np.inf

        self._DetectorMinActivityForStart = self._TrackerDiameter * self._DetectorMinRelativeStartActivity

        self.Plotter = PlotterClass(self)
        self.LocksSubscribers = [self.FeatureManager.AddLock]

        return True

    def _OnSnapModule(self):
        self.RecordedTrackers = self.AliveTrackers + self.JustDeadTrackers
        self.JustDeadTrackers = []

    def RetreiveHistoryData(self, DataName, ID, Snap = None):
        if not Snap is None and Snap < 0:
            Snap = len(self.History['t']) + Snap
        if '@' not in DataName:
            DataName = 'RecordedTrackers@'+DataName
        if DataName not in self.History.keys():
            raise Exception("{0} is not a tracker data saved name".format(DataName))
        Data = self.History[DataName]
        RetreivedValues = []
        Ts = []
        LastEncounteredValue = None
        for nSnap, tIDs in enumerate(zip(self.History['t'], self.History['RecordedTrackers@ID'])):
            t, IDs = tIDs
            try: # We try to find it in this snap's recorded IDs
                IDIndex = IDs.index(ID)
                LastEncounteredValue = Data[nSnap][IDIndex] # If we found it, we store the wanted value as the last value encountered
            except ValueError: # If we havent found it then its either not present yet in the tracker list, thus we leave None as the LastEncounteredValue, or already dead in which case we want to keed the LastEncounteredValue as it is.
                pass

            if (not Snap is None): # If a specific snap was requested
                if nSnap == Snap: # If this is the one, we send what we have found so far
                    return t, LastEncounteredValue
                else: # Otherwise, it's useless to store anything, we continue
                    continue 
            if not LastEncounteredValue is None: # If we want a batch of data, then we only store values once we have started encountering them. 
                Ts += [t]
                RetreivedValues += [LastEncounteredValue]
        return Ts, np.array(RetreivedValues)

    def TrackerEventCondition(self, Tracker):
        if Tracker.State.Status in self._SendTrackerEventForStatuses:
            if self._SendTrackerEventIfApertureIssue or not Tracker.State.ApertureIssue:
                return True
        return False

    def _OnEventModule(self, event):
        self.NewTrackersAsked = 0
        Associated = False
        self.LastEvent = event
        for Tracker in self.AliveTrackers:
            TrackerID = Tracker.ID
            Associated = Tracker.RunEvent(event)
            if Associated and self.TrackerEventCondition(Tracker):
                event.Attach(TrackerEvent, TrackerLocation = np.array(Tracker.Position[:2]), TrackerID = TrackerID, TrackerAngle = Tracker.Position[2], TrackerScaling = Tracker.Position[3], TrackerColor = Tracker.State.GetColor(), TrackerMarker = Tracker.State.GetMarker())
                if Tracker.State.Locked:
                    event.Attach(TauEvent, tau = Tracker.TimeConstant)

        if self.NewTrackersAsked:
            self._PlaceNewTrackers()
            self.NewTrackersAsked = 0

        if self._TrackerStopIfAllDead and not self.AliveTrackers:
            self.__Framework__.Paused = self.__Name__
            
        return

    def _OnTrackerLock(self, Tracker):
        for SubscriberMethod in self.LocksSubscribers:
            SubscriberMethod(Tracker)

    def AddLocksSubscriber(self, ListenerMethod):
        self.LocksSubscribers += [ListenerMethod]

    def _KillTracker(self, Tracker, t, Reason=''):
        self.Log("Tracker {0} died".format(Tracker.ID) + int(bool(Reason)) * (" ("+Reason+")"))
        Tracker.State.SetStatus(Tracker.State._STATUS_DEAD)
        self.LastEvent.Attach(TrackerEvent, TrackerLocation = np.array(Tracker.Position[:2]), TrackerID = Tracker.ID, TrackerAngle = Tracker.Position[2], TrackerScaling = 0, TrackerColor = Tracker.State.GetColor(), TrackerMarker = Tracker.State.GetMarker())
        self.DeathTimes[Tracker.ID] = t
        self.AliveTrackers.remove(Tracker)
        self.JustDeadTrackers += [Tracker]

        if self._DetectorAutoRestart:
            self.NewTrackersAsked += 1

    def _PlaceNewTrackers(self):
        for NewID in range(self.NewTrackersAsked):
            SelectedPositionID = None
            MaxDistance = 0
            
            for InitialPositionID, InitialPosition in enumerate(self.DetectorInitialPositions):
                InitialPositionMinDistance = np.inf
                for Tracker in self.AliveTrackers:
                    InitialPositionMinDistance = min(InitialPositionMinDistance, np.linalg.norm(InitialPosition - Tracker.Position[:2]))
                if InitialPositionMinDistance >= MaxDistance:
                    MaxDistance = InitialPositionMinDistance
                    SelectedPositionID = InitialPositionID
            self._AddTracker(self.DetectorInitialPositions[SelectedPositionID])

    def _AddTracker(self, InitialPosition):
        ID = len(self.Trackers)
        NewTracker = TrackerClass(self, ID, InitialPosition)
        self.Trackers += [NewTracker]
        self.AliveTrackers += [NewTracker]
        self.StartTimes += [None]
        self.DeathTimes += [None]

    def _SaveAdditionalData(self, ExternalDict):
        # As GT generation can be quite long, we can save it here.
        ExternalDict['GTMaker'] = self.GTMaker._SaveDataToDict()
    def _RecoverAdditionalData(self, ExternalDict):
        self.GTMaker._RecoverDataFromDict(ExternalDict['GTMaker'])

class StateClass:
    _STATUS_DEAD = 0
    _STATUS_IDLE = 1
    _STATUS_STABILIZING = 2
    _STATUS_CONVERGED = 3
    _STATUS_LOCKED = 4

    # Properties can stack, so they are given as powers on binary number
    _PROPERTY_APERTURE = 0
    _PROPERTY_OFFCENTERED = 1
    _PROPERTY_DISENGAGED = 2
    
    _COLORS = {_STATUS_DEAD:'k', _STATUS_IDLE: 'r', _STATUS_STABILIZING:'m', _STATUS_CONVERGED: 'b', _STATUS_LOCKED:'g'}
    _MARKERS = {0: 'o', 1: '_', 2:'s', 3: 'D', 4:'X', 5:'X', 6:'X', 7:'X'} # Properties 5 to 7 should not be possible
    _StatusesNames = {_STATUS_DEAD: 'Dead', _STATUS_IDLE:'Idle', _STATUS_STABILIZING:'Stabilizing', _STATUS_CONVERGED:'Converged', _STATUS_LOCKED: 'Locked'} # Statuses names with associated int values.
    _PropertiesNames = {0: 'None', 1: 'Aperture issue', 2:'OffCentered', 3:'Disengaged'} # Properties names with associated int values.
    _PropertiesLetters = {_PROPERTY_APERTURE: 'a', _PROPERTY_OFFCENTERED:'o', _PROPERTY_DISENGAGED: 'd'}

    def __init__(self):
        self.Status = self._STATUS_IDLE
        self.ApertureIssue = False
        self.OffCentered = False
        self.Disengaged = False
    def __repr__(self):
        return str(self.Value)

    @property
    def Properties(self):
        return (self.OffCentered << self._PROPERTY_OFFCENTERED | self.ApertureIssue << self._PROPERTY_APERTURE | self.Disengaged << self._PROPERTY_DISENGAGED)
    @property
    def Value(self):
        return (self.Status, self.Properties)
    def SetStatus(self, Value):
        self.Status = Value
    def __eq__(self, RHS):
        return self.Status == RHS
    @property
    def Idle(self):
        return self.Status == self._STATUS_IDLE
    @property
    def Stabilizing(self):
        return self.Status == self._STATUS_STABILIZING
    @property
    def Converged(self):
        return self.Status == self._STATUS_CONVERGED
    @property
    def Locked(self):
        return self.Status == self._STATUS_LOCKED
    @property
    def Dead(self):
        return self.Status == self._STATUS_DEAD

    def GetMarker(self):
        return self._MARKERS[self.Properties]
    def GetColor(self):
        return self._COLORS[self.Status]
    def GetStr(self):
        return ','.join([s for pad, s in self._PropertiesLetters.items() if (self.Properties >> pad) & 0b1])

class LockClass:
    def __init__(self, Time, TrackerActivity, FlowActivity, Events):
        self.Time = Time
        self.TrackerActivity = TrackerActivity
        self.FlowActivity = FlowActivity
        self.Events = Events
        self.ReleaseTime = None
        self.Reason = ''

class EstimatorTemplate:
    def __init__(self):
        self.LastUpdate = -np.inf
        self.W = 0
        self._DecayingVars = ['W']
        self._GeneralVars = []
    def __iter__(self):
        self._IterVar = 0
        return self
    def __next__(self):
        if self._IterVar >= len(self._DecayingVars):
            raise StopIteration
        self._IterVar += 1
        Var = self.__dict__[self._DecayingVars[self._IterVar-1]]
        if type(Var)==np.ndarray:
            return np.array(Var)
        else:
            return type(Var)(Var)
    def AddDecayingVar(self, VarName, Dimension = 1, GeneralVar = False):
        if Dimension == 1:
            self.__dict__['_'+VarName] = 0.
            if GeneralVar:
                self.__dict__[VarName] = 0.
                self._GeneralVars += [VarName]
        else:
            self.__dict__['_'+VarName] = np.zeros(Dimension)
            if GeneralVar:
                self.__dict__[VarName] = np.zeros(Dimension)
                self._GeneralVars += [VarName]
        self._DecayingVars += ['_'+VarName]
    def RecoverGeneralData(self):
        if self.W:
            for VarName in self._GeneralVars:
                self.__dict__[VarName] = self.__dict__['_'+VarName] / NonZeroNumber(self.W)
    def EstimatorStep(self, newT, Tau, WeightIncrease = 1):
        DeltaUpdate = newT - self.LastUpdate
        self.LastUpdate = newT
        Decay = np.e**(-DeltaUpdate/Tau)
        for VarName in self._DecayingVars:
            self.__dict__[VarName] *= Decay
        self.W += WeightIncrease

class DynamicsEstimatorClass(EstimatorTemplate):
    _FallBackToIdealMatrix = False
    _FallBackToSimpleTranslation = True
    _CenterTo = 'Tracker' # 'Tracker' defines position relative to tracker center. 'Events' defines position to average events position
    _HomogeneityRadiusFactor = 0
    def __init__(self, Radius, T = True, R = True, S = True):
        EstimatorTemplate.__init__(self)
        self._Radius = Radius
        self._Dim = 2+R+S
        self._R = int(R)
        self._S = int(S);self._Si = 2+self._R
        self._HomogeneityVector = np.ones(self._Dim)
        if self._HomogeneityRadiusFactor != 0:
            if R:
                self._HomogeneityVector[2] = 1 / (self._Radius * self._HomogeneityRadiusFactor)
            if S:
                self._HomogeneityVector[self._Si] = 1 / (self._Radius * self._HomogeneityRadiusFactor)

        self._UpToDateMatrix = False
        self._IdealMatrix = np.identity(self._Dim) * 2
        if R:
            self._IdealMatrix[2,2] /= Radius**2 / 3
        if S:
            self._IdealMatrix[self._Si,self._Si] /= Radius**2 / 3
        self._LowerInvertLimit = (Radius/3)**(self._Dim - 2)*2 * 0.5**self._Dim

        self._M = np.zeros((self._Dim, self._Dim))
        self.MDet = 0.
        self._InvM = np.zeros((self._Dim, self._Dim))

        self.AddDecayingVar('Es', self._Dim, GeneralVar = True)
        self.AddDecayingVar('Ed', self._Dim, GeneralVar = True)

        self.AddDecayingVar('X', 2, GeneralVar = True)
        self.AddDecayingVar('r2', 1, GeneralVar = True)

        for ConvergingSum in ['Scc', 'Sxxcc', 'Sxxss', 'Syycc', 'Syyss']:
            self.AddDecayingVar(ConvergingSum)
        self.AddDecayingVar('RXcc', 2)
        self.AddDecayingVar('RXcs', 2)
        for OneDimensionResidue in ['Rcs', 'Rxxcs', 'Ryycs', 'Rxycs', 'Rxycc', 'Rxyss']:
            self.AddDecayingVar(OneDimensionResidue)

    def _EstimatorVariation(self, Observable, x, y):
        Var = np.zeros(self._Dim)
        Var[:2] = Observable
        if self._R:
            Var[2] = Observable[1]*x-Observable[0]*y
        if self._S:
            Var[self._Si] = Observable[0]*x+Observable[1]*y
        return Var
    def _EstimatorShift(self, Estimator, dx, dy):
        Var = self._EstimatorVariation(Estimator[:2], dx, dy) # It can be done this way, shortens the code
        Var[:2] = 0 # The raw optical flow are not affected by a shift in position
        return Var # The minus sign will be placed outside

    def TrackerShift(self, PositionVariation): # An event initially in (x,y) is now in (x-dx, y-dy)
        dx, dy = PositionVariation
        N2PositionVariation = (PositionVariation**2).sum()
        self._Es -= self._EstimatorShift(self._Es, dx, dy) # W is hidden in the first two  terms
        self._Ed -= self._EstimatorShift(self._Ed, dx, dy)

        self._X -= self.W*PositionVariation # If X > 0, the events are too far right in tracker frame. The tracker compensates by moving right, thus PositionVariation > 0, X' is thus lowered.
        self._r2 -= 2*(self._X*PositionVariation).sum() + self.W*N2PositionVariation # We can use the new _X coordinates to simplify the process

        # Scc, Sss and Rcs are unchanged since they only use optical flow orientations
    
        self._RXcs  -= self._Rcs * PositionVariation
        self._RXcc  -= self._Scc * PositionVariation

        self._Sxxcc -= 2* self._RXcc[0] * dx + self._Scc * dx**2
        self._Syycc -= 2* self._RXcc[1] * dy + self._Scc * dy**2
        _RXss = self._X - self._RXcc
        _Sss  = self.W  - self._Scc
        self._Sxxss -= 2* _RXss[0] * dx + _Sss * dx**2
        self._Syyss -= 2* _RXss[1] * dy + _Sss * dy**2

        self._Rxxcs -= 2* self._RXcs[0] * dx + self._Rcs * dx**2
        self._Ryycs -= 2* self._RXcs[1] * dy + self._Rcs * dy**2
        self._Rxycs -= self._RXcs[0] * dy + self._RXcs[1] * dx + self._Rcs * dx * dy
        self._Rxycc -= self._RXcc[0] * dy + self._RXcc[1] * dx + self._Scc * dx * dy
        self._Rxyss -= _RXss[0] * dy + _RXss[1] * dx + _Sss * dx * dy

    def AddData(self, t, TrackerRelativeLocation, Flow, Displacement, Tau):
        F2 = Flow**2
        N2 = F2.sum()
        if N2 == 0:
            return
        self.EstimatorStep(t, Tau)

        self._X += TrackerRelativeLocation
        if self._CenterTo == 'Events':
            x, y = (TrackerRelativeLocation - (self._X / NonZeroNumber(self.W))) # If using mean position inside tracker
        elif self._CenterTo == 'Tracker':
            x, y = TrackerRelativeLocation  # If using tracker center
        xx, yy = x**2, y**2
        xy = x*y

        N = np.sqrt(N2)
        c , s  = Flow/N
        cc, ss = F2/N2
        cs = c*s

        self._Es += self._EstimatorVariation(Flow, x, y)
        self._Ed += self._EstimatorVariation(Displacement, x, y)

        self._r2 += xx+yy

        self._Scc += cc
        self._Sxxcc += xx*cc
        self._Sxxss += xx*ss
        self._Syycc += yy*cc
        self._Syyss += yy*ss

        self._Rcs  += cs
        self._RXcs += np.array([x,y])*cs
        self._RXcc += np.array([x,y])*cc

        self._Rxxcs += xx*cs
        self._Ryycs += yy*cs
        self._Rxycs += xy*cs
        self._Rxycc += xy*cc
        self._Rxyss += xy*ss

        self._UpToDateMatrix = False

    def _GetInverseMatrix(self):
        if self._UpToDateMatrix:
            return (self._FallBackToIdealMatrix or (abs(self.MDet) > self._LowerInvertLimit))

        if self._CenterTo == 'Events':
            RXss = - self._RXcc # If using mean position inside tracker #In which case <X> = 0, not to confuse with self.X variable that actually describes <X> by respect to tracker center
        elif self._CenterTo == 'Tracker':
            RXss = self._X - self._RXcc # If using tracker center

        self._M[0,0] = self._Scc
        self._M[1,1] = self.W - self._Scc
        if self._R:
            self._M[2,2] = self._Sxxss + self._Syycc - 2*self._Rxycs
        if self._S:
            self._M[self._Si,self._Si] = self._Sxxcc + self._Syyss + 2*self._Rxycs

        self._M[1,0] = self._M[0,1] = self._Rcs
        if self._R:
            self._M[2,0] = self._M[0,2] = self._RXcs[0] - self._RXcc[1]
            self._M[2,1] = self._M[1,2] = (RXss[0]) - self._RXcs[1]
        if self._S:
            self._M[self._Si,0] = self._M[0,self._Si] = self._RXcc[0] + self._RXcs[1]
            self._M[self._Si,1] = self._M[1,self._Si] = self._RXcs[0] + (RXss[1])
        if self._S and self._R:
            self._M[self._Si,2] = self._M[2,self._Si] = self._Rxxcs + self._Rxyss - self._Rxycc - self._Ryycs

        self.MDet = np.linalg.det(self._M) / (NonZeroNumber(self.W) ** self._Dim)
        self._UpToDateMatrix = True
        if abs(self.MDet) > self._LowerInvertLimit: # Should be homogeneous to px^4, so proportionnal to radius**4
            self._InvM = np.linalg.inv(self._M)
            return True
        else:
            if self._FallBackToIdealMatrix:
                self._InvM = self._IdealMatrix
                return True
            else:
                return False

    @property
    def Speed(self):
        return self.GetSpeed()

    def GetSpeed(self):
        if self._GetInverseMatrix():
            return self.ReDimVector(self._HomogeneityVector * self._InvM.dot(self._Es))
        elif self._FallBackToSimpleTranslation:
            SimpleTranslation = self._Es / NonZeroNumber(self.W)
            SimpleTranslation[2:] = 0
            return self.ReDimVector(SimpleTranslation)
        else:
            return None

    @property
    def Displacement(self):
        return self.GetDisplacement()
    def GetDisplacement(self):
        if self._GetInverseMatrix():
            return self.ReDimVector(self._HomogeneityVector * self._InvM.dot(self._Ed))
        elif self._FallBackToSimpleTranslation:
            SimpleTranslation = self._Ed / NonZeroNumber(self.W)
            SimpleTranslation[2:] = 0
            return self.ReDimVector(SimpleTranslation)
        else:
            return None
    def ReDimVector(self, UnderDimVector):
        TRS = np.zeros(4)
        TRS[:2] = UnderDimVector[:2]
        if self._R:
            TRS[2] = UnderDimVector[2]
        if self._S:
            TRS[3] = UnderDimVector[self._Si]
        return TRS

class SpeedConvergenceEstimatorClass(EstimatorTemplate):
    def __init__(self, Type):
        EstimatorTemplate.__init__(self)
        if Type == 'CF':
            self.AddData = self._AddDataCF
            self.GetValue = self._GetValueCF
            self.AddDecayingVar('Vector', Dimension = 2, GeneralVar = True)
        elif Type == 'PD':
            self.AddData = self._AddDataPD
            self.GetValue = self._GetValuePD
            self.AddDecayingVar('Epsilon', Dimension = 1, GeneralVar = True)
            self.AddDecayingVar('Sigma2', Dimension = 1, GeneralVar = True)


    @property
    def Value(self):
        return self.GetValue()

    def _AddDataCF(self, t, Flow, Tau):
        NFlow = np.linalg.norm(Flow)
        if NFlow == 0:
            return
        self.EstimatorStep(t, Tau)
        self._Vector += Flow / NFlow
    def _GetValueCF(self):
        self.RecoverGeneralData()
        return np.linalg.norm(self.Vector), 0.

    def _AddDataPD(self, t, ProjectionError, Tau):
        NProjectionError = np.linalg.norm(ProjectionError)
        self.EstimatorStep(t, Tau)
        self._Epsilon += NProjectionError
        self._Sigma2 += (NProjectionError - self._Epsilon/self.W)**2
    def _GetValuePD(self):
        self.RecoverGeneralData()
        return self.Epsilon, np.sqrt(self.Sigma2)

class ApertureEstimatorClass(EstimatorTemplate):
    def __init__(self):
        EstimatorTemplate.__init__(self)

        self.AddDecayingVar('Vector', 2, GeneralVar = True)
        self.AddDecayingVar('Deviation', 1, GeneralVar = True)

    def AddData(self, t, LocalVector, Tau):
        N = (LocalVector**2).sum()
        if N == 0:
            return
        self.EstimatorStep(t, Tau, WeightIncrease = np.sqrt(N))

        self._Vector += LocalVector
        self._Deviation += np.linalg.norm(LocalVector - self._Vector / NonZeroNumber(self.W))
    @property
    def Value(self):
        self.RecoverGeneralData()
        return np.linalg.norm(self.Vector)

class DynamicsModifierClass:
    _SaveData = False
    def __init__(self, Tracker):
        self.Tracker = Tracker
        self.PositionMods = {}
        self.SpeedMods = {}
        self.BoolFactors = {"Speed":{}, "Position":{}}
        if self._SaveData:
            self.ModificationsHistory = {'t':[], 'Position': {}, 'Speed': {}}
        self._NoModValue = np.array([0., 0., 0., 0.])
        self.LastRecordedSpeed = np.array(Tracker.Speed)

    def AddModifier(self, Name, AffectSpeed = [False, False, False], AffectPosition = [False, False, False]):
        if not (True in AffectSpeed) and not (True in AffectPosition):
            self.Tracker.LogError("Modifier {0} affects nothing".format(Name))
        if True in AffectSpeed:
            self.SpeedMods[Name] = self._NoModValue
            self.BoolFactors["Speed"][Name] = np.array([AffectSpeed[0], AffectSpeed[0], AffectSpeed[1], AffectSpeed[2]])
            if self._SaveData:
                self.ModificationsHistory['Speed'][Name] = []
        if True in AffectPosition:
            self.PositionMods[Name] = self._NoModValue
            self.BoolFactors["Position"][Name] = np.array([AffectPosition[0], AffectPosition[0], AffectPosition[1], AffectPosition[2]])
            if self._SaveData:
                self.ModificationsHistory['Position'][Name] = []
    def Compile(self):
        AddSnap = False
        if self._SaveData and (not self.ModificationsHistory['t'] or self.ModificationsHistory['t'][-1] != self.Tracker.LastUpdate):
            self.ModificationsHistory['t'] += [self.Tracker.LastUpdate]
            AddSnap = True
        #if (self.LastRecordedSpeed != self.Tracker.Speed).any():
        #    self.Tracker.TM.LogWarning("Speedwas modified (ID = {0})".format(self.Tracker.ID))
        for Origin, Value in self.SpeedMods.items():
            self.Tracker.Speed += Value
            if self._SaveData:
                if AddSnap:
                    self.ModificationsHistory['Speed'][Origin] += [np.array(Value)]
                else:
                    self.ModificationsHistory['Speed'][Origin][-1] += np.array(Value)
            self.SpeedMods[Origin] = self._NoModValue
        self.LastRecordedSpeed = np.array(self.Tracker.Speed)
        for Origin, Value in self.PositionMods.items():
            self.Tracker.Position += Value
            if self._SaveData:
                if AddSnap:
                    self.ModificationsHistory['Position'][Origin] += [np.array(Value)]
                else:
                    self.ModificationsHistory['Position'][Origin][-1] += np.array(Value)
            self.PositionMods[Origin] = self._NoModValue

    def ModSpeed(self, Origin, Value):
        self.SpeedMods[Origin] = Value * self.BoolFactors["Speed"][Origin]# * np.array([1., 1., 0.3, 0.5])
    def ModPosition(self, Origin, Value):
        self.PositionMods[Origin] = Value * self.BoolFactors["Position"][Origin]# * np.array([1., 1., 0.3, 0.5])

    def PlotModifications(self):
        if not self._SaveData:
            raise Exception("Unable to plot precise modifications as no data was saved")
        f, axs = plt.subplots(4, 2)
        for nColumn, ModType in enumerate(['Position', 'Speed']):
            Ts, Data = self.Tracker.TM.RetreiveHistoryData('RecordedTrackers@'+ModType, self.Tracker.ID)
            for nDoF in range(4):
                for Origin in self.ModificationsHistory[ModType].keys():
                    if nDoF == 0:
                        axs[nDoF, nColumn].plot(self.ModificationsHistory['t'], np.array(self.ModificationsHistory[ModType][Origin])[:,nDoF], label = Origin)
                    else:
                        axs[nDoF, nColumn].plot(self.ModificationsHistory['t'], np.array(self.ModificationsHistory[ModType][Origin])[:,nDoF])
                ax = axs[nDoF, nColumn].twinx()
                ax.plot(Ts, Data[:,nDoF], 'k')
            axs[0, nColumn].set_title(ModType)
            axs[0, nColumn].legend(loc = 'upper left')

class TrackerClass:
    _Modifiers = {'Speed':((False, False, False), (True, True, True)), # Speed modification (T,R,S), Position modification (T,R,S)
                  'Flow': ((True, True, False),   (True, True, False)),
                  'MeanPos': ((False, False, False), (True, False, False)),
                  'Disengage': ((True, True, False), (True, True, False))}
    def __init__(self, TrackerManager, ID, InitialPosition):
        self.TM = TrackerManager
        self.ID = ID

        self.State = StateClass()
        self.Lock = None
        self.LocksSaves = []

        self.Position = np.array(list(InitialPosition) + [0., 1.], dtype = float) # We now include all 4 parameters of the tracker in a single variable. Easier resolution and code compacity
        self.Speed    = np.array([0., 0., 0., 0.], dtype = float)
        # self.Position = [tx, ty, theta, s]
        self.Radius = self.TM._TrackerDiameter / 2
        self.SquaredRadius = self.Radius**2

        self.TimeConstant = self.TM._TrackerDefaultTimeConstant
        self.EdgeBinTC = self.TimeConstant * self.TM._EdgeBinRatio
        
        self.ProjectedEvents = []
        self.AssociatedFlows = []
        self.AssociatedOffsets = []

        self.LastUpdate = 0.
        self.LastValidFlow = 0.
        self.LastRecenter = 0.

        self.TrackerActivity = 0.
        self.FlowActivity = 0.
        self.DynamicsEstimator = DynamicsEstimatorClass(self.Radius, R = self._Modifiers['Flow'][0][1]|self._Modifiers['Flow'][1][1], S = self._Modifiers['Flow'][0][2]|self._Modifiers['Flow'][1][2])
        self.ApertureEstimator = ApertureEstimatorClass()
        self.SpeedConvergenceEstimator = SpeedConvergenceEstimatorClass(self.TM._ConvergenceEstimatorType)

        self.MeanPosCorrection = np.array([0., 0.]) # Computed in tracker frame

        self.DynamicsModifier = DynamicsModifierClass(self)
        for Modifier, Params in self._Modifiers.items():
            self.DynamicsModifier.AddModifier(Modifier, AffectSpeed = Params[0], AffectPosition = Params[1])

        self._ComputeSpeedError = self.__class__.__dict__['_ComputeSpeedError' + self.TM._ComputeSpeedErrorMethod]
        self._ComputeRecenter = self.__class__.__dict__['_Center' + self.TM._CenteringProcess]

    def UpdateWithEvent(self, event):
        DeltaUpdate = event.timestamp - self.LastUpdate
        self.LastUpdate = event.timestamp
        Decay = np.e**(-DeltaUpdate / self.TimeConstant)
        self._AutoUpdateSpeed(DeltaUpdate)
        
        if not self.State.Disengaged:
            self.DynamicsModifier.ModPosition('Speed', self.Speed * DeltaUpdate)
            self.DynamicsModifier.Compile()
            self.Position[3] = min(max(self.Position[3], 0.1), 10.)
        else:
            self.DynamicsModifier.ModSpeed('Disengage', self.Speed * (Decay-1)) 
            self.DynamicsModifier.Compile()
            self._UpdateTC()

        self.TrackerActivity *= Decay
        self.FlowActivity *= Decay
        self.MeanPosCorrection *= Decay

        if self.State.Idle:
            return True
        if not self.State.Disengaged and ((self.Position[:2] - self.TM._OutOfBondsDistance < 0).any() or (self.Position[:2] + self.TM._OutOfBondsDistance >= np.array(self.Geometry[:2])).any()): # out of bounds, cannot happen if disengaged
            if self.Lock:
                self.Unlock('out of bounds')
            self.TM._KillTracker(self, event.timestamp, Reason = "out of bounds")
            return False
        if self.State.Locked:
            if not self.State.Disengaged and self.TM._AllowDisengage and self.TrackerActivity < self.TM._TrackerDiameter * (self.TM._TrackerDisengageActivityThreshold - self.TM._TrackerDisengageActivityHysteresis):
                self.Disengage(event.timestamp)
                return True
        if self.TrackerActivity < self.TM._TrackerDiameter * self.TM._TrackerMinDeathActivity: # Activity remaining too low
            if self.Lock:
                self.Unlock('low activity')
            self.TM._KillTracker(self, event.timestamp, "low activity")
            return False

        return True

    def _AutoUpdateSpeed(self, DeltaUpdate):
        # Takes care of the acceleration, specifically the scaling factor that self evolves.
        # If decaying speeds want to be considered, they should be put here
        #self.Speed[3] += (2 * self.Speed[3]**2 * DeltaUpdate) / self.Position[3]
        return

    def RunEvent(self, event):
        if not self.UpdateWithEvent(event): # We update the position and activity. They do not depend on the event location. False return means the tracker has died : out of screen or activity too low (we should potentially add 1 for current event but shoud be marginal)
            return False # We also consider that even if the event may have matched, the tracker still dies. The activity increase of 1 is not enough

        RC, RS = np.cos(self.Position[2]), -np.sin(self.Position[2])
        Dx, Dy = event.location - self.Position[:2]
        Dx2, Dy2 = Dx**2, Dy**2
        R2 = Dx2 + Dy2
        if R2 > self.SquaredRadius:
            return False
        self.TrackerActivity += 1
        if self.State.Disengaged and self.TrackerActivity > self.TM._TrackerDiameter * (self.TM._TrackerDisengageActivityThreshold + self.TM._TrackerDisengageActivityHysteresis):
            self.Reengage()

        self.TrackerActivity = min(self.TrackerActivity, self.Radius * 2 * self.TM._NPixelsEdge * self.TM._RelativeActivityCap)
        
        R = np.sqrt(R2)
        CurrentProjectedEvent = np.array([event.timestamp, (Dx*RC - Dy*RS) / self.Position[3], (Dy*RC + Dx*RS) / self.Position[3]])
        SavedProjectedEvent = np.array(CurrentProjectedEvent) 

        if self.State.Idle:
            self.ProjectedEvents += [SavedProjectedEvent]
            self.AssociatedFlows += [np.array([0., 0.])]
            self.AssociatedOffsets += [np.array([0., 0.])]
            if len(self.ProjectedEvents) > self.TM._TrackerMaxEventsBuffer:
                self.ProjectedEvents.pop(0)
                self.AssociatedFlows.pop(0)
                self.AssociatedOffsets.pop(0)
            if self.TrackerActivity > self.TM._DetectorMinActivityForStart: # This StatusUpdate is put here, since it would be costly in computation to have it somewhere else and always be checked 
                self.State.SetStatus(self.State._STATUS_STABILIZING)
                self.TM.StartTimes[self.ID] = event.timestamp
            return True

        if self.Lock:
            CurrentProjectedEvent[0] = self.Lock.Time + self.TimeConstant
            UsedEventsList = self.Lock.Events
            self.ProjectedEvents = self.ProjectedEvents[-(self.TM._TrackerLockingMaxRecoveryBuffer-1):] + [SavedProjectedEvent]
        else:
            while len(self.ProjectedEvents) > self.TM._TrackerMaxEventsBuffer or (len(self.ProjectedEvents) > 0 and self.LastUpdate - self.ProjectedEvents[0][0] >= self.EdgeBinTC): # We look for the first event recent enough. We remove for events older than N Tau, meaning a participation of 5%
                self.ProjectedEvents.pop(0)
                self.AssociatedFlows.pop(0)
                self.AssociatedOffsets.pop(0)
            UsedEventsList = self.ProjectedEvents
            self.ProjectedEvents += [SavedProjectedEvent]

        FlowSuccess, FlowError, ProjectionError, LocalEdge = self._ComputeLocalErrorAndEdge(CurrentProjectedEvent, UsedEventsList) # In the TRS model, the flow is different from a simple speed error.
        #FlowError, ProjectionError = self._CorrectFlowOrientationAndNorm(FlowError, ProjectionError) # The events are compensated in rotation and scaling, but we need to keep them relative to their retated location for now. 
        #The translation speed is the one to be corrected
        if self.Lock:
            self.AssociatedFlows = self.AssociatedFlows[:len(self.ProjectedEvents)-1] + [FlowError]
            self.AssociatedOffsets = self.AssociatedFlows[:len(self.ProjectedEvents)-1] + [ProjectionError]
        else:
            self.AssociatedFlows += [FlowError]
            self.AssociatedOffsets += [ProjectionError]
        if not FlowSuccess: # Computation could not be performed, due to not enough events
            return True

        self.LastValidFlow = event.timestamp
        self.FlowActivity += 1
        self.DynamicsEstimator.AddData(event.timestamp, CurrentProjectedEvent[1:], FlowError, ProjectionError, self.TimeConstant*self.TM._DynamicsEstimatorTMRatio)
        self.DynamicsEstimator.RecoverGeneralData()

        if self.TM._ConvergenceEstimatorType == 'CF':
            self.SpeedConvergenceEstimator.AddData(event.timestamp, FlowError, self.TimeConstant)
        elif self.TM._ConvergenceEstimatorType == 'PD':
            self.SpeedConvergenceEstimator.AddData(event.timestamp, ProjectionError, self.TimeConstant)
        else:
            raise Exception('Ill-defined convergence estimator type')
        self.ApertureEstimator.AddData(event.timestamp, LocalEdge, self.TimeConstant)

        if self.Lock or not self.TM._RSLockModel: #RSLockModel
            SpeedError = self.DynamicsEstimator.GetSpeed()
            if SpeedError is None: # We could not access TRS speed errors, as matrix is singular. We have to wait for more events to aggregate
                return True
            DisplacementError = self.DynamicsEstimator.GetDisplacement()
        else:
            SpeedError = np.array([FlowError[0], FlowError[1], 0., 0.])
            DisplacementError = np.array([ProjectionError[0], ProjectionError[1], 0., 0.])
            #SpeedError = self.DynamicsEstimator.ReDimVector(self.DynamicsEstimator.Es)
            #SpeedError[2:] = 0
            #DisplacementError = self.DynamicsEstimator.ReDimVector(self.DynamicsEstimator.Ed)
            #DisplacementError[2:] = 0

        if self.Lock:
            SpeedMod = self.TM._NPixelsEdge * self.TM._LockedSpeedModReduction * SpeedError / self.Lock.TrackerActivity ** self.TM._TrackerLockedSpeedModActivityPower
            PositionMod = self.TM._NPixelsEdge * self.TM._LockedPosModReduction * DisplacementError / self.Lock.TrackerActivity ** self.TM._TrackerLockedDisplacementModActivityPower
        else:
            SpeedMod = self.TM._NPixelsEdge * SpeedError / self.TrackerActivity ** self.TM._TrackerUnlockedSpeedModActivityPower
            PositionMod = self.TM._NPixelsEdge * DisplacementError / self.TrackerActivity ** self.TM._TrackerUnlockedDisplacementModActivityPower

        self.DynamicsModifier.ModSpeed('Flow', self._CorrectModificationOrientationAndNorm(SpeedMod * self.TM._TrackerAccelerationFactor))
        self.DynamicsModifier.ModPosition('Flow', self._CorrectModificationOrientationAndNorm(PositionMod * self.TM._TrackerDisplacementFactor))

        self._ComputeRecenter(self)

        self._UpdateTC()

        return self.ComputeCurrentStatus(event.timestamp)

    def ComputeCurrentStatus(self, t): # Cannot be called when DEAD or IDLE. Thus, we first check evolutions for aperture issue and lock properties, then update the status
        self.State.OffCentered = (not self.State.Locked) and (np.linalg.norm(self.DynamicsEstimator.X) > self.TM._TrackerOffCenterThreshold * self.Radius)

        if not self.State.ApertureIssue and self.ApertureEstimator.Value > self.TM._TrackerApertureIssueThreshold + self.TM._TrackerApertureIssueHysteresis:
            self.State.ApertureIssue = True
            Reason = "aperture issue"
        elif self.State.ApertureIssue and self.ApertureEstimator.Value < self.TM._TrackerApertureIssueThreshold - self.TM._TrackerApertureIssueHysteresis:
            self.State.ApertureIssue = False

        CanBeLocked = True
        if self.State.ApertureIssue:
            CanBeLocked = False
        elif self.TrackerActivity > self.TM._TrackerLockMaxRelativeActivity * self.TM._TrackerDiameter:
            CanBeLocked = False
            Reason = "excessive absolute activity"
        elif self.FlowActivity < (self.TM._TrackerLockedRelativeCorrectionsFailures - self.TM._TrackerLockedRelativeCorrectionsHysteresis) * self.TrackerActivity:
            CanBeLocked = False
            Reason = "unsufficient number of points matching"

        if self.TM._LockOnlyCentered:
            if self.State.OffCentered:
                CanBeLocked = False
                Reason = "non centered"
            elif not self.State.Locked and t < self.LastRecenter + self.TimeConstant: # We dont want to release a locked tracker that got so much speed that the second condition gets checked
                CanBeLocked = False
                Reason = "recent recenter"

        if self.State.Locked and not CanBeLocked:
            self.Unlock(Reason)
            if not self.TM._TrackerUnlockedSurvives:
                self.TM._KillTracker(self, t, Reason = Reason)
                return False
            return True

        if self.State.Converged or self.State.Locked:
            if self.SpeedConvergenceEstimator.Value[0] > (self.TM._TrackerConvergenceThreshold + self.TM._TrackerConvergenceHysteresis): # Part where we downgrade to stabilizing, when the corrections are too great
                if self.State.Locked:
                    if not self.TM._TrackerLockedCanHardCorrect:
                        Reason = "excessive correction"
                        self.Unlock(Reason)
                        if not self.TM._TrackerUnlockedSurvives:
                            self.TM._KillTracker(self, t, Reason = Reason)
                            return False
                        self.State.SetStatus(self.State._STATUS_STABILIZING)
                else:
                    self.State.SetStatus(self.State._STATUS_STABILIZING)
                return True
            if not self.State.Locked and CanBeLocked and self.TM._TrackerAllowShapeLock:
                Reprojection, Spread = self.ProjectionSpread()
                if Reprojection < self.TM._LockupMaximumSpread:
                    if Spread > self.TM._LockupMaximumSpread:
                        self.TM._KillTracker(self, t, 'excessive spread')
                        return False
                    
                    self.State.SetStatus(self.State._STATUS_LOCKED)
                    self.Lock = LockClass(t, self.TrackerActivity, self.FlowActivity, list(self.ProjectedEvents + [np.array([0., 0., 0.])])) # Added event at the end allows for simplicity in neighbours search
                    self.LocksSaves += [LockClass(t, self.TrackerActivity, self.FlowActivity, list(self.ProjectedEvents))]
                    
                    self.TM._OnTrackerLock(self)
                    
                    self.TM.Log("Tracker {0} has locked".format(self.ID), 3)
        elif self.State.Stabilizing:
            if self.SpeedConvergenceEstimator.Value[0] < (self.TM._TrackerConvergenceThreshold - self.TM._TrackerConvergenceHysteresis):
                if self.FlowActivity >= (self.TM._TrackerLockedRelativeCorrectionsFailures + self.TM._TrackerLockedRelativeCorrectionsHysteresis) * self.TrackerActivity:
                    self.State.SetStatus(self.State._STATUS_CONVERGED)
        else:
            self.TM.LogWarning("Unexpected status, ref. 001")
        return True

    def Unlock(self, Reason):
        self.LocksSaves[-1].ReleaseTime = self.LastUpdate
        self.LocksSaves[-1].Reason = Reason
        self.Lock = None
        if self.TM._RSLockModel: # If we can't correct rotation and scaling speeds, we cannot let them stay for now
            SpeedCancelation = -np.array(self.Speed)
            SpeedCancelation[:2] = 0
            self.DynamicsModifier.ModSpeed('Flow', SpeedCancelation)
        self.TM.FeatureManager.RemoveLock(self)
        self.TM.LogWarning("Tracker {0} was released ({1})".format(self.ID, Reason))
        self.State.SetStatus(self.State._STATUS_CONVERGED)

    def ProjectionSpread(self):
        Support = np.zeros((int(2*self.Radius+1), int(2*self.Radius+1)), dtype = int)
        for Event in self.ProjectedEvents:
            Support[int(round((Event[1] + self.Radius))), int(round((Event[2] + self.Radius)))] += 1
        return ((Support==1).sum()/len(self.ProjectedEvents), (Support>0).sum() / (np.pi * self.Radius**2))

    def Disengage(self, t):
        self.State.Disengaged = True
        self.DynamicsModifier.ModPosition('Disengage', -(t - self.LastValidFlow) * self.Speed) # We go back to the last valid flow position. Nothing should have been able to change the speed between those two moments, and only the inertia modified the position
        self.TM.LogWarning("Tracker {0} disengaged".format(self.ID))
    def Reengage(self): # We put that inside dedicated function for clarity.
        self.State.Disengaged = False
        self.TM.Log("Tracker {0} has re-engaged".format(self.ID), 3)

    def _UpdateTC(self):
        vx, vy, w, s = self.Speed
        TrackerAverageSpeed = np.sqrt(vx**2 + vy**2 + self.DynamicsEstimator.r2 * (w**2 + s**2) + 2 * (self.DynamicsEstimator.X * np.array([vy*w + vx*s, -vx*w + vy*s])).sum()) # If dynamics estimator is computed from tracker center
        #TrackerAverageSpeed = np.sqrt(vx**2 + vy**2 + self.DynamicsEstimator.r2 * (w**2 + s**2)) # If dynamics estimator is computed from events center
        if TrackerAverageSpeed == 0:
            self.TimeConstant = self.TM._TrackerDefaultTimeConstant
        else:
            self.TimeConstant = min(self.TM._TrackerDefaultTimeConstant, self.TM._NPixelsEdge / TrackerAverageSpeed)
        self.EdgeBinTC = self.TimeConstant * self.TM._EdgeBinRatio

    def _CorrectModificationOrientationAndNorm(self, Mod):
        cs, ss = self.Position[3]*np.cos(self.Position[2]), self.Position[3]*np.sin(self.Position[2])
        M = np.array([[cs, -ss], [ss, cs]])
        Mod[:2] = M.dot(Mod[:2])
        return Mod

    def _CenterContinuous(self):
        if (not self.State.Locked) and np.linalg.norm(self.DynamicsEstimator.X) > self.TM._TrackerSpringThreshold * self.Radius:
            PositionMod = np.array([0., 0., 0., 0.])
            PositionMod[:2] = self.TM._TrackerMeanPositionDisplacementFactor * self.DynamicsEstimator.X / self.TrackerActivity
            self.MeanPosCorrection += PositionMod[:2]
            self.DynamicsModifier.ModPosition('MeanPos', self._CorrectModificationOrientationAndNorm(PositionMod))
    def _CenterStep(self):
        if self.State.Converged and self.State.OffCentered and self.LastUpdate > self.LastRecenter + self.TimeConstant*2:
            Shift = np.array([0., 0., 0., 0.])
            Shift[:2] = self.TM._TrackerMeanPositionDisplacementFactor * np.array(self.DynamicsEstimator.X)
            self.DynamicsEstimator.TrackerShift(Shift[:2])

            ProjectedEvents = []
            for Event in self.ProjectedEvents:
                NewLoc = Event[1:] - Shift[:2]
                if np.linalg.norm(NewLoc) <= self.Radius:
                    ProjectedEvents += [np.array([Event[0], NewLoc[0], NewLoc[1]])]
            self.ProjectedEvents = ProjectedEvents

            self.DynamicsModifier.ModPosition('MeanPos', Shift)
            self.LastRecenter = self.LastUpdate
    def _CenterNone(self):
        pass

    def _ComputeLocalErrorAndEdge(self, CurrentProjectedEvent, PreviousEvents):
        ConsideredNeighbours = []
        LocalEdgePoints = []
        LocalEdgeSquaredPoints = []
        CurrentProjectedEvent[1:] += self.TM._MeanPositionDampeningFactor * self.MeanPosCorrection # Minimize effects of recentering
        for PreviousEvent in reversed(PreviousEvents[:-1]): # We reject the last event, that is either the CPE, or a fake event in Lock.Events
            D = np.linalg.norm(CurrentProjectedEvent[1:] - PreviousEvent[1:])
            if D < self.TM._LocalEdgeRadius:
                if len(LocalEdgePoints) < self.TM._LocalEdgeNumberOfEvents:
                    LocalEdgePoints += [PreviousEvent[1:]]
#                    LocalEdgeSquaredPoints += [PreviousEvent[1:]**2] # 87449
                if D < self.TM._ClosestEventProximity:
                    if self.Lock or (CurrentProjectedEvent[0]- PreviousEvent[0]) > self.TM._ObjectEdgePropagationTC:
                    #if self.Lock or (CurrentProjectedEvent[0]- PreviousEvent[0]) > self.TimeConstant/(2*self.TM._NPixelsEdge): #Experimental
                        ConsideredNeighbours += [np.array(PreviousEvent)]
                        if len(ConsideredNeighbours) == self.TM._MaxConsideredNeighbours:
                            break

        if len(ConsideredNeighbours) < self.TM._MinConsideredNeighbours:
            return False, np.array([0., 0.]), np.array([0., 0.]), np.array([0., 0.])

        SpeedError, DeltaPos, MeanPoint = self._ComputeSpeedError(self, CurrentProjectedEvent, np.array(ConsideredNeighbours))

        
#        LocalEdgePoints = np.array(LocalEdgePoints) # 87449
#        SquareMeanPoint = np.mean(LocalEdgeSquaredPoints, axis = 0)
#        LocalEdge = np.array([(SquareMeanPoint[0] - MeanPoint[0]**2) - (SquareMeanPoint[1] - MeanPoint[1]**2), 2*((LocalEdgePoints[:,0]*LocalEdgePoints[:,1]).mean() - MeanPoint[0]*MeanPoint[1])])

        LocalEdgePoints = np.array(LocalEdgePoints)
        LocalEdgeVectors = np.zeros(LocalEdgePoints.shape)
        LocalEdgeVectors[:,0] = LocalEdgePoints[:,0] - MeanPoint[0]
        LocalEdgeVectors[:,1] = LocalEdgePoints[:,1] - MeanPoint[1]
        LocalEdgeNorms = np.linalg.norm(LocalEdgeVectors, axis = 1)
        #LocalEdgeVectors[:,0] /= LocalEdgeNorms
        #LocalEdgeVectors[:,1] /= LocalEdgeNorms

        LocalEdgeRotatedVectors = RotateVectors(LocalEdgeVectors)
        #LocalEdgeRotatedVectors[:,0] *= LocalEdgeNorms
        #LocalEdgeRotatedVectors[:,0] *= LocalEdgeNorms
        LocalEdge = LocalEdgeRotatedVectors.mean(axis = 0)
        NEdge = np.linalg.norm(LocalEdge)
        if NEdge > 0:
            LocalEdge /= NEdge

        if self.TM._ProjectFlow:
            xy = (1+np.array([LocalEdge[0], -LocalEdge[0]]))/2
            deltaxy = np.sqrt(abs(xy))
            if LocalEdge[1] < 0:
                deltaxy[0] *= -1
            SpeedError = SpeedError - ((SpeedError*deltaxy).sum()) * deltaxy
            DeltaPos = DeltaPos - ((DeltaPos*deltaxy).sum()) * deltaxy

        return True, SpeedError, DeltaPos, LocalEdge

    def _ComputeSpeedErrorPlanFit(self, CurrentProjectedEvent, ConsideredNeighbours):
        ConsideredNeighbours = np.concatenate((ConsideredNeighbours, np.array([CurrentProjectedEvent])))
        tMean = ConsideredNeighbours[:,0].mean()
        xMean = ConsideredNeighbours[:,1].mean()
        yMean = ConsideredNeighbours[:,2].mean()
        DeltaPos = CurrentProjectedEvent[1:] - np.array([xMean, yMean])

        tDeltas = ConsideredNeighbours[:,0] - tMean
        xDeltas = ConsideredNeighbours[:,1] - xMean
        yDeltas = ConsideredNeighbours[:,2] - yMean

        Sx2 = (xDeltas **2).sum()
        Sy2 = (yDeltas **2).sum()
        Sxy = (xDeltas*yDeltas).sum()
        Stx = (tDeltas*xDeltas).sum()
        Sty = (tDeltas*yDeltas).sum()

        Det = Sx2*Sy2 - Sxy**2
        Fx = Sy2*Stx - Sxy*Sty
        Fy = Sx2*Sty - Sxy*Stx
        DetOverN2 = Det / (Fx**2 + Fy**2)

        SpeedError = np.array([Fx * DetOverN2, Fy * DetOverN2])
        return SpeedError, DeltaPos
    def _ComputeSpeedErrorLinearMean(self, CurrentProjectedEvent, ConsideredNeighbours):
        MeanPoint = np.array([(ConsideredNeighbours[:,1]*ConsideredNeighbours[:,0]).sum(), (ConsideredNeighbours[:,2]*ConsideredNeighbours[:,0]).sum()]) / ConsideredNeighbours[:,0].sum()
        MeanTS = ConsideredNeighbours[:,0].mean()

        DeltaPos = (CurrentProjectedEvent[1:] - MeanPoint)
        SpeedError = DeltaPos / (CurrentProjectedEvent[0] - MeanTS)
        return SpeedError, DeltaPos, MeanPoint
    def _ComputeSpeedErrorBareMean(self, CurrentProjectedEvent, ConsideredNeighbours):
        MeanPoint = ConsideredNeighbours.mean(axis = 0)

        DeltaPos = (CurrentProjectedEvent - MeanPoint)
        SpeedError = DeltaPos[1:] / DeltaPos[0]
        return SpeedError, DeltaPos[1:], MeanPoint[1:]
    def _ComputeSpeedErrorExponentialMean(self, CurrentProjectedEvent, ConsideredNeighbours):
        Weights = np.e**((ConsideredNeighbours[:,0] - CurrentProjectedEvent[0]) / (self.TimeConstant * self.TM._EdgeBinRatio))
        MeanPoint = np.array([(ConsideredNeighbours[:,1] * Weights).sum(), (ConsideredNeighbours[:,2] * Weights).sum()]) / Weights.sum()
        MeanTS = (ConsideredNeighbours[:,0] * Weights).sum() / Weights.sum()

        DeltaPos = (CurrentProjectedEvent[1:] - MeanPoint)
        SpeedError = DeltaPos / (CurrentProjectedEvent[0] - MeanTS)
        return SpeedError, DeltaPos

    def PlotShape(self, LockSave = -1, Center = np.array([0., 0.]), Ratio = 1., OnlyReferenceShape = False, MinAlpha = 0., AddText = True, fax = None, CircleColor = 'g', Interactive = True, ms = 1):
        if fax is None:
            f, ax = plt.subplots(1,1)
        else:
            f, ax = fax
        ax.set_aspect('equal')
        if (Center == 0).all() and Ratio == 1:
            ax.set_xlim(-self.Radius*1.02*Ratio + Center[0], self.Radius*1.02*Ratio + Center[0])
            ax.set_ylim(-self.Radius*1.02*Ratio + Center[1], self.Radius*1.02*Ratio + Center[1])
        Zone = plt.Circle(Center, self.Radius * Ratio, color=CircleColor, fill=False, linewidth = 2)
        if Interactive:
            EventZone = plt.Circle((-100, -100), self.TM._ClosestEventProximity, color='k', fill=False, linestyle = '--')
            OffsetUsed = plt.plot(-100, -100, 'vg', zorder = 10)[0]
            FlowUsed = plt.plot([-100, -101], [-100, -101], 'r')[0]
            ax.add_artist(EventZone)
        ax.add_artist(Zone)
        title = "Tracker {0}, Current time : t = {1:.3f}".format(self.ID, self.LastUpdate)
        if not LockSave is None and self.LocksSaves:
            Lock = self.LocksSaves[LockSave]
        else:
            Lock = None
        if Lock:
            title+=", Lock Time : {0:.3f}".format(Lock.Time)
            if not Lock.ReleaseTime is None:
                title += ", Release : {0:.3f}".format(Lock.ReleaseTime)
        title +="\nEvent time : {0:.3f}"
        if AddText:
            ax.set_title(title)
        MaxFlow = np.linalg.norm(self.AssociatedFlows, axis = 1).max()
        PlottedList = list(self.ProjectedEvents)
        if not Lock or not OnlyReferenceShape:
            for E in self.ProjectedEvents:
                alpha = np.e**((E[0]-self.ProjectedEvents[-1][0])/self.TimeConstant)
                if alpha < MinAlpha:
                    continue
                ax.plot(Center[0] + E[1]*Ratio, Center[1] + E[2]*Ratio, 'ob', alpha = alpha, markersize = ms)
        if Lock:
            PlottedList += Lock.Events[:-1]
            for E in Lock.Events[:-1]:
                alpha = np.e**((E[0]-Lock.Time)/self.TimeConstant)
                if alpha < MinAlpha:
                    continue
                ax.plot(Center[0] + E[1]*Ratio, Center[1] + E[2]*Ratio, 'og', alpha = alpha, markersize = ms)
        if not Interactive:
            return f, ax
        PlottedList = np.array(PlottedList)
        def onclick(event):
            EventConsidered = np.linalg.norm(PlottedList[:,1:] - np.array([event.xdata, event.ydata]), axis = 1).argmin()
            if EventConsidered < len(self.ProjectedEvents):
                E = PlottedList[EventConsidered]
                Flow = self.AssociatedFlows[EventConsidered]
                Offset = self.AssociatedOffsets[EventConsidered]
                N = np.linalg.norm(Flow)
                if N != 0:
                    Flow = Flow * 5 / N
                    FlowUsed.set_data([E[1], E[1]+Flow[0]], [E[2], E[2]+Flow[1]])
                    FlowUsed.set_alpha(N/MaxFlow)
                else:
                    FlowUsed.set_data([-100, -101], [-100, -101])
                OffsetUsed.set_data(E[1] - Offset[0], E[2] - Offset[1])
            else:
                OffsetUsed.set_data(-100, -100)
                FlowUsed.set_data([-100, -101], [-100, -101])
            if np.linalg.norm(PlottedList[EventConsidered,1:]-np.array([event.xdata, event.ydata])) < 1:
                ax.set_title(title.format(PlottedList[EventConsidered,0]))
                EventZone.set_center(tuple(PlottedList[EventConsidered,1:]))
            else:
                ax.set_title(title)
                EventZone.set_center((-100,-100))
            plt.show()

        cid = f.canvas.mpl_connect('button_press_event', onclick)
        return f, ax
