import numpy as np

class _EventContainerClass: # Holds the timestamp and manages the subStreamIndexes and extensions. Should in theory never be used as such.
    def __init__(self, timestamp = None, FirstEvent = None, FirstSubStreamIndex = None, Bare = False):
        if not Bare:
            self.timestamp = timestamp
            self.Events = {FirstSubStreamIndex: [FirstEvent]}
        else:
            self.BareEvent = _BareEventClass(self)
            self.timestamp = None
            self.Events = {}
    def _AddEvent(self, Event):
        Index = Event.SubStreamIndex
        if not Index in self.Events:
            self.Events[Index] = [Event]
        else:
            self.Events[Index].append(Event)
    def GetEvents(self, SubStreamRestriction = {}):
        RequestedEvents = []
        for SubStreamIndex, Events in self.Events.items():
            if not SubStreamRestriction or SubStreamIndex in SubStreamRestriction:
                RequestedEvents += Events
        return RequestedEvents
    def Filter(self, event):
        self.Events[event.SubStreamIndex].remove(event)
        if len(self.Events[event.SubStreamIndex]) == 0:
            del self.Events[event.SubStreamIndex]
    @property
    def IsEmpty(self):
        return len(self.Events) == 0
    @property
    def IsFilled(self):
        return len(self.Events) != 0
    def __eq__(self, rhs):
        return self.timestamp == rhs.timestamp
    def __lt__(self, rhs):
        return self.timestamp < rhs.timestamp
    def __le__(self, rhs):
        return self.timestamp <= rhs.timestamp
    def __gt__(self, rhs):
        return self.timestamp > rhs.timestamp
    def __ge__(self, rhs):
        return self.timestamp >= rhs.timestamp

class _BareEventClass: # First basic event given to an input module. That input module is expected to join another event to this one, restructuring internally the event packet
    def __init__(self, Container):
        self._Container = Container
    def Join(self, Extension, **kwargs):
        if 'SubStreamIndex' in kwargs:
            SubStreamIndex = kwargs['SubStreamIndex']
            del kwargs['SubStreamIndex']
        else:
            raise Exception("No SubStreamIndex specified during first event creation")
        if self._Container.timestamp is None:
            if 'timestamp' in kwargs:
                self._Container.timestamp = kwargs['timestamp']
                del kwargs['timestamp']
            else:
                raise Exception("No timestamp specified during first event creation")
        self._Container._AddEvent(Extension(Container = self._Container, SubStreamIndex = SubStreamIndex, **kwargs))
        del self._Container.__dict__['BareEvent']
        return self._Container.Events[SubStreamIndex][-1]
    def SetTimestamp(self, t):
        self._Container.timestamp = t

class _EventClass:
    def __init__(self, **kwargs):
        if not 'Container' in kwargs:
            self._Container = _EventContainerClass(kwargs['timestamp'], self, kwargs['SubStreamIndex'])
        else:
            self._Container = kwargs['Container']
            del kwargs['Container']
        self.SubStreamIndex = kwargs['SubStreamIndex']
        del kwargs['SubStreamIndex']
        self._Extensions = set()
        if 'Extensions' in kwargs:
            for Extension in kwargs['Extensions']:
                self.Attach(Extension, **kwargs)
    def _Attach(self, Extension, **kwargs):
        # Used to assess that the event was just created, so event information have to be attached anyway
        self._Extensions.add(Extension)
        for Field in Extension._Fields:
            self.__dict__[Field] = kwargs[Field]
    def Attach(self, Extension, **kwargs):
        if not Extension._CanAttach or Extension in self._Extensions:
            self.Join(Extension, **kwargs) # For now, its better to join when instance is already there (ex: multiple TrackerEvents)
            return
        self._Extensions.add(Extension)
        for Field in Extension._Fields:
            self.__dict__[Field] = kwargs[Field]
    def Join(self, ExtensionOrEvent, **kwargs):
        if type(ExtensionOrEvent) == _EventClass:
            self._Container._AddEvent(ExtensionOrEvent)
            return ExtensionOrEvent
        if 'SubStreamIndex' in kwargs:
            SubStreamIndex = kwargs['SubStreamIndex']
            del kwargs['SubStreamIndex']
        else:
            SubStreamIndex = self.SubStreamIndex
        self._Container._AddEvent(ExtensionOrEvent(Container = self._Container, SubStreamIndex = SubStreamIndex, **kwargs))
        return self._Container.Events[SubStreamIndex][-1]
    def Filter(self):
        self._Container.Filter(self)
    def AsList(self, Keys = ()):
        Output = [self.timestamp]
        for Extension in self._Extensions:
            if Keys and Extension._Key not in Keys:
                continue
            Output += [[Extension._Key]+[self.__dict__[Field]for Field in Extension._Fields]]
        return Output
    def AsDict(self, Keys = ()):
        Output = {0:self.timestamp}
        for Extension in self._Extensions:
            if Keys and Extension._Key not in Keys:
                continue
            Output[Extension._Key] = [self.__dict__[Field] for Field in Extension._Fields]
        return Output
    def Copy(self, SubStreamIndex = None):
        if SubStreamIndex is None:
            SubStreamIndex = self.SubStreamIndex
        kwargs = {'timestamp':self.timestamp, 'SubStreamIndex':SubStreamIndex, 'Extensions':self._Extensions}
        for Extension in self._Extensions:
            for Field in Extension._Fields:
                if type(self.__dict__[Field]) != np.ndarray:
                    kwargs[Field] = type(self.__dict__[Field])(self.__dict__[Field])
                else:
                    kwargs[Field] = np.array(self.__dict__[Field])
        return self.__class__(**kwargs)
    @property
    def timestamp(self):
        return self._Container.timestamp
    def Has(self, Extension):
        return (Extension in self._Extensions)
    def __eq__(self, rhs):
        return self.timestamp == rhs.timestamp
    def __lt__(self, rhs):
        return self.timestamp < rhs.timestamp
    def __le__(self, rhs):
        return self.timestamp <= rhs.timestamp
    def __gt__(self, rhs):
        return self.timestamp > rhs.timestamp
    def __ge__(self, rhs):
        return self.timestamp >= rhs.timestamp
    def __repr__(self):
        return "{0:.3f}s".format(self.timestamp)

# Listing all the events existing

class _EventExtensionClass:
    _Key = -1 # identifier for this type of events
    _CanAttach = True
    _Fields = ()
    def __new__(cls, *args, **kwargs):
        Event = _EventClass(**kwargs)
        Event._Attach(cls, **kwargs)
        return Event

class CameraEvent(_EventExtensionClass):
    _Key = 1
    _Fields = ('location', 'polarity')
class TrackerEvent(_EventExtensionClass):
    _Key = 2
    _CanAttach = False # From experience, many trackers can be updated upon a single event. For equity, all trackers are joined, not attached
    _Fields = ('TrackerLocation', 'TrackerID', 'TrackerAngle', 'TrackerScaling', 'TrackerColor', 'TrackerMarker')
class DisparityEvent(_EventExtensionClass):
    _Key = 3
    _Fields = ('disparity', 'sign')
class PoseEvent(_EventExtensionClass):
    _Key = 4
    _Fields = ('poseHomography', 'worldHomography', 'reprojectionError')
class TauEvent(_EventExtensionClass):
    _Key = 5
    _Fields = ('tau',)
class FlowEvent(_EventExtensionClass):
    _Key = 6
    _Fields = ('flow',)
class TwistEvent(_EventExtensionClass):
    _Key = 7
    _Fields = ('omega', 'v')

_AvailableEventsClassesNames = [EventType.__name__ for EventType in _EventExtensionClass.__subclasses__()]

