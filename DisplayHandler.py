import tools
import atexit
import socket

class DisplayHandler:
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to handle the stream Display.
        '''
        self._ReferencesAsked = []
        self._Name = Name
        self._Framework = Framework
        self._Type = 'Display'
        self._CreationReferences = dict(argsCreationReferences)

        self._PostTransporters = {'Event':tools.SendEvent, 'Segment':tools.SendSegment}
        self._MainAddress =  ("localhost", 54242)
        self._Socket = None

        atexit.register(self.EndTransmission)

    def _Initialize(self):
        DisplayUp = tools.IsDisplayUp()
        if not DisplayUp:
            print "Aborting initialization process"
            self._Initialized = False
            return False

        print "Initializing DisplayHandler sockets."
        tools.DestroySocket(self._Socket)
        self._Socket = tools.GetDisplaySocket(self._Framework.StreamsGeometries[self._Framework.StreamHistory[-1]])

        tools.CleanMapForStream(self._Socket)

        tools.SendStreamData(self._Framework.ProjectFile, self._Framework.StreamHistory[-1], self._Socket)
        self.MainUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

        self.PostBox = []

        self._Initialized = True


    def _OnEvent(self, event):
        if not self._Initialized:
            return event
        self.PostBox += [event]
        self.Post()

        return event

    def Post(self):
        for item in self.PostBox:
            self._PostTransporters[item.__class__.__name__](item, self.MainUDP, self._MainAddress, self._Socket)
        self.PostBox = []

    def EndTransmission(self):
        tools.DestroySocket(self._Socket)
