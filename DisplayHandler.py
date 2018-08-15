import tools
import atexit
import socket

class DisplayHandler:
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to handle the stream Display.
        '''
        self.__ReferencesAsked__ = []
        self.__Name__ = Name
        self.__Framework__ = Framework
        self.__Type__ = 'Display'
        self.__CreationReferences__ = dict(argsCreationReferences)
        self.__Started__ = False

        self.__PostTransporters__ = {'Event':tools.SendEvent, 'Segment':tools.SendSegment}

        self._MainAddress =  ("localhost", 54242)

    def _Initialize(self):
        self.Socket = None
        atexit.register(self.EndTransmission)
        self.PostBox = []

        DisplayUp = tools.IsDisplayUp()
        if not DisplayUp:
            print "Aborting initialization process"
            self.__Started__ = False
            return False

        print "Initializing DisplayHandler sockets."
        tools.DestroySocket(self.Socket)
        self.Socket = tools.GetDisplaySocket(self.__Framework__.StreamsGeometries[self.__Framework__.StreamHistory[-1]])

        tools.CleanMapForStream(self.Socket)

        tools.SendStreamData(self.__Framework__.ProjectFile, self.__Framework__.StreamHistory[-1], self.Socket)
        self.MainUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

        self.__Started__ = True


    def _OnEvent(self, event):
        if self.__Started__:
            self.PostBox += [event]
            self.Post()

        return event

    def Post(self):
        for item in self.PostBox:
            self.__PostTransporters__[item.__class__.__name__](item, self.MainUDP, self._MainAddress, self.Socket)
        self.PostBox = []

    def EndTransmission(self):
        if not self.Socket is None:
            tools.DestroySocket(self.Socket)
