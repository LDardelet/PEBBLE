import tools
import atexit
import socket

class DisplayHandler:
    def __init__(self, argsCreationDict): 
        '''
        Class to handle the stream Display.
        '''
        self.PostTransporters = {'Event':tools.SendEvent, 'Segment':tools.SendSegment}
        self.MainAddress =  ("localhost", 54242)
        self.Socket = None

        self._Type = 'Display'

        for key in argsCreationDict.keys():
            self.__dict__[key.split('.')[2]] = argsCreationDict[key]

        atexit.register(self.EndTransmission)

    def _Initialize(self, argsInitializationDict):
        '''
        Expects:
        'Framework.StreamsGeometries' -> Dict of Streams Geometries
        'Framework.StreamHistory' -> List of previous streams
        'Framework.ProjectFile' -> Name of the current project file
        '''

        DisplayUp = tools.IsDisplayUp()
        if not DisplayUp:
            print "Aborting initialization process"
            self._Initialized = False
            return False

        print "Initializing DisplayHandler sockets."
        tools.DestroySocket(self.Socket)
        self.Socket = tools.GetDisplaySocket(argsInitializationDict['Framework.StreamsGeometries'][argsInitializationDict['Framework.StreamHistory'][-1]])

        tools.CleanMapForStream(self.Socket)

        tools.SendStreamData(argsInitializationDict['Framework.ProjectFile'], argsInitializationDict['Framework.StreamHistory'][-1], self.Socket)
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
            self.PostTransporters[item.__class__.__name__](item, self.MainUDP, self.MainAddress, self.Socket)
        self.PostBox = []

    def EndTransmission(self):
        tools.DestroySocket(self.Socket)
