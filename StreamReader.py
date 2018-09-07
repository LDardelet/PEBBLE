import  tools
import numpy as np
from struct import unpack
from sys import stdout
import atexit

from event import Event

class Reader:
    DefaultGeometry = [304,240,2]
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to read events streams files.
        '''
        self.__ReferencesAsked__ = []
        self.__Name__ = Name
        self.__Framework__ = Framework
        self.__Type__ = 'Input'
        self.__CreationReferences__ = dict(argsCreationReferences)

        self._RemoveNegativeTimeDifferences = True
        self._SaveStream = False
        self._BatchEventSize = 1000
        self._TryUseDefaultGeometry = True

        self.CurrentFile = None

    def _Initialize(self, **kwargs):
        self._CloseFile()
        self.StreamName = self.__Framework__.StreamHistory[-1]

        if not '.dat' in self.StreamName:
            print "Invalid filename."
            return False

        self.CurrentFile = open(self.StreamName,'rb')

        HeaderHandled = self.DealWithHeader()
        if self.CurrentGeometry == self.DefaultGeometry:
            print "No geometry found."
            if self._TryUseDefaultGeometry:
                print "Using default geometry : {0}".format(self.CurrentGeometry)
            else:
                self._CloseFile()
                return False
        self.__Framework__.StreamsGeometries[self.StreamName] = self.CurrentGeometry
        self.yMax = self.CurrentGeometry[1] - 1

        atexit.register(self._CloseFile)

        if self._SaveStream:
            self.CurrentStream = []
            self._StorageFunction = self._StoreEvent
        else:
            self._StorageFunction = self._DoNothing

        self.tMask = 0x00000000FFFFFFFF
        self.pMask = 0x1000000000000000
        self.xMask = 0x00003FFF00000000
        self.yMask = 0x0FFFC00000000000
        self.pPadding = 60
        self.yPadding = 46
        self.xPadding = 32
        
        #self.pMask = 0x0002000000000000
        #self.xMask = 0x000001FF00000000
        #self.yMask = 0x0001FE0000000000
        #self.pPadding = 49
        #self.yPadding = 41
        #self.xPadding = 32

        # Analyzing file size
        start = self.CurrentFile.tell()
        self.CurrentFile.seek(0,2)
        stop = self.CurrentFile.tell()
        self.CurrentFile.seek(start)
    
        self.NEvents = int( (stop-start)/self.Event_Size)
        dNEvents = self.NEvents/100
        print("> The file contains %d events." %self.NEvents)

        self.nEvent = 0
        self.CurrentByteBatch = ''

        self.PreviousEventTs = -np.inf

        return True

    def FastForward(self, t):
        try:
            while True:
                self.nEvent += 1
                NextEvent = self._NextEvent()
                if self.nEvent%2048 == 0:
                    stdout.write("t = {0:.3f} / {1:.3f}\r".format(NextEvent.timestamp, t))
                    stdout.flush()
                if self.nEvent >= self.NEvents:
                    self.__Framework__.Running = False
                    print ""
                    print "Input reached EOF."
                    break
                if NextEvent.timestamp >= t:
                    print ""
                    print "Done."
                    break
        except KeyboardInterrupt:
            print ""
            print "Stopped fast forward at t = {0:.3f}".format(NextEvent.timestamp)


    def _OnEvent(self, event):
        if self.nEvent >= self.NEvents:
            self.__Framework__.Running = False
            print "Input reached EOF."
        
        self.nEvent += 1
        NextEvent = self._NextEvent()

        # Possibly save the event
        self._StorageFunction(NextEvent)

        # Send event in Framework
        return NextEvent

    def _NextEvent(self):
        if not len(self.CurrentByteBatch):
            self._LoadNewBatch()

        event = unpack('Q', self.CurrentByteBatch[:self.Event_Size])
        self.CurrentByteBatch = self.CurrentByteBatch[self.Event_Size:]

        ts = event[0] & self.tMask 
        if self._RemoveNegativeTimeDifferences:
            if ts < self.PreviousEventTs:
                self.NEvents -= 1 # Here we remove one event from counter, since we want nEvent to be continous but still stop when nEvent == NEvents
                return self._NextEvent()
            self.PreviousEventTs = ts

        # padding = event[0] & 0xFFFC000000000000
        p = (event[0] & self.pMask) >> self.pPadding
        y = (event[0] & self.yMask) >> self.yPadding
        x = (event[0] & self.xMask) >> self.xPadding

        return Event(float(ts) * 10**-6, np.array([x, self.yMax - y]), int(p))

    def _LoadNewBatch(self):
        self.CurrentByteBatch = self.CurrentFile.read(self._BatchEventSize * self.Event_Size)

    def _StoreEvent(self, event):
        self.CurrentStream += [event]

    def _DoNothing(self, event):
        None

    def _CloseFile(self):
        if not self.CurrentFile is None:
            self.CurrentFile.close()
            print "Closed file {0}".format(self.StreamName)
            self.CurrentFile = None

    def DealWithHeader(self):
        self.Header = False
        self.CurrentGeometry = list(self.DefaultGeometry)
        FoundHeightOrWidth = False
        while tools.peek(self.CurrentFile) == b'%':
            HeaderNextLine = self.CurrentFile.readline()
            self.Header = True
            if 'height' in HeaderNextLine.lower():
                FoundHeightOrWidth = True
                self.CurrentGeometry[1] = int(HeaderNextLine.lower().split('height')[1].strip())
            if 'width' in HeaderNextLine.lower():
                FoundHeightOrWidth = True
                self.CurrentGeometry[0] = int(HeaderNextLine.lower().split('width')[1].strip())

        if FoundHeightOrWidth:
            print "> Found stream geometry of {0}".format(self.CurrentGeometry)
        if self.Header:
            self.Event_Type = unpack('B',self.CurrentFile.read(1))[0]
            self.Event_Size = unpack('B',self.CurrentFile.read(1))[0]
        return self.Header

