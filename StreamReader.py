import  tools
import numpy as np
from struct import unpack
from sys import stdout
import atexit

from event import Event
from Framework import Module

_ES_HEADER_SIZE = 15

_EVENT_TYPE_HANDLED = {0: False, 1:True, 2:True, 3:False, 4:False}
_EVENT_TYPE_OVERFLOW_VALUE = {0:0b11111110, 1:0b1111111, 2:0b111111, 3:0b11111110, 4:0b11111110}
_EVENT_TYPE_RESET_BYTE = {0:0b1111111, 1:0b1111111, 2:0b111111, 3:0b1111111, 4:0b1111111}
_EVENT_TYPE_RESET_PADDING = {0: 1, 1:1, 2:2, 3:1, 4:1}
_EVENT_TYPE_NAME = {0:'Generic events', 1:'DVS events', 2:'ATIS events', 3:'Asynchronous & Modular Display events', 4:'Color events'}

_EVENT_TYPE_ADITIONNAL_BYTES = {0: None, 1:4, 2:4, 3:3, 4:7}
_EVENT_TYPE_ANALYSIS_FUNCTIONS_NAMES = {0: None, 1: '_DVS_BYTES_ANALYSIS', 2: '_ATIS_BYTES_ANALYSIS', 3: None, 4:None} # All necessary variables are hardcoded in these functions, as they don't have necessarily the same output and needs

class Reader(Module):
    DefaultGeometry = [304,240,2]
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to read events streams files.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Input'

        self._RemoveNegativeTimeDifferences = True
        self._SaveStream = False
        self._BatchEventSize = 1000
        self._TryUseDefaultGeometry = True
        self._yInvert = True

        self.CurrentFile = None

    def _Initialize(self, **kwargs):
        Module._Initialize(self, **kwargs)

        self._CloseFile()
        self.StreamName = self.__Framework__.StreamHistory[-1]

        if '.dat' in self.StreamName:
            self._NextEvent = self._NextEventDat
            return self._InitializeDat()
        elif '.es' in self.StreamName:
            self._NextEvent = self._NextEventEs
            return self._InitializeEs()
        else:
            print "Invalid filename."
            return False

    def _OnEvent(self, event):
        self.nEvent += 1
        NextEvent = self._NextEvent()

        if self.__Framework__.Running:
            # Possibly save the event
            self._StorageFunction(NextEvent)
            
            # Send event in Framework
            return NextEvent

        else:
            return None

    def FastForward(self, t):
        try:
            while self.__Framework__.Running:
                self.nEvent += 1
                NextEvent = self._NextEvent()
                if self.nEvent%2048 == 0:
                    stdout.write("t = {0:.3f} / {1:.3f}\r".format(NextEvent.timestamp, t))
                    stdout.flush()
                if NextEvent.timestamp >= t:
                    print ""
                    print "Done."
                    break
        except KeyboardInterrupt:
            print ""
            print "Stopped fast forward at t = {0:.3f}".format(NextEvent.timestamp)

    def _InitializeEs(self):
        self.CurrentFile = open(self.StreamName,'rb')

        if not self._DealWithHeaderEs():
            return False
        
        self.__Framework__.StreamsGeometries[self.StreamName] = self.CurrentGeometry

        self.yMax = self.CurrentGeometry[1] - 1

        atexit.register(self._CloseFile)

        if self._SaveStream:
            self.CurrentStream = []
            self._StorageFunction = self._StoreEvent
        else:
            self._StorageFunction = self._DoNothing

        self.nEvent = 0 # Global counter to locate oneself in the stream
        self.nByte = -1 # Local counter for the position in the buffer
        self.CurrentByteBatch = ''

        self.PreviousEventTs = 0

        return True

    def _NextEventEs(self):
        while True:

            if len(self.CurrentByteBatch) - self.nByte < 6:
                self.CurrentByteBatch = self.CurrentByteBatch[self.nByte + 1:]
                self.nByte = -1
                self._LoadNewBatch()

                if not self.__Framework__.Running:
                    return None

            self.nByte += 1
            Bytes = [unpack('B', self.CurrentByteBatch[self.nByte])[0]]
            if (Bytes[0] >> self.ResetPadding) == self.ResetByte: # Check for reset or overflow
                Overflow = Bytes[0] & 0x03
                
                self.PreviousEventTs += self.OverflowValue * Overflow
                continue

            for i in range(self.AditionnalBytes):
                self.nByte += 1
                Bytes.append(unpack('B', self.CurrentByteBatch[self.nByte])[0])

            CreatedEvent = self._BytesAnalysisFunction(self, Bytes)

            if not CreatedEvent is None:
                return CreatedEvent

    def _DVS_BYTES_ANALYSIS(self, Bytes):
        td = Bytes[0] >> 1
        self.PreviousEventTs += td

        p = Bytes[0] & 0b00000001
        
        x = (Bytes[2] << 8) | (Bytes[1])
        if self._yInvert:
            y = self.yMax - ((Bytes[4] << 8) | (Bytes[3]))
        else:
            y = (Bytes[4] << 8) | (Bytes[3])

        return Event(float(self.PreviousEventTs) * 10**-6, np.array([x, y]), int(p))

    def _ATIS_BYTES_ANALYSIS(self, Bytes):
        #print "Current Batch:"
        #print Bytes
        #for Byte in Bytes:
        #    print "{0:8b}".format(Byte)
        
        td = Bytes[0] >> 2
        self.PreviousEventTs += td

        if Bytes[0] & 0b00000001: # if is_tc == 1
            return None

        p = (Bytes[0] & 0b00000010) >> 1
        x = (Bytes[2] << 8) | (Bytes[1])
        if self._yInvert:
            y = self.yMax - ((Bytes[4] << 8) | (Bytes[3]))
        else:
            y = (Bytes[4] << 8) | (Bytes[3])
        #print "Gives :"
        #print "t = {0:.6f} (td = {1:.6f})".format(self.PreviousEventTs, td)
        #print "p = {0}".format(p)
        #print "x = {0}".format(x)
        #print "y = {0}".format(y)

        return Event(float(self.PreviousEventTs) * 10**-6, np.array([x, y]), int(p))

    def _DealWithHeaderEs(self):
        self.CurrentGeometry = list(self.DefaultGeometry)

        Header = self.CurrentFile.read(_ES_HEADER_SIZE)
        Version = ord(Header[12])
        if Version != 2:
            print "Wrong .es version."
            return False

        self.Event_Type = ord(self.CurrentFile.read(1))
        self.Event_Size = 5

        print "Reading Event Stream file, with {0} types".format(_EVENT_TYPE_NAME[self.Event_Type])

        if not _EVENT_TYPE_HANDLED[self.Event_Type]:
            print "Event type not handled yet. Too bad..."
            return False

        self.OverflowValue = _EVENT_TYPE_OVERFLOW_VALUE[self.Event_Type]
        self.AditionnalBytes = _EVENT_TYPE_ADITIONNAL_BYTES[self.Event_Type]
        self._BytesAnalysisFunction = self.__class__.__dict__[_EVENT_TYPE_ANALYSIS_FUNCTIONS_NAMES[self.Event_Type]]
        self.ResetByte = _EVENT_TYPE_RESET_BYTE[self.Event_Type]
        self.ResetPadding = _EVENT_TYPE_RESET_PADDING[self.Event_Type]

        Events_Header = self.CurrentFile.read(4)

        #  TODO : Create subroutine to handle the different event types. This current one is ATIS/DVS
        Width = (unpack('B', Events_Header[1])[0] << 8) | unpack('B', Events_Header[0])[0]
        Height = (unpack('B', Events_Header[3])[0] << 8) | unpack('B', Events_Header[2])[0]

        self.CurrentGeometry[0] = Width
        self.CurrentGeometry[1] = Height

        self.BatchSize = self._BatchEventSize * self.Event_Size

        print "> Found stream geometry of {0}".format(self.CurrentGeometry)
        return True

    # .DAT FILES METHODS

    def _InitializeDat(self):
        self.CurrentFile = open(self.StreamName,'rb')

        HeaderHandled = self._DealWithHeaderDat()
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

    def _DealWithHeaderDat(self):
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
        self.BatchSize = self._BatchEventSize * self.Event_Size
        return self.Header

    def _NextEventDat(self):
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

    # GENERIC METHODS

    def _LoadNewBatch(self):
        Buffer = self.CurrentFile.read(self.BatchSize)
        if not Buffer:
            print ""
            print "Input reached EOF."
            self.__Framework__.Running = False
        
        self.CurrentByteBatch = self.CurrentByteBatch + Buffer

    def _StoreEvent(self, event):
        self.CurrentStream += [event]

    def _DoNothing(self, event):
        None

    def _CloseFile(self):
        if not self.CurrentFile is None:
            self.CurrentFile.close()
            print "Closed file {0}".format(self.StreamName)
            self.CurrentFile = None

