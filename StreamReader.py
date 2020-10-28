import numpy as np
from struct import unpack
from sys import stdout
import atexit
import h5py

from Framework import Module, Event

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

        self._CameraIndex = 0
        self._DatVersion = 2
        self._RemoveNegativeTimeDifferences = True
        self._SaveStream = False
        self._BatchEventSize = 1000
        self._TryUseDefaultGeometry = True
        self._AutoZeroOffset = True
        self._yInvert = True

        self._TxtFileStructure = ['t', 'x', 'y', 'p']
        self._TxtTimestampMultiplier = 1.
        self._TxtDefaultGeometry = [240, 180, 2]
        self._TxtPosNegPolarities = False

        self._Rewinded = False
        self._NeedsLogColumn = False
        self.CurrentFile = None

    @property
    def Geometry(self):
        return self._Geometry
    @Geometry.setter
    def Geometry(self, value):
        self._Geometry = value

    def _SetOutputCameraIndexes(self):
        if self.__CameraInputRestriction__:
            if len(self.__CameraInputRestriction__) == 1:
                self._CameraIndex = self.__CameraInputRestriction__[0]
            else:
                self.LogWarning("Unvalid cameras index for this module")
        self.__CameraOutputRestriction__ = [self._CameraIndex]

    def _InitializeModule(self, **kwargs):

        self._CloseFile()

        if self._SaveStream:
            self.CurrentStream = []
            self._StorageFunction = self._StoreEvent
        else:
            self._StorageFunction = self._DoNothing
            self.__RewindForbidden__ = True
        self.TsOffset = None

        Extension = self.StreamName.split('.')[-1]
        if 'dat' == Extension:
            self._NextEvent = self._NextEventDat
            self.Log("Using dat extension")
            return self._InitializeDat()
        elif 'es' == Extension:
            self._NextEvent = self._NextEventEs
            self.Log("Using es extension")
            return self._InitializeEs()
        elif 'txt' == Extension:
            self._NextEvent = self._NextEventTxt
            self.Log("Using txt extension")
            return self._InitializeTxt()
        elif 'aedat' == Extension:
            self._NextEvent = self._NextEventAedat
            self.Log("Using aedat extension")
            return self._InitializeAedat()
        elif 'hdf5' == Extension:
            self._NextEvent = self._NextEventH5
            self.Log("Using hdf5 extension")
            return self._InitializeH5()
        else:
            self.LogWarning("Invalid filename.")
            return False

    def _OnEventModule(self, event):
        self.nEvent += 1

        if not self._Rewinded:
            self.NextOwnEvent = self._NextEvent()
        else:
            if self.nEvent < len(self.CurrentStream):
                self.NextOwnEvent = self.CurrentStream[self.nEvent]
            else:
                self._Rewinded = False
                self.NextOwnEvent = self._NextEvent()
        if not self.NextOwnEvent is None:
            self.NextOwnEvent.cameraIndex = self._CameraIndex
        if not self._Rewinded:
            # Possibly save the event
            self._StorageFunction(self.NextOwnEvent)
        
        if self.__Framework__.Running:
            return self.NextOwnEvent
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
                    self.Log("Done.")
                    break
        except KeyboardInterrupt:
            self.LogWarning("Stopped fast forward at t = {0:.3f}".format(NextEvent.timestamp))

    def _Rewind(self, tNew):
        if tNew >= self.CurrentStream[-1].timestamp:
            return None
        self._Rewinded = True
        if tNew == 0:
            self.nEvent = 0
            return None
        for Event in reversed(self.CurrentStream):
            self.nEvent -= 1
            if Event.timestamp < tNew:
                break

# .h5 Files methods

    def _InitializeH5(self):
        StreamName = str(self.StreamName)
        if "@" not in StreamName:
            self.LogError("No path specified into h5 file. Use KEYS.TO.EVENTS@FileName as stream input")
            return False
        Path, FileName = StreamName.split("@")
        self.CurrentFile = h5py.File(FileName, mode = 'r')
        atexit.register(self._CloseFile)
        Keys = Path.split('.')
        self.Data = self.CurrentFile[Keys.pop(0)]
        while Keys:
            self.Data = self.Data[Keys.pop(0)]
        #self.Geometry = np.array([self.Data[:,0].max() + 1, self.Data[:,1].max() + 1, 2], dtype = int)
        self.Geometry = np.array([346,260,2])
        self.NEvents = self.Data.shape[0]
        self.nEvent = 0
        self.tOffset = self.Data[0,2]

        return True

    def _NextEventH5(self):
        if self.nEvent == self.NEvents:
            self.__Framework__.Running = False
            return None
        Line = self.Data[self.nEvent,:]
        self.nEvent += 1
        return Event(timestamp = (Line[2] - self.tOffset), location = np.array([int(Line[0]), self.Geometry[1] - int(Line[1])-1]), polarity = int(Line[3] == 1))

# .AEDAT Files methods

    def _InitializeAedat(self):
        self.CurrentFile = open(self.StreamName,'rb')
        atexit.register(self._CloseFile)
        self.nEvent = 0
        self.CurrentByteBatch = b''

        self.PreviousEventTs = 0
        self.Geometry = np.array([346, 260, 2])
        self.xMax = self.Geometry[0]-1
        self._Version = self.CurrentFile.readline().split(b'AER-DAT')[1].split(b'\r')

        self.Event_Size = 8
        self.BatchSize = self._BatchEventSize * self.Event_Size
        self.nByte = -self.Event_Size

        self.CurrentByteBatch = self.CurrentFile.readline()
        while self.CurrentByteBatch[0] == 35:
            self.CurrentByteBatch = self.CurrentFile.readline()
        self.Offset = 0
        self.Offset = self._NextEventAedat().timestamp

        return True

    def _NextEventAedat(self):
        while True:
            self.nByte += self.Event_Size
            if len(self.CurrentByteBatch) - self.nByte < self.Event_Size:
                self.CurrentByteBatch = self.CurrentByteBatch[self.nByte:]
                self.nByte = 0
                self._LoadNewBatch()

                if not self.__Framework__.Running:
                    return None

            Bytes = self.CurrentByteBatch[self.nByte:self.nByte+self.Event_Size]

            if Bytes[0] >> 7: # Not DVS event
                continue
            if (Bytes[2] >> 2) & 0B1: # External trigger, we leave it for now
                continue

            x = (Bytes[1] & 0B111111) << 4 | Bytes[2] >> 4
            y = (Bytes[0] & 0B1111111) << 2 | Bytes[1] >> 6
            p = (Bytes[2] >> 3) & 0B1
            t = Bytes[-4] << 24 | Bytes[-3] << 16 | Bytes[-2] << 8 | Bytes[-1]

            return Event(timestamp = t / 1e6 - self.Offset, location = np.array([self.xMax - x, y]), polarity = p)

# .ES Files methods

    def _InitializeEs(self):
        self.CurrentFile = open(self.StreamName,'rb')

        if not self._DealWithHeaderEs():
            return False
        
        self.yMax = self.Geometry[1] - 1

        atexit.register(self._CloseFile)

        self.nEvent = 0 # Global counter to locate oneself in the stream
        self.nByte = -1 # Local counter for the position in the buffer
        self.CurrentByteBatch = b''

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
            Bytes = [self.CurrentByteBatch[self.nByte]]
            if (Bytes[0] >> self.ResetPadding) == self.ResetByte: # Check for reset or overflow
                Overflow = Bytes[0] & 0x03
                
                self.PreviousEventTs += self.OverflowValue * Overflow
                continue

            for i in range(self.AditionnalBytes):
                self.nByte += 1
                Bytes.append(self.CurrentByteBatch[self.nByte])

            CreatedEvent = self._BytesAnalysisFunction(self, Bytes)

            if not CreatedEvent is None:
                if self.TsOffset is None and self._AutoZeroOffset:
                    self.TsOffset = CreatedEvent.timestamp
                    if self.TsOffset:
                        self.Log("Setting AutoOffset to {0:.3f}".format(self.TsOffset))
                    CreatedEvent.timestamp = 0
                else:
                    CreatedEvent.timestamp -= self.TsOffset

                #if self._yInvert: Written in ANALYSIS_FUNCTIONS
                #    CreatedEvent.location[1] = self.yMax - CreatedEvent.location[1]
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

        return Event(timestamp = float(self.PreviousEventTs) * 10**-6, location = np.array([x, y]), polarity = int(p))

    def _ATIS_BYTES_ANALYSIS(self, Bytes):
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

        return Event(timestap = float(self.PreviousEventTs) * 10**-6, location = np.array([x, y]), polarity = int(p))

    def _DealWithHeaderEs(self):
        self.Geometry = list(self.DefaultGeometry)

        Header = self.CurrentFile.read(_ES_HEADER_SIZE)
        Version = Header[12]
        if Version != 2:
            self.LogWarning("Wrong .es version.")
            return False

        self.Event_Type = ord(self.CurrentFile.read(1))
        self.Event_Size = 5

        self.Log("Reading Event Stream file, with {0} types".format(_EVENT_TYPE_NAME[self.Event_Type]))

        if not _EVENT_TYPE_HANDLED[self.Event_Type]:
            self.LogWarning("Event type not handled yet. Too bad...")
            return False

        self.OverflowValue = _EVENT_TYPE_OVERFLOW_VALUE[self.Event_Type]
        self.AditionnalBytes = _EVENT_TYPE_ADITIONNAL_BYTES[self.Event_Type]
        self._BytesAnalysisFunction = self.__class__.__dict__[_EVENT_TYPE_ANALYSIS_FUNCTIONS_NAMES[self.Event_Type]]
        self.ResetByte = _EVENT_TYPE_RESET_BYTE[self.Event_Type]
        self.ResetPadding = _EVENT_TYPE_RESET_PADDING[self.Event_Type]

        Events_Header = self.CurrentFile.read(4)

        #  TODO : Create subroutine to handle the different event types. This current one is ATIS/DVS
        Width = (Events_Header[1] << 8) | Events_Header[0]
        Height = (Events_Header[3] << 8) | Events_Header[2]

        self.Geometry[0] = Width
        self.Geometry[1] = Height

        self.BatchSize = self._BatchEventSize * self.Event_Size

        self.Log("Found stream geometry of {0}".format(self.Geometry))
        return True

    # .DAT FILES METHODS

    def _InitializeDat(self):
        self.CurrentFile = open(self.StreamName,'rb')

        HeaderHandled = self._DealWithHeaderDat()
        if self.Geometry == self.DefaultGeometry:
            self.Log("No geometry found.")
            if self._TryUseDefaultGeometry:
                self.LogWarning("Using default geometry : {0}".format(self.Geometry))
            else:
                self._CloseFile()
                return False
        self.yMax = self.Geometry[1] - 1

        atexit.register(self._CloseFile)

        self.tMask = 0x00000000FFFFFFFF

        if self._DatVersion == 2:
            self.pMask = 0x1000000000000000
            self.xMask = 0x00003FFF00000000
            self.yMask = 0x0FFFC00000000000
            self.pPadding = 60
            self.yPadding = 46
            self.xPadding = 32
        
        else:
            self.pMask = 0x0002000000000000
            self.xMask = 0x000001FF00000000
            self.yMask = 0x0001FE0000000000
            self.pPadding = 49
            self.yPadding = 41
            self.xPadding = 32

        # Analyzing file size
        start = self.CurrentFile.tell()
        self.CurrentFile.seek(0,2)
        stop = self.CurrentFile.tell()
        self.CurrentFile.seek(start)
    
        self.NEvents = int( (stop-start)/self.Event_Size)
        dNEvents = self.NEvents/100
        self.Log("The file contains %d events." %self.NEvents)

        self.nEvent = 0
        self.CurrentByteBatch = b''

        self.PreviousEventTs = -np.inf

        return True

    def _DealWithHeaderDat(self):
        self.Header = False
        self.Geometry = list(self.DefaultGeometry)
        FoundHeightOrWidth = False
        while peek(self.CurrentFile) == b'%':
            HeaderNextLine = self.CurrentFile.readline()
            self.Header = True
            if b'height' in HeaderNextLine.lower():
                FoundHeightOrWidth = True
                self.Geometry[1] = int(HeaderNextLine.lower().split(b'height')[1].strip())
            if b'width' in HeaderNextLine.lower():
                FoundHeightOrWidth = True
                self.Geometry[0] = int(HeaderNextLine.lower().split(b'width')[1].strip())

        if FoundHeightOrWidth:
            self.Log("Found stream geometry of {0}".format(self.Geometry))
        if self.Header:
            self.Event_Type = unpack('B',self.CurrentFile.read(1))[0]
            self.Event_Size = unpack('B',self.CurrentFile.read(1))[0]
        else:
            self.Event_Type = 0
            self.Event_Size = 8
        self.BatchSize = self._BatchEventSize * self.Event_Size
        return self.Header

    def _NextEventDat(self):
        if not len(self.CurrentByteBatch):
            self._LoadNewBatch()
            if not self.__Framework__.Running:
                return None

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

        if self.TsOffset is None and self._AutoZeroOffset:
            self.TsOffset = ts
            if self.TsOffset:
                self.Log("Setting AutoOffset to {0:.3f}".format(self.TsOffset))
            ts = 0
        else:
            ts -= self.TsOffset
        return Event(timestamp = float(ts) * 10**-6, location = np.array([x, self.yMax - y]), polarity = int(p))

    # .TXT METHODS

    def _InitializeTxt(self):
        self.Geometry = list(self._TxtDefaultGeometry)
        self.CurrentFile = open(self.StreamName,'r')
        atexit.register(self._CloseFile)

        Separators = [' ', '&', '_']
        l = self.CurrentFile.readline()[:-1]
        Found = False
        for self._Separator in Separators:
            if l.count(self._Separator) > 2:
                Found = True
                break
        if not Found:
            "No defined separator found. Aborting."
            return False
        self.CurrentFile.seek(0)
        self.nEvent = 0
        return True

    def _NextEventTxt(self):
        Line = self.CurrentFile.readline()
        if Line[-1:] == '\n':
            Line = Line[:-1]
        else:
            if len(Line) == 0:
                self.__Framework__.Running = False
                return None
        #t_str, x_str, y_str, p_str = Line.split(self._Separator)[:4]
        Data = [Value.strip() for Value in Line.strip().split(self._Separator) if Value.strip()]
        ts = float(Data[self._TxtFileStructure.index('t')]) * self._TxtTimestampMultiplier
        x = int(float(Data[self._TxtFileStructure.index('x')]))
        y = int(float(Data[self._TxtFileStructure.index('y')]))
        if self._yInvert:
            y = (self.Geometry[1]-1) - y
        p = (self._TxtPosNegPolarities + int(float(Data[self._TxtFileStructure.index('p')]))) / (1 + self._TxtPosNegPolarities)

        if self.TsOffset is None and self._AutoZeroOffset:
            self.TsOffset = ts
            if self.TsOffset:
                self.Log("Setting AutoOffset to {0:.3f}".format(self.TsOffset))
            ts = 0
        else:
            ts -= self.TsOffset

        return Event(timestamp = ts, location = np.array([x, y]), polarity = int(p))

    # GENERIC METHODS

    def _LoadNewBatch(self):
        Buffer = self.CurrentFile.read(self.BatchSize)
        if not Buffer:
            self.Log("Input reached EOF.")
            self.__Framework__.Running = False
        
        self.CurrentByteBatch = self.CurrentByteBatch + Buffer

    def _StoreEvent(self, event):
        self.CurrentStream += [event]

    def _DoNothing(self, event):
        None

    def _CloseFile(self):
        if not self.CurrentFile is None:
            self.CurrentFile.close()
            self.Log("Closed file {0}".format(self.StreamName))
            self.CurrentFile = None

def peek(f, length=1):
    pos = f.tell()
    data = f.read(length)
    f.seek(pos)
    return data
