#import tools
import atexit
import socket
import random
import time
import _pickle as cPickle

from Framework import Module

class DisplayHandler(Module):
    Address = "localhost"
    EventPort = 54242
    QuestionPort = 54243
    ResponsePort = 54244

    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to handle the stream Display.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Display'

        self.__Started__ = False
        self._CompulsoryModule = False

        self._PostBoxLimit = 10

        self.__PostTransporters__ = {'Event':self._SendEvent, 'Segment':self._SendSegment}
        self.Socket = None

    def _InitializeModule(self, **kwargs):
        if not self.Socket is None:
            self.EndTransmission()
        self.Socket = None
        self.PostBox = []

        DisplayUp = self._IsDisplayUp()

        if not DisplayUp:
            self.LogWarning("Aborting initialization process")
            self.__Started__ = False
            return not self._CompulsoryModule

        self.Log("Initializing DisplayHandler sockets.")
        self._DestroySocket()
        self._GetDisplaySocket()

        self._CleanMapForStream()

        self._SendStreamData()
        self.MainUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

        self.__Started__ = True
        atexit.register(self.EndTransmission)

        time.sleep(0.1)

        return True

    def Restart(self):
        self.EndTransmission()
        self._InitializeModule()

    def _OnEventModule(self, event):
        if self.__Started__:
            self.PostBox += [event._AsList()]

            if len(self.PostBox) > self._PostBoxLimit:
                self.Post()

        return event

    def Post(self):
        #for item in self.PostBox:
        #    self.__PostTransporters__[item.__class__.__name__](item)
        self._SendPackage()
        self.PostBox = []

    def EndTransmission(self):
        if not self.Socket is None:
            self._DestroySocket()

    def _SendPackage(self):
        Package = [self.Socket] + self.PostBox
        data = cPickle.dumps(Package)
        self.MainUDP.sendto(data, (self.Address, self.EventPort))

    def _SendEvent(self, ev):
        ev.socket = self.Socket
        data = cPickle.dumps(ev)
        self.MainUDP.sendto(data, (self.Address, self.EventPort))

    def _SendSegment(self, seg):
        segment.socket = self.Socket
        data = cPickle.dumps(segment)
        self.MainUDP.sendto(data, (self.Address, self.EventPort))

    def _SendStreamData(self):
        ProjectFile = self.__Framework__.ProjectFile
        StreamName = self.__Framework__._GetStreamFormattedName(self)

        ResponseUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listen_addr = ("", self.ResponsePort)
        ResponseUDP.bind(listen_addr)

        id_random = random.randint(100000,200000)
        QuestionDict = {'id': id_random, 'socket': self.Socket, 'infosline1': ProjectFile+' -> ' + self.__Name__, 'infosline2': StreamName, 'command':'socketdata'}

        QuestionUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        QuestionUDP.sendto(cPickle.dumps(QuestionDict), (self.Address, self.QuestionPort))
        QuestionUDP.close()
        ResponseUDP.settimeout(1.)

        try:
            data_raw, addr = ResponseUDP.recvfrom(1064)
            data = cPickle.loads(data_raw)
            if data['id'] == id_random and data['answer'] == 'datareceived':
                pass
            else:
                self.LogWarning("Could not transmit data")
        except:
            self.LogWarning("Display seems down (SendStreamData)")
        ResponseUDP.close()

    def _Rewind(self, tNew):
        ResponseUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listen_addr = ("",self.ResponsePort)
        ResponseUDP.bind(listen_addr)

        id_random = random.randint(100000,200000)
        QuestionDict = {'id': id_random, 'socket': self.Socket, 'command':'rewind', 'tNew': tNew}

        QuestionUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        QuestionUDP.sendto(cPickle.dumps(QuestionDict), (self.Address, self.QuestionPort))
        QuestionUDP.close()
        ResponseUDP.settimeout(1.)

        try:
            data_raw, addr = ResponseUDP.recvfrom(1064)
            data = cPickle.loads(data_raw)
            if data['id'] == id_random and data['answer'] == 'rewinded':
                self.Log("Rewinded")
            else:
                self.LogWarning("Could not clean map")
        except:
            self.LogWarning("Display seems down (Rewind)")
        ResponseUDP.close()

    def _CleanMapForStream(self):
        ResponseUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listen_addr = ("",self.ResponsePort)
        ResponseUDP.bind(listen_addr)

        id_random = random.randint(100000,200000)
        QuestionDict = {'id': id_random, 'socket': self.Socket, 'command':'cleansocket'}

        QuestionUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        QuestionUDP.sendto(cPickle.dumps(QuestionDict), (self.Address, self.QuestionPort))
        QuestionUDP.close()
        ResponseUDP.settimeout(1.)

        try:
            data_raw, addr = ResponseUDP.recvfrom(1064)
            data = cPickle.loads(data_raw)
            if data['id'] == id_random and data['answer'] == 'socketcleansed':
                self.Log("Cleansed")
            else:
                self.LogWarning("Could not clean map")
        except:
            self.LogWarning("Display seems down (CleanMapForStream)")
        ResponseUDP.close()

    def _IsDisplayUp(self):
        ResponseUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        listen_addr = ("",self.ResponsePort)
        ResponseUDP.bind(listen_addr)

        id_random = random.randint(100000,200000)
        QuestionDict = {'id': id_random, 'command':'isup'}

        QuestionUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        QuestionUDP.sendto(cPickle.dumps(QuestionDict), (self.Address, self.QuestionPort))
        QuestionUDP.close()
        ResponseUDP.settimeout(1.)

        try:
            data_raw, addr = ResponseUDP.recvfrom(1064)
            data = cPickle.loads(data_raw)
            if data['id'] == id_random and data['answer']:
                ResponseUDP.close()
                return True
            else:
                ResponseUDP.close()
                return False
        except:
            self.LogWarning("No answer, display is down")
            ResponseUDP.close()
            return False

    def _DestroySocket(self):
        if self.Socket is None:
            return None
        ResponseUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listen_addr = ("",self.ResponsePort)
        ResponseUDP.bind(listen_addr)

        id_random = random.randint(100000,200000)
        QuestionDict = {'id': id_random, 'socket': self.Socket, 'command':'destroysocket'}

        QuestionUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        QuestionUDP.sendto(cPickle.dumps(QuestionDict), (self.Address, self.QuestionPort))
        QuestionUDP.close()
        ResponseUDP.settimeout(1.)

        try:
            data_raw, addr = ResponseUDP.recvfrom(1064)
            data = cPickle.loads(data_raw)
            if data['id'] == id_random and data['answer'] == 'socketdestroyed':
                self.Log("Destroyed socket {0}".format(self.Socket))
            else:
                self.LogWarning("Could not destroy socket {0}".format(self.Socket))
        except:
            self.LogWarning("Display seems down (DestroySocket)")
        ResponseUDP.close()

    def _GetDisplaySocket(self):
        Geometry = self.__Framework__._GetStreamGeometry(self)

        ResponseUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listen_addr = ("",self.ResponsePort)
        ResponseUDP.bind(listen_addr)

        id_random = random.randint(100000,200000)
        QuestionDict = {'id': id_random, 'shape':Geometry}
        if self.Socket is None:
            QuestionDict['command'] = "asksocket"
        else:
            QuestionDict['command'] = "askspecificsocket"
            QuestionDict['socket'] = self.Socket

        QuestionUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        QuestionUDP.sendto(cPickle.dumps(QuestionDict), (self.Address, self.QuestionPort))
        QuestionUDP.close()
        ResponseUDP.settimeout(1.)

        try:
            data_raw, addr = ResponseUDP.recvfrom(1064)
            data = cPickle.loads(data_raw)
            if data['id'] == id_random:
                if data['answer'] == 'socketexists':
                    self.LogWarning("Socket refused")
                else:
                    self.Socket = data['answer']
                    self.Log("Got socket {0}".format(self.Socket))
            else:
                self.LogWarning("Socket refused")
        except:
            self.LogWarning("Display seems down (GetDisplaySocket)")
        ResponseUDP.close()
