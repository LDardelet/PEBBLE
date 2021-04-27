import socket
import random
import time
import _pickle as cPickle
import types
import numpy as np

import threading
from multiprocessing import Queue

from PEBBLE import Module

class TransmissionInfo:
    _EventPort = 54242
    _PacketSizeLimit = 8192
    _Address = "localhost"
    _QuestionPort = 54243
    _ResponsePort = 54244

class PostServiceClass(threading.Thread):
    def __init__(self, Socket, PostBoxLimit, PostBox):
        self.Socket = Socket
        self.PostBoxLimit = PostBoxLimit
        self.Running = False

        threading.Thread.__init__(self)

        self.MainUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

        self.PostBox = PostBox
        self.NextBatchPackages = []

    def run(self):
        self.Running = True
        while self.Running:
            if self.NextBatchPackages or self.PostBox.qsize() >= self.PostBoxLimit:
                self._SendPackage()

    def _SendPackage(self):
        Package = [self.Socket] + self.NextBatchPackages + [self.PostBox.get() for nPackage in range(self.PostBox.qsize())]
        self.NextBatchPackages = []
        data = cPickle.dumps(Package)
        while len(data) > TransmissionInfo._PacketSizeLimit:
            ExcessPackage = Package.pop(-1)
            data = cPickle.dumps(Package)
            if len(cPickle.dumps([ExcessPackage])) <= TransmissionInfo._PacketSizeLimit: # If the last removed package is small enough to be sent
                self.NextBatchPackages.insert(0, ExcessPackage)
        self.MainUDP.sendto(data, (TransmissionInfo._Address, TransmissionInfo._EventPort))

class DisplayHandler(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to handle the stream Display.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Output'

        self.__Started__ = False

        self._MultiThread = False # USELESS.
        self._PostBoxLimit = 7

        self._NeedsLogColumn = False

        self.PostService = None
        self.Socket = None

    def _InitializeModule(self):
        if not self.Socket is None:
            self.EndTransmission()
        self.Socket = None

        DisplayUp = self._IsDisplayUp()

        if not DisplayUp:
            self.__Started__ = False
            return True

        self.Log("Initializing DisplayHandler sockets.")
        self._DestroySocket()
        self._GetDisplaySocket()

        self._CleanMapForStream()

        self._SendStreamData()

        if self._MultiThread:
            self.PostBox = Queue()
            self.PostService = PostServiceClass(self.Socket, self._PostBoxLimit, self.PostBox)
            self.PostService.start()
            def RCV(self, event):
                self.PostBox.put(event.AsDict())
            def Check(self):
                pass
        else:
            self.MainUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            self.PostBox = []
            def RCV(self, event):
                self.PostBox.append(event.AsDict())
            def Check(self):
                if len(self.PostBox) >= self._PostBoxLimit:
                    self._SendPackage()
        self.Check = types.MethodType(Check, self)
        self.RCV = types.MethodType(RCV, self)

        self.__Started__ = True

        time.sleep(0.1)

        return True

    def _Restart(self):
        self.EndTransmission()
        self._InitializeModule()
    def _Pause(self, Origin):
        if self._MultiThread:
            self.PostService.Running = False
    def _Resume(self):
        if self._MultiThread:
            self.PostService.run()

    def _OnEventModule(self, event):
        if self.__Started__:
            self.RCV(event)
            self.Check()

    def _OnClosing(self):
        self.EndTransmission()

    def EndTransmission(self):
        if self._MultiThread and not self.PostService is None:
            self.PostService.Running = False
        if not self.Socket is None:
            self._DestroySocket()

    def _SendPackage(self):
        Package = [self.Socket] + self.PostBox
        self.PostBox = []
        data = cPickle.dumps(Package)
        while len(data) > TransmissionInfo._PacketSizeLimit:
            ExcessPackage = Package.pop(-1)
            data = cPickle.dumps(Package)
            if len(cPickle.dumps([ExcessPackage])) <= TransmissionInfo._PacketSizeLimit: # If the last removed package is small enough to be sent
                self.PostBox.insert(0, ExcessPackage)
        self.MainUDP.sendto(data, (TransmissionInfo._Address, TransmissionInfo._EventPort))

    def _SendStreamData(self):
        ProjectFile = self.__Framework__.ProjectFile
        StreamName = self.__Framework__._GetStreamFormattedName(self)

        ResponseUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listen_addr = ("", TransmissionInfo._ResponsePort)
        ResponseUDP.bind(listen_addr)

        id_random = random.randint(100000,200000)
        QuestionDict = {'id': id_random, 'socket': self.Socket, 'infosline1': ProjectFile+' -> ' + self.__Name__, 'infosline2': StreamName, 'command':'socketdata'}

        QuestionUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        QuestionUDP.sendto(cPickle.dumps(QuestionDict), (TransmissionInfo._Address, TransmissionInfo._QuestionPort))
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

    def _CleanMapForStream(self):
        ResponseUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listen_addr = ("",TransmissionInfo._ResponsePort)
        ResponseUDP.bind(listen_addr)

        id_random = random.randint(100000,200000)
        QuestionDict = {'id': id_random, 'socket': self.Socket, 'command':'cleansocket'}

        QuestionUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        QuestionUDP.sendto(cPickle.dumps(QuestionDict), (TransmissionInfo._Address, TransmissionInfo._QuestionPort))
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
        listen_addr = ("",TransmissionInfo._ResponsePort)
        ResponseUDP.bind(listen_addr)

        id_random = random.randint(100000,200000)
        QuestionDict = {'id': id_random, 'command':'isup'}

        QuestionUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        QuestionUDP.sendto(cPickle.dumps(QuestionDict), (TransmissionInfo._Address, TransmissionInfo._QuestionPort))
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
        listen_addr = ("",TransmissionInfo._ResponsePort)
        ResponseUDP.bind(listen_addr)

        id_random = random.randint(100000,200000)
        QuestionDict = {'id': id_random, 'socket': self.Socket, 'command':'destroysocket'}

        QuestionUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        QuestionUDP.sendto(cPickle.dumps(QuestionDict), (TransmissionInfo._Address, TransmissionInfo._QuestionPort))
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
        self.Socket = None

    def _GetDisplaySocket(self):
        ResponseUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listen_addr = ("",TransmissionInfo._ResponsePort)
        ResponseUDP.bind(listen_addr)

        id_random = random.randint(100000,200000)
        QuestionDict = {'id': id_random, 'shape':np.array(self.Geometry)}
        if self.Socket is None:
            QuestionDict['command'] = "asksocket"
        else:
            QuestionDict['command'] = "askspecificsocket"
            QuestionDict['socket'] = self.Socket

        QuestionUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        QuestionUDP.sendto(cPickle.dumps(QuestionDict), (TransmissionInfo._Address, TransmissionInfo._QuestionPort))
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
