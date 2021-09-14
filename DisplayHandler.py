import socket
import random
import time
import _pickle as cPickle
import types
import numpy as np

import threading
from multiprocessing import Queue

from ModuleBase import ModuleBase

class TransmissionInfo:
    EventPort = 54242
    PacketSizeLimit = 8192
    QuestionPort = 54243
    ResponsePort = 54244
    Timeout = 1.
    def __init__(self, Address):
        self.Address = Address

class DisplayHandler(ModuleBase):
    def _OnCreation(self):
        '''
        Class to handle the stream Display.
        '''
        self.__Started__ = False

        self._PostBoxLimit = 7

        self._NeedsLogColumn = False
        self._Address = "localhost"

        self.PostService = None
        self.Socket = None

    def _OnInitialization(self):
        self.TI = TransmissionInfo(self._Address)

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

        self.MainUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.PostBox = []

        self.__Started__ = True

        time.sleep(0.1)

        return True

    def RCV(self, event):
        self.PostBox.append(event.AsDict())
    def Check(self):
        if len(self.PostBox) >= self._PostBoxLimit:
            self._SendPackage()

    def Restart(self):
        '''
        Specific method, hardcoded in PEBBLE to restart all displays at once.
        '''
        self.EndTransmission()
        self._OnInitialization()

    def _OnEventModule(self, event):
        if self.__Started__:
            self.RCV(event)
            self.Check()

    def _OnClosing(self):
        self.EndTransmission()

    def EndTransmission(self):
        if not self.Socket is None:
            self._DestroySocket()

    def _SendPackage(self):
        Package = [self.Socket] + self.PostBox
        self.PostBox = []
        data = cPickle.dumps(Package)
        while len(data) > self.TI.PacketSizeLimit:
            ExcessPackage = Package.pop(-1)
            data = cPickle.dumps(Package)
            if len(cPickle.dumps([ExcessPackage])) <= self.TI.PacketSizeLimit: # If the last removed package is small enough to be sent
                self.PostBox.insert(0, ExcessPackage)
        self.MainUDP.sendto(data, (self.TI.Address, self.TI.EventPort))

    def _SendQuestion(self, command, QuestionData):
        ResponseUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listen_addr = ("", self.TI.ResponsePort)
        ResponseUDP.bind(listen_addr)

        id_random = random.randint(100000,200000)
        PacketDict = {'id': id_random, 'socket': self.Socket, 'command':command}
        for dataKey, dataValue in QuestionData.items():
            PacketDict[dataKey] = dataValue
        QuestionUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        QuestionUDP.sendto(cPickle.dumps(PacketDict), (self.TI.Address, self.TI.QuestionPort))
        QuestionUDP.close()
        ResponseUDP.settimeout(self.TI.Timeout)

        try:
            data_raw, addr = ResponseUDP.recvfrom(1064)
            data = cPickle.loads(data_raw)
            if data['id'] == id_random:
                ResponseUDP.close()
                return data['answer']
            else:
                self.LogWarning("Got wrong question ID answer.")
        except:
            self.LogWarning("Display seems down ({0})".format(command))
        ResponseUDP.close()
        return False, None

    def _SendStreamData(self):
        ProjectFile = self.__Framework__.ProjectFile
        StreamName = self.__Framework__._GetStreamFormattedName(self)

        Success, _ = self._SendQuestion('socketdata', {'infosline1': ProjectFile.split("/")[-1].split(".json")[0] +' -> ' + self.__Name__, 'infosline2': StreamName})
        if not Success:
            self.LogWarning("Could not transmit data")
        else:
            self.LogSuccess("Stream data sent")

    def _CleanMapForStream(self):
        Success, _ = self._SendQuestion('cleansocket', {})

        if Success:
            self.Log("Cleansed")
        else:
            self.LogWarning("Could not clean map")

    def _IsDisplayUp(self):
        Success, _ = self._SendQuestion('isup', {})

        if Success:
            self.Log("Display is up")
            return True
        else:
            self.LogWarning("Display is down")
            return False

    def _DestroySocket(self):
        if self.Socket is None:
            return None
        Success, _ = self._SendQuestion('destroysocket', {})
        if Success:
            self.Log("Destroyed socket {0}".format(self.Socket))
        else:
            self.LogWarning("Could not destroy socket {0}".format(self.Socket))
        self.Socket = None

    def _GetDisplaySocket(self):
        if self.Socket is None:
            Success, self.Socket = self._SendQuestion('asksocket', {'shape':np.array(self.Geometry)})
        else:
             Success, self.Socket = self._SendQuestion('askspecificsocket', {'shape':np.array(self.Geometry)})

        if Success:
            self.Log("Got socket {0}".format(self.Socket))
        else:
            self.LogWarning("Socket refused")
