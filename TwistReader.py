from ModuleBase import ModuleBase
from Events import TwistEvent

import numpy as np

class TwistReader(ModuleBase):
    def _OnCreation(self):
        '''
        '''
        self.__ModulesLinksRequested__ = []     

        self._NeedsLogColumn = False            

        self.__IsInput__ = False
        self.__GeneratesSubStream__ = False     

        self._MonitoredVariables = []           
        self._MonitorDt = 0     
        self._InputFile = ''
        self._Separator = ' '

        self._VFactor = np.array([1., 1., 1.])
        self._OmegaFactor = np.array([1., 1., 1.])

    def _OnInitialization(self):
        if self._InputFile:
            self.CurrentFile = open(self._InputFile, 'r')
            self.RetreiveNextIMUData()
        else:
            self.CurrentFile = None
        return True

    def _OnEventModule(self, event):
        if not self.CurrentFile is None:
            if event.timestamp >= self.StoredIMUData['t']:
                event.Attach(TwistEvent, v = self.StoredIMUData['V'], omega = self.StoredIMUData['Omega'])
                self.RetreiveNextIMUData()
        return

    def RetreiveNextIMUData(self):
        NextLine = self.CurrentFile.readline().strip()
        if not NextLine:
            self.StoredIMUData = {'t':np.inf}
            return
        t, wx, wy, wz, vx, vy, vz = [float(RawData) for RawData in NextLine.split(self._Separator)]
        self.StoredIMUData = {'t':t, 'V':self._VFactor * np.array([vx, vy, vz]), 'Omega': self._OmegaFactor * np.array([wx, wy, wz])}

    def _OnClosing(self):
        '''
        Method to close any file or connexions that would have been created by the framework, upon frameork destruction - typically when interpreter exits.
        Use that instead of an atexit instance.
        '''
        if not self.CurrentFile is None:
            self.CurrentFile.close()
