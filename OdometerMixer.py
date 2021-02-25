from PEBBLE import Module, OdometryEvent
import numpy as np

class OdometerMixer(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Module template to be filled foe specific purpose
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Computation'

        self.__ReferencesAsked__ = []
        self._MonitorDt = 0. # By default, a module does not stode any date over time.
        self._NeedsLogColumn = False
        self._MonitoredVariables = [('D', float),
                                    ('Omega', np.array),
                                    ('StereoRectifiedV', np.array)]

        self._ReferenceIndex = 0

    def _InitializeModule(self, **kwargs):
        self.ReferenceOmega = np.zeros(3)
        self.ReferenceV = np.zeros(3)

        self.StereoOmega = np.zeros(3)
        self.StereoV = np.zeros(3)
        return True

    def _OnEventModule(self, event):
        if event.Has(OdometryEvent):
            if event.cameraIndex == self._ReferenceIndex:
                self.ReferenceV = np.array(event.v)
                self.ReferenceOmega = np.array(event.omega)
            else:
                self.StereoV = np.array(event.v)
                self.StereoOmega = np.array(event.omega)
        return event

    @property
    def Omega(self):
        return (self.ReferenceOmega + self.StereoOmega)/2
    @property
    def V(self):
        return (self.Referencev + self.StereoV)/2

    @property
    def D(self):
        Omega = self.Omega
        if np.linalg.norm(self.Omega[1:]) == 0:
            return 0
        return -(Omega[2] * (self.ReferenceV[1] - self.StereoV[1]) - Omega[1] * (self.ReferenceV[2] - self.StereoV[2])) / (Omega[1]**2 + Omega[2]**2)

    @property
    def StereoRectifiedV(self):
        D = self.D
        Omega = self.Omega
        return self.StereoV - np.array([0., D * Omega[2], - D * Omega[1]])
