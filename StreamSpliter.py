import numpy as np
from Framework import Module, Event

class StreamSpliter(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Module to split an event stream into two different substreams. Camera indexes are used to address each subsequent camera
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Computation'

        self._SplitType = 'Random' # Either 'Random', each event is the assigned randomly to a camera, or 'Sequential', and each camera is addressed sequencially
        self._NumberOfCameras = 2
        self._OverrideInputIndex = False

        self._MonitorDt = 0. # By default, a module does not stode any date over time.
        self._MonitoredVariables = []

    def _InitializeModule(self, **kwargs):
        if self._SplitType == 'Random':
            self._CameraIndexMethod = self._CameraRandomIndex
        elif self._SplitType == 'Sequential':
            self._CameraIndexMethod = self._NextCamera
            self._CurrentIndex = 0
        else:
            self.LogError("Invalid SplitType method")
        return True

    def _SetOutputCameraIndexes(self):
        self.__CameraOutputRestriction__ = [CameraIndex + int(self._OverrideInputIndex) for CameraIndex in range(self._NumberOfCameras)]

    def _CameraRandomIndex(self):
        return np.random.randint(self._NumberOfCameras)

    def _NextCamera(self):
        self._CurrentIndex = (self._CurrentIndex + 1)%self._NumberOfCameras
        return self._CurrentIndex

    def _OnEventModule(self, event):
        event.cameraIndex = self._CameraIndexMethod()
        return event

    @property
    def StreamName(self):
        return self.__Framework__._GetStreamFormattedName(self)
    @property
    def Geometry(self):
        return self.__Framework__._GetStreamGeometry(self)
