from PEBBLE import Module, Event, DisparityEvent, FlowEvent
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

_FOLDER = '/home/dardelet/Pictures/Recordings/'

class PictureSaver(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Creates PNG images of anything plottable that comes though
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Computation'

        self.__ReferencesAsked__ = []
        self._NeedsLogColumn = False

        self._FramesDt = 0.01
        self._Outputs = ['Events', 'Events+Flows', 'Disparities']
        self._Tau = 0.005

    def _InitializeModule(self, **kwargs):
        self.ScreenSize = tuple(self.Geometry[:2])
        self.StreamsContainers = {}
        self.StreamsLastFrames = {}
        self.StreamsFAxs = {}

        self.UsedContainers = set()
        for Output in self._Outputs:
            for DisplayedType in Output.split('+'):
                self.UsedContainers.add(DisplayedType)

        self.TemplateContainers = {'Events':EventsContainer, 'Flows':FlowsContainer, 'Disparities':DisparitiesContainer}

        if not self.__CameraInputRestriction__:
            self.LogWarning("No camera input restriction would slow the framework down too much.")
            self.LogWarning("No pictures will be created")
            self.Active = False
            return True
        else:
            for subStreamIndex in self.__CameraInputRestriction__:
                self.AddSubStreamData(subStreamIndex)
            self.Active = True

        self.FolderName = _FOLDER
        if len(self.__CameraInputRestriction__) == 1:
            self.FolderName = self.FolderName + 'Mono_'
        else:
            self.FolderName = self.FolderName + 'Stereo_'
        self.FolderName = self.FolderName + '_'.join(self.UsedContainers) + '_'
        self.FolderName = self.FolderName + datetime.datetime.now().strftime("%d-%m-%Y_%H-%M") + '/'
        try:
            os.mkdir(self.FolderName)
            self.LogSuccess("Creating pictures in folder :")
            self.LogSuccess(self.FolderName)
        except:
            self.LogWarning("Unable to create output folder")
            self.LogWarning("No pictures will be created")
            self.Active = False
            return True

        return True

    def _OnEventModule(self, event):
        if not self.Active:
            return event
        for Container in self.StreamsContainers[event.cameraIndex].values():
            Container.OnEvent(event)
        if event.timestamp > self.StreamsLastFrames[event.cameraIndex][0] + self._Tau:
            self.Draw(event.cameraIndex, event.timestamp)
        return event

    def AddSubStreamData(self, subStreamIndex):
        self.StreamsContainers[subStreamIndex] = {Container:self.TemplateContainers[Container](self.ScreenSize) for Container in self.UsedContainers}
        self.StreamsFAxs[subStreamIndex] = {}
        for output in self._Outputs:
            self.StreamsFAxs[subStreamIndex][output] = plt.subplots(1,1)
            self.StreamsFAxs[subStreamIndex][output][1].tick_params('both', bottom =  False, labelbottom = False, left = False, labelleft = False)
            plt.close(self.StreamsFAxs[subStreamIndex][output][0])
        self.StreamsLastFrames[subStreamIndex] = (0, -1)

    def Draw(self, subStreamIndex, t):
        FrameIndexes = self.StreamsLastFrames[subStreamIndex][1] + 1
        for output in self._Outputs:
            self.StreamsFAxs[subStreamIndex][output][1].cla()
            for Container in output.split('+'):
                self.StreamsContainers[subStreamIndex][Container].Draw(t, self._Tau, self.StreamsFAxs[subStreamIndex][output][1])
            self.StreamsFAxs[subStreamIndex][output][1].set_xlim(0, self.ScreenSize[0])
            self.StreamsFAxs[subStreamIndex][output][1].set_ylim(0, self.ScreenSize[1])
            self.StreamsFAxs[subStreamIndex][output][0].savefig(self.FolderName + output + '_{0}_{1}.png'.format(subStreamIndex, FrameIndexes))
        self.StreamsLastFrames[subStreamIndex] = (t, FrameIndexes)

class EventsContainer:
    def __init__(self, ScreenSize):
        self.STContext = np.zeros(ScreenSize + (2,))

    def OnEvent(self, event):
        self.STContext[event.location[0], event.location[1], event.polarity] = event.timestamp

    def Draw(self, t, Tau, ax):
        Map = np.zeros(self.STContext.shape[:2])
        for Polarity in range(2):
            Map = (Map + ((t - self.STContext[:,:,Polarity]) < Tau))
        ax.imshow(np.transpose(Map), origin = 'lower', cmap = 'binary', vmin = 0, vmax = 2)

class FlowsContainer:
    def __init__(self, ScreenSize):
        self.FlowsList = []

    def OnEvent(self, event):
        if not event.Has(FlowEvent):
            return
        self.FlowsList += [(event.timestamp, event.location, event.flow)]

    def Draw(self, t, Tau, ax):
        while self.FlowsList and self.FlowsList[0][0] < t - Tau:
            self.FlowsList.pop(0)
        for _, (x, y), flow in self.FlowsList:
            n = np.linalg.norm(flow)
            if n == 0:
                continue
            nf = flow / n
            r = max(0., nf[0])
            g = max(0., (nf * np.array([-1, np.sqrt(3)])).sum()/2)
            b = max(0., (nf * np.array([-1, -np.sqrt(3)])).sum()/2)
            color = np.sqrt(np.minimum(np.array([r, g, b, 1]), np.ones(4, dtype = float)))
            df = flow * Tau
            ax.plot([x,x+df[0]], [y, y+df[1]], color = color)

class DisparitiesContainer:
    def __init__(self, ScreenSize):
        self.STDContext = np.zeros(ScreenSize + (2,))
        self.MaxD = 5

    def OnEvent(self, event):
        if not event.Has(DisparityEvent):
            return
        self.STDContext[event.location[0], event.location[1],:] = np.array([event.disparity, event.timestamp])

    def Draw(self, t, Tau, ax):
        Map = self.STDContext[:,:,0] * ((t - self.STDContext[:,:,1]) < Tau)
        self.MaxD = max(self.MaxD, self.STDContext[:,:,0].max())
        ax.imshow(np.transpose(Map), origin = 'lower', vmin = 0, vmax = self.MaxD, cmap = 'hot')