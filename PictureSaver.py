from PEBBLE import ModuleBase, CameraEvent, DisparityEvent, FlowEvent
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

_PICTURES_VIEWER = "eog"

class PictureSaver(ModuleBase):
    def _OnCreation(self):
        '''
        Creates PNG images of anything plottable that comes though
        '''
        self._FramesDt = 0.01
        self._TauOutputs = ['Default', 'Framework']
        self._Outputs = ['Events', 'Disparities']
        self._Tau = 0.005
        self._DisparityRange = [0, 60]

    def _OnInitialization(self):
        if self._FramesDt == 0:
            self.LogWarning("Sampling interval (_FramesDt) is null")
            self.LogWarning("No pictures will be created")
            self.Active = False
            return True

        self.ScreenSize = tuple(self.Geometry)
        self.StreamsContainers = {}
        self.StreamsLastFrames = {}
        self.StreamsFAxs = {}

        self.UsedContainers = set()
        for Output in self._Outputs:
            for DisplayedType in Output.split('+'):
                self.UsedContainers.add(DisplayedType)

        class ModuleDisparitiesContainer(DisparitiesContainer):
            _DisparityRange = self._DisparityRange


        self.TemplateContainers = {'Events':EventsContainer, 'Flows':FlowsContainer, 'Disparities':ModuleDisparitiesContainer}

        for subStreamIndex in self.__SubStreamInputIndexes__:
            self.AddSubStreamData(subStreamIndex)
        self.Active = True

        self.FolderName = self.PicturesFolder + self.__Name__ + '/'
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
            return
        for Container in self.StreamsContainers[event.SubStreamIndex].values():
            Container.OnEvent(event)
        if event.timestamp > self.StreamsLastFrames[event.SubStreamIndex][0] + self._FramesDt:
            self.Draw(event.SubStreamIndex, event.timestamp)
        return

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
        for TauName in self._TauOutputs:
            if TauName == 'Default':
                Tau = self._Tau
            elif TauName == 'Framework':
                Tau = self.FrameworkAverageTau
                if Tau is None or Tau == 0:
                    Tau = self._Tau
            for output in self._Outputs:
                self.StreamsFAxs[subStreamIndex][output][1].cla()
                for Container in output.split('+'):
                    self.StreamsContainers[subStreamIndex][Container].Draw(t, Tau, self.StreamsFAxs[subStreamIndex][output][1])
                self.StreamsFAxs[subStreamIndex][output][1].set_xlim(0, self.ScreenSize[0])
                self.StreamsFAxs[subStreamIndex][output][1].set_ylim(0, self.ScreenSize[1])
                self.StreamsFAxs[subStreamIndex][output][1].set_title("{0:.3f}s".format(t))
                self.StreamsFAxs[subStreamIndex][output][0].savefig(self.FolderName + output + '_{0}Tau'.format(TauName) + '_{0}_{1:06d}.png'.format(subStreamIndex, FrameIndexes))
            self.StreamsLastFrames[subStreamIndex] = (t, FrameIndexes)

    def OpenPictures(self, output = None, subStreamIndex = 0, TauName = 'Default'):
        if output is None:
            output = self._Outputs[0]
        os.system(_PICTURES_VIEWER + ' ' + self.FolderName + output + '_{0}Tau'.format(TauName) + '_{0}_0.png'.format(subStreamIndex))

class EventsContainer:
    _TauMultiplier = 2
    def __init__(self, ScreenSize):
        self.STContext = np.zeros(ScreenSize + (2,))

    def OnEvent(self, event):
        if event.Has(CameraEvent):
            self.STContext[event.location[0], event.location[1], event.polarity] = event.timestamp

    def Draw(self, t, Tau, ax):
        Map = np.zeros(self.STContext.shape[:2])
        for Polarity in range(2):
            Map = (Map + ((t - self.STContext[:,:,Polarity]) < Tau*self._TauMultiplier))
        ax.imshow(np.transpose(Map), origin = 'lower', cmap = 'binary', vmin = 0, vmax = 2)

class FlowsContainer:
    _TauMultiplier = 1
    def __init__(self, ScreenSize):
        self.FlowsList = []

    def OnEvent(self, event):
        if not event.Has(FlowEvent):
            return
        self.FlowsList += [(event.timestamp, event.location, event.flow)]

    def Draw(self, t, Tau, ax):
        while self.FlowsList and self.FlowsList[0][0] < t - Tau*self._TauMultiplier:
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
            df = flow * Tau*self._TauMultiplier
            ax.plot([x,x+df[0]], [y, y+df[1]], color = color)

class DisparitiesContainer:
    _TauMultiplier = 2
    _DisparityRange = [0, 60]
    def __init__(self, ScreenSize):
        self.STDContext = np.zeros(ScreenSize + (2,))
        self.MaxD = 5

    def OnEvent(self, event):
        if not event.Has(DisparityEvent):
            return
        self.STDContext[event.location[0], event.location[1],:] = np.array([event.disparity, event.timestamp])

    def Draw(self, t, Tau, ax):
        Map = self.STDContext[:,:,0] * ((t - self.STDContext[:,:,1]) < Tau*self._TauMultiplier)
        self.MaxD = max(self.MaxD, self.STDContext[:,:,0].max())
        ax.imshow(np.transpose(Map), origin = 'lower', vmin = self._DisparityRange[0], vmax = self._DisparityRange[1], cmap = 'hot')
