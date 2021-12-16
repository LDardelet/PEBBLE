from ModuleBase import ModuleBase
from Events import CameraEvent, DisparityEvent, FlowEvent, TrackerEvent, TwistEvent

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
        self._TauOutputs = ['Framework'] # among 'Framework', 'Default'
        self._Outputs = ['Disparities', 'Events+Trackers']
        self._DefaultTau = 0.005
        self._RemoveTicks = True
        self._DisparityRange = [0, 60]

        self._EventsMonotonicLastT = True
        
        self._TrackersOnlyLocked = True
        self._TrackersIDs = True

    def _OnInitialization(self):
        if self._FramesDt == 0:
            self.LogWarning("Sampling interval (_FramesDt) is null")
            self.LogWarning("No pictures will be created")
            self.Active = False
            return True

        self.ScreenSize = tuple(self.Geometry)
        self.StreamsContainers = {}
        self.StreamsLastFrames = {}
        self.StreamsFramesInfo = {}
        self.StreamsFAxs = {}

        self.UsedContainers = set()
        for Output in self._Outputs:
            for DisplayedType in Output.split('+'):
                self.UsedContainers.add(DisplayedType)

        class ModuleEventsContainer(EventsContainer):
            _MonotonicLastT = self._EventsMonotonicLastT
        class ModuleFlowsContainer(FlowsContainer):
            pass
        class ModuleDisparitiesContainer(DisparitiesContainer):
            _DisparityRange = self._DisparityRange
        class ModuleTrackersContainer(TrackersContainer):
            _TrackersOnlyLocked = self._TrackersOnlyLocked
            _TrackersIDs = self._TrackersIDs
        class ModuleTwistContainer(TwistContainer):
            pass

        self.TemplateContainers = {'Events':ModuleEventsContainer, 'Flows':ModuleFlowsContainer, 'Disparities':ModuleDisparitiesContainer, 'Trackers':ModuleTrackersContainer, 'Twist': ModuleTwistContainer}

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
        self.StreamsFramesInfo[subStreamIndex] = {Container:[] for Container in self.UsedContainers}
        self.StreamsFAxs[subStreamIndex] = {}
        for output in self._Outputs:
            self.StreamsFAxs[subStreamIndex][output] = plt.subplots(1,1)
            plt.close(self.StreamsFAxs[subStreamIndex][output][0])
        self.StreamsLastFrames[subStreamIndex] = (0, -1)

    def Draw(self, subStreamIndex, t):
        FrameIndexes = self.StreamsLastFrames[subStreamIndex][1] + 1
        for TauName in self._TauOutputs:
            if TauName == 'Default':
                Tau = self._DefaultTau
            elif TauName == 'Framework':
                Tau = self.FrameworkAverageTau
                if Tau is None or Tau == 0:
                    Tau = self._DefaultTau
            for output in self._Outputs:
                self.StreamsFAxs[subStreamIndex][output][1].cla()
                if self._RemoveTicks:
                    self.StreamsFAxs[subStreamIndex][output][1].tick_params('both', bottom =  False, labelbottom = False, left = False, labelleft = False, top = False, labeltop = False, right = False, labelright = False)
                self.StreamsFAxs[subStreamIndex][output][1].set_title("{0:.3f}s".format(t))
                for Container in output.split('+'):
                    self.StreamsFramesInfo[subStreamIndex][Container].append(self.StreamsContainers[subStreamIndex][Container].Draw(t, Tau, self.StreamsFAxs[subStreamIndex][output][1]))
                self.StreamsFAxs[subStreamIndex][output][1].set_xlim(0, self.ScreenSize[0])
                self.StreamsFAxs[subStreamIndex][output][1].set_ylim(0, self.ScreenSize[1])
                self.StreamsFAxs[subStreamIndex][output][0].savefig(self.FolderName + output + '_{0}Tau'.format(TauName) + '_{0}_{1:06d}.png'.format(subStreamIndex, FrameIndexes))
            self.StreamsLastFrames[subStreamIndex] = (t, FrameIndexes)

    def OpenPictures(self, output = None, subStreamIndex = 0, TauName = 'Default'):
        if output is None:
            output = self._Outputs[0]
        os.system(_PICTURES_VIEWER + ' ' + self.FolderName + output + '_{0}Tau'.format(TauName) + '_{0}_0.png'.format(subStreamIndex))

class EventsContainer:
    _TauMultiplier = 2
    _MonotonicLastT = True
    def __init__(self, ScreenSize):
        self.STContext = np.zeros(ScreenSize + (2,))
        self.LastTConsidered = -np.inf

    def OnEvent(self, event):
        if event.Has(CameraEvent):
            self.STContext[event.location[0], event.location[1], event.polarity] = event.timestamp

    def Draw(self, t, Tau, ax):
        Map = np.zeros(self.STContext.shape[:2])
        tMin = t - Tau*self._TauMultiplier
        if self._MonotonicLastT:
            tMin = max(self.LastTConsidered, tMin)
        self.LastTConsidered = tMin
        for Polarity in range(2):
            Map = (Map + (self.STContext[:,:,Polarity] > tMin))
        ax.imshow(np.transpose(Map), origin = 'lower', cmap = 'binary', vmin = 0, vmax = 2)
        return ((Map == 1).sum(), (Map == 2).sum(), (Map > 0).sum() / np.prod(self.STContext.shape[:2]))

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
        NFlows = 0
        for _, (x, y), flow in self.FlowsList:
            n = np.linalg.norm(flow)
            if n == 0:
                continue
            NFlows += 1
            nf = flow / n
            r = max(0., nf[0])
            g = max(0., (nf * np.array([-1, np.sqrt(3)])).sum()/2)
            b = max(0., (nf * np.array([-1, -np.sqrt(3)])).sum()/2)
            color = np.sqrt(np.minimum(np.array([r, g, b, 1]), np.ones(4, dtype = float)))
            df = flow * Tau*self._TauMultiplier
            ax.plot([x,x+df[0]], [y, y+df[1]], color = color)
        return NFlows

class DisparitiesContainer:
    _TauMultiplier = 2
    _DisparityRange = [0, 60]
    def __init__(self, ScreenSize):
        self.STDContext = np.zeros(ScreenSize + (2,))
        self.MaxD = 5

    def OnEvent(self, event):
        if not event.Has(DisparityEvent) or not event.Has(CameraEvent):
            return
        self.STDContext[event.location[0], event.location[1],:] = np.array([event.disparity, event.timestamp])

    def Draw(self, t, Tau, ax):
        Map = self.STDContext[:,:,0] * ((t - self.STDContext[:,:,1]) < Tau*self._TauMultiplier)
        self.MaxD = max(self.MaxD, self.STDContext[:,:,0].max())
        ax.imshow(np.transpose(Map), origin = 'lower', vmin = self._DisparityRange[0], vmax = self._DisparityRange[1], cmap = 'hot')
        return (Map > 0).sum() / np.prod(self.STDContext.shape[:2])

class TrackersContainer:
    _TauMultiplier = 3
    _TrackersOnlyLocked = True
    _TrackersIDs = False
    def __init__(self, ScreenSize):
        self.Trackers = {}
        from TrackerTRS import StateClass 
        self.TrackerStateC = StateClass

    def OnEvent(self, event):
        if not event.Has(TrackerEvent):
            return
        if self._TrackersOnlyLocked and (event.TrackerColor != self.TrackerStateC._COLORS[self.TrackerStateC._STATUS_LOCKED] or event.TrackerMarker != self.TrackerStateC._MARKERS[0]):
            return
        self.Trackers[event.TrackerID] = (event.timestamp, event.TrackerLocation, event.TrackerColor, event.TrackerMarker)

    def Draw(self, t, Tau, ax):
        Tau *= self._TauMultiplier
        NTrackers = 0
        for ID in list(self.Trackers.keys()):
            if self.Trackers[ID][0] < t - Tau:
                del self.Trackers[ID]
                continue
            _, (x, y), color, marker = self.Trackers[ID]
            ax.plot(x, y, color = color, marker = marker)
            NTrackers += 1
            if self._TrackersIDs:
                ax.text(x+5, y, str(ID), color = color)
        return NTrackers

class TwistContainer:
    def __init__(self, ScreenSize):
        self.LastV = np.zeros(3)
        self.LastOmega = np.zeros(3)

    def OnEvent(self, event):
        if not event.Has(TwistEvent):
            return
        self.LastV = event.v
        self.LastOmega = event.omega

    def Draw(self, t, Tau, ax):
        previous_title = ax.get_title()
        if previous_title:
            previous_title = previous_title + '\n'
        previous_title = previous_title + "Vx = {0:.3f}, Vy = {1:.3f}, Vz = {2:.3f}\n".format(*self.LastV.tolist())
        previous_title = previous_title + "Wx = {0:.3f}, Wy = {1:.3f}, Wz = {2:.3f}".format(*self.LastOmega.tolist())
        ax.set_title(previous_title)
        return (np.array(self.LastV), np.array(self.LastOmega))
