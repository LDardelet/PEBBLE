import numpy as np
import DisplayHandler
import Memory

from PEBBLE import Module, Event, Framework

class ResolutionEnhancer(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to enhance a stream resolution by creating artificial timestamps depending on a prior flow computation.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Computation'
        self.__ReferencesAsked__ = ['Flow Computer', 'Memory']

        self._AddMemory = True
        self._AddDisplay = True

        self._TemporalWindow = 0.03
        self._EnhancementFactor = 1
        self._ExtensionFactor = 3

    def _InitializeModule(self, **kwargs):

        self._EnhancementFactor = int(self._EnhancementFactor)
        self.InitialGeometry = self.__Framework__._GetStreamGeometry(self)
        self._LinkedMemory = self.__Framework__.Tools[self.__CreationReferences__['Memory']]
        self.EnhancedGeometry = self.InitialGeometry * np.array([self._EnhancementFactor, self._EnhancementFactor, 1])

        SavedFrameworkGeometryMethod = self.__Framework__._GetStreamGeometry
        self.__Framework__._GetStreamGeometry = lambda Tool: self.EnhancedGeometry
        if self._AddMemory:
            self.CreatedMemory = Memory.Memory(self.__Name__+'->EnhancedMemory', self.__Framework__, {})
            Module.__Initialize__(self.CreatedMemory)
        if self._AddDisplay:
            self.CreatedDisplay = DisplayHandler.DisplayHandler(self.__Name__+'->EnhancedDisplay', self.__Framework__, {})
            Module.__Initialize__(self.CreatedDisplay)
        self.__Framework__._GetStreamGeometry = SavedFrameworkGeometryMethod

        self._R = self._ExtensionFactor*int((self._EnhancementFactor - 1)/2)
        self._LookupRadius = int((self._ExtensionFactor - 1)/2)
        self.FlowMap = self.__Framework__.Tools[self.__CreationReferences__['Flow Computer']].FlowMap
        self.N2Map = self.__Framework__.Tools[self.__CreationReferences__['Flow Computer']].NormMap
        self.DetMap = self.__Framework__.Tools[self.__CreationReferences__['Flow Computer']].DetMap
        return True

    def _OnEventModule(self, event):
        EnhancedEventCenterLocation = np.array(event.location)*self._EnhancementFactor

        Det = self.DetMap[event.location[0], event.location[1], event.polarity]
        if Det < 50:
            return event
        Flow = self.FlowMap[event.location[0], event.location[1], event.polarity,:]
        NFlow = np.linalg.norm(Flow)
        FlowUnitVector = Flow/NFlow

        for dx in range(-self._LookupRadius, self._LookupRadius+1):
            for dy in range(-self._LookupRadius, self._LookupRadius+1):
                Offset = np.array([dx, dy])
                LookupLocation = Offset + event.location
                if (LookupLocation < 0).any() or (LookupLocation + 1 > self.InitialGeometry[:2]).any():
                    continue
                TOffset = (Offset*Flow).sum()/self.N2Map[event.location[0], event.location[1], event.polarity]
                if TOffset > self._TemporalWindow and TOffset < event.timestamp - self._LinkedMemory.STContext[LookupLocation[0], LookupLocation[1], event.polarity]:
                    return event
        
        for dx in range(-self._R, self._R+1):
            for dy in range(-self._R, self._R+1):
                dLocation = np.array([dx, dy])
                ScalarValue = (dLocation*Flow).sum()
                if ScalarValue <= 0: # Inverse to flow, so in the past, and less than half a pixel away
                    dt = ScalarValue / self.N2Map[event.location[0], event.location[1], event.polarity]/self._EnhancementFactor
                    CreatedEventLocation = EnhancedEventCenterLocation + dLocation
                    if (CreatedEventLocation < 0).any() or (CreatedEventLocation+1 > self.CreatedMemory.STContext.shape[:2]).any():
                        continue
                    CreatedEvent = Event(event.timestamp + dt, CreatedEventLocation, event.polarity)
                    self._LocalPipelineOnEvent(CreatedEvent)
        return event

    def _LocalPipelineOnEvent(self, event):
        if self._AddMemory:
            self.CreatedMemory.__OnEvent__(event)
        if self._AddDisplay:
            self.CreatedDisplay.__OnEvent__(event)

