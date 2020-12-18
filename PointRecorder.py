import matplotlib.pyplot as plt
import numpy as np
import datetime

import json
import atexit

from PEBBLE import Module

class Clicker(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to record points in a sequence
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__ReferencesAsked__ = ['Memory']
        self.__Type__ = 'Analysis'

        self._TW = 0.01
        self._BinDt = 0.05
        self._FirstPictureAt = 0.01

    def _InitializeModule(self, **kwargs):

        self._Mem = self.__Framework__.Tools[self.__CreationReferences__['Memory']]

        self.RecordedPoints = []
        self.PreviousFramePoints = []
        self.CurrentT = self._FirstPictureAt - self._BinDt

        self.Figure = plt.figure()
        self.Ax = self.Figure.add_subplot(1,1,1)
        self.Image = None
        
        self.UnsavedData = False
        self.WasActive = []
        self._RewindLimit = None

        atexit.register(self._OnClosing)

        return True

    def _OnEventModule(self, event):
        if event.timestamp > self.CurrentT + self._BinDt:
            if self.RecordedPoints:
                self._RewindLimit = self.RecordedPoints[-1][1]
            self.CurrentT = event.timestamp
            if self.Image is None:
                self.Image = self.Ax.imshow(np.transpose(self._Mem.STContext.max(axis = 2) > event.timestamp - self._TW), origin = 'lower')
            else:
                self.Image.set_data(np.transpose(self._Mem.STContext.max(axis = 2) > event.timestamp - self._TW))
            
            self.PreviousPs = [self.Ax.plot(Point[2], Point[3], 'vg'*(nPoint > 0) + '*b'*(nPoint == 0))[0] for nPoint, Point in enumerate(self.PreviousFramePoints)]
            self.PreviousLinks = []
            for nPoint in range(len(self.PreviousPs[:-1])):
                self.PreviousLinks.append(self.Ax.plot([self.PreviousFramePoints[nPoint][2], self.PreviousFramePoints[nPoint+1][2]], [self.PreviousFramePoints[nPoint][3], self.PreviousFramePoints[nPoint+1][3]], 'g')[0])

            self.NewID = -1
            self.NewPoints = []
            self.Ps = []

            self._NextID()

            self.Recording = True
            self.NewPoints += [[None, self._Mem.LastEvent.timestamp, 0, 0]]

            self.Ps += [self.Ax.plot(self.NewPoints[-1][2], self.NewPoints[-1][3], 'vr')[0]]
            self.PreviousFramePoints = []

            self.Ax.set_title("Setting NewID = {0} ({2}) at t = {1:.2f}".format(self.NewID, self._Mem.LastEvent.timestamp, 'None'*int(self.NewPoints[self.NewID][0] is None) + 'Set'*(1 - int(self.NewPoints[self.NewID][0] is None))))

            self.CurrentCIDButt = self.Figure.canvas.mpl_connect('button_press_event', self.RecordEvent)
            self.CurrentCIDKey = self.Figure.canvas.mpl_connect('key_press_event', self.RecordEvent)
            while self.Recording:
                plt.pause(0.1)
            for P in self.Ps:
                P.remove()
            for P in self.PreviousPs:
                P.remove()
            for L in self.PreviousLinks:
                L.remove()
            self.Figure.canvas.mpl_disconnect(self.CurrentCIDButt)
            self.Figure.canvas.mpl_disconnect(self.CurrentCIDKey)

        return event

    def _NextID(self):
        if not self.WasActive:
            self.NewID += 1
            return None
        else:
            if max(self.WasActive) <= self.NewID:
                self.NewID += 1
                return None
            else:
                PossibleIDs = [ID for ID in self.WasActive if ID > self.NewID]
                self.NewID = min(PossibleIDs)
                for i in range(self.NewID - len(self.NewPoints)):
                    self.NewPoints += [[None, self._Mem.LastEvent.timestamp, 0, 0]]
                    self.Ps += [self.Ax.plot(self.NewPoints[-1][2], self.NewPoints[-1][3], 'vr')[0]]
                return None

    def RecordEvent(self, event):
        if event.name == 'button_press_event':
            if event.button == 1:
                self.NewPoints[self.NewID][0] = self.NewID
                self.NewPoints[self.NewID][2] = event.xdata
                self.NewPoints[self.NewID][3] = event.ydata
                self.Ps[self.NewID].set_data(self.NewPoints[self.NewID][2], self.NewPoints[self.NewID][3])
                self.Ax.set_title("Setting NewID = {0} ({2}) at t = {1:.2f}".format(self.NewID, self._Mem.LastEvent.timestamp, 'None'*int(self.NewPoints[self.NewID][0] is None) + 'Set'*(1 - int(self.NewPoints[self.NewID][0] is None))))
        else:
            if event.key == 'right':
                self.Ps[self.NewID].set_color('k')
                self._NextID()
                if len(self.NewPoints) == self.NewID:
                    self.NewPoints += [[None, self._Mem.LastEvent.timestamp, 0, 0]]
                    self.Ps += [self.Ax.plot(self.NewPoints[self.NewID][2], self.NewPoints[self.NewID][3], 'vr')[0]]
                else:
                    self.Ps[self.NewID].set_color('r')

                self.Ax.set_title("Setting NewID = {0} ({2}) at t = {1:.2f}".format(self.NewID, self._Mem.LastEvent.timestamp, 'None'*int(self.NewPoints[self.NewID][0] is None) + 'Set'*(1 - int(self.NewPoints[self.NewID][0] is None))))
            elif event.key == 'left' and self.NewID >= 1:
                self.Ps[self.NewID].set_color('k')
                self.NewID -= 1
                self.Ps[self.NewID].set_color('r')
                self.Ax.set_title("Setting NewID = {0} ({2}) at t = {1:.2f}".format(self.NewID, self._Mem.LastEvent.timestamp, 'None'*int(self.NewPoints[self.NewID][0] is None) + 'Set'*(1 - int(self.NewPoints[self.NewID][0] is None))))
            elif event.key == 'down':
                self.Ax.set_title("Generating next frame")
                self.WasActive = []
                for Point in self.NewPoints:
                    if not Point[0] is None:
                        self.RecordedPoints += [Point]
                        self.PreviousFramePoints += [Point]
                        self.WasActive += [Point[0]]
                self.UnsavedData = True
                self.Recording = False
            elif event.key == 'escape':
                self.Recording = False
                self.__Framework__.Paused = self.__Name__
                for Point in self.NewPoints:
                    if not Point[0] is None:
                        self.RecordedPoints += [Point]
                    self.UnsavedData = True
                self.Ax.set_title("Stopped")
            elif event.key == 'delete':
                self.NewPoints[self.NewID][0] = None
                self.NewPoints[self.NewID][2] = 0
                self.NewPoints[self.NewID][3] = 0
                self.Ax.set_title("Setting NewID = {0} ({2}) at t = {1:.2f}".format(self.NewID, self._Mem.LastEvent.timestamp, 'None'*int(self.NewPoints[self.NewID][0] is None) + 'Set'*(1 - int(self.NewPoints[self.NewID][0] is None))))
                self.Ps[self.NewID].set_data(self.NewPoints[self.NewID][2], self.NewPoints[self.NewID][3])
            elif event.key == 'backspace':
                tAimed = self._Mem.LastEvent.timestamp - self._BinDt/10
                if (not self._RewindLimit is None and tAimed < self._RewindLimit):
                    print "Unable to rewind more"
                    return None
                self.Ax.set_title("Rewinding")
                ForbiddingModules = self.__Framework__.Rewind(tAimed)
                if ForbiddingModules:
                    self.Ax.set_title("Rewind failed")
                    print "Unable to rewind due to module " + " ,".join(ForbiddingModules)
                else:
                    self.WasActive = []
                    self.Recording = False

    def SaveRecordedData(self):
        DataDict = {}
        DataDict['StreamName'] = self.__Framework__._GetStreamFormattedName(self)
        DataDict['RecordedPoints'] = list(self.RecordedPoints)
        DataDict['RecordingDate'] = datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')

        StreamName = DataDict['StreamName']
        NameParts = StreamName.split('/')
        NewName = NameParts[-1].replace('.', '_') + '.gnd'
        SaveName = '/'.join(NameParts[:-1]) + '/' + NewName
        ans = raw_input("Autosave to {0} (y), enter new name (.gnd) or discard data (D) : ".format(SaveName))

        if ans == 'y':
            with open(SaveName, 'w') as outfile:  
                json.dump(DataDict, outfile)
                self.UnsavedData = False
                print "Data saved in {0} json file.".format(SaveName)
        elif ans == 'D':
            return None
        elif '.gnd' in ans:
            with open(ans, 'w') as outfile:
                json.dump(DataDict, outfile)
                self.UnsavedData = False
                print "Data saved in {0} json file.".format(ans)
        else:
            print "No valid answer. Stop typing random stuff please."
            self.SaveRecordedData()

    def _Rewind(self, tNew):
        self.CurrentT = tNew - self._BinDt

    def _OnClosing(self):
        if self.UnsavedData and self.RecordedPoints:
            print "Unsaved data recorded ({0} points).".format(len(self.RecordedPoints))
            self.SaveRecordedData()
