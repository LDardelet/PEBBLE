import matplotlib.pyplot as plt
import numpy as np
import datetime

import json
import atexit

class Clicker:
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to record points in a sequence
        '''
        self.__ReferencesAsked__ = ['Memory']
        self.__Name__ = Name
        self.__Framework__ = Framework
        self.__Type__ = 'Analysis'
        self.__CreationReferences__ = dict(argsCreationReferences)

        self._BinDt = 0.01
        self._FirstPictureAt = 0.01

    def _Initialize(self):
        self._Mem = self.__Framework__.Tools[self.__CreationReferences__['Memory']]

        self.RecordedPoints = []
        self.CurrentT = self._FirstPictureAt - self._BinDt

        self.Figure = plt.figure()
        self.Ax = self.Figure.add_subplot(1,1,1)
        self.Image = None
        
        self.UnsavedData = False

        atexit.register(self._OnClosing)

    def _OnEvent(self, event):
        if event.timestamp > self.CurrentT + self._BinDt:
            self.CurrentT = event.timestamp
            if self.Image is None:
                self.Image = self.Ax.imshow(np.transpose(self._Mem.STContext.max(axis = 2) > event.timestamp - self._BinDt/2), origin = 'lower')
            else:
                self.Image.set_data(np.transpose(self._Mem.STContext.max(axis = 2) > event.timestamp - self._BinDt/2))
            self.NewID = 0
            self.Recording = True
            self.NewPoints = [[None, self._Mem.LastEvent.timestamp, 0, 0]]
            self.Ps = [self.Ax.plot(self.NewPoints[-1][2], self.NewPoints[-1][3], 'vr')[0]]
            self.Ax.set_title("Currently setting NewID = {0} at t = {1:.2f}".format(self.NewID, self._Mem.LastEvent.timestamp))

            self.CurrentCIDButt = self.Figure.canvas.mpl_connect('button_press_event', self.RecordEvent)
            self.CurrentCIDKey = self.Figure.canvas.mpl_connect('key_press_event', self.RecordEvent)
            while self.Recording:
                plt.pause(0.3)
            for P in self.Ps:
                P.remove()
            self.Figure.canvas.mpl_disconnect(self.CurrentCIDButt)
            self.Figure.canvas.mpl_disconnect(self.CurrentCIDKey)

        return event

    def RecordEvent(self, event):
        if event.name == 'button_press_event':
            if event.button == 1:
                self.NewPoints[self.NewID][0] = self.NewID
                self.NewPoints[self.NewID][2] = event.xdata
                self.NewPoints[self.NewID][3] = event.ydata
                self.Ps[self.NewID].set_data(self.NewPoints[self.NewID][2], self.NewPoints[self.NewID][3])
        else:
            if event.key == 'right':
                self.Ps[self.NewID].set_color('k')
                self.NewID += 1
                if len(self.NewPoints) == self.NewID:
                    self.NewPoints += [[None, self._Mem.LastEvent.timestamp, 0, 0]]
                    self.Ps += [self.Ax.plot(self.NewPoints[self.NewID][2], self.NewPoints[self.NewID][3], 'vr')[0]]
                else:
                    self.Ps[self.NewID].set_color('r')

                self.Ax.set_title("Currently setting NewID = {0} at t = {1:.2f}".format(self.NewID, self._Mem.LastEvent.timestamp))
            elif event.key == 'left' and self.NewID >= 1:
                self.Ps[self.NewID].set_color('k')
                self.NewID -= 1
                self.Ps[self.NewID].set_color('r')
                self.Ax.set_title("Currently setting NewID = {0} at t = {1:.2f}".format(self.NewID, self._Mem.LastEvent.timestamp))
            elif event.key == 'down':
                for Point in self.NewPoints:
                    if not Point[0] is None:
                        self.RecordedPoints += [Point]
                self.UnsavedData = True
                self.Recording = False
            elif event.key == 'escape':
                self.Recording = False
                self._Framework.Running = False
                for Point in self.NewPoints:
                    if not Point[0] is None:
                        self.RecordedPoints += [Point]
                    self.UnsavedData = True
            elif event.key == 'delete':
                self.NewPoints.pop(self.NewID)
                for nPoint in range(len(self.NewPoints)):
                    Point = self.NewPoints[nPoint]
                    if not Point[0] is None:
                        Point[0] = nPoint
                self.Ps[self.NewID].remove()
                self.Ps.pop(self.NewID)
                if len(self.NewPoints) == self.NewID:
                    self.NewPoints += [[None, self._Mem.LastEvent.timestamp, 0, 0]]
                    self.Ps += [self.Ax.plot(self.NewPoints[self.NewID][2], self.NewPoints[self.NewID][3], 'vr')[0]]
                else:
                    self.Ps[self.NewID].set_color('r')
                    

    def SaveRecordedData(self):
        DataDict = {}
        DataDict['StreamName'] = self.__Framework__.StreamHistory[-1]
        DataDict['RecordedPoints'] = list(self.RecordedPoints)
        DataDict['RecordingDate'] = datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')

        StreamName = self.__Framework__.StreamHistory[-1]
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

    def _OnClosing(self):
        if self.UnsavedData:
            print "Unsaved data recorded."
            self.SaveRecordedData()
