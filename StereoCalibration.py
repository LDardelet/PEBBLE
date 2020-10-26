from Framework import Module, Event
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import imageio

class StereoCalibration(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Module template to be filled foe specific purpose
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Computation'

        self.__ReferencesAsked__ = []
        self._MonitorDt = 0. # By default, a module does not store any data over time.
        self._NeedsLogColumn = False
        self._MonitoredVariables = []

        self._CalibrationInput = ''
        self._SendUncalibratedEvents = False
        self._TriggerCalibrationAfterRatio = 0.1

    def _InitializeModule(self, **kwargs):
        self.UsedGeometry = np.array(self.Geometry[:2])
        if type(self._CalibrationInput) == str:
            if self._CalibrationInput:
                self.CalibrationData = pickle.load(open(self._CalibrationInput, 'rb'))
                self.CalibrationMatrix[0] = self.CalibrationData['M0']
                self.CalibrationMatrix[1] = self.CalibrationData['M1']
                self.FundamentalMatrix = self.CalibrationData['F']
                self.Calibrated = True
            else:
                self.STContext = -np.inf*np.ones(tuple(self.UsedGeometry) + (2,))
                self.Calibrated = False
                self.EventsReceived = np.array([0,0])
                self.MinEventsForCalibration = self.UsedGeometry[:2].prod() * self._TriggerCalibrationAfterRatio
                return True
        elif type(self._CalibrationInput) == tuple:
            if type(self._CalibrationInput[0]) == str:
                Data = np.zeros(tuple(self.UsedGeometry) + (2,))
                Data[:,:,0] = 1-np.flip(np.transpose(imageio.imread(self._CalibrationInput[0])[:,:,0]), axis = 1)/255
                Data[:,:,1] = 1-np.flip(np.transpose(imageio.imread(self._CalibrationInput[1])[:,:,0]), axis = 1)/255
                self.Calibrate(Data, False)
            else:
                self.CalibrationMatrix = list(self._CalibrationInput[0])
                self.FundamentalMatrix = np.array(self._CalibrationInput[1])
                self.Calibrated = True
        return True

    def _OnEventModule(self, event):
        if self.Calibrated:
            XProj = self.CalibrationMatrix[event.cameraIndex].dot(np.array([event.location[0], event.location[1], 1]))
            event.location[:] = np.rint(XProj[:2] / XProj[-1]).astype(int)
            if (event.location < 0).any() or (event.location >= self.UsedGeometry).any():
                return None
            return event
        else:
            self.LastEvent = event.Copy()
            position = tuple(self.LastEvent.location.tolist() + [event.cameraIndex])
            
            self.STContext[position] = event.timestamp
            self.EventsReceived[event.cameraIndex] += 1
            if (self.EventsReceived >= self.MinEventsForCalibration).all():
                self.Calibrate(self.STContext, True)
                delattr(self, 'LastEvent')
                delattr(self, 'STContext')
                delattr(self, 'EventsReceived')
                
            if self._SendUncalibratedEvents:
                return event
            else:
                return None

    def Calibrate(self, Data, AutoEnhance):
        self.Calibrating = True
        C = CalibrationSystem(self, Data, AutoEnhance)
        while(self.Calibrating):
            plt.pause(0.01)
        delattr(self, 'Calibrating')
        self.Calibrated = True

class CalibrationSystem:
    def __init__(self, Module, Data, AutoEnhancePictures):
        self.Data = Data
        self.AutoEnhancePictures = AutoEnhancePictures
        self.Module = Module
        self.StoredPoints = [[],[]]
        self.CurrentPoints = [None, None]
        self.Zoomed = [False, False]
        plt.close('all')
        self.f, self.axs = plt.subplots(2,2)
        self.IniAxs = list(self.axs[0,:])
        for nrow in range(2):
            for ncol in range(2):
                self.axs[nrow, ncol].set_xlim(0, self.Module.UsedGeometry[0])
                self.axs[nrow, ncol].set_ylim(0, self.Module.UsedGeometry[1])
                self.axs[nrow, ncol].set_aspect('equal')
        self.RectImages = [None, None]
        
        self.FundamentalMatrix = np.identity(3)
        self.RectificationMatrices = [np.identity(3), np.identity(3)]
        
        self.PlottedPoints = []
        self.PlottedRectifiedPoints = [[],[]]
        self.PlottedRectifiedEpipolarLines = [[],[]]
        self.STContexts = []

        self.Tau = self.Data.max()/2
        for i in range(2):
            ST = self.Data[:,:,i]
            Map = np.e**(-(ST.max() - ST) / self.Tau)
            self.STContexts += [self.IniAxs[i].imshow(np.transpose(Map), origin = 'lower', cmap='binary')]
            self.RectImages[i] = self.axs[1,i].imshow(np.transpose(Map), origin = 'lower', cmap='binary')
            xs, ys = np.where(Map > 0)
            vs = Map[xs, ys]
            self.PlottedPoints += [np.array([xs, ys, np.ones(xs.shape[0]), vs])]
        self.f.canvas.mpl_connect('button_press_event', self.OnClick)
        self.f.canvas.mpl_connect('close_event', self.OnClosing)
        self.f.canvas.mpl_connect('resize_event', self.OnResize)
    def OnClick(self, event):
        if event.button == 3:
            self.Zoom()
            return
        if event.inaxes is None:
            return
        try:
            nax = self.IniAxs.index(event.inaxes)
        except:
            return
        print("Axes {0}, x = {1}, y = {2}".format(nax, event.xdata, event.ydata))
        if not self.Zoomed[nax]:
            self.Zoom((event.xdata, event.ydata), nax)
            return
        if not self.CurrentPoints[nax] is None:
            self.CurrentPoints[nax].set_xdata([event.xdata])
            self.CurrentPoints[nax].set_ydata([event.ydata])
            return
        self.CurrentPoints[nax] = self.IniAxs[nax].plot(event.xdata, event.ydata, 'or')[0]
        
        if not None in self.CurrentPoints:
            for nax, dot in enumerate(self.CurrentPoints):
                x, y = dot.get_xdata()[0], dot.get_ydata()[0]
                self.StoredPoints[nax] += [np.array([x, y, 1])]
                dot.set_color('g')
                self.PlottedRectifiedPoints[nax] += [self.axs[1,nax].plot(x, y, 'og')[0]]
                self.PlottedRectifiedEpipolarLines[nax] += [self.axs[1,1-nax].plot([0, self.Module.UsedGeometry[0]], [y, y], '--g')[0]]
            self.CurrentPoints = [None, None]
            self.Zoom()
            if len(self.StoredPoints[0]) >= 8:
                self.Rectify()
    def OnClosing(self, event):
        self.Module.CalibrationMatrix = [np.array(self.RectificationMatrices[0]), np.array(self.RectificationMatrices[1])]
        self.Module.FundamentalMatrix = np.array(self.FundamentalMatrix)
        self.Module.Calibrating = False

    def EnhanceView(self, Center = None, nax = None, Radius = 10):
        if not self.AutoEnhancePictures:
            return
        if Center is None:
            for nax in range(2):
                ST = self.Data[:,:,nax]
                Map = np.e**(-(ST.max() - ST) / self.Tau)
                self.STContexts[nax].set_data(np.transpose(Map))
            return
        else:
            ST = self.Data[int(max(0,Center[0]-Radius)):int(Center[0]+Radius) + 1,int(max(0,Center[1]-Radius)):int(Center[1]+Radius) + 1,nax]
            timestamps = ST[np.where(ST >= 0)]
            NActivePixels = int(0.1 * (2*Radius+1)**2)
            if NActivePixels > timestamps.shape[0]:
                tMin = self.Tau * 2
            else:
                tMin = np.sort(timestamps)[-NActivePixels]
            print("Setting tMin = {0:.3f}".format(tMin))
            Data = np.e**((self.Data[:,:,nax] - timestamps.max()) / tMin)
            self.STContexts[nax].set_data(np.transpose(Data))

    def Zoom(self, Center = None, nax = None, Radius = 20):
        self.EnhanceView(Center, nax, Radius)
        if Center is None:
            for ncol in range(2):
                self.axs[0, ncol].set_xlim(0, self.Module.UsedGeometry[0])
                self.axs[0, ncol].set_ylim(0, self.Module.UsedGeometry[1])
                self.Zoomed[ncol] = False
        else:
            self.axs[0, nax].set_xlim(Center[0] - Radius, Center[0] + Radius)
            self.axs[0, nax].set_ylim(Center[1] - Radius, Center[1] + Radius)
            self.Zoomed[nax] = True

    def OnResize(self, event):
        self.f.tight_layout()

    def Rectify(self):
        LXs, RXs = np.array(self.StoredPoints[0])[:,:2], np.array(self.StoredPoints[1])[:,:2]
        self.FundamentalMatrix = cv2.findFundamentalMat(LXs, RXs)[0]
        self.RectificationMatrices = cv2.stereoRectifyUncalibrated(LXs, RXs, self.FundamentalMatrix, tuple(self.Module.UsedGeometry))[1:]
        self.UpdateRectifiedImages()
        
    def UpdateRectifiedImages(self):
        for nax in range(2):
            Xs = self.RectificationMatrices[nax].dot(self.PlottedPoints[nax][:3,:])
            Ys = (Xs[:2] / Xs[-1]).astype(int)
            ValidIndexes = np.where((Ys[0,:] >= 0) * (Ys[1,:] >= 0) * (Ys[0,:] < self.Module.UsedGeometry[0]) * (Ys[1,:] < self.Module.UsedGeometry[1]))
            Map = np.zeros(self.Module.UsedGeometry)
            Map[Ys[0,ValidIndexes], Ys[1,ValidIndexes]] = self.PlottedPoints[nax][3,ValidIndexes]
            self.RectImages[nax].set_data(np.transpose(Map))
            for nPoint, Point in enumerate(self.StoredPoints[nax]):
                ProjPoint = self.RectificationMatrices[nax].dot(Point)
                ProjPoint = ProjPoint[:2]/ProjPoint[-1]
                self.PlottedRectifiedPoints[nax][nPoint].set_xdata([ProjPoint[0]])
                self.PlottedRectifiedPoints[nax][nPoint].set_ydata([ProjPoint[1]])
                self.PlottedRectifiedEpipolarLines[nax][nPoint].set_ydata([ProjPoint[1], ProjPoint[1]])
