from ModuleBase import ModuleBase
from Events import CameraEvent
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import imageio

class StereoCalibration(ModuleBase):
    def _OnCreation(self):
        '''
        Module template to be filled foe specific purpose
        '''
        self._CalibrationInput = ''
        self._CalibrationStoredPoints = [[],[]]
        self._SendUncalibratedEvents = False
        self._TriggerCalibrationAfterRatio = 0.1
        self._EnhanceSTContext = True

    def _OnInitialization(self):
        self.UsedGeometry = np.array(self.Geometry)
        self.RectifyFunction = self.MatrixRectification
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
                self.Calibrate(Data, False, self._CalibrationStoredPoints)
            else:
                self.CalibrationMatrix = list(self._CalibrationInput[0])
                self.FundamentalMatrix = np.array(self._CalibrationInput[1])
                self.Calibrated = True
        elif type(self._CalibrationInput) == dict:
            self._MappingCalibration()
            self.Calibrated = True
        return True

    def _MappingCalibration(self):
        def ExtractData(fName):
            f = open(fName, 'r')
            lines = f.readlines()
            f.close()
            Data = np.array([[float(value) for value in line.strip().split(' ')] for line in lines])
            return np.flip(np.transpose(Data), axis = 1)
        self.Maps = [np.zeros(tuple(self.UsedGeometry) + (2,)), np.zeros(tuple(self.UsedGeometry) + (2,))]
        self.Maps[0][:,:,0] = ExtractData(self._CalibrationInput['0x'])
        self.Maps[0][:,:,1] = ExtractData(self._CalibrationInput['0y'])

        self.Maps[1][:,:,0] = ExtractData(self._CalibrationInput['1x'])
        self.Maps[1][:,:,1] = ExtractData(self._CalibrationInput['1y'])

        for i in range(2):
            self.Maps[i][:,:,1] = self.UsedGeometry[1]-1-self.Maps[i][:,:,1]
        self.RectifyFunction = self.MappingRectification

    def MatrixRectification(self, location, index):
        XProj = self.CalibrationMatrix[index].dot(np.array([location[0], location[1], 1]))
        return np.rint(XProj[:2] / XProj[-1]).astype(int)
    def MappingRectification(self, location, index):
        return self.Maps[index][location[0], location[1],:]

    def _OnEventModule(self, event):
        if self.Calibrated:
            event.location[:] = self.RectifyFunction(event.location, event.SubStreamIndex)
            if (event.location < 0).any() or (event.location >= self.UsedGeometry).any():
                event.Filter()
            return 
        else:

            self.LastEvent = event.Copy()
            position = tuple(self.LastEvent.location.tolist() + [event.SubStreamIndex])
            
            self.STContext[position] = event.timestamp
            self.EventsReceived[event.SubStreamIndex] += 1
            if (self.EventsReceived >= self.MinEventsForCalibration).all():
                self.Calibrate(self.STContext, True)
                delattr(self, 'LastEvent')
                delattr(self, 'STContext')
                delattr(self, 'EventsReceived')
                
            if not self._SendUncalibratedEvents:
                event.Filter()

    def Calibrate(self, Data, AutoEnhance, StoredPoints=[[],[]]):
        self.Calibrating = True
        self.C = CalibrationSystem(self, Data, AutoEnhance, StoredPoints)
        while(self.Calibrating):
            plt.pause(0.01)
        delattr(self, 'Calibrating')
        self.Calibrated = True

    def GenerateCalibrationDataFromFolder(self, Folder, ReferenceCamera, yInvert = True, yMax = None, MetricFactor = 1e-3):
        if yInvert and yMax is None:
            raise Exception("Cannot invert the homographies towards bottom-left origin without screen height")
        if yInvert:
            RyInvert = np.array([[1., 0., 0.],
                      [0., -1, yMax-1],
                      [0., 0., 1.]])
            RyInvertInv = np.linalg.inv(RyInvert)
        else:
            RyInvert = np.identity(3)
            RyInvertInv = np.identity(3)
        CamerasNames = [None, None]
        for ModuleName, Module in self.__Framework__.Modules.items():
            if not Module.__IsInput__:
                continue
            UsedIndex = Module.SubStreamIndex
            if 'right' in ModuleName.lower():
                CamerasNames[UsedIndex] = 'right'
            elif 'left' in ModuleName.lower():
                CamerasNames[UsedIndex] = 'left'
            else:
                raise Exception("Unable to automatically set camera indexes to camera names")
        def ExtractArrayFromFile(FileName, NLines, NCols):
            with open(FileName, 'r') as f:
                Data = np.zeros((NLines, NCols))
                for nLine, Line in enumerate(f.readlines()):
                    if ',' in Line:
                        Terms = [Term.strip() for Term in Line.strip().split(',')]
                    else:
                        Terms = Line.strip().split()
                    for nCol, Term in enumerate(Terms):
                        Data[nLine, nCol] = float(Term)
            return Data
        RFileName = 'Rotation_camera2.txt'
        R = ExtractArrayFromFile(Folder+RFileName, 3, 3)
        TFileName = 'Translation_camera2.txt'
        T = ExtractArrayFromFile(Folder+TFileName, 3, 1).reshape(3) * MetricFactor

        CalibrationData = {'H:Phy->Used':[None, None], 'K:SB->Used':[None, None], 'T:Ref->Stereo':np.array(T), 'R:Ref->Stereo':np.array(R), 'Delta':np.linalg.norm(T), 'refName':ReferenceCamera, 'refIndex': CamerasNames.index(ReferenceCamera)}

        for CameraIndex, CameraName in enumerate(CamerasNames):
            HFileName = 'H'+CameraName+'.txt'
            H = ExtractArrayFromFile(Folder+HFileName, 3, 3)
            CalibrationData['H:Phy->Used'][CameraIndex] = RyInvert.dot(H.dot(RyInvertInv))

            KFileName = 'K'+CameraName+'.txt'
            K = ExtractArrayFromFile(Folder+KFileName, 3, 3)
            CalibrationData['K:SB->Used'][CameraIndex] = RyInvert.dot(H).dot(K)
        return CalibrationData 
            

class CalibrationSystem:
    def __init__(self, Module, Data, AutoEnhancePictures, StoredPoints = [[],[]]):
        self.Data = Data
        self.AutoEnhancePictures = AutoEnhancePictures
        self.Module = Module
        self.CurrentPoints = [None, None]
        self.Zoomed = [False, False]
        plt.close('all')
        self.RectImages = [None, None]
        
        self.FundamentalMatrix = np.identity(3)
        self.RectifiedFundamentalMatrix = np.identity(3)
        self.RectificationMatrices = [np.identity(3), np.identity(3)]
        self.RectificationImages = []
        
        self.PlottedPointsImage = []
        self.PlottedPointsLeft = [[],[]]
        self.PlottedPointsMid = [[],[]]
        self.PlottedEpipolarLines = [[],[]]
        self.PlottedRectifiedPoints = [[],[]]
        self.PlottedRectifiedEpipolarLines = [[],[]]
        self.STContexts = []

        self.BeingRemoved = None

        self.f, self.axs = plt.subplots(2,4)
        self.IniAxs = list(self.axs[:,0])
        self.MatAxs = list(self.axs[:,3])
        for nrow in range(2):
            for ncol in range(3):
                self.axs[nrow, ncol].set_xlim(0, self.Module.UsedGeometry[0])
                self.axs[nrow, ncol].set_ylim(0, self.Module.UsedGeometry[1])
                self.axs[nrow, ncol].set_aspect('equal')
            self.RectificationImages += [self.axs[nrow,3].imshow(np.transpose(self.RectificationMatrices[nrow]), origin = 'lower', cmap = 'hot')]

        self.Tau = self.Data.max()/2
        for i in range(2):
            ST = self.Data[:,:,i]
            Map = np.e**(-(ST.max() - ST) / self.Tau)
            self.STContexts += [self.IniAxs[i].imshow(np.transpose(Map), origin = 'lower', cmap='binary')]
            self.axs[i,1].imshow(np.transpose(Map), origin = 'lower', cmap='binary')
            self.RectImages[i] = self.axs[i,2].imshow(np.transpose(Map), origin = 'lower', cmap='binary')
            xs, ys = np.where(Map > 0)
            vs = Map[xs, ys]
            self.PlottedPointsImage += [np.array([xs, ys, np.ones(xs.shape[0]), vs])]
        self.f.canvas.mpl_connect('button_press_event', self.OnClick)
        self.f.canvas.mpl_connect('scroll_event', self.OnScroll)
        self.f.canvas.mpl_connect('close_event', self.OnClosing)
        self.f.canvas.mpl_connect('resize_event', self.OnResize)

        self.StoredPoints = [[],[]]
        if StoredPoints[0]:
            print("Received previous points")
            for X1, X2 in zip(StoredPoints[0], StoredPoints[1]):
                print("Adding", X1, X2)
                FE1 = FakeEvent(self.IniAxs[0], X1[0], X1[1])
                self.OnClick(FE1)
                FE2 = FakeEvent(self.IniAxs[1], X2[0], X2[1])
                self.OnClick(FE2)
    

    def OnScroll(self, event):
        try:
            nax = self.MatAxs.index(event.inaxes)
        except:
            return
        x = int(event.xdata + 0.5)
        y = int(event.ydata + 0.5)
        self.RectificationMatrices[nax][y,x] += event.step * 0.0001
        self.UpdateRectificationView()

    def OnClick(self, event):
        if event.inaxes is None:
            return
        try:
            nax = self.IniAxs.index(event.inaxes)
        except:
            return
        if event.button == 3:
            if not self.Zoomed[nax]:
                self.Zoom((event.xdata, event.ydata), nax)
                return
            else:
                self.Zoom()
                return
        if event.button == 2:
            if self.CurrentPoints[nax] is None and not self.StoredPoints[nax]:
                return
            xy = np.array([event.xdata, event.ydata])
            ClosestPoint, ClosestDistance = None, np.inf
            for nPoint, Point in enumerate(self.StoredPoints[nax]):
                D = np.linalg.norm(Point[:2] - xy)
                if D < ClosestDistance:
                    ClosestPoint = nPoint
                    ClosestDistance = D
            tbr = (nax, ClosestPoint)
            if not self.CurrentPoints[nax] is None:
                xyCurrent = np.array([self.CurrentPoints[nax].get_xdata(), self.CurrentPoints[nax].get_ydata()])
                D = np.linalg.norm(xyCurrent - xy)
                if D < ClosestDistance:
                    tbr = (nax, -1)
                    ClosestDistance = D
            if ClosestDistance > 10:
                tbr = None

            if not self.BeingRemoved is None and self.BeingRemoved != tbr:
                nax = self.BeingRemoved[0]
                if self.BeingRemoved[1] == -1:
                    dot = self.IniAxs[nax].plot(self.CurrentPoints[nax].get_xdata(), self.CurrentPoints[nax].get_ydata(), marker = 'x')
                    self.CurrentPoints[nax].remove()
                    self.CurrentPoints[nax] = dot
                else:
                    nP = self.BeingRemoved[1]
                    self.PlottedPointsLeft[nax][nP].set_color(self.PlottedPointsLeft[1-nax][nP].get_color())
                self.BeingRemoved = None
            if tbr is None:
                return
            if self.BeingRemoved is None:
                if tbr[1] == -1:
                    self.CurrentPoints[nax].set_color('k')
                    return
                self.PlottedPointsLeft[tbr[0]][tbr[1]].set_color('k')
                self.BeingRemoved = tbr
                return
            else:
                if tbr[1] == -1:
                    self.CurrentPoints[tbr[0]].remove()
                else:
                    for nax in range(2):
                        nP = tbr[1]
                        self.PlottedPointsLeft[nax][nP].remove()
                        self.PlottedPointsLeft[nax].pop(nP)
                        self.PlottedPointsMid[nax][nP].remove()
                        self.PlottedPointsMid[nax].pop(nP)
                        self.PlottedEpipolarLines[nax][nP].remove()
                        self.PlottedEpipolarLines[nax].pop(nP)
                        self.PlottedRectifiedPoints[nax][nP].remove()
                        self.PlottedRectifiedPoints[nax].pop(nP)
                        self.PlottedRectifiedEpipolarLines[nax][nP].remove()
                        self.PlottedRectifiedEpipolarLines[nax].pop(nP)
                        self.StoredPoints[nax].pop(nP)
                self.BeingRemoved = None
                return


        print("Axes {0}, x = {1}, y = {2}".format(nax, event.xdata, event.ydata))
        if not self.CurrentPoints[nax] is None:
            self.CurrentPoints[nax].set_xdata([event.xdata])
            self.CurrentPoints[nax].set_ydata([event.ydata])
            return
        self.CurrentPoints[nax] = self.IniAxs[nax].plot(event.xdata, event.ydata, marker='x')[0]
        
        if not None in self.CurrentPoints:
            for nax, dot in enumerate(self.CurrentPoints):
                x, y = dot.get_xdata()[0], dot.get_ydata()[0]
                self.StoredPoints[nax] += [np.array([x, y, 1])]
                dot.set_marker('o')

                self.PlottedPointsLeft[nax] += [dot]
                self.PlottedPointsMid[nax] += [self.axs[nax,1].plot(x, y, marker='o', color = dot.get_color())[0]]
                self.PlottedEpipolarLines[nax] += [self.axs[1-nax,1].plot([0,self.Module.UsedGeometry[0]], [-10,-10], color = dot.get_color())[0]]
                self.PlottedRectifiedPoints[nax] += [self.axs[nax,2].plot(x, y, marker='o', color = dot.get_color())[0]]
                self.PlottedRectifiedEpipolarLines[nax] += [self.axs[1-nax,2].plot([0, self.Module.UsedGeometry[0]], [y, y], ls='--', color = dot.get_color())[0]]
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
            for nrow in range(2):
                self.axs[nrow, 0].set_xlim(0, self.Module.UsedGeometry[0])
                self.axs[nrow, 0].set_ylim(0, self.Module.UsedGeometry[1])
                self.Zoomed[nrow] = False
        else:
            self.axs[nax, 0].set_xlim(Center[0] - Radius, Center[0] + Radius)
            self.axs[nax, 0].set_ylim(Center[1] - Radius, Center[1] + Radius)
            self.Zoomed[nax] = True

    def OnResize(self, event):
        self.f.tight_layout()

    def Rectify(self):
        LXs, RXs = np.array(self.StoredPoints[0])[:,:2], np.array(self.StoredPoints[1])[:,:2]
        self.FundamentalMatrix = cv2.findFundamentalMat(LXs, RXs)[0]
        self.UpdateEpipolarLines()
        self.RectificationMatrices = list(cv2.stereoRectifyUncalibrated(LXs, RXs, self.FundamentalMatrix, tuple(self.Module.UsedGeometry))[1:])
        self.UpdateRectificationView()

    def UpdateRectificationView(self):
        for nrow in range(2):
            self.RectificationMatrices[nrow] /= abs(self.RectificationMatrices[nrow]).max()
            self.RectificationImages[nrow].set_data(self.RectificationMatrices[nrow])
        self.UpdateRectifiedImages()

    def UpdateEpipolarLines(self):
        for nPoint in range(len(self.StoredPoints[0])):
            X1 = self.StoredPoints[0][nPoint]
            X2 = self.StoredPoints[1][nPoint]
            Line1 = self.PlottedEpipolarLines[0][nPoint]
            Line2 = self.PlottedEpipolarLines[1][nPoint]
            lTh1 = self.FundamentalMatrix.dot(X1)
            lTh2 = self.FundamentalMatrix.T.dot(X2)
            Line1.set_ydata([-lTh1[2]/lTh1[1], -(lTh1[2] + self.Module.UsedGeometry[0] * lTh1[0]) / lTh1[1]])
            Line2.set_ydata([-lTh2[2]/lTh2[1], -(lTh2[2] + self.Module.UsedGeometry[0] * lTh2[0]) / lTh2[1]])
        
    def UpdateRectifiedImages(self):
        self.RectifiedPoints = [[],[]]
        self.Disparities = [[],[]]
        for nax in range(2):
            Xs = self.RectificationMatrices[nax].dot(self.PlottedPointsImage[nax][:3,:])
            Ys = (Xs[:2] / Xs[-1]).astype(int)
            ValidIndexes = np.where((Ys[0,:] >= 0) * (Ys[1,:] >= 0) * (Ys[0,:] < self.Module.UsedGeometry[0]) * (Ys[1,:] < self.Module.UsedGeometry[1]))
            Map = np.zeros(self.Module.UsedGeometry)
            Map[Ys[0,ValidIndexes], Ys[1,ValidIndexes]] = self.PlottedPointsImage[nax][3,ValidIndexes]
            self.RectImages[nax].set_data(np.transpose(Map))
        for nPoint in range(len(self.StoredPoints[0])):
            for nax in range(2):
                Point = self.StoredPoints[nax][nPoint]
                ProjPoint = self.RectificationMatrices[nax].dot(Point)
                ProjPoint = ProjPoint[:2]/ProjPoint[-1]
                self.RectifiedPoints[nax] += [np.array(ProjPoint)]
            for nax in range(2):
                ProjPoint = self.RectifiedPoints[nax][nPoint]
                PairPoint = self.RectifiedPoints[1-nax][nPoint]
                self.Disparities[nax] += [ProjPoint[0] - PairPoint[0]]
                self.PlottedRectifiedPoints[nax][nPoint].set_xdata([ProjPoint[0]])
                self.PlottedRectifiedPoints[nax][nPoint].set_ydata([ProjPoint[1]])
                self.PlottedRectifiedEpipolarLines[nax][nPoint].set_ydata([ProjPoint[1], ProjPoint[1]])
        self.RectifiedFundamentalMatrix = cv2.findFundamentalMat(np.array(self.RectifiedPoints[0]), np.array(self.RectifiedPoints[1]))[0]
        for nax in range(2):
            DispSign = np.sign(np.mean(self.Disparities[nax]))
            if nax == 0:
                F = np.array(self.RectifiedFundamentalMatrix)
            else:
                F = np.array(self.RectifiedFundamentalMatrix.T)
            MaxDisp = abs(np.array(self.Disparities)).max()
            for nPoint in range(len(self.StoredPoints[0])):
                color = np.array([max(0, min(1, self.Disparities[nax][nPoint] / MaxDisp)), max(0, min(1, -self.Disparities[nax][nPoint] / MaxDisp)), 0])
                if np.sign(self.Disparities[nax][nPoint]) == DispSign:
                    marker = 'o'
                else:
                    marker = 'x'
                self.PlottedRectifiedPoints[nax][nPoint].set_color(color)
                self.PlottedRectifiedPoints[nax][nPoint].set_marker(marker)
                self.PlottedRectifiedEpipolarLines[nax][nPoint].set_color(color)
                line = F.dot(np.array([self.RectifiedPoints[nax][nPoint][0], self.RectifiedPoints[nax][nPoint][1], 1]))
                self.PlottedRectifiedEpipolarLines[nax][nPoint].set_ydata([-line[2] / line[1], -(line[2] + self.Module.UsedGeometry[0] * line[0]) / line[1]])
        plt.show()

class FakeEvent:
    def __init__(self, ax, x, y):
        self.xdata = x
        self.ydata = y
        self.button = 1
        self.inaxes = ax
