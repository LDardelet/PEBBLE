import matplotlib.pyplot as plt
import numpy as np
import datetime

import json

import cv2

from PEBBLE import Module

class HoughCD(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to record points in a sequence
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__ReferencesAsked__ = ['Memory']
        self.__Type__ = 'Analysis'

        self._BinDt = 0.002

        self._ExpositionDt = 0.004
        self._FirstPictureAt = 0.01

    def _InitializeModule(self):

        self.RecordedPoints = []
        self.CurrentT = self._FirstPictureAt - self._BinDt

        self.Mem = self.__Framework__.Tools[self.__CreationReferences__['Memory']]
        self.Tini = 100
        self.Edges = []
        self.Grays = []

        self.UnsavedData = False

        return True

    def _OnEventModule(self, event):
        if event.timestamp > self.CurrentT + self._BinDt:
            self.CurrentT = event.timestamp

            self.gray_image = np.array(np.transpose(self.Mem.STContext.max(axis = 2) > event.timestamp - self._ExpositionDt)*200, dtype = np.uint8)

            Corners, edges = GetCorners(np.array(self.gray_image), self.Tini)
            self.Edges += [edges]
            self.Grays += [np.array(self.gray_image)]
            if Corners is None:
                return

            NewPoints = []

            Feature0Index = np.array(Corners)[:,0].argmax()
            NewPoints += [[0, self.CurrentT] + Corners[Feature0Index]]
            Corners.pop(Feature0Index)

            Feature1Index = np.array(Corners)[:,1].argmax()
            NewPoints += [[1, self.CurrentT] + Corners[Feature1Index]]
            Corners.pop(Feature1Index)

            Feature2Index = np.array(Corners)[:,0].argmin()
            NewPoints += [[2, self.CurrentT] + Corners[Feature2Index]]
            Corners.pop(Feature2Index)

            NewPoints += [[3, self.CurrentT] + Corners[0]]

            self.RecordedPoints += NewPoints

            self.UnsavedData = True

        return

    def SaveRecordedData(self):
        DataDict = {}
        StreamName = self.__Framework__._GetStreamFormattedName(self)
        DataDict['StreamName'] = StreamName
        DataDict['RecordedPoints'] = list(self.RecordedPoints)
        DataDict['RecordingDate'] = datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')

        NameParts = StreamName.split('/')
        NewName = NameParts[-1].replace('.', '_') + '_hough.gnd'
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

def GetLines(gray, Tini = 100):
    lines = []
    types = []
    edges = cv2.Canny(np.array(gray),100,250,3)
    print Tini, type(Tini)
    linesFound = cv2.HoughLines(gray,2,np.pi/180,Tini)
    if linesFound is None:
        print "No lines found"
        return GetLines(gray, Tini / 2)
    linesFound = linesFound[0]
    for line in linesFound:
        if abs(line[1] - 45*np.pi/180) < 20*np.pi/180:
            lineType = 0
        elif abs(line[1] - 135*np.pi/180) < 20*np.pi/180:
            lineType = 1
        else:
            continue
        nThisType = types.count(lineType)
        if nThisType == 2:
            continue
        elif nThisType == 0:
            lines += [line]
            types += [lineType]
            continue
        else:
            indexOtherLine = types.index(lineType)
            otherLine = lines[indexOtherLine]
            if abs(otherLine[1] - line[1]) < 10*np.pi/180 and abs(otherLine[0] - line[0]) > 100:
                lines += [line]
                types += [lineType]
                continue
            else:
                x, y = IntersectLines(line, otherLine)
                if x > gray.shape[1] or x < 0 or y > gray.shape[0] or y < 0:
                    lines += [line]
                    types += [lineType]
                    continue
        if len(lines) == 4:
            break
    print lines, " & ", types
    if len(lines) != 4 and Tini > 1:
        print "Not enough lines found"
        return GetLines(gray, Tini / 2)
    elif len(lines) != 4 and Tini == 1:
        return linesFound, edges

    return np.array(lines), edges

def GetLinesOld(gray, Tini):
    lines = None
    SignIni = None
    while lines is None or len(lines[0]) != 4:
        edges = cv2.Canny(np.array(gray),100,250,3)
        lines = cv2.HoughLines(edges,7,np.pi/60,Tini)

        if lines is None or len(lines[0]) < 4:
            Tini -= 1
            if SignIni is None:
                SignIni = -1
            else:
                if SignIni == +1:
                    print "Unable to find 4 lines exactly (+ -> -)"
                    return lines[0], Tini, edges
        elif len(lines[0]) > 4:
            Tini += 1
            if SignIni is None:
                SignIni = +1
            else:
                if SignIni == -1:
                    print "Unable to find 4 lines exactly (- -> +)"
                    return lines[0], Tini, edges
        
    return lines[0], Tini, edges

def GetCorners(gray,  Tini = 500):
    lines, edges = GetLines(gray, Tini)
    if lines is None or len(lines) != 4:
        print "Lines ill detected"
        return None, edges

    edges = np.array(edges)
    for rho,theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 2000*(-b))
        y1 = int(y0 + 2000*(a))
        x2 = int(x0 - 2000*(-b))
        y2 = int(y0 - 2000*(a))
    
        cv2.line(edges,(x1,y1),(x2,y2),(150,150,150),2)

    Pairs = []
    if (abs(lines[:,1] - lines[0,1]) < 20*np.pi/180).sum() == 2:
        Pairs += [np.where(abs(lines[:,1] - lines[0,1]) < 20*np.pi/180)[0].tolist()]
        Pairs += [list(set(range(4)) - set(Pairs[0]))]
    else:
        print "Wrong lines detected"
        print lines
        return None, edges
    Corners = []
    
    for nLine0 in Pairs[0]:
        for nLine1 in Pairs[1]:
            x, y = IntersectLines(lines[nLine0,:], lines[nLine1,:])
            Corners += [[float(x), float(y)]]
    return Corners, edges

def IntersectLines(l1, l2):
    lambda2 = (l1[0] * np.cos(l1[1]) - l2[0]*np.cos(l2[1]) - l2[0] * np.sin(l2[1] - l1[1]) * np.sin(l1[1])) / (np.cos(l2[1] - l1[1]) * np.sin(l1[1]) - np.sin(l2[1]))
    xf = l2[0] * np.cos(l2[1]) - lambda2 * np.sin(l2[1])
    yf = l2[0] * np.sin(l2[1]) + lambda2 * np.cos(l2[1])
    
    return xf, yf
