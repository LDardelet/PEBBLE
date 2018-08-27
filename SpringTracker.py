import numpy as np
from event import Event

from roster import roster

class Tracker:
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Tool to track features by assigning a deformable spring string minimizing its energy
        '''
        self.__ReferencesAsked__ = ['Memory']
        self.__Name__ = Name
        self.__Framework__ = Framework
        self.__Type__ = 'Computation'
        self.__CreationReferences__ = dict(argsCreationReferences)

        self._StringDefaultNPoints = 10
        self._MaxDistanceHandled = 3

        self._MinBuildingCos = -0.5 # Avoids having points not following each other in the proper order

    def _Initialize(self):
        self.Strings = []

    def _OnEvent(self, event):
        self.LastEvent = Event(original = event)
        position = tuple(self.LastEvent.location.tolist() + [self.LastEvent.polarity])

        self.STContext[position] = self.LastEvent.timestamp

        return event

    def GenerateString(self, STContext, PositionCenter, ObservationRadius):
        STContext = self.__Framework__.Tools[self.__CreationReferences__['Memory']].CreateSnapshot().max(axis = 2)
        
        LocalMap = STContext[PositionCenter[0] - ObservationRadius: PositionCenter[0] + ObservationRadius + 1, PositionCenter[1] - ObservationRadius: PositionCenter[1] + ObservationRadius + 1]
        StoredPoints = roster([])

        Xs, Ys = np.where(LocalMap > 0)
        Values = LocalMap[Xs, Ys]
        SortedArgsValues = np.argsort(Values)

        for i in range(SortedValues.shape[0]):
            nPoint = -(i+1)
            item = SortedArgsValues[nPoint]
            X, Y = Xs[item], Ys[item]

            P = _Point(np.array([X, Y]))
            
            PossiblePointsNotAvailable = 0
            for ComparedPoint in StoredPoints:
                if np.linalg.norm(P.Position - ComparedPoint.Position) <= self._MaxDistanceHandled:
                    if ComparedPoint.nLinks == 0:
                        ComparedPoint.AddLink(P)
                        P.AddLink(ComparedPoint)
                    elif ComparedPoint.nLinks == 1:
                        V1 = ComparedPoint.Position - ComparedPoint.LinkedPoints[0].Position
                        V1 = V1 / np.linalg.norm(V1)
                        V2 = P.Position - ComparedPoint.Position
                        V2 = V2 / np.linalg.norm(V2)
                        if (V1*V2).sum() >= self._MinBuildingCos:
                            ComparedPoint.AddLink(P)
                            P.AddLink(ComparedPoint)
                    else:
                        PossiblePointsNotAvailable += 1
                    break
             if not PossiblePointsNotAvailable == 2:
                StoredPoints.append(P)
            if len(StoredPoints) == self._StringDefaultNPoints:
                break
        self.Strings.append(_String(StoredPoints, PositionCenter, Radius)

class _String:
    def __init__(self, Points, Center, Radius):
        self.Points = Points
        self.Energy = 0
        self.Center = Center
        self.Radius = Radius

        self._SpringLenghtConstant = 1
        self._SpringTorqueConstant = 1

class _Point:
    def __init__(self, Position, LinkedPoints = [None, None]):
        self.Position = Position
        self.LinkedPoints = LinkedPoints
        self.nLinks = 0
        self.AngleRef = None

    def AddLink(self, Point):
        self.LinkedPoints[self.nLinks] = Point
        self.nLinks += 1

        if nLinks == 2:
            self.ComputeAngle()

    def ComputeAngle(self):
        V0 = self.LinkedPoints[0].Position - self.Position
        V1 = self.LinkedPoints[1].Position - self.Position

        V0 = V0 / np.linalg.norm(V0)
        V1 = V1 / np.linalg.norm(V1)

        Angle = np.sign(V0[0] * V1[1] - V1[0]*V0[1]) * np.arccos((V0*V1).sum())

        if self.AngleRef is None:
            self.AngleRef = Angle
        return Angle
