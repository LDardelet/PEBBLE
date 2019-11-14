import numpy as np

from BaseObjects import TypedList

import matplotlib.pyplot as plt

from Framework import Module, Event

class Tracker(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Tool to track features by assigning a deformable spring string minimizing its energy
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences):
        self.__ReferencesAsked__ = ['Memory']
        self.__Type__ = 'Computation'

        self.__Started__ = False

        self._StringDefaultNPoints = 3
        self._StringMinNPoints = 3

        self._MaxDistanceHandled = 20
        self._RunningMeanLenght = 15
        self._MinDistancePerPoint = 5

        self._SegmentAngleRef = np.pi/2 / (self._StringDefaultNPoints - 2)
        self._CornerAngleRef = np.pi/2 / (self._StringDefaultNPoints - 2)

        self._MaxBuildingTime = 0.01
        self._MaxBuildingFailures = 2

        self._MinBuildingCos = -0.7 # Avoids having points not following each other in the proper order

        #self.Centers = np.array([[353, 138], [255, 246], [371, 344], [463, 227]])
        self.Centers = np.array([[241, 163]])

    def _InitializeModule(self, **kwargs):
        self.DefaultRadius = int(self._StringDefaultNPoints/2 * (self._MaxDistanceHandled + self._MinDistancePerPoint)/2)
        self.BuiltCenters = [False for nCenter in range(self.Centers.shape[0])]
        self.BuildingStrings = TypedList(_String)
        self.InitializedStrings = TypedList(_String)
        self.nStrings = 0

        return True

    def _OnEventModule(self, event):
        # First if no string is initialized and the event is not located in one of the predefined centers, we leave the module
        if not self.InitializedStrings and (abs(event.location - self.Centers).max(axis = 1) > self.DefaultRadius).all():
            return event

        # If we haven't left the module, we consider one of the initialized strings to be updated with this event
        for String in self.InitializedStrings:
            if (abs(String.MeanLocation - event.location) <= self.DefaultRadius * 2.).all():
                String.UpdateWith(event)
                return event

        # If we still haven't left the module, we try to build one of the strings with this event
        if len(self.InitializedStrings) < self.Centers.shape[0]:
            for String in self.BuildingStrings:
                if (abs(String._InitialCenter - event.location) <= self.DefaultRadius).all():
                    String.BuildWith(event)
                    return event

        # If we STILL haven't left the module, well we need to start a string at this place
        if (abs(event.location - self.Centers).max(axis = 1) <= self.DefaultRadius).any():
            CenterIndex = abs(event.location - self.Centers).max(axis = 1).argmin()
            if not self.BuiltCenters[CenterIndex]:
                self.BuiltCenters[CenterIndex] = True
                self.BuildingStrings.append(_String(self, self.nStrings, event, self.Centers[CenterIndex,:]))
        return event

class _String:
    def __init__(self, Tracker = None, nString = None, StartEvent = None, Center = None, fake = False):
        self.Points = TypedList(_Point)
        self.nPoints = 0
        self.LastUpdate = 0
        self.MeanLocation = np.array([0., 0.])
        self.LenghtInnerEnergy = 0
        self.TorqueInnerEnergy = 0
        self.OuterEnergy = 0
        self.nUpdates = 0
        self.nUpdatesTrials = 0

        if fake:
            return None

        self.LastUpdate = StartEvent.timestamp
        print "Initiating string around {0}".format(Center)
        self._Tracker = Tracker
        self._nString = nString
        self._InitialCenter = Center
        self.StartTs = StartEvent.timestamp
        
        self._SpringLenghtConstant = 1./self._Tracker._RunningMeanLenght**2
        self._SpringTorqueConstant = 1./self._Tracker._SegmentAngleRef**2
        self._PositionConstant = 0.5
        self._DefaultTimeConstant = 0.002
        self._MaxEnergyRatioIncrease = 1.02
        self.Initialized = False

        self.AddPoint(_Point(StartEvent, self.nPoints))

    def AddPoint(self, Point):
        self.Points.append(Point)
        self.nPoints += 1
        Point.String = self

    def RemovePoint(self, Point):
        self.Points.remove(Point)
        for Neighbour in Point.LinkedPoints:
            if not Neighbour is None:
                Neighbour.LinkedPoints.remove(Point)
                Neighbour.LinkedPoints += [None]
                Neighbour.nLinks -= 1
        self.nPoints -= 1

    def BuildWith(self, event):
        if event.timestamp > self.StartTs + self._Tracker._MaxBuildingTime and self.nPoints >= self._Tracker._StringMinNPoints:
            self._EndBuilding(event.timestamp)
            return None
        NewPoint = _Point(event, self.nPoints)

        for Point in self.Points:
            D_Neighbour = np.linalg.norm(NewPoint.Position - Point.Position)
            if self._Tracker._MinDistancePerPoint <= D_Neighbour <= self._Tracker._MaxDistanceHandled:
                if Point.nLinks == 2:
                    return None
                if Point.nLinks == 0:
                    self.AddPoint(NewPoint)
                    self.LastUpdate = event.timestamp
                    Point.AddLink(NewPoint)
                    NewPoint.AddLink(Point)
                    AddedPoint = True
                    return None
                P_Prev = Point
                P_Next = Point.LinkedPoints[0]
                nPoint = 1
                while not P_Next is None:
                    D_Next = np.linalg.norm(NewPoint.Position - P_Next.Position)
                    if D_Next - nPoint * self._Tracker._MinDistancePerPoint  < D_Neighbour:
                        Point.BuildingFails += 1
                        return None
                    if P_Next.LinkedPoints[0] == P_Prev:
                        P_Next, P_Prev = P_Next.LinkedPoints[1], P_Next
                    else:
                        P_Next, P_Prev = P_Next.LinkedPoints[0], P_Next
                    nPoint += 1

                V1 = Point.Position - Point.LinkedPoints[0].Position
                V1 = V1 / np.linalg.norm(V1)
                V2 = NewPoint.Position - Point.Position
                V2 = V2 / np.linalg.norm(V2)
                if (V1*V2).sum() > self._Tracker._MinBuildingCos:
                    self.AddPoint(NewPoint)
                    self.LastUpdate = event.timestamp
                    Point.AddLink(NewPoint)
                    NewPoint.AddLink(Point)
                    AddedPoint = True
                else:
                    Point.BuildingFails += 1
                break

        if Point.nLinks == 1 and Point.BuildingFails == self._Tracker._MaxBuildingFailures:
            self.RemovePoint(Point)

        if self.nPoints == self._Tracker._StringDefaultNPoints:
            self._EndBuilding(event.timestamp)

    def _EndBuilding(self, t):
        self._Tracker.BuildingStrings.remove(self)
        self._Tracker.InitializedStrings.append(self)
        self.Initialized = True
        
        self._SortPoints()

        self._ComputeMeanLocation()

        SumAngle = 0
        for Point in self.Points:
            Point.Angle = Point.ComputeAngle()
            SumAngle += Point.Angle
        MeanAngle = SumAngle / self.nPoints
        for nPoint, Point in enumerate(self.Points):
            if Point.nLinks == 1:
                Point.AngleRef = 0
            else:
                if nPoint == self.nPoints/2:
                    Point.AngleRef = np.sign(MeanAngle) * self._Tracker._CornerAngleRef
                else:
                    Point.AngleRef = np.sign(MeanAngle) * self._Tracker._SegmentAngleRef
            Point.AngleEnergy = self._SpringTorqueConstant * (Point.Angle - Point.AngleRef) ** 2
        self._ComputeInitialInnerEnergy()
        self._ComputeOuterEnergy(t)

        self._Tracker.nStrings += 1
        if self._Tracker.nStrings == self._Tracker.Centers.shape[0]:
            self._Tracker.__Started__ = True

    def UpdateWith(self, event):
        self.nUpdatesTrials += 1
        MinEnergyVariation = np.inf
        MinAssociatedPoint = None
        MinAssociatedLenghtInnerVariation = None
        MinAssociatedTorqueInnerVariation = None
        MinAssociatedOuterVariation = None
        self._ComputeOuterEnergy(event.timestamp)
        for Point in self.Points:
            if True or np.linalg.norm(event.location - Point.Position) <= self._Tracker._MaxDistanceHandled:
                LenghtInnerVariation = 0
                TorqueInnerVariation = 0
                TorqueInnerVariation += self._SpringTorqueConstant * ((Point.ComputeAngle(With = (Point, event.location)) - Point.AngleRef) ** 2) - Point.AngleEnergy
                OuterVariation = (-self._PositionConstant - Point.OuterEnergy)
                for LinkedPoint in Point.LinkedPoints:
                    LenghtInnerVariation += self._SpringLenghtConstant * (((np.linalg.norm(event.location - LinkedPoint.Position) - self._Tracker._RunningMeanLenght) ** 2) - ((np.linalg.norm(Point.Position - LinkedPoint.Position) - self._Tracker._RunningMeanLenght) ** 2))
                    TorqueInnerVariation += self._SpringTorqueConstant * ((LinkedPoint.ComputeAngle(With = (Point, event.location)) - LinkedPoint.AngleRef) ** 2) - LinkedPoint.AngleEnergy
                if LenghtInnerVariation + TorqueInnerVariation + OuterVariation < MinEnergyVariation:
                    MinEnergyVariation = LenghtInnerVariation + TorqueInnerVariation + OuterVariation
                    MinAssociatedPoint = Point
                    MinAssociatedLenghtInnerVariation = LenghtInnerVariation
                    MinAssociatedTorqueInnerVariation = TorqueInnerVariation
                    MinAssociatedOuterVariation = OuterVariation
        #if not MinAssociatedPoint is None and self.Energy + MinEnergyVariation <= self.Energy * self._MaxEnergyRatioIncrease:
        if not MinAssociatedPoint is None and MinEnergyVariation <= 0:
            self._UpdatePoint(MinAssociatedPoint, event)
            self.LenghtInnerEnergy += MinAssociatedLenghtInnerVariation
            self.TorqueInnerEnergy += MinAssociatedTorqueInnerVariation
            self.OuterEnergy += MinAssociatedOuterVariation
            self.LastUpdate = event.timestamp
            self.nUpdates += 1

    def _UpdatePoint(self, Point, event):
        Point.Position = event.location
        Point.LastUpdate = event.timestamp
        Point.Angle = Point.ComputeAngle()
        Point.AngleEnergy = self._SpringTorqueConstant * (Point.Angle - Point.AngleRef) ** 2
        for LinkedPoint in Point.LinkedPoints:
            LinkedPoint.Angle = LinkedPoint.ComputeAngle()
            LinkedPoint.AngleEnergy = self._SpringTorqueConstant * (LinkedPoint.Angle - LinkedPoint.AngleRef) ** 2
        self._ComputeMeanLocation()

    def _SortPoints(self):
        TmpList = TypedList(self.Points.__elems_type__)
        for PrevPoint in self.Points:
            if PrevPoint.nLinks == 1:
                break
        PrevPoint.LinkedPoints.pop(1)
        TmpList.append(PrevPoint)
        NextPoint = PrevPoint.LinkedPoints[0]
        while NextPoint.nLinks == 2:
            if NextPoint.LinkedPoints[1] != PrevPoint:
                NextPoint.LinkedPoints[1], NextPoint.LinkedPoints[0] = NextPoint.LinkedPoints[0], NextPoint.LinkedPoints[1]
            TmpList += [NextPoint]
            NextPointIndex = 1 - NextPoint.LinkedPoints.index(PrevPoint)
            NextPoint, PrevPoint = NextPoint.LinkedPoints[NextPointIndex], NextPoint
        NextPoint.LinkedPoints.pop(1)
        TmpList.append(NextPoint)
        self.Points = TmpList

    def _ComputeMeanLocation(self):
        SumLocation = np.array([0., 0.])
        for P in self.Points:
            SumLocation = SumLocation + P.Position
        self.MeanLocation = SumLocation / self.nPoints

    def _ComputeInitialInnerEnergy(self):
        PrevPoint = self.Points[0]
        self.TorqueInnerEnergy += PrevPoint.AngleEnergy
        for NextPoint in self.Points[1:]:
            self.TorqueInnerEnergy += NextPoint.AngleEnergy
            self.LenghtInnerEnergy += self._SpringLenghtConstant * ((np.linalg.norm(PrevPoint.Position - NextPoint.Position) - self._Tracker._RunningMeanLenght) ** 2)
            PrevPoint = NextPoint

    def _ComputeOuterEnergy(self, t):
        self.OuterEnergy = 0
        for Point in self.Points:
            Point.OuterEnergy = -self._PositionConstant * np.e**((Point.LastUpdate - t)/self._DefaultTimeConstant)
            self.OuterEnergy += Point.OuterEnergy

class _Point:
    def __init__(self, event = None, nPoint = None, fake = False):
        self.Angle = 0
        self.OuterEnergy = 0

        if fake:
            self.LastUpdate = 0
            self.Position = np.array([0., 0.])
            return None

        self.LastUpdate = event.timestamp
        self.Position = event.location
        self.LinkedPoints = [None, None]
        self.nLinks = 0
        self.AngleRef = None
        self.nPoint = nPoint
        self.BuildingFails = 0

    def AddLink(self, Point):
        self.LinkedPoints[self.nLinks] = Point
        self.nLinks += 1

        if self.nLinks == 2:
            self.Angle = self.ComputeAngle()

    def ComputeAngle(self, With = (None, None)):
        if self.nLinks == 1:
            return 0
        CurrPos = self.Position
        PrevPos = self.LinkedPoints[0].Position
        NextPos = self.LinkedPoints[1].Position
        if not With[0] is None:
            if With[0] == self:
                CurrPos = With[1]
            elif With[0] == self.LinkedPoints[0]:
                PrevPos = With[1]
            elif With[0] == self.LinkedPoints[1]:
                NextPos = With[1]
        V0 = PrevPos - CurrPos
        V1 = CurrPos - NextPos
        n0 = np.linalg.norm(V0)
        if n0 == 0:
            return 0
        n1 = np.linalg.norm(V1)
        if n1 == 0:
            return 0
        V0 = V0 / n0

        V1 = V1 / n1

        Angle = np.sign(V0[0] * V1[1] - V1[0]*V0[1]) * np.arccos((V0*V1).sum())
        if np.isnan(Angle):
            if (V0*V1).sum() < -1:
                Angle = np.pi
            elif (V0*V1).sum() > 1:
                Angle = 0
        #Angle = np.arccos((V0*V1).sum())

        return Angle
