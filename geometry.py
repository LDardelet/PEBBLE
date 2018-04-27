import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import rgb2hex

import numpy as np

import random
import pylab
import sys

def ComputeNewItems(InitialList, NewElement, Area = 0):
    ElementsAppearing = [NewElement]
    NewParts = []
    IntersectionParts = []
    while len(ElementsAppearing) > 0:
        CurrentElement = ElementsAppearing[0]
        FoundIntersection = False
        for ComparedElement in InitialList:
            if (ComparedElement[0] <= CurrentElement[0] < ComparedElement[2] or CurrentElement[0] <= ComparedElement[0] < CurrentElement[2]) and (ComparedElement[1] <= CurrentElement[1] < ComparedElement[3] or CurrentElement[1] <= ComparedElement[1] < CurrentElement[3]):
                SubParts = []
                FoundIntersection = True

                ymin = CurrentElement[1]

                if ymin < ComparedElement[1]:
                    SubParts += [(CurrentElement[0], ymin, CurrentElement[2], ComparedElement[1])]
                    ymin = ComparedElement[1]
                
                if CurrentElement[0] < ComparedElement[0]: # In case the new element has margin on the left
                    SubParts += [(CurrentElement[0], ymin, ComparedElement[0], CurrentElement[3])]
                    top_part_x_min = ComparedElement[0]
                else:
                    top_part_x_min = CurrentElement[0]
                if CurrentElement[2] > ComparedElement[2]: # In case the new element has margin on the right
                    SubParts += [(ComparedElement[2], ymin, CurrentElement[2], CurrentElement[3])]
                    top_part_x_max = ComparedElement[2]
                else:
                    top_part_x_max = CurrentElement[2]
                IntersectionParts += [(top_part_x_min, ymin, top_part_x_max, min(CurrentElement[3], ComparedElement[3]))]
                if CurrentElement[3] > ComparedElement[3]:
                    SubParts += [(top_part_x_min, ComparedElement[3], top_part_x_max, CurrentElement[3])]
                
                ElementsAppearing += SubParts
                break
        if not FoundIntersection:
            InitialList += [tuple(CurrentElement)]
            NewParts += [tuple(CurrentElement)]
            Area += (CurrentElement[2] - CurrentElement[0]) * (CurrentElement[3] - CurrentElement[1])
        ElementsAppearing = ElementsAppearing[1:]

    return NewParts, IntersectionParts, Area


def DrawSubdivision(List, fig = None, ax = None):
    if fig == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

    r = lambda: random.randint(0,255)
    
    for part in List:
        ax.add_patch(patches.Rectangle((part[0], part[1]), part[2]-part[0], part[3]-part[1], color = '#%02X%02X%02X' % (r(),r(),r())))
    ax.autoscale(True)
    fig.show()
    return fig, ax

def CreateElementFromPosition(x, y, R = 0.5):
    return (x - R, y - R, x + R, y + R)

def ComputeArea(PartsList):
    Area = 0
    for Part in PartsList:
        Area += (Part[2]-Part[0])*(Part[3]-Part[1])
    return Area

def CompareEvolutionOf(PartsList, AreasHistory, Ts, Speeds, n1, n2, ts_points = None, default_n_points = 4, defaultMaxAreaRatio = 2.):
    Figure = plt.figure()

    if ts_points == None:
        n_points = default_n_points
    else:
        n_points = len(ts_points)
    axs = [[],[]]
    argmax = int(AreasHistory[-1][n2] > AreasHistory[-1][n1])
    if argmax:
        ns = [n2, n1]
    else:
        ns = [n1, n2]
    for i in range(n_points):
        axs[0] += [plt.subplot2grid((3, n_points), (0, i))]
        axs[0][-1].tick_params('x', bottom = 'off', labelbottom = 'off')
        axs[0][-1].tick_params('y', left = 'off', labelleft = 'off')
        axs[1] += [plt.subplot2grid((3, n_points), (2, i))]
        axs[1][-1].tick_params('x', bottom = 'off', labelbottom = 'off')
        axs[1][-1].tick_params('y', left = 'off', labelleft = 'off')
    AreasAx = plt.subplot2grid((3, n_points), (1, 0), colspan = n_points)

    MaxArea = min(min(AreasHistory[-1])*defaultMaxAreaRatio, AreasHistory[-1][ns[0]])
    AreasHistory = np.array(AreasHistory)

    AreasLimits = [[],[]]
    AreasLimits[0] = [(i+1)*float(MaxArea)/n_points for i in range(n_points)]
    indexes = [abs(AreasHistory[:,ns[0]] - AreaLimit).argmin() for AreaLimit in AreasLimits[0]]
    AreasLimits[1] = [AreasHistory[index, ns[1]] for index in indexes]

    last_displayed_index = min(indexes[-1] + (indexes[-1] - indexes[-2])/2, AreasHistory.shape[0]-1)
    AreasAx.plot(Ts[1:1+last_displayed_index], AreasHistory[:last_displayed_index,ns[0]], label = "vx = {0}, vy = {1}".format(Speeds[ns[0]][0], Speeds[ns[0]][1]))
    AreasAx.plot(Ts[1:1+last_displayed_index], AreasHistory[:last_displayed_index,ns[1]], label = "vx = {0}, vy = {1}".format(Speeds[ns[1]][0], Speeds[ns[1]][1]))
    AreasAx.legend(loc = 'upper right')

    UpperAreaLimit = AreasHistory[-1][ns[0]]

    cmap1 = pylab.cm.get_cmap('Blues', n_points + 3)
    cmap2 = pylab.cm.get_cmap('Oranges', n_points + 3)
    A1 = 0
    A2 = 0
    n1 = 0
    n2 = 0

    for n_step in range(n_points):
        print "Treating step {0}/{1} at t = {2}".format(n_step+1, n_points, Ts[indexes[n_step]])
        AreasAx.plot([Ts[indexes[n_step]], Ts[indexes[n_step]]], [0, UpperAreaLimit], '--k')
        while A1 < AreasLimits[0][n_step]:
            part = PartsList[ns[0]][n1]
            for i in range(n_step, n_points):
                axs[0][i].add_patch(patches.Rectangle((part[0], part[1]), part[2]-part[0], part[3]-part[1], color = rgb2hex(cmap1(n_step + 2)[:3])))
            n1 += 1
            A1 += (part[2]-part[0])*(part[3]-part[1])

        while A2 < AreasLimits[1][n_step]:
            part = PartsList[ns[1]][n2]
            for i in range(n_step, n_points):
                axs[1][i].add_patch(patches.Rectangle((part[0], part[1]), part[2]-part[0], part[3]-part[1], color = rgb2hex(cmap2(n_step + 2 )[:3])))
            n2 += 1
            A2 += (part[2]-part[0])*(part[3]-part[1])
    for i in range(n_points):
        axs[0][i].autoscale(True)
        axs[1][i].autoscale(True)
    Figure.show()

    return Figure, AreasAx, axs

def GetMinMaxAreaValues(PartsList):
    xMin = np.inf
    yMin = np.inf
    xMax = -np.inf
    yMax = -np.inf

    for Part in PartsList:
        xMin = min(xMin, Part[0])
        yMin = min(yMin, Part[1])
        xMax = max(xMax, Part[2])
        yMax = max(yMax, Part[3])
    return xMin, yMin, xMax, yMax

def CreateDensityProjection(PartsList, IntersectingPartsList, resolution = 10): # resolution in dpp
    FinalList = PartsList + IntersectingPartsList
    nItems = len(FinalList)
    Limits = GetMinMaxAreaValues(PartsList)

    xLenght = Limits[2] - Limits[0]
    yLenght = Limits[3] - Limits[1]

    nX = int(xLenght*resolution) + 1
    nY = int(yLenght*resolution) + 1

    Density = np.zeros((nX, nY))
    print "Meshgrid created"
    nItem = 0
    for Part in FinalList:
        if nItem % 64 == 0:
            sys.stdout.write("Filling .. {0}%\r".format(int(100.*nItem/nItems)))
            sys.stdout.flush()
        for x in range(int(resolution*(Part[0] - Limits[0])), int(resolution*(Part[2] - Limits[0])) +1):
            for y in range(int(resolution*(Part[1] - Limits[1])), int(resolution*(Part[3] - Limits[1])) +1):
                Density[x,y] += 1
        nItem += 1
    print "Filling complete."
    return Density
