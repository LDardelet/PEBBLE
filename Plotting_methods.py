import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
import json

import os

from sys import stdout
from matplotlib import collections  as mc

import datetime

def PlotConvergence(S, SpeedIDs = None, GroundTruthFile = None, AddSpeedIdLabel = True, AddFeatureIdLabel = True):
    Vxs = [];Vys = [];Tss = []
    for speed_id in SpeedIDs:
        Tss += [[]]
        Vxs += [[]]
        Vys += [[]]
        for Change in S.SpeedsChangesHistory[speed_id]:
            Tss[-1] += [Change[0]]
            Vxs[-1] += [Change[1][0]]
            Vys[-1] += [Change[1][1]]
    f, axs = plt.subplots(1,2)
    axx = axs[0]
    axy = axs[1]
    axx.set_title('Vx Benchmark')
    axy.set_title('Vy Benchmark')
    maxTs = 0
    minTs = np.inf
    for i in range(len(SpeedIDs)):
        maxTs = max(maxTs, max(Tss[i]))
        minTs = min(minTs, min(Tss[i]))
        if AddSpeedIdLabel or AddFeatureIdLabel:
            label = ''
            if AddFeatureIdLabel:
                for Zone in S.Zones.keys():
                    if SpeedIDs[i] in S.Zones[Zone]:
                        Feature_id = Zone[4]
                label += 'Feature {0}'.format(Feature_id + 1)
                if AddSpeedIdLabel:
                    label += ', '
            if AddSpeedIdLabel:
                label += 'ID = {0}'.format(SpeedIDs[i])
            axx.plot(Tss[i], Vxs[i], label = label)
            axy.plot(Tss[i], Vys[i], label = label)
        else:
            axx.plot(Tss[i], Vxs[i])
            axy.plot(Tss[i], Vys[i])

    D = None
    if not GroundTruthFile is None:
        with open(GroundTruthFile, 'rb') as openfile:
            D = json.load(openfile)
    else:
        StreamName = S._Framework.StreamHistory[-1]
        NameParts = StreamName.split('/')
        NewName = NameParts[-1].replace('.', '_') + '.gnd'
        LoadName = '/'.join(NameParts[:-1]) + '/' + NewName
        try:
            openfile = open(LoadName, 'r')
            D = json.load(openfile)
            openfile.close()
            print "Using default .gnd file found"
        except IOError:
            print "Unable to find default .gnd file"
    if not D is None:
        PointsList = D['RecordedPoints']
        for PointID in range(int(np.array(PointsList)[:,0].max() + 1)):
            PointList = []
            for Point in PointsList:
                if Point[0] == PointID:
                    PointList += [Point[1:]]
            TsTh = np.array(PointList)[:,0]
            Xs = np.array(PointList)[:,1]
            Ys = np.array(PointList)[:,2]
            VxTh = ((Xs - Xs.mean())**2).sum()/((TsTh - TsTh.mean()) * (Xs - Xs.mean())).sum()
            VyTh = ((Ys - Ys.mean())**2).sum()/((TsTh - TsTh.mean()) * (Ys - Ys.mean())).sum()
            axx.plot([minTs, maxTs], [VxTh, VxTh], '--', label = 'Ground truth, Feature {0}'.format(PointID + 1))
            axy.plot([minTs, maxTs], [VyTh, VyTh], '--', label = 'Ground truth, Feature {0}'.format(PointID + 1))
    axx.legend()
    axx.set_xlabel('t(s)')
    axx.set_ylabel('vx(px/s)')
    axy.set_xlabel('t(s)')
    axy.set_ylabel('vy(px/s)')
    return f, axs

def Plot3dSurface(M, ax = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    X = np.arange(M.shape[0])
    Y = np.arange(M.shape[1])

    X, Y = np.meshgrid(X, Y)

    surf = ax.plot_surface(X, Y, np.transpose(M), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()
    return surf

def PlotCurrentStreaksMaps(SpeedProjector, ZoneNumber = 0, MaxRatio = 0.3):
    S = SpeedProjector
    
    IniSpeeds = np.array(S.InitialSpeeds)
    SeedsX = np.unique(IniSpeeds[:,0]).tolist()
    SeedsY = np.unique(IniSpeeds[:,1]).tolist()

    f, axs = plt.subplots(len(SeedsX), len(SeedsX))
    OW = S.OWAPT[S.Zones.values()[ZoneNumber][0]]
    f.suptitle("Zone number {0}, at t = {1}\nx : {2} -> {3}\ny : {4} -> {5}".format(ZoneNumber, S._Framework.Tools[S._CreationReferences['Memory']].LastEvent.timestamp, OW[0], OW[2], OW[1], OW[3]))
    for speed_id in S.Zones.values()[ZoneNumber]:
        xIndex = SeedsX.index(S.InitialSpeeds[speed_id][0])
        yIndex = SeedsY.index(S.InitialSpeeds[speed_id][1])
        axs[xIndex, yIndex].imshow(np.transpose(S.StreaksMaps[speed_id] >= MaxRatio*S.StreaksMaps[speed_id].max()), origin = 'lower')
        if speed_id in S.ActiveSpeeds:
            color = 'g'
        else:
            color = 'r'
        axs[xIndex, yIndex].set_title("vx = {0:.1f}, vy = {1:.1f}, id = {2}".format(S.Speeds[speed_id][0], S.Speeds[speed_id][1], speed_id), color = color)
        axs[xIndex, yIndex].tick_params('both', left = 'off', bottom = 'off', labelleft = 'off', labelbottom = 'off')
    return f, axs

def PlotDecayingMaps(SpeedProjector, ZoneNumber = 0):
    S = SpeedProjector
    
    IniSpeeds = np.array(S.InitialSpeeds)
    SeedsX = np.unique(IniSpeeds[:,0]).tolist()
    SeedsY = np.unique(IniSpeeds[:,1]).tolist()

    f, axs = plt.subplots(len(SeedsX), len(SeedsX))
    OW = S.OWAPT[S.Zones.values()[ZoneNumber][0]]
    f.suptitle("Zone number {0}, at t = {1}\nx : {2} -> {3}\ny : {4} -> {5}".format(ZoneNumber, S._Framework.Tools[S._CreationReferences['Memory']].LastEvent.timestamp, OW[0], OW[2], OW[1], OW[3]))
    for speed_id in S.Zones.values()[ZoneNumber]:
        xIndex = SeedsX.index(S.InitialSpeeds[speed_id][0])
        yIndex = SeedsY.index(S.InitialSpeeds[speed_id][1])
        axs[xIndex, yIndex].imshow(np.transpose(S.DecayingMaps[speed_id]), origin = 'lower')
        if speed_id in S.ActiveSpeeds:
            color = 'g'
        else:
            color = 'r'
        axs[xIndex, yIndex].set_title("vx = {0:.1f}, vy = {1:.1f}, id = {2}".format(S.Speeds[speed_id][0], S.Speeds[speed_id][1], speed_id), color = color)
    for ax_row in axs:
        for ax in ax_row:
            ax.tick_params('both', left = 'off', bottom = 'off', labelleft = 'off', labelbottom = 'off')
    return f, axs

def PlotTracking(S, Speed_id, SnapsIDs, orientation = 'vertical'):
    nSnaps = len(SnapsIDs)
    if orientation == 'vertical':
        f, axs = plt.subplots(nSnaps, 1)
    else:
        f, axs = plt.subplots(1, nSnaps)

    Ref = S.MeanPositionsReferences[Speed_id]*S.DensityDefinition
    for n_ax in range(nSnaps):
        axs[n_ax].plot(Ref[0], Ref[1], 'vr', markersize = 10)

        Snap = S.DMSnaps[SnapsIDs[n_ax]][Speed_id]
        axs[n_ax].imshow(np.transpose(Snap), origin = 'lower')

        Xs, Ys = np.where(Snap > 0)
        Weights = Snap[Xs, Ys]
        Xm, Ym = (Weights*Xs).sum() / Weights.sum(), (Weights*Ys).sum() / Weights.sum()
        axs[n_ax].plot(Xm, Ym, 'v', color = 'grey', markersize = 10)

        tSnap = S.TsSnaps[SnapsIDs[n_ax]]
        axs[n_ax].set_title("t = {0:.2}s".format(tSnap))

def CreateTrackingShot(F, S, SpeedIDs, SnapshotNumber = 0, BinDt = 0.005, fontsize = 15):
    f, ax = plt.subplots(1,1)
    ax.imshow(np.transpose(F.Mem.Snapshots[SnapshotNumber][1].max(axis = 2) > F.Mem.Snapshots[SnapshotNumber][1].max()-BinDt), origin = 'lower') 
    for speed_id in SpeedIDs:
        for Zone in S.Zones.keys():
            if speed_id in S.Zones[Zone]:
                Feature_id = Zone[4]
        Box = [S.OWAPT[speed_id][0] + S.DisplacementSnaps[SnapshotNumber][speed_id][0], S.OWAPT[speed_id][1] + S.DisplacementSnaps[SnapshotNumber][speed_id][1], S.OWAPT[speed_id][2] + S.DisplacementSnaps[SnapshotNumber][speed_id][0], S.OWAPT[speed_id][3] + S.DisplacementSnaps[SnapshotNumber][speed_id][1]]
        
        ax.plot([Box[0], Box[0]], [Box[1], Box[3]], 'r')
        ax.plot([Box[0], Box[2]], [Box[3], Box[3]], 'r')
        ax.plot([Box[2], Box[2]], [Box[3], Box[1]], 'r')
        ax.plot([Box[2], Box[0]], [Box[1], Box[1]], 'r')
        
        ax.text(Box[2] + 5, Box[1] + (Box[3] - Box[1])*0.4, 'Feature {0}'.format(Feature_id + 1), color = 'r', fontsize = fontsize)
    ax.tick_params(bottom = 'off', left = 'off', labelbottom = 'off', labelleft = 'off')
    return f, ax

def GenerateTrackingGif(F, S, SpeedIDs, Folder = '/home/dardelet/Pictures/AutoGeneratedTrackingGIFs/', BinDt = 0.005):
    os.system('rm '+ Folder + '*.png')
    print "Generating {0} png frames.".format(len(S.TsSnaps))
    for snap_id in range(len(S.TsSnaps)):
        f, ax = plt.subplots(1,1)
        ax.imshow(np.transpose(F.Mem.Snapshots[snap_id][1].max(axis = 2) > F.Mem.Snapshots[snap_id][1].max()-BinDt), origin = 'lower') 
        for speed_id in SpeedIDs:
            Box = [S.OWAPT[speed_id][0] + S.DisplacementSnaps[snap_id][speed_id][0], S.OWAPT[speed_id][1] + S.DisplacementSnaps[snap_id][speed_id][1], S.OWAPT[speed_id][2] + S.DisplacementSnaps[snap_id][speed_id][0], S.OWAPT[speed_id][3] + S.DisplacementSnaps[snap_id][speed_id][1]]
            
            ax.plot([Box[0], Box[0]], [Box[1], Box[3]], 'r')
            ax.plot([Box[0], Box[2]], [Box[3], Box[3]], 'r')
            ax.plot([Box[2], Box[2]], [Box[3], Box[1]], 'r')
            ax.plot([Box[2], Box[0]], [Box[1], Box[1]], 'r')
        f.savefig(Folder + 't_{0:03d}.png'.format(snap_id))
        plt.close(f.number)
    GifFile = 'tracking_'+datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')+'.gif'
    print "Generating gif."
    os.system('convert -delay {0} -loop 0 '.format(int(1000*S.SnapshotDt))+Folder+'*.png ' + Folder + GifFile)
    os.system('kde-open ' + Folder + GifFile)

    ans = raw_input('Rate this result (1-nice/0-bad/(d)elete) : ')
    if '1' in ans.lower() or 'nice' in ans.lower():
        os.system('mv ' + Folder + GifFile + ' ' + Folder + 'Nice/' + GifFile)
        print "Moving gif file to Nice folder."
    elif '0' in ans.lower() or 'bad' in ans.lower():
        os.system('mv ' + Folder + GifFile + ' ' + Folder + 'Bad/' + GifFile)
        print "Moving gif file to Bad folder."
    elif len(ans) > 0 and ans.lower()[0] == 'd' or 'delete' in ans.lower():
        os.system('rm ' + Folder + GifFile)
        print "Deleted Gif file. RIP."
    else:
        os.system('mv ' + Folder + GifFile + ' ' + Folder + 'Meh/' + GifFile)
        print "Moving gif file to Meh folder."

def CreatePositionsHistory(S, speed_id, Tmax = np.inf):
    PositionsHistoryx = []
    PositionsHistoryy = []
    OW = S.OWAPT[speed_id]
    CenterPos = np.array([OW[2] + OW[0], OW[3] + OW[1]], dtype = float) / 2
    PTList = list(S.ProjectionTimesHistory[speed_id])
    CurrentProjectionTime = PTList.pop(0)
    CurrentBaseDisplacement = np.array([0.,0.])
    for n in range(len(S.SpeedsChangesHistory[speed_id]) - 1):
        t = S.SpeedsChangesHistory[speed_id][n][0]
        if len(PTList) > 0 and t > PTList[0]:
            CurrentProjectionTime = PTList.pop(0)
            CurrentBaseDisplacement = np.array([PositionsHistoryx[-1][1][1] - CenterPos[0], PositionsHistoryy[-1][1][1] - CenterPos[1]])
        PositionsHistoryx += [[(t, (CenterPos + CurrentBaseDisplacement + (t - CurrentProjectionTime)*S.SpeedsChangesHistory[speed_id][n][1])[0]), (S.SpeedsChangesHistory[speed_id][n+1][0], (CenterPos + CurrentBaseDisplacement + (S.SpeedsChangesHistory[speed_id][n+1][0] - CurrentProjectionTime)*S.SpeedsChangesHistory[speed_id][n][1])[0])]]
        PositionsHistoryy += [[(t, (CenterPos + CurrentBaseDisplacement + (t - CurrentProjectionTime)*S.SpeedsChangesHistory[speed_id][n][1])[1]), (S.SpeedsChangesHistory[speed_id][n+1][0], (CenterPos + CurrentBaseDisplacement + (S.SpeedsChangesHistory[speed_id][n+1][0] - CurrentProjectionTime)*S.SpeedsChangesHistory[speed_id][n][1])[1])]]
        if S.SpeedsChangesHistory[speed_id][n+1][0] > Tmax:
            break
    return PositionsHistoryx, PositionsHistoryy

def PlotPositionTracking(S, SpeedIDs, GroundTruthFile = None, AddSpeedIdLabel = True, AddFeatureIdLabel = True, Tmax = np.inf):
    f, axs = plt.subplots(1,2)

    D = None
    if not GroundTruthFile is None:
        with open(GroundTruthFile, 'rb') as openfile:
            D = json.load(openfile)
    else:
        StreamName = S._Framework.StreamHistory[-1]
        NameParts = StreamName.split('/')
        NewName = NameParts[-1].replace('.', '_') + '.gnd'
        LoadName = '/'.join(NameParts[:-1]) + '/' + NewName
        try:
            openfile = open(LoadName, 'r')
            D = json.load(openfile)
            openfile.close()
            print "Using default .gnd file found"
        except IOError:
            print "Unable to find default .gnd file"
    AddToLegend = []
    if not D is None:
        GndColors = []
        PointsList = D['RecordedPoints']
        for PointID in range(int(np.array(PointsList)[:,0].max() + 1)):
            PointList = []
            for Point in PointsList:
                if Point[0] == PointID:
                    PointList += [Point[1:]]
            TsTh = np.array(PointList)[:,0]
            Xs = np.array(PointList)[:,1]
            Ys = np.array(PointList)[:,2]

            if Tmax == 'gnd':
                Tmax = TsTh.max()
            KeptPos = np.where(TsTh <= Tmax)
            TsTh = TsTh[KeptPos]
            Xs = Xs[KeptPos]
            Ys = Ys[KeptPos]
            axs[0].plot(TsTh, Xs, '--')
            GndColors += [axs[0].get_lines()[-1].get_color()]
            axs[1].plot(TsTh, Ys, '--', color = GndColors[-1])

            AddToLegend += ['Ground truth, Feature {0}'.format(PointID + 1)]
            if not AddSpeedIdLabel and not AddFeatureIdLabel:
                axs[0].legend(AddToLegend)
    else:
        GndColors = None

    Labels = []
    for speed_id in SpeedIDs:
        stdout.write("\r > {0}/{1}".format(SpeedIDs.index(speed_id) + 1, len(SpeedIDs)))
        stdout.flush()

        PHx, PHy = CreatePositionsHistory(S, speed_id, Tmax)
        
        Labels += ['']
        for Zone in S.Zones.keys():
            if speed_id in S.Zones[Zone]:
                Feature_id = Zone[4]
        if AddFeatureIdLabel:
            Labels[-1] += 'Feature {0}'.format(Feature_id + 1)
            if AddSpeedIdLabel:
                Labels[-1] += ', '
        if AddSpeedIdLabel:
            Labels[-1] += 'ID = {0}'.format(speed_id)
        
        if not GndColors is None:
            c = GndColors[Feature_id]
        else:
            c = numpy.random.rand(3,1)
        lcx = mc.LineCollection(PHx, colors=c)
        lcy = mc.LineCollection(PHy, colors=c)
        axs[0].add_collection(lcx)
        axs[1].add_collection(lcy)
    
    print ""

    if AddSpeedIdLabel or AddFeatureIdLabel:
        axs[0].legend(AddToLegend + Labels)

    axs[0].set_title('X tracked position')
    axs[1].set_title('Y tracked position')
    axs[0].set_xlabel('t(s)')
    axs[0].set_ylabel('x(px)')
    axs[1].set_xlabel('t(s)')
    axs[1].set_ylabel('y(px)')
    axs[0].autoscale()
    axs[0].margins(0.1)
    axs[1].autoscale()
    axs[1].margins(0.1)

    return f, axs
