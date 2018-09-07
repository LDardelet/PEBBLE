import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
import json

import os

from sys import stdout
from matplotlib import collections  as mc

import datetime

def PlotConvergence(S, SpeedIDs = None, GroundTruthFile = None, AddSpeedIdLabel = True, AddFeatureIdLabel = True, GivenColors = None, PlotUpdatePTMarkers = True):
    if GivenColors is None:
        GivenColors = [None for speed_id in SpeedIDs]

#    SortedSpeedIDs = []
#    for ZoneID in range(len(S.Zones.keys())):
#        for Zone in S.Zones.keys():
#            if Zone[4] == ZoneID:
#                for speed_id in S.Zones[Zone]:
#                    if speed_id in SpeedIDs:
#                        SortedSpeedIDs += [speed_id]
#                break
    ZonesAsked = {}
    SortedSpeedIDs = []
    for ZoneID in range(len(S.Zones.keys())):
        FoundOneIDForZone = False
        for Zone in S.Zones.keys():
            ZonesAsked[Zone[4]] = False
            if Zone[4] == ZoneID:
                for speed_id in S.Zones[Zone]:
                    if speed_id in SpeedIDs:
                        if FoundOneIDForZone:
                            SeveralIDsForOneZone = True
                        SortedSpeedIDs += [speed_id]
                        FoundOneIDForZone = True
                        ZonesAsked[Zone[4]] = True
                break
    print 'ZonesAsked : ', ZonesAsked

    Vxs = [];Vys = [];Tss = []
    for speed_id in SortedSpeedIDs:
        Tss += [[]]
        Vxs += [[]]
        Vys += [[]]
        for Change in S.SpeedsChangesHistory[speed_id]:
            Tss[-1] += [Change[0]]
            Vxs[-1] += [Change[1][0]]
            Vys[-1] += [Change[1][1]]
    f, axs = plt.subplots(2,2)
    axx = axs[0,0]
    axy = axs[0,1]
    axEx = axs[1,0]
    axEy = axs[1,1]
    axx.set_title('Vx Benchmark')
    axy.set_title('Vy Benchmark')
    axEx.set_title('Error in Vx')
    axEy.set_title('Error in Vy')
    maxTs = 0
    minTs = np.inf

    for n_speed, speed_id in enumerate(SortedSpeedIDs):
        maxTs = max(maxTs, max(Tss[n_speed]))
        minTs = min(minTs, min(Tss[n_speed]))
        label = ''
        if AddSpeedIdLabel or AddFeatureIdLabel:
            if AddFeatureIdLabel:
                label += 'Feature {0}'.format(n_speed + 1)
                if AddSpeedIdLabel:
                    label += ', '
            if AddSpeedIdLabel:
                label += 'ID = {0}'.format(speed_id)
        axx.plot(Tss[n_speed], Vxs[n_speed], label = label, color = GivenColors[n_speed])
        if GivenColors[n_speed] is None:
            GivenColors[n_speed] = axx.get_lines()[-1].get_color()
        axy.plot(Tss[n_speed], Vys[n_speed], label = label, color = GivenColors[n_speed])
        
        axx.plot(S.TsSnaps, np.array(S.AimedSpeedSnaps)[:,speed_id,0], color = GivenColors[n_speed], marker = '.')
        axy.plot(S.TsSnaps, np.array(S.AimedSpeedSnaps)[:,speed_id,1], color = GivenColors[n_speed], marker = '.')

        axEx.plot(S.TsSnaps, np.array(S.SpeedErrorSnaps)[:,speed_id,0], color = GivenColors[n_speed])
        axEy.plot(S.TsSnaps, np.array(S.SpeedErrorSnaps)[:,speed_id,1], color = GivenColors[n_speed])

        if S._UpdatePT and PlotUpdatePTMarkers:
            PTH = S.ProjectionTimesHistory[speed_id]
            SpeedTss = np.array(Tss[n_speed])
            for Update in PTH:
                try:
                    Index_At_PTChange = (SpeedTss > Update).tolist().index(True)
                except:
                    Index_At_PTChange = -1
                axx.plot(Update, Vxs[n_speed][Index_At_PTChange], color = GivenColors[n_speed], marker = '^')
                axy.plot(Update, Vys[n_speed][Index_At_PTChange], color = GivenColors[n_speed], marker = '^')

    D = None
    if not GroundTruthFile is None:
        with open(GroundTruthFile, 'rb') as openfile:
            D = json.load(openfile)
    else:
        StreamName = S.__Framework__.StreamHistory[-1]
        NameParts = StreamName.split('/')
        NewName = NameParts[-1].replace('.', '_') + '_hough.gnd'
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
            axx.plot([minTs, maxTs], [VxTh, VxTh], '--', label = 'Ground truth, Feature {0}'.format(PointID + 1), color = GivenColors[PointID])
            axy.plot([minTs, maxTs], [VyTh, VyTh], '--', label = 'Ground truth, Feature {0}'.format(PointID + 1), color = GivenColors[PointID])
    axx.legend()
    axx.set_xlabel('t(s)')
    axx.set_ylabel('vx(px/s)')
    axy.set_xlabel('t(s)')
    axy.set_ylabel('vy(px/s)')
    return f, axs, GivenColors

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
    f.suptitle("Zone number {0}, at t = {1}\nx : {2} -> {3}\ny : {4} -> {5}".format(ZoneNumber, S.__Framework__.Tools[S.__CreationReferences__['Memory']].LastEvent.timestamp, OW[0], OW[2], OW[1], OW[3]))
    for Zone in S.Zones.keys():
        if Zone[4] == ZoneNumber:
            break
    for speed_id in S.Zones[Zone]:
        xIndex = SeedsX.index(S.InitialSpeeds[speed_id][0])
        yIndex = SeedsY.index(S.InitialSpeeds[speed_id][1])
        axs[xIndex, yIndex].imshow(np.transpose(S.StreaksMaps[speed_id] >= MaxRatio*S.StreaksMaps[speed_id].max()), origin = 'lower')
        if speed_id in S.ActiveSpeeds:
            color = 'g'
        else:
            color = 'r'
        axs[xIndex, yIndex].set_title("vx = {0:.1f}, vy = {1:.1f}, id = {2}".format(S.Speeds[speed_id][0], S.Speeds[speed_id][1], speed_id), color = color)

    for axx in axs:
        for ax in axx:
            ax.tick_params('both', left = 'off', bottom = 'off', labelleft = 'off', labelbottom = 'off')
    return f, axs

def PlotDecayingMaps(SpeedProjector, ZoneNumber = 0):
    S = SpeedProjector
    
    IniSpeeds = np.array(S.InitialSpeeds)
    SeedsX = np.unique(IniSpeeds[:,0]).tolist()
    SeedsY = np.unique(IniSpeeds[:,1]).tolist()

    f, axs = plt.subplots(len(SeedsX), len(SeedsX))
    OW = S.OWAPT[S.Zones.values()[ZoneNumber][0]]
    f.suptitle("Zone number {0}, at t = {1}\nx : {2} -> {3}\ny : {4} -> {5}".format(ZoneNumber, S.__Framework__.Tools[S.__CreationReferences__['Memory']].LastEvent.timestamp, OW[0], OW[2], OW[1], OW[3]))
    for Zone in S.Zones.keys():
        if Zone[4] == ZoneNumber:
            break
    for speed_id in S.Zones[Zone]:
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

def PlotTracking(S, SpeedIDs, SnapsIDs, orientation = 'horizontal', cmap = None, markercolors = ['green', 'grey'], markersize = 15, markertypes = ['v', 'v'], AddIDs = True):
    nSnaps = len(SnapsIDs)
    nSpeeds = len(SpeedIDs)
    if orientation == 'vertical':
        f, axs = plt.subplots(nSnaps, nSpeeds)
    else:
        f, axs = plt.subplots(nSpeeds, nSnaps)
    if nSpeeds == 1 and nSnaps == 1:
        axs = np.array(axs)
    else:
        if nSpeeds == 1:
            axs = np.array([axs])
        elif nSnaps == 1:
            axs = np.array([axs])
            if orientation == 'vertical':
                axs = axs.T

    n_x_legend = len(SnapsIDs)/2
    n_y_legend = len(SpeedIDs) - 1
                
    if orientation == 'horizontal':
        n_x_legend, n_y_legend = n_y_legend, n_x_legend
    for n_y_ini, speed_id in enumerate(SpeedIDs):
        Ref = S.MeanPositionsReferences[speed_id]*S._DensityDefinition
        if orientation == 'horizontal' and AddIDs:
            axs[n_y_ini,0].set_ylabel("ID = {0}".format(speed_id))
        for n_x_ini, snap_id in enumerate(SnapsIDs):
            if orientation == 'vertical':
                n_x, n_y = n_x_ini, n_y_ini
            else:
                n_y, n_x = n_x_ini, n_y_ini
            if n_x == n_x_legend and n_y == n_y_legend:
                label = 'Reference position'
            else:
                label = None
            axs[n_x, n_y].plot(Ref[0], Ref[1], markertypes[0], color = markercolors[0], markersize = markersize, label = label)

            Snap = S.DMSnaps[snap_id][speed_id]
            if cmap is None:
                axs[n_x, n_y].imshow(np.transpose(Snap), origin = 'lower')
            else:
                axs[n_x, n_y].imshow(np.transpose(Snap), origin = 'lower', cmap = plt.get_cmap(cmap))

            Xs, Ys = np.where(Snap > 0)
            Weights = Snap[Xs, Ys]
            Xm, Ym = (Weights*Xs).sum() / Weights.sum(), (Weights*Ys).sum() / Weights.sum()
            if n_x == n_x_legend and n_y == n_y_legend:
                label = 'Instant mean position'
            else:
                label = None
            axs[n_x, n_y].plot(Xm, Ym, markertypes[1], color = markercolors[1], markersize = markersize, label = label)

            tSnap = S.TsSnaps[snap_id]
            Title = "t = {0:.2}s\nA = {1:.1f}".format(tSnap, Weights.sum())
            if AddIDs and orientation == 'vertical' and n_x_ini == 0:
                Title = "ID = {0}\n".format(speed_id) + Title
            axs[n_x, n_y].set_title(Title)
    axs[n_x_legend,n_y_legend].legend(loc=9, bbox_to_anchor=(0.5, -0.3), ncol=2)

def CreateTrackingShot(F, S, SpeedIDs, SnapshotNumber = 0, BinDt = 0.005, fontsize = 15, ax_given = None, cmap = None, addSpeedIDs = False, removeTicks = True):
    if ax_given is None:
        f, ax = plt.subplots(1,1)
    else:
        ax = ax_given
    if cmap is None:
        ax.imshow(np.transpose(F.Mem.Snapshots[SnapshotNumber][1].max(axis = 2) > F.Mem.Snapshots[SnapshotNumber][1].max()-BinDt), origin = 'lower') 
    else:
        ax.imshow(np.transpose(F.Mem.Snapshots[SnapshotNumber][1].max(axis = 2) > F.Mem.Snapshots[SnapshotNumber][1].max()-BinDt), origin = 'lower', cmap = plt.get_cmap(cmap))
    for speed_id in SpeedIDs:
        for Zone in S.Zones.keys():
            if speed_id in S.Zones[Zone]:
                Feature_id = Zone[4]
        Box = [S.OWAPT[speed_id][0] + S.DisplacementSnaps[SnapshotNumber][speed_id][0], S.OWAPT[speed_id][1] + S.DisplacementSnaps[SnapshotNumber][speed_id][1], S.OWAPT[speed_id][2] + S.DisplacementSnaps[SnapshotNumber][speed_id][0], S.OWAPT[speed_id][3] + S.DisplacementSnaps[SnapshotNumber][speed_id][1]]
        
        ax.plot([Box[0], Box[0]], [Box[1], Box[3]], 'r')
        ax.plot([Box[0], Box[2]], [Box[3], Box[3]], 'r')
        ax.plot([Box[2], Box[2]], [Box[3], Box[1]], 'r')
        ax.plot([Box[2], Box[0]], [Box[1], Box[1]], 'r')
        
        ax.text(Box[2] + 5, Box[1] + (Box[3] - Box[1])*0.4, 'Feature {0}'.format(Feature_id) + addSpeedIDs * ', ID = {0}'.format(speed_id), color = 'r', fontsize = fontsize)
    if removeTicks:
        ax.tick_params(bottom = 'off', left = 'off', labelbottom = 'off', labelleft = 'off')
    if ax_given is None:
        return f, ax

def GenerateTrackingGif(F, S, SpeedIDs, SnapRatio = 1, tMax = np.inf, Folder = '/home/dardelet/Pictures/GIFs/AutoGeneratedTracking/', BinDt = None, add_timestamp_title = True, cmap = None, GivenColors = None):
    if BinDt is None:
        BinDt = S._SnapshotDt
    if GivenColors is None:
        GivenColors = ['r' for speed_id in SpeedIDs]
    Snaps_IDs = [snap_id for snap_id in range(len(S.TsSnaps)) if snap_id % SnapRatio == 0]
    os.system('rm '+ Folder + '*.png')
    print "Generating {0} png frames on {1} possible ones.".format(len(Snaps_IDs), len(S.TsSnaps))
    for snap_id in Snaps_IDs:
        stdout.write("\r > {0}/{1}".format(snap_id/SnapRatio + 1, len(Snaps_IDs)))
        stdout.flush()
        if S.TsSnaps[snap_id] > tMax:
            break
        f, ax = plt.subplots(1,1)

        _CreateTrackingPicture(ax, snap_id, F, S, SpeedIDs, GivenColors, BinDt, cmap, add_timestamp_title)

        f.savefig(Folder + 't_{0:03d}.png'.format(snap_id))
        plt.close(f.number)
    print "\r > Done.          "
    GifFile = 'tracking_'+datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')+'.gif'
    print "Generating gif."
    os.system('convert -delay {0} -loop 0 '.format(int(1000*S._SnapshotDt))+Folder+'*.png ' + Folder + GifFile)
    print "GIF generated : {0}".format(GifFile)
    os.system('kde-open ' + Folder + GifFile)

    ans = raw_input('Rate this result (1-nice/0-bad/(d)elete) : ')
    if '->' in ans:
        ans, name = ans.split('->')
        ans = ans.strip()
        name = name.strip()
        if '.gif' in name:
            name = name.split('.gif')[0]
        NewGifFile = name + '_' + GifFile
    else:
        NewGifFile = GifFile
    if '1' in ans.lower() or 'nice' in ans.lower():
        os.system('mv ' + Folder + GifFile + ' ' + Folder + 'Nice/' + NewGifFile)
        print "Moving gif file to Nice folder."
    elif '0' in ans.lower() or 'bad' in ans.lower():
        os.system('mv ' + Folder + GifFile + ' ' + Folder + 'Bad/' + NewGifFile)
        print "Moving gif file to Bad folder."
    elif len(ans) > 0 and ans.lower()[0] == 'd' or 'delete' in ans.lower():
        os.system('rm ' + Folder + GifFile)
        print "Deleted Gif file. RIP."
    else:
        os.system('mv ' + Folder + GifFile + ' ' + Folder + 'Meh/' + NewGifFile)
        print "Moving gif file to Meh folder."

def _CreateTrackingPicture(ax, snap_id, F, S, SpeedIDs, BoxColors, BinDt, cmap, add_timestamp_title, titleSize = 15, lw = 1, BorderMargin = 5):
    if cmap is None:
        ax.imshow(np.transpose(F.Mem.Snapshots[snap_id][1].max(axis = 2) > F.Mem.Snapshots[snap_id][1].max()-BinDt), origin = 'lower') 
    else:
        ax.imshow(np.transpose(F.Mem.Snapshots[snap_id][1].max(axis = 2) > F.Mem.Snapshots[snap_id][1].max()-BinDt), origin = 'lower', cmap = plt.get_cmap(cmap))

    SortedSpeedIDs = []
    for ZoneID in range(len(S.Zones.keys())):
        for Zone in S.Zones.keys():
            if Zone[4] == ZoneID:
                for speed_id in S.Zones[Zone]:
                    if speed_id in SpeedIDs:
                        SortedSpeedIDs += [speed_id]
                break
    for n_speed, speed_id in enumerate(SortedSpeedIDs):
        Box = [S.OWAPT[speed_id][0] + S.DisplacementSnaps[snap_id][speed_id][0], S.OWAPT[speed_id][1] + S.DisplacementSnaps[snap_id][speed_id][1], S.OWAPT[speed_id][2] + S.DisplacementSnaps[snap_id][speed_id][0], S.OWAPT[speed_id][3] + S.DisplacementSnaps[snap_id][speed_id][1]]
        if (np.array(Box) < BorderMargin).any() or Box[2] >= F.Mem.Snapshots[snap_id][1].shape[0] - BorderMargin or Box[3] >= F.Mem.Snapshots[snap_id][1].shape[1] - BorderMargin:
            continue
        
        ax.plot([Box[0], Box[0]], [Box[1], Box[3]], c = BoxColors[n_speed], lw = lw)
        ax.plot([Box[0], Box[2]], [Box[3], Box[3]], c = BoxColors[n_speed], lw = lw)
        ax.plot([Box[2], Box[2]], [Box[3], Box[1]], c = BoxColors[n_speed], lw = lw)
        ax.plot([Box[2], Box[0]], [Box[1], Box[1]], c = BoxColors[n_speed], lw = lw)
    if add_timestamp_title:
        ax.set_title("t = {0:.2f}s".format(S.TsSnaps[snap_id]), fontsize = titleSize)

def GenerateTrackingPanel(F, S, SnapsIDs, SpeedIDs, Colors = None, BinDt = 0.005, add_timestamp_title = True, cmap = None, RemoveTicksAndTicksLabels = True, titleSize = 15, lw = 1):
    if Colors is None:
        Colors = ['r' for speed_id in SpeedIDs]
    f, axs = plt.subplots(1, len(SnapsIDs))
    if len(SnapsIDs) == 1:
        axs = [axs]
    for n_snap, snap_id in enumerate(SnapsIDs):
        _CreateTrackingPicture(axs[n_snap], snap_id, F, S, SpeedIDs, Colors, BinDt, cmap, add_timestamp_title, titleSize, lw = lw)

    if RemoveTicksAndTicksLabels:
        for ax in axs:
            ax.tick_params('both', bottom = 'off', left = 'off', labelbottom = 'off', labelleft = 'off')
    return f, axs

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

def PlotPositionTracking(S, SpeedIDs, GroundTruthFile = None, SnapshotsTss = [], AddSpeedIdLabel = True, AddFeatureIdLabel = True, Tmax = np.inf, legendsize = 20, axissize = 20, titlesize = 20, orientation = 'horizontal', legend = True, legendloc = None, boxToAnchor = None, legendNcols = 1, GndColorsIni = None):
    ZonesAsked = {}
    SeveralIDsForOneZone = False
    SortedSpeedIDs = []
    for ZoneID in range(len(S.Zones.keys())):
        FoundOneIDForZone = False
        for Zone in S.Zones.keys():
            ZonesAsked[Zone[4]] = False
            if Zone[4] == ZoneID:
                for speed_id in S.Zones[Zone]:
                    if speed_id in SpeedIDs:
                        if FoundOneIDForZone:
                            SeveralIDsForOneZone = True
                        SortedSpeedIDs += [speed_id]
                        FoundOneIDForZone = True
                        ZonesAsked[Zone[4]] = True
                break
    print 'ZonesAsked : ', ZonesAsked
    if orientation == 'horizontal':
        f, axs = plt.subplots(1,2)
    else:
        f, axs = plt.subplots(2,1)

    D = None
    if not GroundTruthFile is None:
        with open(GroundTruthFile, 'rb') as openfile:
            D = json.load(openfile)
    else:
        StreamName = S.__Framework__.StreamHistory[-1]
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
        
        GndColors = {}
        
        PointsList = D['RecordedPoints']
        for PointID in range(int(np.array(PointsList)[:,0].max() + 1)):
            if PointID not in ZonesAsked.keys() or not ZonesAsked[PointID]:
                continue
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
            
            if GndColorsIni is None:
                axs[0].plot(TsTh, Xs, '--')
                GndColors[PointID] = axs[0].get_lines()[-1].get_color()
            else:
                GndColors[PointID] = GndColorsIni[PointID]
                axs[0].plot(TsTh, Xs, '--', color = GndColors[PointID])
            axs[1].plot(TsTh, Ys, '--', color = GndColors[PointID])

            AddToLegend += ['Ground truth, Feature {0}'.format(PointID + 1)]
            if legend and not AddSpeedIdLabel and not AddFeatureIdLabel:
                axs[0].legend(AddToLegend)
    else:
        GndColors = None

    ChosenColors = {}
    Labels = []
    for speed_id in SortedSpeedIDs:
        stdout.write("\r > {0}/{1}".format(SpeedIDs.index(speed_id) + 1, len(SpeedIDs)))
        stdout.flush()

        PHx, PHy = CreatePositionsHistory(S, speed_id, Tmax)
        
        Labels += ['']
        for Zone in S.Zones.keys():
            if speed_id in S.Zones[Zone]:
                Feature_id = Zone[4]
        if AddFeatureIdLabel:
            Labels[-1] += 'Feature {0}'.format(Feature_id + 1)
            if AddSpeedIdLabel or SeveralIDsForOneZone:
                Labels[-1] += ', '
        if AddSpeedIdLabel or SeveralIDsForOneZone:
            Labels[-1] += 'ID = {0}'.format(speed_id)
        
        if not GndColors is None:
            c = GndColors[Feature_id]
        else:
            c = np.random.rand(3,1)
        lcx = mc.LineCollection(PHx, colors=c)
        lcy = mc.LineCollection(PHy, colors=c)
        axs[0].add_collection(lcx)
        axs[1].add_collection(lcy)
        ChosenColors[Feature_id] = c
    
    print ""

    if legend and (AddSpeedIdLabel or AddFeatureIdLabel):
        print "Adding legend"
        if legendloc is None:
            axs[1].legend(AddToLegend + Labels, fontsize = legendsize, ncol = legendNcols)
        else:
            if boxToAnchor is None:
                axs[1].legend(AddToLegend + Labels, fontsize = legendsize, loc = legendloc, ncol = legendNcols)
            else:
                axs[1].legend(AddToLegend + Labels, fontsize = legendsize, loc = legendloc, bbox_to_anchor = boxToAnchor, ncol = legendNcols)

    for ts in SnapshotsTss:
        axs[0].plot([ts, ts], [0, 700], 'k--')
        axs[1].plot([ts, ts], [0, 700], 'k--')

    axs[0].set_title('X tracked position', fontsize = titlesize)
    axs[1].set_title('Y tracked position', fontsize = titlesize)
    axs[0].set_xlabel('t(s)', fontsize = axissize)
    axs[0].set_ylabel('x(px)', fontsize = axissize)
    axs[1].set_xlabel('t(s)', fontsize = axissize)
    axs[1].set_ylabel('y(px)', fontsize = axissize)
    axs[0].autoscale()
    axs[0].margins(0.1)
    axs[1].autoscale()
    axs[1].margins(0.1)

    return f, axs, ChosenColors

from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D

from colour import Color

def Plot3DTrackingView(S, Speed_ids, SnapRatio, tmax = np.inf, WMinRatio = 0.2, MarkersSize = 5):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    MeanPosX = []
    MeanPosY = []
    MeanPosT = []

    PosX = []
    PosY = []
    PosT = []
    PosRGBA = []

    colors = list(Color("red").range_to(Color("green"),len(S.TsSnaps)/2))[:-1] + list(Color("green").range_to(Color("blue"),len(S.TsSnaps) - len(S.TsSnaps)/2 + 1))
    for speed_id in Speed_ids:
        OWCenter = np.array([float(S.OWAPT[speed_id][0] + S.OWAPT[speed_id][2])/2, float(S.OWAPT[speed_id][1] + S.OWAPT[speed_id][3])/2])
        for snap_id in range(len(S.TsSnaps)):
            stdout.write("\r > speed_id : {2}/{3} -> {0}/{1}".format(snap_id, len(S.TsSnaps), Speed_ids.index(speed_id) + 1, len(Speed_ids)))
            stdout.flush()
            if not (snap_id%SnapRatio == 0):
                continue
            if S.TsSnaps[snap_id] > tmax:
                break
            Map = S.DMSnaps[snap_id][speed_id]
            Displacement = S.DisplacementSnaps[snap_id][speed_id] + OWCenter - np.array(Map.shape, dtype = float)/(2*S._DensityDefinition)
            Xs, Ys = np.where(Map > 0)
            Ws = Map[Xs, Ys]
            MeanPosX += [(Xs*Ws).sum()/Ws.sum()/S._DensityDefinition + Displacement[0]]
            MeanPosY += [(Ys*Ws).sum()/Ws.sum()/S._DensityDefinition + Displacement[1]]
            MeanPosT += [S.TsSnaps[snap_id]]

            continue
            for i in range(Xs.shape[0]):
                if Ws[i]/Ws.max() > WMinRatio:
                    PosX += [Xs[i]/S._DensityDefinition + Displacement[0]]
                    PosY += [Ys[i]/S._DensityDefinition + Displacement[1]]
                    PosT += [S.TsSnaps[snap_id]]
                    PosRGBA += [colors[snap_id].get_rgb() + tuple([Ws[i]/Ws.max()])]
                    #R = Rectangle((Xs[i]/S._DensityDefinition + Displacement[0], Ys[i]/S._DensityDefinition + Displacement[1]), 2./ S._DensityDefinition ,2./ S._DensityDefinition, color = colors[snap_id].get_hex_l(), alpha = Ws[i]/Ws.max())
                    #ax.add_patch(R)
                    #art3d.pathpatch_2d_to_3d(R, z=S.TsSnaps[snap_id], zdir="y")
            #R = Rectangle((MeanPosX[-1], MeanPosY[-1]), 4./ S._DensityDefinition ,4./ S._DensityDefinition, color = 'k', alpha = 1)
            #ax.add_patch(R)
            #art3d.pathpatch_2d_to_3d(R, z=S.TsSnaps[snap_id] + 0.0001, zdir="y")
    print ""
    print "Scattering {0} points".format(len(PosX))
    PosX = np.array(PosX)
    PosY = np.array(PosY)
    PosT = np.array(PosT)
    #ax.scatter(PosX, PosT, PosY, color= np.array(PosRGBA), marker ='o', s = MarkersSize)

    MeanPosX = np.array(MeanPosX)
    MeanPosY = np.array(MeanPosY)
    MeanPosT = np.array(MeanPosT)
    ax.scatter(MeanPosX, MeanPosT, MeanPosY, color= 'k', marker ='o')

    Dx = MeanPosX.max() - MeanPosX.min() + 30
    Dy = MeanPosY.max() - MeanPosY.min() + 30
    Mx = (MeanPosX.max() + MeanPosX.min())/2
    My = (MeanPosY.max() + MeanPosY.min())/2
    D = max(Dx, Dy)
    ax.set_xlim(Mx - D/2, Mx + D/2)
    ax.set_zlim(My - D/2, My + D/2)
    ax.set_ylim(S.TsSnaps[-1], S.TsSnaps[0])
    ax.set_xlabel('X position (px)')
    ax.set_zlabel('Y position (px)')
    ax.set_ylabel('t (s)')
    print " > Done !"

    plt.show()
    return fig, ax, PosX, PosY, PosT, MeanPosX, MeanPosY, MeanPosT
