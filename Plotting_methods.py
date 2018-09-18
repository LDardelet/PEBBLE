import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
import json

import os

from sys import stdout
from matplotlib import collections  as mc

import datetime

def PlotConvergence(S, SpeedDuos = None, GroundTruthFile = None, AddSpeedIdLabel = True, AddFeatureIdLabel = True, GivenColors = None, PlotUpdatePTMarkers = True, FeatureNumerotation = 'computer', FeatureInitialOriginKept = True):
    if SpeedDuos is None:
        SpeedDuos = S.RecoverCurrentBestSpeeds()

    if GivenColors is None:
        GivenColors = {zone_id: None for speed_id, zone_id in SpeedDuos}

    Vxs = [];Vys = [];Tss = []
    for speed_id, zone_id in SpeedDuos:
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

    for n_speed, duo in enumerate(SpeedDuos):
        speed_id, zone_id = duo

        maxTs = max(maxTs, max(Tss[n_speed]))
        minTs = min(minTs, min(Tss[n_speed]))
        label = ''
        if AddSpeedIdLabel or AddFeatureIdLabel:
            if AddFeatureIdLabel:
                label += 'Feature {0}'.format((FeatureInitialOriginKept * zone_id + (not FeatureInitialOriginKept) * n_speed) + (FeatureNumerotation == 'human'))
                if AddSpeedIdLabel:
                    label += ', '
            if AddSpeedIdLabel:
                label += 'ID = {0}'.format(speed_id)
        axx.plot(Tss[n_speed], Vxs[n_speed], label = label, color = GivenColors[zone_id])
        if GivenColors[zone_id] is None:
            GivenColors[zone_id] = axx.get_lines()[-1].get_color()
        axy.plot(Tss[n_speed], Vys[n_speed], label = label, color = GivenColors[zone_id])
        
        axx.plot(S.TsSnaps, np.array(S.AimedSpeedSnaps)[:,speed_id,0], color = GivenColors[zone_id], marker = '.')
        axy.plot(S.TsSnaps, np.array(S.AimedSpeedSnaps)[:,speed_id,1], color = GivenColors[zone_id], marker = '.')

        axEx.plot(S.TsSnaps, np.array(S.SpeedErrorSnaps)[:,speed_id,0], color = GivenColors[zone_id])
        axEy.plot(S.TsSnaps, np.array(S.SpeedErrorSnaps)[:,speed_id,1], color = GivenColors[zone_id])

        if S._UpdatePT and PlotUpdatePTMarkers:
            PTH = S.ProjectionTimesHistory[speed_id]
            SpeedTss = np.array(Tss[n_speed])
            for Update in PTH:
                try:
                    Index_At_PTChange = (SpeedTss > Update).tolist().index(True)
                except:
                    Index_At_PTChange = -1
                axx.plot(Update, Vxs[n_speed][Index_At_PTChange], color = GivenColors[zone_id], marker = '^')
                axy.plot(Update, Vys[n_speed][Index_At_PTChange], color = GivenColors[zone_id], marker = '^')

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
        for n_speed, duo in enumerate(SpeedDuos):
            speed_id, zone_id = duo

            PointList = []
            for Point in PointsList:
                if Point[0] == zone_id:
                    PointList += [Point[1:]]
            TsTh = np.array(PointList)[:,0]
            Xs = np.array(PointList)[:,1]
            Ys = np.array(PointList)[:,2]
            VxTh = ((Xs - Xs.mean())**2).sum()/((TsTh - TsTh.mean()) * (Xs - Xs.mean())).sum()
            VyTh = ((Ys - Ys.mean())**2).sum()/((TsTh - TsTh.mean()) * (Ys - Ys.mean())).sum()
            axx.plot([minTs, maxTs], [VxTh, VxTh], '--', label = 'Ground truth, Feature {0}'.format((FeatureInitialOriginKept * zone_id + (not FeatureInitialOriginKept) * n_speed) + (FeatureNumerotation == 'human')), color = GivenColors[zone_id])
            axy.plot([minTs, maxTs], [VyTh, VyTh], '--', label = 'Ground truth, Feature {0}'.format((FeatureInitialOriginKept * zone_id + (not FeatureInitialOriginKept) * n_speed) + (FeatureNumerotation == 'human')), color = GivenColors[zone_id])
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

def PlotTracking(S, SpeedDuos = None, SnapsIDs = None, orientation = 'horizontal', cmap = None, markercolors = ['green', 'grey'], markersize = 15, markertypes = ['v', 'v'], AddIDs = True):
    if SnapsIDs is None:
        SnapsIDs = range(len(S.TsSnaps))
    if SpeedDuos is None:
        SpeedDuos = S.RecoverCurrentBestSpeeds()

    nSnaps = len(SnapsIDs)
    nSpeeds = len(SpeedDuos)
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
    n_y_legend = len(SpeedDuos) - 1
                
    if orientation == 'horizontal':
        n_x_legend, n_y_legend = n_y_legend, n_x_legend
    for n_y_ini, duo in enumerate(SpeedDuos):
        speed_id, zone_id = duo
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

def CreateTrackingShot(F, S, SpeedDuos = None, SnapshotNumber = 0, BinDt = 0.005, fontsize = 15, ax_given = None, cmap = None, addSpeedIDs = False, removeTicks = True, FeatureNumerotation = 'computer', FeatureInitialOriginKept = True):
    if SpeedDuos is None:
        SpeedDuos = S.RecoverCurrentBestSpeeds()

    if ax_given is None:
        f, ax = plt.subplots(1,1)
    else:
        ax = ax_given
    if cmap is None:
        ax.imshow(np.transpose(F.Mem.Snapshots[SnapshotNumber][1].max(axis = 2) > F.Mem.Snapshots[SnapshotNumber][1].max()-BinDt), origin = 'lower') 
    else:
        ax.imshow(np.transpose(F.Mem.Snapshots[SnapshotNumber][1].max(axis = 2) > F.Mem.Snapshots[SnapshotNumber][1].max()-BinDt), origin = 'lower', cmap = plt.get_cmap(cmap))
    for n_speed, duo in enumerate(SpeedDuos):
        speed_id, zone_id = duo
        Box = [S.OWAPT[speed_id][0] + S.DisplacementSnaps[SnapshotNumber][speed_id][0], S.OWAPT[speed_id][1] + S.DisplacementSnaps[SnapshotNumber][speed_id][1], S.OWAPT[speed_id][2] + S.DisplacementSnaps[SnapshotNumber][speed_id][0], S.OWAPT[speed_id][3] + S.DisplacementSnaps[SnapshotNumber][speed_id][1]]
        
        ax.plot([Box[0], Box[0]], [Box[1], Box[3]], 'r')
        ax.plot([Box[0], Box[2]], [Box[3], Box[3]], 'r')
        ax.plot([Box[2], Box[2]], [Box[3], Box[1]], 'r')
        ax.plot([Box[2], Box[0]], [Box[1], Box[1]], 'r')
        
        ax.text(Box[2] + 5, Box[1] + (Box[3] - Box[1])*0.4, 'Feature {0}'.format((FeatureInitialOriginKept * zone_id + (not FeatureInitialOriginKept) * n_speed) + (FeatureNumerotation == 'human')) + addSpeedIDs * ', ID = {0}'.format(speed_id), color = 'r', fontsize = fontsize)
    if removeTicks:
        ax.tick_params(bottom = 'off', left = 'off', labelbottom = 'off', labelleft = 'off')
    if ax_given is None:
        return f, ax

def GenerateTrackingGif(F, S, SpeedDuos = None, SnapRatio = 1, tMax = np.inf, Folder = '/home/dardelet/Pictures/GIFs/AutoGeneratedTracking/', BinDt = None, add_timestamp_title = True, cmap = None, GivenColors = None, add_Feature_label_Fontsize = 0, FeatureNumerotation = 'computer', FeatureInitialOriginKept = True, IncludeSpeedError = False):
    if BinDt is None:
        BinDt = S._SnapshotDt
    if SpeedDuos is None:
        SpeedDuos = S.RecoverCurrentBestSpeeds()

    if GivenColors is None:
        GivenColors = {zone_id: 'r' for speed_id, zone_id in SpeedDuos}

    Snaps_IDs = [snap_id for snap_id in range(len(S.TsSnaps)) if snap_id % SnapRatio == 0]
    os.system('rm '+ Folder + '*.png')
    print "Generating {0} png frames on {1} possible ones.".format(len(Snaps_IDs), len(S.TsSnaps))
    for snap_id in Snaps_IDs:
        stdout.write("\r > {0}/{1}".format(snap_id/SnapRatio + 1, len(Snaps_IDs)))
        stdout.flush()
        if S.TsSnaps[snap_id] > tMax:
            break
        f, ax = plt.subplots(1,1)

        _CreateTrackingPicture(ax, snap_id, F, S, SpeedDuos, GivenColors, BinDt, cmap, add_timestamp_title, add_Feature_label_Fontsize, FeatureNumerotation = FeatureNumerotation, FeatureInitialOriginKept = FeatureInitialOriginKept, IncludeSpeedError = IncludeSpeedError)

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

def _CreateTrackingPicture(ax, snap_id, F, S, SpeedDuos, BoxColors, BinDt, cmap, add_timestamp_title, add_Feature_label_Fontsize = 0, titleSize = 15, lw = 1, BorderMargin = 5, FeatureNumerotation = 'computer', FeatureInitialOriginKept = True, IncludeSpeedError = 0):
    if cmap is None:
        ax.imshow(np.transpose(F.Mem.Snapshots[snap_id][1].max(axis = 2) > F.Mem.Snapshots[snap_id][1].max()-BinDt), origin = 'lower') 
    else:
        ax.imshow(np.transpose(F.Mem.Snapshots[snap_id][1].max(axis = 2) > F.Mem.Snapshots[snap_id][1].max()-BinDt), origin = 'lower', cmap = plt.get_cmap(cmap))

    for n_speed, duo in enumerate(SpeedDuos):
        speed_id, zone_id = duo
        Box = [S.OWAPT[speed_id][0] + S.DisplacementSnaps[snap_id][speed_id][0], S.OWAPT[speed_id][1] + S.DisplacementSnaps[snap_id][speed_id][1], S.OWAPT[speed_id][2] + S.DisplacementSnaps[snap_id][speed_id][0], S.OWAPT[speed_id][3] + S.DisplacementSnaps[snap_id][speed_id][1]]
        if (np.array(Box) < BorderMargin).any() or Box[2] >= F.Mem.Snapshots[snap_id][1].shape[0] - BorderMargin or Box[3] >= F.Mem.Snapshots[snap_id][1].shape[1] - BorderMargin:
            continue
        color = BoxColors[zone_id]
        if IncludeSpeedError:
            Error = S.SpeedErrorSnaps[snap_id][speed_id]
            t = S.TsSnaps[snap_id]
            Speed = np.array([0., 0.])
            for SpeedChange in S.SpeedsChangesHistory[speed_id]:
                if t < SpeedChange[0]:
                    break
                Speed = SpeedChange[1]
            AlphaValue = max(0., 1. - IncludeSpeedError * np.linalg.norm(Error)/np.linalg.norm(Speed))
        ax.plot([Box[0], Box[0]], [Box[1], Box[3]], c = color, lw = lw, alpha = AlphaValue)
        ax.plot([Box[0], Box[2]], [Box[3], Box[3]], c = color, lw = lw, alpha = AlphaValue)
        ax.plot([Box[2], Box[2]], [Box[3], Box[1]], c = color, lw = lw, alpha = AlphaValue)
        ax.plot([Box[2], Box[0]], [Box[1], Box[1]], c = color, lw = lw, alpha = AlphaValue)

        if add_Feature_label_Fontsize:
            ax.text(Box[2] + 5, Box[1] + (Box[3] - Box[1])*0.4, 'Feature {0}'.format((FeatureInitialOriginKept * zone_id + (not FeatureInitialOriginKept) * n_speed) + (FeatureNumerotation == 'human')) + ', ID = {0}'.format(speed_id), color = np.array(BoxColors[zone_id]).reshape(3), fontsize = add_Feature_label_Fontsize)
    if add_timestamp_title:
        ax.set_title("t = {0:.2f}s".format(S.TsSnaps[snap_id]), fontsize = titleSize)

def GenerateTrackingPanel(F, S, SnapsIDs, SpeedDuos, GivenColors = None, BinDt = 0.005, add_timestamp_title = True, cmap = None, RemoveTicksAndTicksLabels = True, titleSize = 15, lw = 1):
    if SpeedDuos is None:
        SpeedDuos = S.RecoverCurrentBestSpeeds()

    if GivenColors is None:
        GivenColors = {zone_id: 'r' for speed_id, zone_id in SpeedDuos}

    f, axs = plt.subplots(1, len(SnapsIDs))
    if len(SnapsIDs) == 1:
        axs = [axs]
    for n_snap, snap_id in enumerate(SnapsIDs):
        _CreateTrackingPicture(axs[n_snap], snap_id, F, S, SpeedDuos, GivenColors, BinDt, cmap, add_timestamp_title, titleSize, lw = lw)

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
    CurrentBaseDisplacement = CenterPos
    for n in range(len(S.SpeedsChangesHistory[speed_id]) - 1):
        t_start = S.SpeedsChangesHistory[speed_id][n][0]
        t_stop = S.SpeedsChangesHistory[speed_id][n+1][0]

        v_line = S.SpeedsChangesHistory[speed_id][n][1]

        if len(PTList) > 0 and t_start >= PTList[0]:
            CurrentProjectionTime = PTList.pop(0)
            CurrentBaseDisplacement = np.array([PositionsHistoryx[-1][1][1], PositionsHistoryy[-1][1][1]])
        PositionsHistoryx += [[(t_start, (CurrentBaseDisplacement + (t_start - CurrentProjectionTime) * v_line)[0]), (t_stop, (CurrentBaseDisplacement + (t_stop - CurrentProjectionTime) * v_line)[0])]]
        PositionsHistoryy += [[(t_start, (CurrentBaseDisplacement + (t_start - CurrentProjectionTime) * v_line)[1]), (t_stop, (CurrentBaseDisplacement + (t_stop - CurrentProjectionTime) * v_line)[1])]]
        if S.SpeedsChangesHistory[speed_id][n+1][0] > Tmax:
            break
    return PositionsHistoryx, PositionsHistoryy

def PlotPositionTracking(S, SpeedDuos = None, GroundTruthFile = 'default', SnapshotsTss = [], AddSpeedIdLabel = True, AddFeatureIdLabel = True, Tmax = np.inf, legendsize = 20, axissize = 20, titlesize = 20, orientation = 'horizontal', legend = True, legendloc = None, boxToAnchor = None, legendNcols = 1, GivenColors = None, FeatureNumerotation = 'computer', FeatureInitialOriginKept = True):
    if SpeedDuos is None:
        SpeedDuos = S.RecoverCurrentBestSpeeds()

    if GivenColors is None:
        GivenColors = {zone_id: None for speed_id, zone_id in SpeedDuos}

    if orientation == 'horizontal':
        f, axs = plt.subplots(1,2)
    else:
        f, axs = plt.subplots(2,1)

    D = None
    if GroundTruthFile:
        if not GroundTruthFile == 'default':
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
                print "Using default {0} file found".format(LoadName)
            except IOError:
                print "Unable to find default .gnd file"
    AddToLegend = []
    if not D is None:
        
        PointsList = D['RecordedPoints']
        for n_speed, duo in enumerate(SpeedDuos):
            speed_id, zone_id = duo

            PointList = []
            for Point in PointsList:
                if Point[0] == zone_id:
                    PointList += [Point[1:]]
            if not PointList:
                continue
            TsTh = np.array(PointList)[:,0]
            Xs = np.array(PointList)[:,1]
            Ys = np.array(PointList)[:,2]

            if Tmax == 'gnd':
                Tmax = TsTh.max()
            KeptPos = np.where(TsTh <= Tmax)
            TsTh = TsTh[KeptPos]
            Xs = Xs[KeptPos]
            Ys = Ys[KeptPos]
            
            axs[0].plot(TsTh, Xs, '--', color = GivenColors[zone_id])
            if GivenColors[zone_id] is None:
                GivenColors[zone_id] = axs[0].get_lines()[-1].get_color()
            axs[1].plot(TsTh, Ys, '--', color = GivenColors[zone_id])

            AddToLegend += ['Ground truth, Feature {0}'.format((FeatureInitialOriginKept * zone_id + (not FeatureInitialOriginKept) * n_speed) + (FeatureNumerotation == 'human'))]
            if legend and not AddSpeedIdLabel and not AddFeatureIdLabel:
                axs[0].legend(AddToLegend)

    Labels = []
    for n_speed, duo in enumerate(SpeedDuos):
        speed_id, zone_id = duo

        stdout.write("\r > {0}/{1}".format(n_speed + 1, len(SpeedDuos)))
        stdout.flush()

        PHx, PHy = CreatePositionsHistory(S, speed_id, Tmax)
        
        Labels += ['']
        if AddFeatureIdLabel:
            Labels[-1] += 'Feature {0}'.format((FeatureInitialOriginKept * zone_id + (not FeatureInitialOriginKept) * n_speed) + (FeatureNumerotation == 'human'))
            if AddSpeedIdLabel or SeveralIDsForOneZone:
                Labels[-1] += ', '
        if AddSpeedIdLabel or SeveralIDsForOneZone:
            Labels[-1] += 'ID = {0}'.format(speed_id)
        
        if GivenColors[zone_id] is None:
            GivenColors[zone_id] = np.random.rand(3,1)

        lcx = mc.LineCollection(PHx, colors = GivenColors[zone_id])
        lcy = mc.LineCollection(PHy, colors = GivenColors[zone_id])
        axs[0].add_collection(lcx)
        axs[1].add_collection(lcy)
    
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

    return f, axs, GivenColors

from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D

from colour import Color

def Plot3DTrackingView(S, SpeedDuos, SnapRatio, tMax = np.inf, WMinRatio = 0.2, MarkersSize = 5, ImshowSnaps = [], ImshowCmap = 'binary', ImshowTW = 0.02, ImshowMarkersSize = 40, ImshowMinAlpha = 0.4, t_scale = 3, azim = -43.5, elev = 34.2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    MeanPosX = []
    MeanPosY = []
    MeanPosT = []
    MeanPosRGB = []

    SpecialMarkerMeanPosX = []
    SpecialMarkerMeanPosY = []
    SpecialMarkerMeanPosT = []
    SpecialMarkerMeanPosRGB = []


    PosX = []
    PosY = []
    PosT = []
    PosRGBA = []

    # Scaling properties
    x_scale=1
    y_scale= t_scale
    z_scale=1
    
    scale=np.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=1.0
    
    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)
    
    ax.get_proj=short_proj

    nShots = (np.array(S.TsSnaps) <= tMax).sum() / SnapRatio + 1
    colors = list(Color("red").range_to(Color("green"), nShots / 2))[:-1] + list(Color("green").range_to(Color("blue"), nShots - (nShots / 2) + 1))
    for n_speed, duo in enumerate(SpeedDuos):
        speed_id, zone_id = duo
        OWCenter = np.array([float(S.OWAPT[speed_id][0] + S.OWAPT[speed_id][2])/2, float(S.OWAPT[speed_id][1] + S.OWAPT[speed_id][3])/2])
        for n_snap in range(nShots):
            snap_id = n_snap * SnapRatio
            stdout.write("\r > speed_id : {2}/{3} -> {0}/{1}".format(n_snap + 1, nShots, n_speed + 1, len(SpeedDuos)))
            stdout.flush()

            Map = S.DMSnaps[snap_id][speed_id]
            Displacement = S.DisplacementSnaps[snap_id][speed_id] + OWCenter - np.array(Map.shape, dtype = float)/(2*S._DensityDefinition)
            Xs, Ys = np.where(Map > 0)
            Ws = Map[Xs, Ys]

            MPX = (Xs*Ws).sum()/Ws.sum()/S._DensityDefinition + Displacement[0]
            MPY = (Ys*Ws).sum()/Ws.sum()/S._DensityDefinition + Displacement[1]

            if MPX < 0 or MPY < 0 or MPX >= 640 or MPY >= 480:
                continue

            if snap_id in ImshowSnaps:
                SpecialMarkerMeanPosX += [(Xs*Ws).sum()/Ws.sum()/S._DensityDefinition + Displacement[0]]
                SpecialMarkerMeanPosY += [(Ys*Ws).sum()/Ws.sum()/S._DensityDefinition + Displacement[1]]
                SpecialMarkerMeanPosT += [S.TsSnaps[snap_id]]
                SpecialMarkerMeanPosRGB += [colors[n_snap].get_rgb()]
            else:
                MeanPosX += [(Xs*Ws).sum()/Ws.sum()/S._DensityDefinition + Displacement[0]]
                MeanPosY += [(Ys*Ws).sum()/Ws.sum()/S._DensityDefinition + Displacement[1]]
                MeanPosT += [S.TsSnaps[snap_id]]
                MeanPosRGB += [colors[n_snap].get_rgb()]



            continue
            for i in range(Xs.shape[0]):
                if Ws[i]/Ws.max() > WMinRatio:
                    PosX += [Xs[i]/S._DensityDefinition + Displacement[0]]
                    PosY += [Ys[i]/S._DensityDefinition + Displacement[1]]
                    PosT += [S.TsSnaps[snap_id]]
                    PosRGBA += [colors[n_snap].get_rgb() + tuple([Ws[i]/Ws.max()])]
                    #R = Rectangle((Xs[i]/S._DensityDefinition + Displacement[0], Ys[i]/S._DensityDefinition + Displacement[1]), 2./ S._DensityDefinition ,2./ S._DensityDefinition, color = colors[snap_id].get_hex_l(), alpha = Ws[i]/Ws.max())
                    #ax.add_patch(R)
                    #art3d.pathpatch_2d_to_3d(R, z=S.TsSnaps[snap_id], zdir="y")
            #R = Rectangle((MeanPosX[-1], MeanPosY[-1]), 4./ S._DensityDefinition ,4./ S._DensityDefinition, color = 'k', alpha = 1)
            #ax.add_patch(R)
            #art3d.pathpatch_2d_to_3d(R, z=S.TsSnaps[snap_id] + 0.0001, zdir="y")
    #print ""
    #print "Scattering {0} points".format(len(PosX))
    #PosX = np.array(PosX)
    #PosY = np.array(PosY)
    #PosT = np.array(PosT)
    #ax.scatter(PosX, PosT, PosY, color= np.array(PosRGBA), marker ='o', s = MarkersSize)

    # Current axis attribution :
    # t -> y axis
    # x -> x axis
    # y -> z axis
    print ""
    print " > Done generating lines"
    print ""
    for n_snap, snap_id in enumerate(ImshowSnaps):
        stdout.write("\r > Imshows : snap {0}/{1}".format(n_snap + 1, len(ImshowSnaps)))
        stdout.flush()
        t, Map = S.__Framework__.Tools[S.__CreationReferences__['Memory']].Snapshots[snap_id]
        Mask = Map.max(axis = 2) > t - ImshowTW
        Xs, Ys = np.meshgrid(range(Map.shape[0]), range(Map.shape[1]))
        ax.contourf(Xs, transpose(Mask), Ys, zdir = 'y', offset = t, cmap = ImshowCmap, alpha = 1 - (1 - ImshowMinAlpha) * float(n_snap) / (len(ImshowSnaps) - 1))
    print ""
    print " > Scattering lines"

    MeanPosX = np.array(MeanPosX)
    MeanPosY = np.array(MeanPosY)
    MeanPosT = np.array(MeanPosT)
    MeanPosRGB = np.array(MeanPosRGB)
    ax.scatter(MeanPosX, MeanPosT, MeanPosY, color= MeanPosRGB, marker = '.', s = MarkersSize)
    ax.scatter(SpecialMarkerMeanPosX, SpecialMarkerMeanPosT, SpecialMarkerMeanPosY, color= SpecialMarkerMeanPosRGB, marker = '+', s = ImshowMarkersSize)

    Dx = MeanPosX.max() - MeanPosX.min() + 30
    Dy = MeanPosY.max() - MeanPosY.min() + 30
    Mx = (MeanPosX.max() + MeanPosX.min())/2
    My = (MeanPosY.max() + MeanPosY.min())/2
    D = max(Dx, Dy)
    ax.set_xlim(0, 640)
    ax.set_zlim(0, 480)
    ax.set_ylim(MeanPosT.max(), MeanPosT.min())
    ax.set_xlabel('\nX position (px)')
    ax.set_zlabel('Y position (px)')
    ax.set_ylabel('t (s)')
    print " > Done !"

    fig.tight_layout()
    ax.view_init(elev, azim)
    plt.show()
    return fig, ax
