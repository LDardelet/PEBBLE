import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
import json

import os

def PlotConvergence(S, SpeedIDs, GroundTruthFile = None):
    Vxs = [];Vys = [];Tss = []
    for speed_id in SpeedIDs:
        Tss += [[]]
        Vxs += [[]]
        Vys += [[]]
        for Change in S.SpeedsChangesHistory[speed_id]:
            Tss[-1] += [Change[0]]
            Vxs[-1] += [Change[1][0]]
            Vys[-1] += [Change[1][1]]
    fx = figure()
    axx = fx.add_subplot(1,1,1)
    axx.set_title('Vx Benchmark')
    fy = figure()
    axy = fy.add_subplot(1,1,1)
    axy.set_title('Vy Benchmark')
    maxTs = 0
    minTs = np.inf
    for i in range(len(SpeedIDs)):
        maxTs = max(maxTs, max(Tss[i]))
        minTs = min(minTs, min(Tss[i]))
        axx.plot(Tss[i], Vxs[i], label = 'Speed ID : {0}'.format(SpeedIDs[i]))
        axy.plot(Tss[i], Vys[i], label = 'Speed ID : {0}'.format(SpeedIDs[i]))
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
        except IOError:
            print "Unable to find default .gnd file"
    if not D is None:
        PointsList = D['RecordedPoints']
        for PointID in range(int(array(PointsList)[:,0].max() + 1)):
            PointList = []
            for Point in PointsList:
                if Point[0] == PointID:
                    PointList += [Point[1:]]
            TsTh = array(PointList)[:,0]
            Xs = array(PointList)[:,1]
            Ys = array(PointList)[:,2]
            VxTh = ((Xs - Xs.mean())**2).sum()/((TsTh - TsTh.mean()) * (Xs - Xs.mean())).sum()
            VyTh = ((Ys - Ys.mean())**2).sum()/((TsTh - TsTh.mean()) * (Ys - Ys.mean())).sum()
            axx.plot([minTs, maxTs], [VxTh, VxTh], label = 'Ground truth, ID = {0}'.format(PointID))
            axy.plot([minTs, maxTs], [VyTh, VyTh], label = 'Ground truth, ID = {0}'.format(PointID))
    axx.legend()
    axx.set_xlabel('t(s)')
    axx.set_ylabel('vx(px/s)')
    axy.set_xlabel('t(s)')
    axy.set_ylabel('vy(px/s)')
    axy.legend()
    return fx, axx, fy, axy

def Plot3dSurface(M, ax = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    X = np.arange(M.shape[0])
    Y = np.arange(M.shape[1])

    X, Y = np.meshgrid(X, Y)

    surf = ax.plot_surface(X, Y, transpose(M), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()
    return surf

def PlotCurrentStreaksMaps(SpeedProjector, ZoneNumber = 0, MaxRatio = 0.3):
    S = SpeedProjector
    
    IniSpeeds = np.array(S.InitialSpeeds)
    SeedsX = np.unique(IniSpeeds[:,0]).tolist()
    SeedsY = np.unique(IniSpeeds[:,1]).tolist()

    f, axs = subplots(len(SeedsX), len(SeedsX))
    OW = S.OWAPT[S.Zones.values()[ZoneNumber][0]]
    f.suptitle("Zone number {0}, at t = {1}\nx : {2} -> {3}\ny : {4} -> {5}".format(ZoneNumber, S._Framework.Tools[S._CreationReferences['Memory']].LastEvent.timestamp, OW[0], OW[2], OW[1], OW[3]))
    for speed_id in S.Zones.values()[ZoneNumber]:
        xIndex = SeedsX.index(S.InitialSpeeds[speed_id][0])
        yIndex = SeedsY.index(S.InitialSpeeds[speed_id][1])
        axs[xIndex, yIndex].imshow(transpose(S.StreaksMaps[speed_id] >= MaxRatio*S.StreaksMaps[speed_id].max()), origin = 'lower')
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

    f, axs = subplots(len(SeedsX), len(SeedsX))
    OW = S.OWAPT[S.Zones.values()[ZoneNumber][0]]
    f.suptitle("Zone number {0}, at t = {1}\nx : {2} -> {3}\ny : {4} -> {5}".format(ZoneNumber, S._Framework.Tools[S._CreationReferences['Memory']].LastEvent.timestamp, OW[0], OW[2], OW[1], OW[3]))
    for speed_id in S.Zones.values()[ZoneNumber]:
        xIndex = SeedsX.index(S.InitialSpeeds[speed_id][0])
        yIndex = SeedsY.index(S.InitialSpeeds[speed_id][1])
        axs[xIndex, yIndex].imshow(transpose(S.DecayingMaps[speed_id]), origin = 'lower')
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
        f, axs = subplots(nSnaps, 1)
    else:
        f, axs = subplots(1, nSnaps)

    Ref = S.MeanPositionsReferences[Speed_id]*S.DensityDefinition
    for n_ax in range(nSnaps):
        axs[n_ax].plot(Ref[0], Ref[1], 'vr', markersize = 10)

        Snap = S.DMSnaps[SnapsIDs[n_ax]][Speed_id]
        axs[n_ax].imshow(transpose(Snap), origin = 'lower')

        Xs, Ys = np.where(Snap > 0)
        Weights = Snap[Xs, Ys]
        Xm, Ym = (Weights*Xs).sum() / Weights.sum(), (Weights*Ys).sum() / Weights.sum()
        axs[n_ax].plot(Xm, Ym, 'v', color = 'grey', markersize = 10)

        tSnap = S.TsSnaps[SnapsIDs[n_ax]]
        axs[n_ax].set_title("t = {0:.2}s".format(tSnap))

def GenerateTrackingGif(F, S, SpeedIDs, Folder = '/home/dardelet/Documents/Papers/STProjection/Pictures/GIFs/AutoGeneratedTracking/', BinDt = 0.005):
    os.system('rm '+ Folder + '*.png')
    print "Generating {0} png frames.".format(len(S.TsSnaps))
    for snap_id in range(len(S.TsSnaps)):
        f = figure()
        imshow(transpose(F.Mem.Snapshots[snap_id][1].max(axis = 2) > F.Mem.Snapshots[snap_id][1].max()-BinDt), origin = 'lower') 
        for speed_id in SpeedIDs:
            Box = [S.OWAPT[speed_id][0] + S.DisplacementSnaps[snap_id][speed_id][0], S.OWAPT[speed_id][1] + S.DisplacementSnaps[snap_id][speed_id][1], S.OWAPT[speed_id][2] + S.DisplacementSnaps[snap_id][speed_id][0], S.OWAPT[speed_id][3] + S.DisplacementSnaps[snap_id][speed_id][1]]
            
            plot([Box[0], Box[0]], [Box[1], Box[3]], 'r')
            plot([Box[0], Box[2]], [Box[3], Box[3]], 'r')
            plot([Box[2], Box[2]], [Box[3], Box[1]], 'r')
            plot([Box[2], Box[0]], [Box[1], Box[1]], 'r')
        f.savefig(Folder + 't_{0:03d}.png'.format(snap_id))
        close(f.number)
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
    elif ans.lower()[0] == 'd' or 'delete' in ans.lower():
        os.system('rm ' + Folder + GifFile)
        print "Deleted Gif file. RIP."
    else:
        os.system('mv ' + Folder + GifFile + ' ' + Folder + 'Meh/' + GifFile)
        print "Moving gif file to Meh folder."
