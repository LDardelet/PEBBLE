import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm

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
        axs[xIndex, yIndex].tick_params('both', left = 'off', bottom = 'off', labelleft = 'off', labelbottom = 'off')
    return f, axs
