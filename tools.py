from event import Event

import numpy as np
import random
import pprofile
import sys
import pickle
import cPickle
import socket
from struct import unpack, pack

def IsDisplayUp(questionPort = 54243, responsePort = 54244, address = "localhost"):
    ResponseUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    listen_addr = ("",responsePort)
    ResponseUDP.bind(listen_addr)
    id_random = random.randint(100000,200000)
    QuestionDict = {'id': id_random, 'command':'isup'}
    QuestionUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    QuestionAddress = (address, questionPort)
    QuestionUDP.sendto(cPickle.dumps(QuestionDict),QuestionAddress)
    QuestionUDP.close()
    ResponseUDP.settimeout(1.)
    try:
        data_raw, addr = ResponseUDP.recvfrom(1064)
        data = cPickle.loads(data_raw)
        if data['id'] == id_random and data['answer']:
            return True
        else:
            return False
    except:
        print "No answer, display is down"
        return False

def KillDisplay(mainPort = 54242, questionPort = 54243, address = "localhost"):
    Question = "kill#"
    QuestionUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    QuestionAddress = (address, questionPort)
    QuestionUDP.sendto(Question,QuestionAddress)
    QuestionUDP.close()

    Main = "kill#"
    MainUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    MainAddress = (address, mainPort)
    MainUDP.sendto(Main,MainAddress)
    MainUDP.close()

def DestroySocket(Socket, questionPort = 54243, responsePort = 54244, address = "localhost"):
    if Socket == None:
        return None
    ResponseUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listen_addr = ("",responsePort)
    ResponseUDP.bind(listen_addr)

    id_random = random.randint(100000,200000)
    QuestionDict = {'id': id_random, 'socket': Socket, 'command':'destroysocket'}
    QuestionUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    QuestionAddress = (address, questionPort)
    QuestionUDP.sendto(cPickle.dumps(QuestionDict),QuestionAddress)
    QuestionUDP.close()
    ResponseUDP.settimeout(1.)

    try:
        data_raw, addr = ResponseUDP.recvfrom(1064)
        data = cPickle.loads(data_raw)
        if data['id'] == id_random and data['answer'] == 'socketdestroyed':
            print "Destroyed socket {0}".format(Socket)
        else:
            print "Could not destroy socket {0}".format(Socket)
            return None
    except:
        print "Display seems down (DestroySocket)"
    ResponseUDP.close()

def GetDisplaySocket(Shape, Socket = None, questionPort = 54243, responsePort = 54244, address = "localhost"):
    ResponseUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listen_addr = ("",responsePort)
    ResponseUDP.bind(listen_addr)

    id_random = random.randint(100000,200000)
    QuestionDict = {'id': id_random, 'shape':Shape}
    if Socket == None:
        QuestionDict['command'] = "asksocket"
    else:
        QuestionDict['command'] = "askspecificsocket"
        QuestionDict['socket'] = Socket
    QuestionUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    QuestionAddress = (address, questionPort)
    QuestionUDP.sendto(cPickle.dumps(QuestionDict),QuestionAddress)
    QuestionUDP.close()
    ResponseUDP.settimeout(1.)

    try:
        data_raw, addr = ResponseUDP.recvfrom(1064)
        data = cPickle.loads(data_raw)
        if data['id'] == id_random:
            if data['answer'] == 'socketexists':
                print "Socket refused"
                return None
            else: 
                Socket = data['answer']
                print "Got socket {0}".format(Socket)
                return Socket
        else:
            print "Socket refused"
            return None
    except:
        print "Display seems down (GetDisplaySocket)"
    ResponseUDP.close()

def CleanMapForStream(Socket, questionPort = 54243, responsePort = 54244, address = "localhost"):
    ResponseUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listen_addr = ("",responsePort)
    ResponseUDP.bind(listen_addr)

    id_random = random.randint(100000,200000)
    QuestionDict = {'id': id_random, 'socket': Socket, 'command':'cleansocket'}
    QuestionUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    QuestionAddress = (address, questionPort)
    QuestionUDP.sendto(cPickle.dumps(QuestionDict),QuestionAddress)
    QuestionUDP.close()
    ResponseUDP.settimeout(1.)

    try:
        data_raw, addr = ResponseUDP.recvfrom(1064)
        data = cPickle.loads(data_raw)
        if data['id'] == id_random and data['answer'] == 'socketcleansed':
            print "Cleansed"
        else:
            print "Could not clean map"
    except:
        print "Display seems down (CleanMapForStream)"
    ResponseUDP.close()

def SendStreamData(Infos1, Infos2, Socket, questionPort = 54243, responsePort = 54244, address = "localhost"):
    ResponseUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listen_addr = ("",responsePort)
    ResponseUDP.bind(listen_addr)

    id_random = random.randint(100000,200000)
    QuestionDict = {'id': id_random, 'socket': Socket, 'infosline1': Infos1, 'infosline2': Infos2, 'command':'socketdata'}
    QuestionUDP = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    QuestionAddress = (address, questionPort)
    QuestionUDP.sendto(cPickle.dumps(QuestionDict),QuestionAddress)
    QuestionUDP.close()
    ResponseUDP.settimeout(1.)

    try:
        data_raw, addr = ResponseUDP.recvfrom(1064)
        data = cPickle.loads(data_raw)
        if data['id'] == id_random and data['answer'] == 'datareceived':
            print "Data transmitted"
        else:
            print "Could not transmit data"
    except:
        print "Display seems down (SendStreamData)"
    ResponseUDP.close()


def SendEvent(ev, Connexion, Address, Socket):
    ev.socket = Socket
    data = pickle.dumps(ev)
    Connexion.sendto(data, Address)

def SendSegment(segment, Connexion, Address, Socket):
    segment.socket = Socket
    data = pickle.dumps(segment)
    Connexion.sendto(data, Address)
    
# END OF DISPLAY TOOLS #####################################################################

# START OF LOADING TOOLS ###################################################################

def CreateMovingCircleStream(Speed, angle, gitter = 0., screen_size = [200, 200], circle_radius = 20):
    center_position = np.array(screen_size)/2
    TimeStamps = np.inf*np.ones(screen_size+[2])
    Stream = []

    u = np.array([-np.sin(angle), np.cos(angle)])
    v = np.array([np.cos(angle), np.sin(angle)])
    print "Creating moving circle with speed vx = {0} and vy = {1}".format(Speed*np.cos(angle), Speed*np.sin(angle))
    P = center_position - v*(center_position.min() - circle_radius - 5)

    for x in range(screen_size[0]):
        for y in range(screen_size[1]):
            X = np.array([x,y])
            l = ((X - P)*u).sum()
            if abs(l) < circle_radius:
                delta = np.sqrt(circle_radius**2 - l ** 2)
                dP_on = ((X - P)*v).sum() - delta
                if dP_on > 0:
                    if gitter != 0:
                        error = random.expovariate(1./(gitter*(1./Speed)))
                    else:
                        error = 0
                    TimeStamps[x,y,0] = dP_on/Speed + error
                dP_off = ((X - P)*v).sum() + delta
                if dP_off > 0:
                    if gitter != 0:
                        error = random.expovariate(1./(gitter*(1./Speed)))
                    else:
                        error = 0
                    TimeStamps[x,y,1] = dP_off/Speed + error
    while TimeStamps.min() < np.inf:
        Xs, Ys, Ps = np.where(TimeStamps == TimeStamps.min())
        for i in range(Xs.shape[0]):
            Stream += [Event(TimeStamps[Xs[i], Ys[i], Ps[i]], np.array([Xs[i], Ys[i]]), Ps[i])]
            TimeStamps[Xs[i], Ys[i], Ps[i]] = np.inf
    return Stream, tuple(screen_size + [2])

def CreateDuoCirclesStream(Speed1, angle1, Speed2, angle2, gitter = 0., screen_size = [200, 200], circle_radius = [20,20]):
    center_position = np.array(screen_size)/2
    Stream = []
    TimeStamps = np.inf*np.ones(screen_size+[2]+[2])

    u1 = np.array([-np.sin(angle1), np.cos(angle1)])
    v1 = np.array([np.cos(angle1), np.sin(angle1)])
    P1 = center_position - v1*(center_position.min() - circle_radius[0] - 5)
    u2 = np.array([-np.sin(angle2), np.cos(angle2)])
    v2 = np.array([np.cos(angle2), np.sin(angle2)])
    P2 = center_position - v2*(center_position.min() - circle_radius[1] - 5)
    print "Creating circles duo, moving at vx_1 = {0} and vy_1 = {1}, and vx_2 = {2} and vy_2 = {3}.".format(Speed1*np.cos(angle1), Speed1*np.sin(angle1), Speed2*np.cos(angle2), Speed2*np.sin(angle2))

    for x in range(screen_size[0]):
        for y in range(screen_size[1]):
            X = np.array([x,y])
            l1 = ((X - P1)*u1).sum()
            l2 = ((X - P2)*u2).sum()
            if abs(l1) < circle_radius[0]:
                delta1 = np.sqrt(circle_radius[0]**2 - l1 ** 2)
                dP_on1 = ((X - P1)*v1).sum() - delta1
                if dP_on1 > 0:
                    if gitter != 0:
                        error = random.expovariate(1./(gitter*(1./Speed1)))
                    else:
                        error = 0
                    TimeStamps[x,y,0,0] = dP_on1/Speed1 + error
                dP_off1 = ((X - P1)*v1).sum() + delta1
                if dP_off1 > 0:
                    if gitter != 0:
                        error = random.expovariate(1./(gitter*(1./Speed1)))
                    else:
                        error = 0
                    TimeStamps[x,y,1,0] = dP_off1/Speed1 + error
            if abs(l2) < circle_radius[1]:
                delta2 = np.sqrt(circle_radius[1]**2 - l2 ** 2)
                dP_on2 = ((X - P2)*v2).sum() - delta2
                if dP_on2 > 0:
                    th_ts = dP_on2/Speed2 # Now we check that the correcponding point is not behind the other circle
                    P1_at_ts = P1 + v1 * Speed1 * th_ts
                    if np.linalg.norm(X - P1_at_ts) > circle_radius[0]:
                        if gitter != 0:
                            error = random.expovariate(1./(gitter*(1./Speed2)))
                        else:
                            error = 0
                        TimeStamps[x,y,0,1] = dP_on2/Speed2 + error
                dP_off2 = ((X - P2)*v2).sum() + delta2
                if dP_off2 > 0:
                    th_ts = dP_off2/Speed2
                    P1_at_ts = P1 + v1 * Speed1 * th_ts
                    if np.linalg.norm(X - P1_at_ts) > circle_radius[0]:
                        if gitter != 0:
                            error = random.expovariate(1./(gitter*(1./Speed2)))
                        else:
                            error = 0
                        TimeStamps[x,y,1,1] = dP_off2/Speed2 + error

    while TimeStamps.min() < np.inf:
        Xs, Ys, Ps, Circle = np.where(TimeStamps == TimeStamps.min())
        for i in range(Xs.shape[0]):
            Stream += [Event(TimeStamps[Xs[i], Ys[i], Ps[i], Circle[i]], np.array([Xs[i], Ys[i]]), Ps[i])]
            TimeStamps[Xs[i], Ys[i], Ps[i], Circle[i]] = np.inf
    return Stream, tuple(screen_size + [2])


def CreateMovingBarStream(Speed, angle, gitter = 0., screen_size = [200, 200], bar_size = [10, 40]):
#Gitter is given as a ratio of the time between two spiking pixels
    
    center_position = np.array(screen_size)/2
    TimeStamps = np.inf*np.ones(screen_size+[2])
    Stream = []

    u = np.array([-np.sin(angle), np.cos(angle)])
    v = np.array([np.cos(angle), np.sin(angle)])
    print "Creating moving bar with speed vx = {0} and vy = {1}".format(Speed*np.cos(angle), Speed*np.sin(angle))
    
    P = center_position - v*(center_position.min() - bar_size[0] - 5)
    P = center_position - v*np.linalg.norm(center_position)/1.5

    P_on = P + bar_size[0]*v
    P_off = P - bar_size[0]*v

    for x in range(screen_size[0]):
        for y in range(screen_size[1]):
            X = np.array([x,y])
            if abs(((P - X)*u).sum()) <= bar_size[1]/2:
                if ((X - P_on)*v).sum() > 0:
                    if gitter != 0:
                        error = random.expovariate(1./(gitter*(1./Speed)))
                    else:
                        error = 0
                    TimeStamps[x,y,0] = ((X - P_on)*v).sum()/Speed + error
                if ((X - P_off)*v).sum() > 0:
                    if gitter != 0:
                        error = random.expovariate(1./(gitter*(1./Speed)))
                    else:
                        error = 0
                    TimeStamps[x,y,1] = ((X - P_off)*v).sum()/Speed + error

    while TimeStamps.min() < np.inf:
        Xs, Ys, Ps = np.where(TimeStamps == TimeStamps.min())
        for i in range(Xs.shape[0]):
            Stream += [Event(TimeStamps[Xs[i], Ys[i], Ps[i]], np.array([Xs[i], Ys[i]]), Ps[i])]
            TimeStamps[Xs[i], Ys[i], Ps[i]] = np.inf
    return Stream, tuple(screen_size + [2])

def load_data_dat(filename, y_invert = True, header_size = 42):
    if '.dat' in filename or '.es' in filename:
        all_ts, coords, all_p, rem = readATIS_td(filename)
        all_x = coords[:,0]
        all_y = coords[:,1]
        del coords
    else:
        f = open(filename, 'rb')
        raw_data = np.fromfile(f, dtype=np.uint8)[header_size:]
        f.close()
        raw_data = np.uint32(raw_data)
        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7
        all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    print "Read file. Seems to contain {0} events until ts = {1}s".format(all_x.shape[0], all_ts[-2]*10**-6)
    print ""
    print "Geometry :"
    xMin = all_x.min()
    xMax = all_x.max()
    yMin = all_y.min()
    yMax = all_y.max()
    pMin = all_p.min()
    pMax = all_p.max()
    print "x goes from {0} to {1}".format(xMin, xMax)
    print "y goes from {0} to {1}".format(yMin, yMax)
    Events = []
    for nEvent in range(all_x.shape[0]):
    #for x,y,p,ts in zip(all_x, all_y, all_p, all_ts):
        x,y,p,ts = all_x[nEvent], all_y[nEvent], all_p[nEvent], all_ts[nEvent]
        if y_invert:
            y_final = yMax - y
        else:
            y_final = y - yMin
        x_final = x - xMin

        t = ts*(10**-6)
        
        X = int(x_final)
        Y = int(y_final)
        
        Events += [Event(float(t), np.array([X,Y]), int(p))]
        if nEvent%100000 == 0:
            sys.stdout.write("> "+'t = '+str(Events[-1].timestamp)+", n = "+str(nEvent)+"\r")
            sys.stdout.flush()
    print ""
    return Events, (xMax - xMin + 1, yMax - yMin + 1, pMax - pMin + 1)

def readATIS_td(file_name, orig_at_zero = True, drop_negative_dt = True, verbose = True, events_restriction = [0, np.inf]):
    if '.v2' in file_name:
        print "V2 detected"
        polmask = 0x1000000000000000
        xmask = 0x00003FFF00000000
        ymask = 0x0FFFC00000000000
        polpadding = 60
        ypadding = 46
        xpadding = 32
    else:
        polmask = 0x0002000000000000
        xmask = 0x000001FF00000000
        ymask = 0x0001FE0000000000
        polpadding = 49
        ypadding = 41
        xpadding = 32

    # This one read _td.dat files generated by kAER
    if verbose:
        print('Reading _td dat file... (' + file_name + ')')
    file = open(file_name,'rb')

    header = False
    while peek(file) == b'%':
        file.readline()
        header = True
    if header:
        ev_type = unpack('B',file.read(1))[0]
        ev_size = unpack('B',file.read(1))[0]
        if verbose:
            print('> Header exists. Event type is ' + str(ev_type) + ', event size is ' + str(ev_size))
        if ev_size != 8:
            print('Wrong event size. Aborting.')
            return -1, -1, -1, -1
    else: # set default ev type and size
        if verbose:
            print('> No header. Setting default event type and size.')
        ev_size = 8
        ev_type = 0

    # Compute number of events in the file
    start = file.tell()
    file.seek(0,2)
    stop = file.tell()
    file.seek(start)

    Nevents = int( (stop-start)/ev_size )
    dNEvents = Nevents/100
    if verbose:
        print("> The file contains %d events." %Nevents)

    # store read data
    timestamps = np.zeros(Nevents, dtype = int)
    polarities = np.zeros(Nevents, dtype = int)
    coords = np.zeros((Nevents, 2), dtype = int)

    ActualEvents = 0
    for i in np.arange(0, int(Nevents)):

        event = unpack('Q',file.read(8))
        ts = event[0] & 0x00000000FFFFFFFF
        # padding = event[0] & 0xFFFC000000000000
        pol = (event[0] & polmask) >> polpadding
        y = (event[0] & ymask) >> ypadding
        x = (event[0] & xmask) >> xpadding
        if i >= events_restriction[0] and ts>timestamps[max(0,i-1)]:
            ActualEvents += 1
            timestamps[i] = ts
            polarities[i] = pol
            coords[i, 0] = x
            coords[i, 1] = y

        if verbose and i%dNEvents == 0:
            sys.stdout.write("> "+str(i/dNEvents)+"% \r")
            sys.stdout.flush()
        if i > events_restriction[1]:
            break
    file.close()
    print "> After loading events, actually found {0} events.".format(ActualEvents)

    timestamps = timestamps[:ActualEvents]
    coords = coords[:ActualEvents, :]
    polarities = polarities[:ActualEvents]

    #check for negative timestamps
    for ts in timestamps:
        if ts < 0:
            print('prout negative delta-ts')

    if orig_at_zero:
        timestamps = timestamps - timestamps[0]

    drop_sum = 0
    if drop_negative_dt:
        if verbose:
            print('> Looking for negative dts...')
        # first check if negative TS differences
        just_dropped = True
        nPasses = 0
        while just_dropped:
            nPasses += 1
            index_neg = []
            just_dropped = False
            ii = 0
            dNEvents = timestamps.size / 100
            while ii < (timestamps.size - 1):
                dt = timestamps[ii+1] - timestamps[ii]
                if dt <= 0:  # alors ts en ii+1 plus petit que ii
                    index_neg += [ii+1]
                    ii += 1
                    just_dropped = True
                if verbose and ii%dNEvents == 0:
                    sys.stdout.write("> "+str(ii/dNEvents)+"% (pass "+str(nPasses)+") \r")
                    sys.stdout.flush()
                ii += 1
            if len(index_neg) > 0:
                drop_sum += len(index_neg)
                index_neg = np.array(index_neg)
                timestamps = np.delete(timestamps, index_neg)
                polarities = np.delete(polarities, index_neg)
                coords = np.delete(coords, index_neg, axis = 0)
                if verbose:
                    print('> Removed {0} events in {1} passes.'.format(drop_sum, nPasses))
        removed_events = drop_sum
    else:
        removed_events = -1
    if verbose:
        print("> Sequence duration: {0:.2f}s, ts[0] = {1}, ts[{2}] = {3}.".format(float(timestamps[-1] - timestamps[0]) / 1e6, timestamps[0], len(timestamps)-1, timestamps[-1]))


    return timestamps, coords, polarities, removed_events


def load_data_csv(filename, geometry_restriction, listeningPolas = [0,1], eventType = [0], time_restriction = [1., np.inf]):
    Events = []

    with open(filename) as csvfile:
        for raw_line in csvfile:
            line = raw_line.split(',')
            if int(line[3].strip('\t').strip('\n')) in eventType:
                P = int(line[4].strip('\t').strip('\n'))
                if P in listeningPolas:
                    t = float(line[2].strip('\t').strip('\n'))*(10**-6)
                    if time_restriction[0] <= t <= time_restriction[1]:
                        X = int(line[0].strip('\t').strip('\n'))
                        if geometry_restriction[0][0] <= X <= geometry_restriction[0][1]:
                            Y = int(line[1].strip('\t').strip('\n'))
                            if geometry_restriction[1][0] <= Y <= geometry_restriction[1][1]:
                                Events += [Event(t, np.array([X-geometry_restriction[0][0],Y-geometry_restriction[1][0]]), P)]
                                if random.random() < 0.001:
                                    sys.stdout.write(str(Events[-1].timestamp) + " \r")
                                    sys.stdout.flush()
    print ""
    return Events
    
# END OF LOADING TOOLS #####################################################################

def StatFunction(function, *args):
    prof = pprofile.Profile()

    with prof():
        try:
            function(*args)
        except KeyboardInterrupt:
            print "Function ended, writing stats in file"
    
    orig_stdout = sys.stdout
    f = file('stats.txt', 'w')
    sys.stdout = f
    prof.print_stats()
    sys.stdout = orig_stdout
    f.close()

def peek(f, length=1):
    pos = f.tell()
    data = f.read(length)
    f.seek(pos)
    return data

def RandomStream(geometry = None, vmax = 80., gitterrange = 1.):
    if geometry is None:
        geometry = random.choice(['Circle', 'Bar', 'Duo'])
    speednorm = random.random()*vmax
    angle = random.random()*2*np.pi
    gitter = random.random()*gitterrange
    StreamName = 'Create-{0}#{1:.1f}#{2:.3f}'.format(geometry, speednorm, angle)
    if geometry == 'Duo':
        StreamName += '#{0:.1f}#{1:.3f}'.format(random.random()*vmax, random.random()*2*np.pi)
    StreamName += '#{0:.1f}'.format(gitter)
    return StreamName


def CompareStreamsOld(S1, S2, geometry, deltaTMax):
	TS1 = 0
	TS2 = 0
	
	Sum1 = 0
	Sum2 = 0
	
	TSList1 = [[[[] for p in range(geometry[2])] for y in range(geometry[1])] for x in range(geometry[0])]
	TSList2 = [[[[] for p in range(geometry[2])] for y in range(geometry[1])] for x in range(geometry[0])]
	
	for ev1 in S1:
		TSList1[ev1.location[0]][ev1.location[1]][ev1.polarity] += [ev1.timestamp]
	for ev2 in S2:
		TSList2[ev2.location[0]][ev2.location[1]][ev2.polarity] += [ev2.timestamp]
	
	for x in range(geometry[0]):
		sys.stdout.write("Treating line {0}/{1} \r".format(x, geometry[0]-1))
		sys.stdout.flush()
		for y in range(geometry[1]):
			for p in range(geometry[2]):
				Pixel1 = list(TSList1[x][y][p])
				Pixel2 = list(TSList2[x][y][p])
				L1 = len(Pixel1)
				L2 = len(Pixel2)
				if L1 != 0 and L2 != 0:
					nts = 0
					for ts in Pixel1:
						nts += 1
						diff2 = abs(np.array(Pixel2) - ts)
						index2 = diff2.argmin()
						if diff2[index2] > deltaTMax:
							Sum1 += deltaTMax
						else:
							Sum1 += diff2[index2]
							Pixel2.pop(index2)
						if len(Pixel2) == 0:
							Sum1 += (L1-nts)*deltaTMax
							break
				
					Pixel1 = list(TSList1[x][y][p])
					Pixel2 = list(TSList2[x][y][p])
				
					nts = 0
					for ts in Pixel2:
						nts += 1
						diff1 = abs(np.array(Pixel1) - ts)
						index1 = diff1.argmin()
						if diff1[index1] > deltaTMax:
							Sum2 += deltaTMax
						else:
							Sum2 += diff1[index1]
							Pixel1.pop(index1)
						if len(Pixel1) == 0:
							Sum2 += (L2-nts)*deltaTMax
							break
				else:
					Sum1 += L1*deltaTMax
					Sum2 += L2*deltaTMax
	print ""
	print "Sum1 : {0}".format(Sum1)
	print "Sum2 : {0}".format(Sum2)
	return (Sum1+Sum2)/2

def CompareStreams(S1, S2, geometry, DeltaPixel, DeltaT):
    Map1 = [[[[] for p in range(geometry[2])] for y in range(geometry[1] + 2*DeltaPixel)] for x in range(geometry[0] + 2*DeltaPixel)]
    Map2 = [[[[] for p in range(geometry[2])] for y in range(geometry[1] + 2*DeltaPixel)] for x in range(geometry[0] + 2*DeltaPixel)]
    # Generate all the volumes for each pixels by merging close events, for each stream.
    print "Empty maps generated"
    I1 = 0
    I2 = 0
    n = 0
    l = len(S1)
    for e1 in S1:
        n+= 1
        sys.stdout.write("Treating event {0}/{1} \r".format(n, l))
        sys.stdout.flush()
        if not np.isnan(e1.timestamp) and not np.isinf(e1.timestamp):
            for dx in range(-DeltaPixel, DeltaPixel+1):
                nX = e1.location[0] + dx + DeltaPixel
                for dy in range(-DeltaPixel, DeltaPixel+1):
                    nY = e1.location[1] + dy + DeltaPixel
                    PixelStream = Map1[nX][nY][e1.polarity]
                    if len(PixelStream) == 0 or PixelStream[-1][1] < e1.timestamp - DeltaT:
                        PixelStream += [[e1.timestamp - DeltaT, e1.timestamp + DeltaT]]
                        I1 += 2*DeltaT
                    else:
                        I1 += e1.timestamp + DeltaT - PixelStream[-1][1]
                        PixelStream[-1][1] = e1.timestamp + DeltaT
    print "Volumes generated for Stream 1"
    n = 0
    l = len(S2)
    for e2 in S2:
        n+= 1
        sys.stdout.write("Treating event {0}/{1} \r".format(n, l))
        sys.stdout.flush()
        if not np.isnan(e2.timestamp) and not np.isinf(e2.timestamp):
            for dx in range(-DeltaPixel, DeltaPixel+1):
                nX = e2.location[0] + dx + DeltaPixel
                for dy in range(-DeltaPixel, DeltaPixel+1):
                    nY = e2.location[1] + dy + DeltaPixel
                    PixelStream = Map2[nX][nY][e2.polarity]
                    if len(PixelStream) == 0 or PixelStream[-1][1] < e2.timestamp - DeltaT:
                        PixelStream += [[e2.timestamp - DeltaT, e2.timestamp + DeltaT]]
                        I2 += 2*DeltaT
                    else:
                        I2 += e2.timestamp + DeltaT - PixelStream[-1][1]
                        PixelStream[-1][1] = e2.timestamp + DeltaT
    print "Volumes generated for Stream 2"
    
    # Now we actually compare the streams through the previously define volumes per pixel
    IInter = 0
    for nX in range(geometry[0] + 2*DeltaPixel):
        sys.stdout.write("Treating line {0}/{1} \r".format(nX, geometry[0] + 2*DeltaPixel))
        sys.stdout.flush()
        for nY in range(geometry[1] + 2*DeltaPixel):
            for p in range(2):
                PixelStreams = [Map1[nX][nY][p], Map2[nX][nY][p]]
                n = [0, -1]
                if len(PixelStreams[0]) > 0 and len(PixelStreams[1]) > 0:
                    e = [PixelStreams[0][0], PixelStreams[1][0]]
                    latest = 1
                    while n[latest] + 1 < len(PixelStreams[latest]):
                        n[latest] += 1
                        e[latest] = PixelStreams[latest][n[latest]]
                        earray= np.array(e)
                        latest = earray[:,1].argmin()
                        if e[1-latest][0] < e[latest][1]:
                            IInter += max(0, earray[:,1].min() - earray[:,0].max())
    print "Volumes intersected"
    return I1, I2, IInter

def ComputeSegmentsDistances(P1, P2):
    u1 = P1[1,:] - P1[0,:]
    AB1 = np.linalg.norm(u1)
    u1 = u1/AB1
    u2 = P2[1,:] - P2[0,:]
    AB2 = np.linalg.norm(u2)
    u2 = u2/AB2

    DeltasAB = P2 - P1

    if (DeltasAB.prod(axis = 0).sum(axis = 0) - (u2*DeltasAB[0,:]).sum()*(u2*DeltasAB[1,:]).sum()) <= 0 and  (DeltasAB.prod(axis = 0).sum(axis = 0) - (u2*DeltasAB[0,:]).sum()*(u2*DeltasAB[1,:]).sum()) <= 0:
        return 0
    Lambdas = []
    LambdasRestricted = []
    Distances = []

    X = P1[0,:]
    Lambdas += [-((P2[0,:]-X)*u2).sum()]
    LambdasRestricted += [max(0, min(AB2, Lambdas[-1]))]
    Distances += [np.linalg.norm(X - (P2[0,:] + LambdasRestricted[-1]*u2))]
    X = P1[1,:]
    Lambdas += [-((P2[0,:]-X)*u2).sum()]
    LambdasRestricted += [max(0, min(AB2, Lambdas[-1]))]
    Distances += [np.linalg.norm(X - (P2[0,:] + LambdasRestricted[-1]*u2))]
    X = P2[0,:]
    Lambdas += [-((P1[0,:]-X)*u1).sum()]
    LambdasRestricted += [max(0, min(AB1, Lambdas[-1]))]
    Distances += [np.linalg.norm(X - (P1[0,:] + LambdasRestricted[-1]*u1))]
    X = P2[1,:]
    Lambdas += [-((P1[0,:]-X)*u1).sum()]
    LambdasRestricted += [max(0, min(AB1, Lambdas[-1]))]
    Distances += [np.linalg.norm(X - (P1[0,:] + LambdasRestricted[-1]*u1))]

    return min(Distances)

############## Part for TSs rotations and rotationnal invariant TS matching

def FlowComputation(Patch, TauForget):
    t_max = Patch.max()
    Positions = np.where(Patch > t_max-TauForget)
    
    if Positions[0].shape[0] > 2:
        Ts = Patch[Positions]
        #if (Ts.max()-Ts.min()) < 0.75*TauForget:
        #    return None, Positions[0].shape[0], 0, 0
        Xm = Positions[0] - Positions[0].mean()
        Ym = Positions[1] - Positions[1].mean()
        Tm = Ts - Ts.mean()
        Sx2 = (Xm **2).sum()
        Sy2 = (Ym **2).sum()
        Sxy = (Xm*Ym).sum()
        Stx = (Tm*Xm).sum()
        Sty = (Tm*Ym).sum()
        D = Sx2*Sy2 - Sxy**2
        
        if D > 0:
            F = np.array([Sy2*Stx-Sxy*Sty, Sx2*Sty-Sxy*Stx, -D])/D
            Points = np.array([Xm, Ym, Tm])
            Distances = (((np.transpose(Points)*F).sum(axis = 1))**2).sum()/np.linalg.norm(F)
            return F[:2], Positions[0].shape[0], D, Distances, Ts.min()
        else:
            return None, Positions[0].shape[0], 0, 0, 0
    else:
        return None, Positions[0].shape[0], 0, 0, 0

def PatchExtraction(STContext, location, Radius):
    return STContext[max(0,location[0]-Radius):location[0]+Radius+1, max(0,location[1]-Radius):location[1]+Radius+1]

def RotationMatrix(angle):
    s = np.sin(angle)
    c = np.cos(angle)
    return np.array([[c, -s], [s, c]])

def ExtractCenterVectors(PatchRadius):
    Center = np.array([[PatchRadius, PatchRadius]])
    Vectors = []
    for x in range(2*PatchRadius+1):
        for y in range(2*PatchRadius+1):
            Vectors += [[x,y]]
    Vectors = np.array(Vectors)
    Vectors -= Center
    return Vectors, Center

def RecoverCenters(RotatedVectors):
    AddedVector = np.array([[0.5,0.5]])
    RotatedVectors += AddedVector
    return np.array(np.floor(RotatedVectors), dtype = int)

def RotateTS(OversizedTS, aimedRadius, angle):
    if OversizedTS.shape[0] < int((2*aimedRadius+1)*1.414):
        print "Insufficient Oversized TS. Initial Radius is {0}, asked radius is {1}".format(OversizedTS.shape[0], aimedRadius)
    NewMatrix = np.zeros((2*aimedRadius+1, 2*aimedRadius+1, OversizedTS.shape[2]))
    R = RotationMatrix(-angle) # -angle since we wonder what is the origin of the new matrix pixels, rather than asking where are the old pixels in the new matrix
    OldCenter = np.array([[(OversizedTS.shape[0]-1)/2, (OversizedTS.shape[0]-1)/2]])

    Vectors, NewCenter = ExtractCenterVectors(aimedRadius)
    RotatedVectors = R.dot(Vectors.T).T
    OldTSCenters = RecoverCenters(RotatedVectors)
    OldTSCenters += OldCenter
    Vectors += NewCenter

    NewMatrix[Vectors[:,0], Vectors[:,1], :] = OversizedTS[OldTSCenters[:,0], OldTSCenters[:,1], :]
    return NewMatrix

def RotateTSWithFlow(OversizedTS, aimedRadius, F):
    angle = -np.arccos(F[0]/np.linalg.norm(F))*np.sign(F[1])
    return RotateTS(OversizedTS, aimedRadius, angle), angle
