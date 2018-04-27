import  tools

class Reader:
    def __init__(self, argsCreationDict):
        '''
        Class to read events streams files.
        Expects :
        'Framework.Self' as 'Framework' -> Access the different Streams variables
        '''
        self._Type = 'Input'

        for key in argsCreationDict.keys():
            self.__dict__[key.split('.')[2]] = argsCreationDict[key]

    def _Initialize(self, argsInitializationDict):
        '''
        Expects:
        '''
        StreamName = self.Framework.StreamHistory[-1]
        if '.csv' in StreamName: # TODO
            print "csv load is  not workin yet"
            self.Framework.Streams[StreamName], self.Framework.StreamsGeometries[StreamName] = tools.load_data_csv(StreamName)
        elif '.dat' in StreamName:
            self.Framework.Streams[StreamName], self.Framework.StreamsGeometries[StreamName] = tools.load_data_dat(StreamName)
        elif 'Create-' in StreamName:
            if 'Bar' in StreamName:
                self.Framework.Streams[StreamName], self.Framework.StreamsGeometries[StreamName] = tools.CreateMovingBarStream(float(StreamName.split('#')[1]), float(StreamName.split('#')[2]), float(StreamName.split('#')[3]))
            elif 'Circle' in StreamName:
                self.Framework.Streams[StreamName], self.Framework.StreamsGeometries[StreamName] = tools.CreateMovingCircleStream(float(StreamName.split('#')[1]), float(StreamName.split('#')[2]), float(StreamName.split('#')[3]))
        else:
            print "No valid loading function found for this type of stream. Aborting."

        self.Framework.nEvents[StreamName] = 0
        self.Framework.NEvents = len(self.Framework.Streams[StreamName])

        print "Loaded stream {0}, containing {1} events, from t = {2} to t = {3}".format(StreamName, self.Framework.NEvents, self.Framework.Streams[StreamName][0].timestamp, self.Framework.Streams[StreamName][-1].timestamp)
