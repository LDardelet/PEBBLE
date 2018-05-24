import  tools

class Reader:
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Class to read events streams files.
        '''
        self._ReferencesAsked = []
        self._Name = Name
        self._Framework = Framework
        self._Type = 'Input'
        self._CreationReferences = dict(argsCreationReferences)

    def _Initialize(self):
        self.StreamName = self._Framework.StreamHistory[-1]
        if len(self._Framework.StreamHistory) > 1 and self.StreamName == self._Framework.StreamHistory[-2]:
            self.nEvents = -1
            print "Recovered previous stream data."
            return None

        if '.csv' in self.StreamName: # TODO
            print "csv load is  not workin yet"
            self.CurrentStream, self._Framework.StreamsGeometries[self.StreamName] = tools.load_data_csv(self.StreamName)
        elif '.dat' in self.StreamName or '.es' in self.StreamName:
            self.CurrentStream, self._Framework.StreamsGeometries[self.StreamName] = tools.load_data_dat(self.StreamName)
        elif 'Create-' in self.StreamName:
            if 'Bar' in self.StreamName:
                self.CurrentStream, self._Framework.StreamsGeometries[self.StreamName] = tools.CreateMovingBarStream(float(self.StreamName.split('#')[1]), float(self.StreamName.split('#')[2]), float(self.StreamName.split('#')[3]))
            elif 'Circle' in self.StreamName:
                self.CurrentStream, self._Framework.StreamsGeometries[self.StreamName] = tools.CreateMovingCircleStream(float(self.StreamName.split('#')[1]), float(self.StreamName.split('#')[2]), float(self.StreamName.split('#')[3]))
            elif 'Duo' in self.StreamName:
                self.CurrentStream, self._Framework.StreamsGeometries[self.StreamName] = tools.CreateDuoCirclesStream(float(self.StreamName.split('#')[1]), float(self.StreamName.split('#')[2]), float(self.StreamName.split('#')[3]), float(self.StreamName.split('#')[4]), float(self.StreamName.split('#')[5]))
        else:
            print "No valid loading function found for this type of stream. Initiating empty stream {0}.".format(self.StreamName)
            self.CurrentStream = []
            self._Framework.StreamsGeometries[self.StreamName] = (1,1,2)

        self.nEvents = -1
        self.NEvents = len(self.CurrentStream)
        if len(self.CurrentStream) > 0:
            print "Loaded stream {0}, containing {1} events, from t = {2} to t = {3}".format(self.StreamName, self.NEvents, self.CurrentStream[0].timestamp, self.CurrentStream[-1].timestamp)

    def _OnEvent(self, event):
        try:
            self.nEvents += 1
            return self.CurrentStream[self.nEvents]
        except IndexError:
            self._Framework.Running = False
            print "Input reached EOF."
