import types
import numpy as np
import copy
import matplotlib.pyplot as plt

class ModuleBase:
    def __init__(self, Name, Framework, ModulesLinked):
        '''
        Default module class.
        Each module in the Framework should inherit this class. More information is given in ModuleTemplate.py
        '''
        self.__ModulesLinksRequested__ = []

        self._NeedsLogColumn = False

        self.__IsInput__ = False
        self.__GeneratesSubStream__ = False

        self._MonitoredVariables = []
        self._MonitorDt = 0

        ####

        self.__Framework__ = Framework
        self.__Name__ = Name

        self.Log("Generating module")
        self.__ModulesLinked__ = dict(ModulesLinked)
        self.__Initialized__ = False
        self.__SavedValues__ = {}
        self.__SubStreamInputIndexes__ = set()
        self.__SubStreamOutputIndexes__ = set()
        self.__GeneratedSubStreams__ = []
        
        self.__ProposesTau = ('_OnTauRequest' in self.__class__.__dict__) # We set this as a variable, for user to change it at runtime if need be
        
        try:
            self.__ModuleIndex__ = self.__Framework__.ModulesOrder[self.__Name__]
        except:
            None

        self._OnCreation()

    # All module method the framework interacts with and that can be changed start with "_On" :

    def _OnInputIndexesSet(self, Indexes):
        '''
        Specific method for input modules.
        Upon calling when framework is created, will allow the module to set its variables accordingly to what was specified in the project file.
        Cannot create a framework with an input module that has not its proper method.
        Be careful to have variables that wont be overwritten during initialization of the framework
        '''
        return False
    def _OnCreation(self):
        # Main method for defining parameters of the module, monitored variables, etc...
        # It must be formatted following the ModuleTemplate.py model, for the creation of the module to work smoothly. All predefined lines can be either modified, or removed
        pass
    def _OnInitialization(self):
        # Template for user-filled module initialization
        # Should return True if everything went normal
        return True
    def _OnEventModule(self, event):
        # Template for user-filled module event running method
        # Should not return anything, as the Framework takes care of events handling
        pass
    def _OnTauRequest(self, event = None):
        # User-filled method to propose a tau for the whole framework. Ideally, the further in the framework, the higher the information level is and the more accurate Tau information is
        # Return None for default, or 0 for non-defined tau yet
        pass
    def _OnSnapModule(self):
        # Template for user-filled module preparation for taking a snapshot of the different variables. 
        pass
    def _OnClosing(self):
        # Template method to be called when python closes. 
        # Used for closing any files or connexions that could have been made. Use that rather than another atexit call
        pass
    def _OnWarp(self, t):
        # Template method for input modules to accelerate warp process when running an experiment starting at t > 0. 
        # Should improve computation time
        pass

    # Log methods are used to print data during the experiments in the most convinient way possible

    def Log(self, Message, MessageType = 0):
        '''
        Log system to be used for verbose. for more clear information.
        Message :  str, message specific to the module
        MessageType : int. 0 for simple information, 1 for warning, 2 for error, stopping the stream, 3 for success green highlight
        '''
        self.__Framework__._Log(Message, MessageType, self)
    def LogWarning(self, Message):
        self.Log(Message, 1)
    def LogError(self, Message):
        self.Log(Message, 2)
        self.__Framework__.Paused = self.__Name__
    def LogSuccess(self, Message):
        self.Log(Message, 3)

    # Modules can access specific data from the framework through specific class methods or properties that follow :

    @property
    def StreamName(self):
        '''
        Method to recover the name of the stream fed to this module.
        Looks for the closest 'Input' module generated a corresponding Camera Index Restriction
        '''
        if self.__IsInput__:
            return self.__Framework__.CurrentInputStreams[self.__Name__]
        else:
            return self.__Framework__._GetStreamFormattedName(self)
    @property
    def Geometry(self):
        '''
        Method to recover the geometry of the stream fed to this module.
        Looks for the closest 'Input' module generated a corresponding Camera Index Restriction
        Input modules should override this property as they are the ones to feed the rest of the framework
        '''
        return self.__Framework__._GetStreamGeometry(self)
    @property
    def OutputGeometry(self):
        return self.Geometry
    @property
    def FrameworkEventTau(self):
        '''
        Method to retreive the highest level information Tau from the framework, depending on all the information of the current event running..
        If no Tau is proposed by any other module, will return None, so default value has to be module-specific
        '''
        return self.__Framework__._GetLowerLevelTau(self._RunningEvent, self)
    @property
    def FrameworkAverageTau(self):
        '''
        Method to retreive the highest level information Tau from the framework, with no information about the current event.
        If no Tau is proposed by any other module, will return None, so default value has to be Module specific
        '''
        return self.__Framework__._GetLowerLevelTau(None, self)
    @property
    def PicturesFolder(self):
        '''
        For modules that generates pictures, a default folder is create for each module, at each experiment.
        '''
        return self.__Framework__.PicturesFolder

    # The following methods are used to interact with the saved history

    def GetSnapIndexAt(self, t):
        return (abs(np.array(self.History['t']) - t)).argmin()

    def PlotHistoryData(self, MonitoredVariable, fax = None, color = None):
        t = np.array(self.History['t'])
        Data = np.array(self.History[MonitoredVariable])
        if len(Data.shape) == 0:
            raise Exception("No data saved yet")
        if len(Data.shape) == 1:
            if fax is None:
                f, ax = plt.subplots(1,1)
            else:
                f, ax = fax
            if color is None:
                ax.plot(t, Data, label = self.__Name__ + ' : ' + MonitoredVariable)
            else:
                ax.plot(t, Data, label = self.__Name__ + ' : ' + MonitoredVariable, color = color)
            ax.legend()
            return f, ax
        if len(Data.shape) == 2:
            if fax is None:
                f, ax = plt.subplots(1,1)
            else:
                f, ax = fax
            if Data.shape[1] == 3:
                Labels = ('x', 'y', 'z')
            else:
                Labels = [str(nCol) for nCol in range(Data.shape[1])]
            for nCol, Label in enumerate(Labels):
                ax.plot(t, Data[:,nCol], label = Label)
            ax.legend()
            return f, ax
        if len(Data.shape) > 2:
            raise Exception("Matrices unfit to be plotted")

    # All following methods should be ignored, as they run core methods no module should change.

    def __repr__(self):
        return self.__Name__

    def _GetParameters(self):
        InputDict = {}
        for Key, Value in self.__dict__.items():
            if type(Value) == types.MethodType:
                continue
            if len(Key) > 1 and Key[0] == '_' and Key[1] != '_' and Key[:7] != '_Module':
                InputDict[Key] = Value
        return InputDict

    def __Initialize__(self, Parameters):
        # First restore all previous values
        self.Log(" > Initializing module")
        self.__LastMonitoredTimestamp = -np.inf
        if self.__SavedValues__:
            for Key, Value in self.__SavedValues__.items():
                self.__dict__[Key] = Value
        self.__SavedValues__ = {}
        # Now change specific values for this initializing module
        for Key, Value in Parameters.items():
            if Key not in self.__dict__:
                self.LogError("Unconsistent parameter for {0} : {1}".format(self.__Name__, Key))
                return False
            else:
                if Key == '_MonitoredVariables' or Key == '_MonitorDt':
                    self.__UpdateParameter__(Key, Value, Log = False)
                    continue
                if type(Value) != type(self.__dict__[Key]) or type(Value) in (list, tuple):
                    self.__UpdateParameter__(Key, Value)
                    continue
                if type(Value) != np.ndarray:
                    if Value != self.__dict__[Key]:
                        self.__UpdateParameter__(Key, Value)
                        continue
                else:
                    if (Value != self.__dict__[Key]).any():
                        self.__UpdateParameter__(Key, Value)
                        continue
        if not self._MonitorDt:
            self.Log("No data monitored (_MonitorDt = 0)")
        elif not self._MonitoredVariables:
            self.Log("No data monitored (_MonitoredVariables empty)")
        else:
            self.Log("Data monitored every {0:.3f}s :".format(self._MonitorDt))
            for Var, Type in self._MonitoredVariables:
                self.Log(" * {0} as {1}".format(Var, Type.__name__))
        
        # We can only link modules at this point, since the framework must have created them all before
        for ModuleLinkRequested, ModuleLinked in self.__ModulesLinked__.items():
            if ModuleLinked:
                self.__dict__[ModuleLinkRequested] = self.__Framework__.Modules[ModuleLinked]

        # Initialize the stuff corresponding to this specific module
        InitAnswer = self._OnInitialization()
        if InitAnswer is None:
            self.LogWarning("Did not properly confirmed initialization ('return true' missing in function _OnInitialization ?).")
            return False
        elif not InitAnswer:
            return False

        # Finalize Module initialization
        if not self.__IsInput__:
            OnEventMethodUsed = self.__OnEventRestricted__
        else:
            OnEventMethodUsed = self.__OnEventInput__

        if self._MonitorDt and self.__ProposesTau: # We check if that method was overloaded
            if not self._MonitoredVariables:
                self.LogSuccess("Enabling monitoring for Tau value")
            self._MonitoredVariables.append(('AverageTauMonitored', float))
            self.__class__.AverageTauMonitored = property(lambda self: self._OnTauRequest(None))

        if self._MonitorDt and self._MonitoredVariables:
            self.History = {'t':[]}
            self.__MonitorRetreiveMethods = {}
            for Variable in self._MonitoredVariables:
                if type(Variable) == tuple:
                    VarName, Type = Variable
                else:
                    VarName = Variable
                    Type = None
                self.History[VarName] = []
                try:
                    self.__MonitorRetreiveMethods[VarName] = self.__GetRetreiveMethod__(VarName, Type)
                except:
                    self.LogWarning("Unable to get a valid retreive method for {0}".format(VarName))

            self.__LastMonitoredTimestamp = -np.inf

            self.__OnEvent__ = lambda eventContainer: self.__OnEventMonitor__(OnEventMethodUsed, eventContainer)
        else:
            self.__OnEvent__ = OnEventMethodUsed

        self._RunningEvent = None
        self.__Initialized__ = True
        return True

    def __UpdateParameter__(self, Key, Value, Log = True):
        self.__SavedValues__[Key] = copy.copy(self.__dict__[Key])
        self.__dict__[Key] = Value
        if Log:
            self.Log("Changed specific value {0} from {1} to {2}".format(Key, self.__SavedValues__[Key], self.__dict__[Key]))

    def __OnEventInput__(self, eventContainer):
        self._OnEventModule(eventContainer.BareEvent)
        if eventContainer.IsFilled:
            return eventContainer
        else:
            return None

    def __OnEventRestricted__(self, eventContainer):
        for event in eventContainer.GetEvents(self.__SubStreamInputIndexes__):
            self._RunningEvent = event
            self._OnEventModule(event)
        return eventContainer.IsFilled

    def __OnEventMonitor__(self, OnEventMethodUsed, eventContainer):
        output = OnEventMethodUsed(eventContainer)
        if eventContainer.timestamp - self.__LastMonitoredTimestamp > self._MonitorDt:
            self.__LastMonitoredTimestamp = eventContainer.timestamp
            self._OnSnapModule()
            self.History['t'] += [eventContainer.timestamp]
            for VarName, RetreiveMethod in self.__MonitorRetreiveMethods.items():
                self.History[VarName] += [RetreiveMethod()]
        return output

    def _IsParent(self, ChildModule):
        if ChildModule.__ModuleIndex__ < self.__ModuleIndex__:
            return False
        for SubStreamIndex in self.__SubStreamOutputIndexes__:
            if SubStreamIndex in ChildModule.__SubStreamInputIndexes__:
                return True
        return False

    def _IsChild(self, ParentModule):
        if ParentModule.__ModuleIndex__ > self.__ModuleIndex__:
            return False
        for SubStreamIndex in self.__SubStreamInputIndexes__:
            if SubStreamIndex in ParentModule.__SubStreamOutputIndexes__:
                return True
        return False

    def __GetRetreiveMethod__(self, VarName, UsedType):
        if '@' in VarName:
            Container, Key = VarName.split('@')
            if '.' in Key:
                Key, Field = Key.split('.')
                SubRetreiveMethod = lambda Instance: getattr(getattr(Instance, Key), Field)
            else:
                SubRetreiveMethod = lambda Instance: getattr(Instance, Key)

            if type(self.__dict__[Container]) == list:
                if UsedType is None:
                    ExampleVar = SubRetreiveMethod(self.__dict__[Container][0])
                    UsedType = type(ExampleVar)
                return lambda :[UsedType(SubRetreiveMethod(Instance)) for Instance in self.__dict__[Container]]
            elif type(self.__dict__[Container]) == dict:
                if UsedType is None:
                    ExampleVar = SubRetreiveMethod(self.__dict__[Container].values()[0])
                    UsedType = type(ExampleVar)
                return lambda :[(LocalDictKey, UsedType(SubRetreiveMethod(Instance))) for LocalDictKey, Instance in self.__dict__[Container].items()]
            else:
                if UsedType is None:
                    ExampleVar = SubRetreiveMethod(self.__dict__[Container])
                    UsedType = type(ExampleVar)
                print(UsedType)
                return lambda :UsedType(SubRetreiveMethod(self.__dict__[Container]))
        else:
            if UsedType is None:
                UsedType = type(self.__dict__[VarName])
            Key = VarName
            if '.' in Key:
                Key, Field = Key.split('.')
                SubRetreiveMethod = lambda Instance: getattr(getattr(Instance, Key), Field)
            else:
                SubRetreiveMethod = lambda Instance: getattr(Instance, Key)

            return lambda :UsedType(SubRetreiveMethod(self))

    def _SaveData(self, BinDataFile, MaxHistoryChunkSizeB = 16777216):
        if self._MonitorDt and self._MonitoredVariables:
            DataDict = {'Dt':self._MonitorDt, 't':self.History['t'], 'vars':self._MonitoredVariables}
            cPickle.dump(('.'.join([self.__Name__, 'Monitor']), DataDict), BinDataFile)

            for Var, Type in self._MonitoredVariables:
                NItems = len(self.History[Var])
                if NItems:
                    if '@' in Var:
                        TypeItem = self.History[Var][0][0]
                        ChunkMultiplier = len(self.History[Var][0])
                    else:
                        TypeItem = self.History[Var][0]
                        ChunkMultiplier = 1
                    if Type == np.array:
                        ItemSizeB = TypeItem.nbytes
                    else:
                        ItemSizeB = sys.getsizeof(TypeItem)
                    ItemSizeB *= ChunkMultiplier
                    NChunks = max(1, int(ItemSizeB * NItems / MaxHistoryChunkSizeB))
                    ChunkSize = int(NItems / NChunks) + 1
                    if NChunks > 1:
                        self.LogWarning("Spliting monitored variable {0} into {1} chunks of data".format(Var, NChunks))
                    else:
                        self.Log("Dumping variable {0} in one chunk".format(Var))
                    for nChunk in range(NChunks):
                        if NChunks>1 and self.__Framework__._SessionLog is sys.stdout:
                            sys.stdout.write("{0}%\r".format(int(100*nChunk/NChunks))),
                        Data = self.History[Var][nChunk * ChunkSize : (nChunk+1) * ChunkSize]
                        if Data:
                            cPickle.dump(('.'.join([self.__Name__, 'Monitor', Var, str(nChunk)]), Data), BinDataFile)
            self.LogSuccess("Saved history data")

    def SaveCSVData(self, FileName, Variables = [], FloatPrecision = 6, Default2DIndexes = 'xy', Default3DIndexes = 'xyz', Separator = ' '):
        if not Variables:
            if not self._MonitoredVariables:
                self.LogWarning("No CSV export as no data was being monitored")
                return
            if not self._MonitorDt:
                self.LogWarning("No CSV export as no sample rate was set")
                return
            Variables = [Key for Key, Type in self._MonitoredVariables if Key != 't']
        else:
            if 't' in Variables:
                Variables.remove('t')
        def FloatToStr(value):
            return str(round(value, FloatPrecision))
        FormatFunctions = [FloatToStr] # for 't'
        CSVVariables = ['t']
        CSVDataAccess = [('t', None)]
        for Key, Type in self._MonitoredVariables:
            if Key not in Variables:
                continue
            if '@' in Key:
                 self.LogWarning("Avoiding monitored variable {0} as lists of data elements are not fitted for CSV files".format(Key))
                 continue
            TemplateVar = self.History[Key][0]
            if Type == np.array:
                Size = TemplateVar.flatten().shape[0]
                if Size > 9: # We remove here anything that would be too big for CSV files, like frames, ST-contexts, ...
                    self.LogWarning("Avoiding monitored variable {0} as its shape is not fitted for CSV files".format(Key))
                    continue
                DataType = type(TemplateVar.flatten()[0])
                if DataType == np.int64:
                    OutputFunction = str
                else:
                    OutputFunction = FloatToStr
                if Size == 1:
                    FormatFunctions.append(OutputFunction)
                    CSVVariables.append(Key)
                    CSVDataAccess.append((Key, 0))
                elif Size == 2 and Default2DIndexes:
                    for nIndex, Index in enumerate(Default2DIndexes):
                        FormatFunctions.append(OutputFunction)
                        CSVVariables.append(Key+'_'+Index)
                        CSVDataAccess.append((Key, nIndex))
                elif Size == 3 and Default3DIndexes:
                    for nIndex, Index in enumerate(Default3DIndexes):
                        FormatFunctions.append(OutputFunction)
                        CSVVariables.append(Key+'_'+Index)
                        CSVDataAccess.append((Key, nIndex))
                else:
                    for nIndex in range(Size):
                        FormatFunctions.append(OutputFunction)
                        CSVVariables.append(Key+'_'+str(nIndex))
                        CSVDataAccess.append((Key, nIndex))
            else:
                CSVVariables.append(Key)
                CSVDataAccess.append((Key, None))
                if type(TemplateVar) == int:
                    FormatFunctions.append(str)
                else:
                    FormatFunctions.append(FloatToStr)
        if len(CSVVariables) == 1:
            self.LogWarning("No CSV export as no monitored data was kept")
            return
        with open(FileName, 'w') as fCSV:
            fCSV.write("# "+Separator.join(CSVVariables) + "\n")
            for nLine in range(len(self.History['t'])):
                Data = []
                for nVar, (FormatFunction, (Key, Index)) in enumerate(zip(FormatFunctions, CSVDataAccess)):
                    if Index is None:
                        Data += [FormatFunction(self.History[Key][nLine])]
                    else:
                        Data += [FormatFunction(self.History[Key][nLine].flatten()[Index])]
                fCSV.write(Separator.join(Data)+'\n')
            self.LogSuccess("Saved {0} data in {1}".format(self.__Name__, FileName))

    def _RecoverData(self, Identifier, Data):
        if Identifier == 'Monitor':
            self._MonitorDt = Data['Dt']
            self.__LastMonitoredTimestamp = Data['t'][-1]
            self._MonitoredVariables = Data['vars']
            self.History = {'t':Data['t']}
            for Var, Type in self._MonitoredVariables:
                self.History[Var] = []
            self._ExpectedHistoryVarsChunks = {Var: 0 for Var, Type in self._MonitoredVariables}
            self.LogSuccess("Recovered Monitor general data")
        else:
            Parts = Identifier.split('.')
            if Parts[0] != 'Monitor':
                raise Exception("Module {0} received wrong data recovery identifier : {1}".format(self.__Name__, Identifier))
            nChunk = int(Parts[-1])
            Var = '.'.join(Parts[1:-1])
            if self._ExpectedHistoryVarsChunks[Var] != nChunk:
                raise Exception("Module {0} lost chunk of data on recovery of MonitoredVariable {1}".format(self.__Name__, Var))
            self.History[Var] += Data
            if len(self.History[Var]) == len(self.History['t']):
                del self._ExpectedHistoryVarsChunks[Var]
                self.LogSuccess("{0}".format(Var))
                if not self._ExpectedHistoryVarsChunks:
                    del self.__dict__['_ExpectedHistoryVarsChunks']
                    self.LogSuccess("Recovered all Monitor data")
            else:
                if self.__Framework__._SessionLog is sys.stdout:
                    sys.stdout.write("{0}%\r".format(int(100*len(self.History[Var])/len(self.History['t'])))),
                self._ExpectedHistoryVarsChunks[Var] += 1
