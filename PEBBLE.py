import numpy as np
import pickle
import sys
import select
import inspect
import atexit
import types
import copy
import os
import shutil
import pathlib
import _pickle as cPickle
import json
from datetime import datetime as dtModule
import matplotlib.pyplot as plt
import ast

_RUNS_DIRECTORY = os.path.expanduser('~/Runs/')
        
_TYPE_TO_STR = {np.array:'np.array', float:'float', int:'int', str:'str', tuple:'tuple', list:'list'}
_STR_TO_TYPE = {value:key for key, value in _TYPE_TO_STR.items()}
_SPECIAL_NPARRAY_CAST_MESSAGE = '$ASNPARRAY$'

def LoadParametersFromFile(FileName):
    with open(FileName, 'r') as fParamsJSON:
        Parameters = json.load(fParamsJSON)
    for ModuleName, ModuleParameters in Parameters.items():
        for Key, Value in ModuleParameters.items():
            if Key == '_MonitoredVariables':
                for nVar, (Name, StrType) in enumerate(list(Value)):
                    Value[nVar] = (Name, _STR_TO_TYPE[StrType])
            else:
                ModuleParameters[Key] = UnSerialize(Value)
    return ParametersDictClass(Parameters)

def Serialize(Value):
    tValue = type(Value)
    if tValue == np.ndarray:
        return [_SPECIAL_NPARRAY_CAST_MESSAGE, Value.tolist()]
    elif tValue in (list, tuple):
        return [_TYPE_TO_STR[tValue]]+[Serialize(Item) for Item in Value]
    else:
        return Value
def UnSerialize(Data):
    tData = type(Data)
    if tData == list:
        if Data[0] == _SPECIAL_NPARRAY_CAST_MESSAGE:
            return np.array(Data[1])
        else:
            return _STR_TO_TYPE[Data[0]]([UnSerialize(Item) for Item in Data[1:]])
    else:
        return Data

class Framework:
    _Default_Color = '\033[0m'
    _LogColors = {0:'\033[0m', 1: "\033[1;33;40m", 2: "\033[1;31;40m", 3: "\033[1;32;40m"}
    _LOG_FILE_EXTENSION = 'log'
    _PROJECT_FILE_EXTENSION = 'json'
    _DATA_FILE_EXTENSION = 'data'

    '''
    Main event-based framework file.
    '''
    def __init__(self, File1 = None, File2 = None, onlyRawData = False):
        self.__Type__ = 'Framework'
        self._LogType = 'raw'
        self._SessionLogs = [sys.stdout]
        try:
            self._Terminal_Width = int(os.popen('stty size', 'r').read().split()[1])
        except:
            self._Terminal_Width = 100
        
        self.Modified = False
        self.StreamHistory = []

        self.PropagatedContainer = None
        self.Running = False
        self._Initializing = False
        self.Paused = ''

        if File1 is None and File2 is None:
            self._ProjectRawData = {}
            self.ProjectFile = None
            self._GenerateEmptyProject()
        else:        
            self._LoadFiles(File1, File2, onlyRawData)

        self._FolderData = {'home':None}
        atexit.register(self._OnClosing)

    def Initialize(self):
        self._Initializing = True
        self.PropagatedContainer = None
        self.InputEvents = {ToolName: None for ToolName in self.ToolsList if self.Tools[ToolName].__Type__ == 'Input'}
        if len(self.InputEvents) == 1:
            self._NextInputEventMethod = self._SingleInputModuleNextInputEventMethod
            del self.__dict__['InputEvents']
        else:
            self._NextInputEventMethod = self._MultipleInputModulesNextInputEventMethod

        self._LogType = 'columns'
        self._LogInit()
        self.PropagatedIndexes = set()
        for ToolName in self.ToolsList:
            InitializationAnswer = Module.__Initialize__(self.Tools[ToolName], self.RunParameters[ToolName])
            if not InitializationAnswer:
                self._Log("Tool {0} failed to initialize. Aborting.".format(ToolName), 2)
                self._DestroyFolder()
                return False
            for Index in self.Tools[ToolName].__SubStreamOutputIndexes__:
                self.PropagatedIndexes.add(Index)
        self._RunToolsMethodTuple = tuple([self.Tools[ToolName].__OnEvent__ for ToolName in self.ToolsList if self.Tools[ToolName].__Type__ != 'Input']) # Faster way to access tools in the right order, and only not input modules as they are dealt with through _NextInputEvent
        self._Log("Framework initialized", 3, AutoSendIfPaused = False)
        self._Log("")
        self._SendLog()
        self._Initializing = False
        return True

    def _GetCameraIndexChain(self, Index):
        ToolsChain = []
        for ToolName in self.ToolsList:
            if not self.Tools[ToolName].__SubStreamOutputIndexes__ or Index in self.Tools[ToolName].__SubStreamOutputIndexes__:
                ToolsChain += [ToolName]
        return ToolsChain

    def _GetParentModule(self, Tool):
        ToolEventsRestriction = Tool.__SubStreamInputIndexes__
        for InputToolName in reversed(self.ToolsList[:Tool.__ToolIndex__]):
            InputTool = self.Tools[InputToolName]
            if not ToolEventsRestriction:
                return InputTool
            if not InputTool.__SubStreamOutputIndexes__:
                return InputTool
            for CameraIndex in InputTool.__SubStreamOutputIndexes__:
                if CameraIndex in ToolEventsRestriction:
                    return InputTool
        self._Log("{0} was unable to find its parent module".format(Tool.__Name__), 1)

    def _GetStreamGeometry(self, Tool):
        '''
        Method to retreive the geometry of the events handled by a tool
        '''
        return self._GetParentModule(Tool).OutputGeometry

    def _GetStreamFormattedName(self, Tool):
        '''
        Method to retreive a formatted name depending on the files providing events to this tool.
        Specifically useful for an Input type tool to get the file it has to process.
        '''
        return self._GetParentModule(Tool).StreamName

    def _GetLowerLevelTau(self, EventConcerned, ModuleAsking):
        NameAsking = ModuleAsking.__Name__
        EventTau = None
        for NameProposing in reversed(self.ToolsList[:ModuleAsking.__ToolIndex__]):
            ModuleProposing = self.Tools[NameProposing]
            if (not EventConcerned is None and EventConcerned.SubStreamIndex in ModuleProposing.__SubStreamOutputIndexes__) or (EventConcerned is None and ModuleProposing._IsParent(ModuleAsking)):
                EventTau = ModuleProposing.EventTau(EventConcerned)
                if not EventTau is None and not EventTau == 0:
                    return EventTau

    def ReRun(self, stop_at = np.inf):
        self.RunStream(self.StreamHistory[-1], stop_at = stop_at)

    def RunStream(self, StreamName = None, Parameters = None, start_at = 0., stop_at = np.inf, stop_after = np.inf, resume = False, AtEventMethod = None):
        if not resume:
            if stop_at != np.inf and stop_after != np.inf:
                raise Exception("Both stop_at and stop_after specified")
            if start_at == -np.inf:
                stop_at = min(0. + stop_after, stop_at)
            else:
                stop_at = min(start_at + stop_after, stop_at)
            if Parameters is None:
                self._Log("Using default parameters for all modules", 1)
                ParametersDict = self.GetModulesParameters()
            elif type(Parameters) == ParametersDictClass:
                ParametersDict = Parameters
            elif type(Parameters) == dict:
                ParametersDict = ParametersDictClass(Parameters)
            elif type(Parameters) == str:
                self._Log("Using parameters stored in {0}".format(Parameters), 1)
                ParametersDict = LoadParametersFromFile(Parameters)
            else:
                self._Log("Parameters input type not understood", 2)
                self._Log("Use either None (default) for defaults modules parameter", 2)
                self._Log("           Dictionary build from Framework.GetModulesParameters()", 2)
                self._Log("           Str /PATH/TO/params.txt for using previous run parameters", 2)
                return

            self._InitiateFolder()
            self._SaveParameters(ParametersDict)
            self._SaveRunStreamData(StreamName, start_at, stop_at, resume)
        else:
            ParametersDict = None
        if self._LogType == 'columns':
            self._LogInit(resume)
        self._RunProcess(StreamName = StreamName, ParametersDict = ParametersDict, start_at = start_at, stop_at = stop_at, resume = resume, AtEventMethod = AtEventMethod)
        self._LogType = 'raw'

    def _GetCommitValue(self):
        try:
            f = open('.git/FETCH_HEAD', 'r')
            commit_value = f.readline().strip()
            f.close()
        except:
            commit_value = "unknown"
        return commit_value

    def _InitiateFolder(self):
        self._FolderData = {'home':_RUNS_DIRECTORY + dtModule.now().strftime("%d-%m-%Y_%H-%M") + '/',
                            'history':None,
                            'pictures':None}
        os.mkdir(self._FolderData['home'])
        self._Log("Created output folder {0}".format(self._FolderData['home']), 1)
        self._SessionLogs = [sys.stdout, open(self._FolderData['home']+'log.txt', 'w')]

    def _OnClosing(self):
        if self.Modified:
            ans = 'void'
            while self._PROJECT_FILE_EXTENSION not in ans and ans != '':
                ans = input('Unsaved changes. Please enter a file name with extension .{0}, or leave blank to discard : '.format(self._PROJECT_FILE_EXTENSION))
            if ans != '':
                self.SaveProject(ans)

        if '_LastStartWarning' in self.__dict__: # If warp was stopped midway, we delete everything, nothing must have been produced
            for ToolName in self.ToolsList:
                self.Tools[ToolName]._OnClosing()
            self._SessionLogs.pop(1).close()
            return

        if self._FolderData['home'] is None:
            return

        if not self._CSVDataSaved:
            self.SaveCSVData() # By default, we save the data. Files shouldn't be too big anyway
        for ToolName in self.ToolsList:
            self.Tools[ToolName]._OnClosing()
        self._SessionLogs.pop(1).close()

    def _DestroyFolder(self):
        shutil.rmtree(self._FolderData['home'])
        self._FolderData['home'] = None

    def _SaveParameters(self, Parameters):
        PickableParameters = {}
        for ModuleName, ModuleParameters in Parameters.items():
            PickableParameters[ModuleName] = {}
            for Key, Value in ModuleParameters.items():
                if Key == '_MonitoredVariables':
                    PickableParameters[ModuleName][Key] = []
                    for Name, Type in Value:
                        PickableParameters[ModuleName][Key] += [(Name, _TYPE_TO_STR[Type])]
                    continue
                PickableParameters[ModuleName][Key] = Serialize(Value)
        with open(self.ParamsLogFile, 'w') as fParamsJSON:
            try:
                json.dump(PickableParameters, fParamsJSON, indent=4, sort_keys=True)
                self._Log("Saved parameters in file {0}".format(self.ParamsLogFile), 3)
            except TypeError:
                self._Log("Unable to save parameters as non serializable object was present", 1)

    def _SaveRunStreamData(self, InputStreams, start_at, stop_at, resume):
        if not resume:
            fInputs = open(self.InputsLogFile, 'w')
            for Module, InputFile in InputStreams.items():
                fInputs.write("{0} : {1}\n".format(Module, str(pathlib.Path(InputFile).resolve())))
            fInputs.write('\n')
            fInputs.write("Starting at {0:.3f}s, ".format(start_at))
            if stop_at == np.inf:
                fInputs.write("stops at end of stream\n")
            else:
                fInputs.write("stops at {0:.3f}s\n".format(stop_at))
        else:
            fInputs = open(self.InputsLogFile, 'a')
            fInputs.write("Resuming at {0:.3f}s".format(self.t))
            if stop_at == np.inf:
                fInputs.write("stops at end of stream\n")
            else:
                fInputs.write("stops at {0:.3f}s\n".format(stop_at))
        self._Log("Saved input parameters in {0}".format(self.InputsLogFile), 3)
        fInputs.close()

    @property
    def HistoryFolder(self):
        if self._FolderData['history'] is None:
            self._FolderData['history'] = self._FolderData['home']+'History/'
            os.mkdir(self._FolderData['history'])
        return self._FolderData['history']
    @property
    def PicturesFolder(self):
        if self._FolderData['pictures'] is None:
            self._FolderData['pictures'] = self._FolderData['home']+'Pictures/'
            os.mkdir(self._FolderData['pictures'])
        return self._FolderData['pictures']
    @property
    def ParamsLogFile(self):
        return self._FolderData['home']+'params.json'
    @property
    def InputsLogFile(self):
        return self._FolderData['home']+'inputs.txt'

    def SaveData(self, Filename, forceOverwrite = False):
        GivenExtention = Filename.split('.')[-1]
        if GivenExtention != self._DATA_FILE_EXTENSION:
            raise Exception("Enter data file with .{0} extension".format(self._DATA_FILE_EXTENSION))
        if not forceOverwrite:
            try:
                f = open(Filename, 'r')
                ans = input("File already exists. Overwrite (y/N) ? ")
                if not ans.lower() == "y":
                    print("Aborted")
                    return
                f.close()
            except:
                pass
        with open(Filename, 'wb') as BinDataFile:
            def BinWrite(data_str):
                BinDataFile.write(str.encode(data_str))
            now = dtModule.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            BinWrite("# Edited on -> "+ dt_string + "\n")
            commit_value = self._GetCommitValue()
            BinWrite("# Framework git commit -> " + commit_value + "\n")
            BinWrite("# Project File -> " + os.path.abspath(self.ProjectFile) + '\n')
            BinWrite("# Project Hash -> " + str(hash(json.dumps(self._ProjectRawData, sort_keys=True))) + "\n")
            BinWrite("# Input Files -> " + str({InputTool : os.path.abspath(InputFile) for InputTool, InputFile in self.CurrentInputStreams.items()}) + "\n")
            BinWrite("# Run arguments -> " + str(self.RunKwargs) + '\n')
            BinWrite("#########\n")

            for ToolName in self.ToolsList:
                self.Tools[ToolName]._SaveData(BinDataFile)

    def SaveCSVData(self, Folder = None, FloatPrecision = 6, Default2DIndexes = 'xy', Default3DIndexes = 'xyz', Separator = ' '):
        if Folder is None:
            Folder = self.HistoryFolder
        else:
            if Folder[-1] != '/':
                Folder = Folder + '/'
        for ToolName in self.ToolsList:
            FileName = Folder + ToolName + '.csv'
            self.Tools[ToolName].SaveCSVData(FileName, FloatPrecision = 6, Default2DIndexes = 'xy', Default3DIndexes = 'xyz', Separator = ' ')
        self._CSVDataSaved = True
    
    def _LoadFiles(self, File1, File2, onlyRawData):
        if File1 is None:
            File1, File2 = File2, File1
        if (not File2 is None and File1.split('.')[-1] == File2.split('.')[-1]): # If they are both None (no input file) or both with the same extension
            print("Input error. Input files can be :")
            print("One project file (.{0}), to launch streams".format(self._PROJECT_FILE_EXTENSION))
            print("One data file (.{0}), to recover data".format(self._DATA_FILE_EXTENSION))
            print("One project file and one data file, to recover data into a specified framework. Specific case")
            raise Exception("")
        if File1.split('.')[-1] == self._DATA_FILE_EXTENSION:
            self._LoadFromDataFile(DataFile = File1, ProjectFile = File2, onlyRawData = onlyRawData)
        elif (not File2 is None) and File2.split('.')[-1] == self._DATA_FILE_EXTENSION:
            self._LoadFromDataFile(DataFile = File2, ProjectFile = File1, onlyRawData = onlyRawData)
        else:
            self._LoadProject(File1, onlyRawData = onlyRawData)

    def _LoadFromDataFile(self, DataFile, ProjectFile = None, onlyRawData = False):
        self._Log('Loading data from saved file')
        with open(DataFile, 'rb') as BinDataFile:
            def BinRead():
                Line = BinDataFile.readline().decode('utf-8')
                return (Line[:2] == '##'), Line[2:].split('->')[0].strip(), Line[2:].split('->')[-1].strip()
            ProjectFileHash = None
            ProjectFileRecovered = None
            InputFiles = None
            RunArgs = None
            while True:
                CommentEnd, Key, Value = BinRead()
                if CommentEnd:
                    break
                self._Log(Key)
                if Key == 'Framework git commit':
                    if Value != self._GetCommitValue():
                        self._Log(Value + " (current : {0})".format(self._GetCommitValue()), 1)
                    else:
                        self._Log(Value, 3)
                elif Key == 'Project File':
                    ProjectFileRecovered = Value
                    if ProjectFile is None:
                        ProjectFile = ProjectFileRecovered
                    else:
                        self._Log("Overwrote with {0}".format(ProjectFile), 3)
                    if '/' not in ProjectFile:
                        PEBBLE_LOCATION = '/'.join(os.path.realpath(__file__).split('/')[:-1])
                        ProjectFile = PEBBLE_LOCATION + '/Projects/' + ProjectFile
                    ProjectRawDataReferred = pickle.load(open(ProjectFile, 'rb'))
                    ProjectFileHash = str(hash(json.dumps(ProjectRawDataReferred, sort_keys=True)))
                    self._Log(Value, 3)
                elif Key == 'Project Hash':
                    if not ProjectFileHash is None and Value != ProjectFileHash:
                        self._Log(Value + " (current : {0})".format(ProjectFileHash), 1)
                    else:
                        self._Log(Value, 3)
                elif Key == 'Input Files':
                    InputFiles = ast.literal_eval(Value)
                    self._Log(Value)
                elif Key == 'Run arguments':
                    RunArgs = ast.literal_eval(Value)
                    self._Log(Value)

            if ProjectFile is None:
                raise Exception("Unable to retreive correct ProjectFile (.json)")
            self._LoadProject(ProjectFile, onlyRawData = onlyRawData)
            if InputFiles is None or RunArgs is None:
                self._Log("No valid input file or Run arguments. Unable to initialize modules", 1)
            else:
                self._RunProcess(StreamName = InputFiles, BareInit = True, **RunArgs)
                self._Log("Initialized Framework", 3)
            self._Log("Starting data recovery")
            while True:
                try:
                    Identifier, Data = cPickle.load(BinDataFile)
                    ToolName = Identifier.split('.')[0]
                    self.Tools[ToolName]._RecoverData(Identifier[len(ToolName)+1:], Data)
                except EOFError:
                    break
            self._Log("Successfully recovered data", 3)

    def _RunProcess(self, StreamName = None, ParametersDict = None, start_at = -np.inf, stop_at = np.inf, resume = False, AtEventMethod = None, BareInit = False):
        if StreamName is None:
            N = 0
            StreamName = "DefaultStream_{0}".format(N)
            while StreamName in self.StreamHistory:
                N += 1
                StreamName = "DefaultStream_{0}".format(N)
        if not resume:
            self._LastStartWarning = 0
            self.RunParameters = dict(ParametersDict)
            self.CurrentInputStreams = {ToolName:None for ToolName in self.ToolsList if self.Tools[ToolName].__Type__ == 'Input'}
            if type(StreamName) == str:
                for ToolName in self.CurrentInputStreams.keys():
                    self.CurrentInputStreams[ToolName] = StreamName
            elif type(StreamName) == dict:
                if len(StreamName) != len(self.CurrentInputStreams):
                    self._Log("Wrong number of stream names specified :", 2)
                    self._Log("Framework contains {0} input tools, while {1} tool(s) has been given a file".format(len(self.CurrentInputStreams), len(StreamName)), 2)
                    return False
                for key in StreamName.keys():
                    if key not in self.CurrentInputStreams.keys():
                        self._Log("Wrong input tool key specified in stream names : {0}".format(key), 2)
                        return False
                    self.CurrentInputStreams[key] = StreamName[key]
            else:
                self._Log("Wrong StreamName type. It can be :", 2)
                self._Log(" - None : Default name is then placed, for None file specific input tools like simulators.", 2)
                self._Log(" - str : The same file is then used for all input tools.", 2)
                self._Log(" - dict : Dictionnary with input tools names as keys and specified filenames as values", 2)
                return False
            
            self.StreamHistory += [self.CurrentInputStreams]
            InitializationAnswer = self.Initialize()
            if not InitializationAnswer or BareInit:
                self._SendLog()
                return False
            self._CSVDataSaved = False

        self.Running = True
        self.Paused = ''
        if resume:
            for ToolName in self.ToolsList:
                self.Tools[ToolName]._Resume()

        self.t = 0.
        while not resume and start_at > -np.inf and self.Running and not self.Paused:
            Container = self._NextInputEventMethod()
            if not Container is None:
                self.t = Container.timestamp
            if self.t >= start_at:
                del self.__dict__['_LastStartWarning']
                self._Log("Warp finished", 3)
                break
            if self.t > stop_at:
                self.Paused = 'Framework'
            if self.t > self._LastStartWarning + 1.:
                self._Log("Warping : {0:.1f}/{1:.1f}s".format(self.t, start_at))
                self._SendLog()
                self._LastStartWarning = self.t

        while self.Running and not self.Paused:
            self.t = self.Next(AtEventMethod)
            self._SendLog()

            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                promptedLine = sys.stdin.readline()
                if 'a' in promptedLine or 'q' in promptedLine:
                    self.Paused = 'user'
            if self.t is None or self.t > stop_at:
                self.Paused = 'Framework'
        self.SaveCSVData()
        if not self.Running:
            self._Log("Main loop finished without error.")
            return True
        else:
            if self.Paused:
                self._Log("Paused at t = {0:.3f}s by {1}.".format(self.t, self.Paused), 1)
                for ToolName in self.ToolsList:
                    self.Tools[ToolName]._Pause(self.Paused)
            return False

    def Resume(self, stop_at = np.inf):
        self._LogType = 'columns'
        self.RunStream(self.StreamHistory[-1], stop_at = stop_at, resume = True)

    def DisplayRestart(self):
        for ToolName in self.ToolsList:
            self.Tools[ToolName]._Restart()

    def Next(self, AtEventMethod = None):
        self.PropagatedContainer = self._NextInputEventMethod()
        if self.PropagatedContainer is None:
            return None
        t = self.PropagatedContainer.timestamp
        for RunMethod in self._RunToolsMethodTuple:
            if not AtEventMethod is None:
                AtEventMethod(self.PropagatedContainer)
            if not RunMethod(self.PropagatedContainer):
                break
        if not AtEventMethod is None and not self.PropagatedContainer is None:
            AtEventMethod(self.PropagatedContainer)

        return t

    def _SingleInputModuleNextInputEventMethod(self):
        return self.Tools[self.ToolsList[0]].__OnEvent__(_EventContainerClass(Bare = True))

    def _MultipleInputModulesNextInputEventMethod(self):
        OldestEvent, ModuleSelected = None, None
        for InputName, EventAwaiting in self.InputEvents.items():
            if EventAwaiting is None:
                EventAwaiting = self.Tools[InputName].__OnEvent__(_EventContainerClass(Bare = True))
            else:
                self.InputEvents[InputName] = None
            if not EventAwaiting is None:
                if OldestEvent is None:
                    OldestEvent = EventAwaiting
                    ModuleSelected = InputName
                else:
                    if EventAwaiting < OldestEvent:
                        self.InputEvents[ModuleSelected] = OldestEvent
                        ModuleSelected = InputName
                        OldestEvent = EventAwaiting
                    else:
                        self.InputEvents[InputName] = EventAwaiting
        return OldestEvent

#### Project Management ####

    def _GenerateEmptyProject(self):
        self.Tools = {}
        self._ToolsCreationReferences = {}
        self._ToolsExternalParameters = {}
        self._ToolsCamerasRestrictions = {}
        self._ToolsClasses = {}
        self._SubStreamIndexes = set()

        self.ToolsOrder = {}
        self.ToolsList = []

    def _LoadProject(self, ProjectFile = None, enable_easy_access = True, onlyRawData = False):
        self._LogType = 'raw'
        self._GenerateEmptyProject()

        if not ProjectFile is None:
            if '/' not in ProjectFile:
                PEBBLE_LOCATION = '/'.join(os.path.realpath(__file__).split('/')[:-1])
                ProjectFile = PEBBLE_LOCATION + '/Projects/' + ProjectFile
            self.ProjectFile = ProjectFile
            self._ProjectRawData = pickle.load(open(self.ProjectFile, 'rb'))

        if onlyRawData:
            return None

        for ToolName in self._ProjectRawData.keys():
            fileLoaded = __import__(self._ProjectRawData[ToolName]['File'])
            self._ToolsClasses[ToolName] = getattr(fileLoaded, self._ProjectRawData[ToolName]['Class'])

            self._ToolsCreationReferences[ToolName] = self._ProjectRawData[ToolName]['CreationReferences']
            self._ToolsExternalParameters[ToolName] = self._ProjectRawData[ToolName]['ExternalParameters']
            if 'CamerasHandled' in self._ProjectRawData[ToolName].keys(): # For previous version support
                self._ToolsCamerasRestrictions[ToolName] = self._ProjectRawData[ToolName]['CamerasHandled']
            else:
                self._ToolsCamerasRestrictions[ToolName] = []

            self.ToolsOrder[ToolName] = self._ProjectRawData[ToolName]['Order']
            self._Log("Imported tool {1} from file {0}.".format(self._ProjectRawData[ToolName]['File'], self._ProjectRawData[ToolName]['Class']))

        if len(self.ToolsOrder.keys()) != 0:
            MaxOrder = max(self.ToolsOrder.values()) + 1
            self.ToolsList = [None] * MaxOrder
        else:
            self.ToolsList = []

        for ToolName in self.ToolsOrder.keys():
            if self.ToolsList[self.ToolsOrder[ToolName]] is None:
                self.ToolsList[self.ToolsOrder[ToolName]] = ToolName
            else:
                self._Log("Double assignement of number {0}. Aborting ProjectFile loading".format(self.ToolsOrder[ToolName]), 2)
                return None
        while None in self.ToolsList:
            self.ToolsList.remove(None)

        self._Log("Successfully generated tools order", 3)
        self._Log("")
        
        for ToolName in self.ToolsList:
            self.Tools[ToolName] = self._ToolsClasses[ToolName](ToolName, self, self._ToolsCreationReferences[ToolName])
            self._UpdateToolsParameters(ToolName)
            for Index in self.Tools[ToolName].__SubStreamOutputIndexes__:
                if Index not in self._SubStreamIndexes:
                    self._SubStreamIndexes.add(Index)
                    self._Log("Added SubStream {0}".format(Index), 3)

            if enable_easy_access and ToolName not in self.__dict__.keys():
                self.__dict__[ToolName] = self.Tools[ToolName]
        self._CSVDataSaved = True
        self._Log("Successfully generated Framework", 3)
        self._Log("")

    def _UpdateToolsParameters(self, ToolName):
        for key, value in self._ToolsExternalParameters[ToolName].items():
            if key in self.Tools[ToolName].__dict__.keys():
                try:
                    self.Tools[ToolName].__dict__[key] = type(self.Tools[ToolName].__dict__[key])(value)
                except:
                    self._Log("Issue with setting the new value of {0} for tool {1}, should be {2}, impossible from {3}".format(key, ToolName, type(self.Tools[ToolName].__dict__[key]), value), 1)
            else:
                self._Log("Key {0} for tool {1} doesn't exist. Please check ProjectFile integrity.".format(key, ToolName), 1)
        if self.Tools[ToolName].__Type__ == 'Input':
            self.Tools[ToolName].__SubStreamInputIndexes__ = set()
            self.Tools[ToolName].__SubStreamOutputIndexes__ = set(self._ToolsCamerasRestrictions[ToolName])
            if not self.Tools[ToolName]._SetInputModuleSubStreamIndexes(self._ToolsCamerasRestrictions[ToolName]):
                return False
        else:
            if not self._ToolsCamerasRestrictions[ToolName]:
                self.Tools[ToolName].__SubStreamInputIndexes__ = set(self._SubStreamIndexes)
            else:
                self.Tools[ToolName].__SubStreamInputIndexes__ = set(self._ToolsCamerasRestrictions[ToolName])
            self.Tools[ToolName].__SubStreamOutputIndexes__ = set(self.Tools[ToolName].__SubStreamInputIndexes__)

    def GetModulesParameters(self):
        ParametersDict = {}
        for ToolName in self.ToolsList:
            ParametersDict[ToolName] = self.Tools[ToolName]._GetParameters()
        return ParametersDictClass(ParametersDict)

    def SaveProject(self, ProjectFile):
        GivenExtention = ProjectFile.split('.')[-1]
        if GivenExtention != self._PROJECT_FILE_EXTENSION:
            raise Exception("Enter log file with .{0} extension".format(self._PROJECT_FILE_EXTENSION))
        pickle.dump(self._ProjectRawData, open(ProjectFile, 'wb'))
        self.ProjectFile = ProjectFile
        self.Modified = False
        self._Log("Project saved.", 3)

    def AddTool(self):
        self._Log("Current project :")
        self.DisplayCurrentProject()
        self._Log("")
        FieldList = [('File', str, False), ('Class', str, False), ('Order', int, True), ('CreationReferences', str, False), ('ExternalParameters', list, True)]
        
        Name = input('Enter the name of the new tool : ')                   # Tool name 
        if Name == '' or Name in self._ProjectRawData.keys():
            self._Log("Invalid entry (empty or already existing).", 2)
            return None
        self._ProjectRawData[Name] = {}
        try:
            entry = ''
            while entry == '' :                                                 # Filename and Class definition
                self._Log("Enter the tool filename :")
                entry = input('')
            if '.py' in entry:
                entry = entry.split('.py')[0]
            self._ProjectRawData[Name]['File'] = entry
            fileLoaded = __import__(entry)
            classFound = False
            PossibleClasses = []
            if not 'Module' in fileLoaded.__dict__.keys():
                self._Log("File does not contain any class derived from 'Module'. Aborting entry", 2)
                del self._ProjectRawData[Name]
                return None
            for key in fileLoaded.__dict__.keys():
                if isinstance(fileLoaded.__dict__[key], type) and key[0] != '_' and fileLoaded.__dict__['Module'] in fileLoaded.__dict__[key].__bases__:
                    PossibleClasses += [key]
            if not classFound:
                if len(PossibleClasses) == 0:
                    self._Log("No possible Class is available in this file. Aborting.", 2)
                    del self._ProjectRawData[Name]
                    return None
                elif len(PossibleClasses) == 1:
                    self._Log("Using class {0}".format(PossibleClasses[0]))
                    self._ProjectRawData[Name]['Class'] = PossibleClasses[0]
                else:
                    entry = ''
                    while entry == '' :
                        self._Log("Enter the tool class among the following ones :")
                        for Class in PossibleClasses:
                            self._Log(" * {0}".format(Class))
                        entry = input('')
                    if entry not in PossibleClasses:
                        self._Log("Invalid class, absent from tool file or not a ClassType.", 2)
                        del self._ProjectRawData[Name]
                        return None
                    self._ProjectRawData[Name]['Class'] = entry
            self._Log("")
                                                                                  # Loading the class to get the references needed and parameters

            TmpClass = fileLoaded.__dict__[self._ProjectRawData[Name]['Class']](Name, self, {})
            ReferencesAsked = TmpClass.__ReferencesAsked__
            self._ProjectRawData[Name]['IsInput'] = (TmpClass.__Type__ == 'Input')

            PossibleVariables = []
            for var in TmpClass.__dict__.keys():
                if var[0] == '_' and var[1] != '_':
                    PossibleVariables += [var]
            if TmpClass.__Type__ != 'Input':
                self._Log("Enter the tool order number :")                             # Order definition
                entry = ''
                while entry == '':
                    entry = input('')
                if int(entry) >= len(self._ProjectRawData):
                    self._Log("Excessive tool number, assuming last position")
                    self._ProjectRawData[Name]['Order'] = len(self._ProjectRawData[Name]['Order'])-1
                else:
                    self._ProjectRawData[Name]['Order'] = int(entry)
            else:
                self._Log("Input Type detected. Setting to next default index.")
                self._ProjectRawData[Name]['Order'] = 0
                for ToolName in self._ProjectRawData.keys():
                    if ToolName != Name and self._ProjectRawData[ToolName]['IsInput']:
                        self._ProjectRawData[Name]['Order'] = max(self._ProjectRawData[Name]['Order'], self._ProjectRawData[ToolName]['Order'] + 1)
            NumberTaken = False
            for ToolName in self._ProjectRawData.keys():
                if ToolName != Name and 'Order' in self._ProjectRawData[ToolName].keys() and self._ProjectRawData[Name]['Order'] == self._ProjectRawData[ToolName]['Order']:
                    NumberTaken = True
            if NumberTaken:
                self._Log("Compiling new order.")
                for ToolName in self._ProjectRawData.keys():
                    if ToolName != Name and 'Order' in self._ProjectRawData[ToolName].keys():
                        if self._ProjectRawData[ToolName]['Order'] >= self._ProjectRawData[Name]['Order']:
                            self._ProjectRawData[ToolName]['Order'] += 1
                self._Log("Done")
                self._Log("")

            self._ProjectRawData[Name]['CreationReferences'] = {}
            if ReferencesAsked:
                self._Log("Fill tool name for the needed references. Currently available tool names:")
                for key in self._ProjectRawData.keys():
                    if key == Name:
                        continue
                    self._Log(" * {0}".format(key))
                for Reference in ReferencesAsked:
                    self._Log("Reference for '" + Reference + "'")
                    entry = ''
                    while entry == '':
                        entry = input('-> ')
                    self._ProjectRawData[Name]['CreationReferences'][Reference] = entry
            else:
                self._Log("No particular reference needed for this tool.")
            self._Log("")
            if TmpClass.__Type__ == 'Input':
                self._Log("Enter indexes created by this module, coma separated.")
            else:
                self._Log("Enter streams index(es) handled by this module, coma separated. Void will not create any restriction.")
            entry = input(" -> ")
            self._ProjectRawData[Name]['CamerasHandled'] = []
            if entry:
                for index in entry.split(','):
                    self._ProjectRawData[Name]['CamerasHandled'] += [int(index.strip())]

            self._ProjectRawData[Name]['ExternalParameters'] = {}
            if PossibleVariables:
                self._Log("Current tool parameters :")
                for var in PossibleVariables:
                    self._Log(" * {0} : {1}".format(var, TmpClass.__dict__[var]))
                entryvar = 'nothing'
                while entryvar != '':
                    self._Log("Enter variable to change :")
                    entryvar = input('-> ')
                    if entryvar != '' and entryvar in PossibleVariables:
                        self._Log("Enter new value :")
                        entryvalue = input('-> ')
                        if entryvalue != '':
                            try:
                                if type(TmpClass.__dict__[entryvar]) == list:
                                    values = entryvalue.strip('][').split(',') 
                                    if TmpClass.__dict__[entryvar]:
                                        aimedType = type(TmpClass.__dict__[entryvar][0])
                                    else:
                                        aimedType = float
                                    self._ProjectRawData[Name]['ExternalParameters'][entryvar] = [aimedType(value) for value in values]
                                else:
                                    self._ProjectRawData[Name]['ExternalParameters'][entryvar] = type(TmpClass.__dict__[entryvar])(entryvalue)
                            except ValueError:
                                self._Log("Could not parse entry into the correct type", 1)
                    elif '=' in entryvar:
                        entryvar, entryvalue = entryvar.split('=')
                        if entryvar.strip() in PossibleVariables:
                            try:
                                self._ProjectRawData[Name]['ExternalParameters'][entryvar.strip()] = type(TmpClass.__dict__[entryvar.strip()])(entryvalue.strip())
                            except ValueError:
                                self._Log("Could not parse entry into the correct type", 1)
                    elif entryvar != '':
                        self._Log("Wrong variable name.", 1)
            self._Log("")

        except KeyboardInterrupt:
            self._Log("Canceling entries.", 1)
            del self._ProjectRawData[Name]
            return None
        except ImportError:
            self._Log("No such file found. Canceling entries", 2)
            del self._ProjectRawData[Name]
            return None

        self._Log("AddTool finished. Reloading project.")
        self._LoadProject()
        self._Log("New project : ")
        self._Log("")
        self.DisplayCurrentProject()
        self.Modified = True

    def DelTool(self, ToolName):
        ToolIndex = self._ProjectRawData[ToolName]['Order']
        del self._ProjectRawData[ToolName]
        for RemainingToolName, RemainingRawData in self._ProjectRawData.items():
            if RemainingRawData['Order'] > ToolIndex:
                RemainingRawData['Order'] -= 1

        self._Log("DelTool finished. Reloading project.")
        self._LoadProject()
        self._Log("New project : ")
        self._Log("")
        self.DisplayCurrentProject()
        self.Modified = True

    def Project(self):
        self._Log("# Framework\n", 3)
        self._Log("")

        nOrder = 0
        for ToolName in self.ToolsList:
            filename = inspect.getfile(self.Tools[ToolName].__class__)
            self._Log("# {0} : {1}, from class {2} in file {3}.".format(nOrder, ToolName, str(self.Tools[ToolName].__class__).split('.')[1][:-2], filename), 3)
            self._Log("     Type : {0}".format(self.Tools[ToolName].__Type__))
            if self.Tools[ToolName].__SubStreamInputIndexes__:
                self._Log("     Uses cameras indexes " + ", ".join([str(CameraIndex) for CameraIndex in self.Tools[ToolName].__SubStreamInputIndexes__]))
            else:
                self._Log("     Uses all cameras inputs.")
            if self.Tools[ToolName].__SubStreamOutputIndexes__  and not self.Tools[ToolName].__SubStreamOutputIndexes__ == self.Tools[ToolName].__SubStreamInputIndexes__:
                self._Log("     Outputs specific indexes " + ", ".join([str(CameraIndex) for CameraIndex in self.Tools[ToolName].__SubStreamOutputIndexes__]))
            else:
                self._Log("     Outputs the same camera indexes.")
            if self._ToolsCreationReferences[ToolName]:
                self._Log("     Creation References:")
                for argName, toolReference in self._ToolsCreationReferences[ToolName].items():
                    self._Log("         -> Access to {0} from tool {1}".format(argName, toolReference))
            else:
                self._Log("     No creation reference.")
            if self._ToolsExternalParameters[ToolName]:
                self._Log("     Modified Parameters:")
                for var, value in  self._ToolsExternalParameters[ToolName].items():
                    self._Log("         -> {0} = {1}".format(var, value))
            else:
                self._Log("     Using default parameters.")
            self._Log("")
            
            nOrder += 1

    def VisualProject(self):
        f, ax = plt.subplots(1,1)
        ax.tick_params('both', left = False, labelleft = False, bottom = False, labelbottom = False)
        SubStreamsToolLevels = {Index:0 for Index in self._SubStreamIndexes}
        SubStreamsToolsOutput = {Index:None for Index in self._SubStreamIndexes}
        SubStreamsColors = {Index:None for Index in self._SubStreamIndexes}
        ToolWidth = 0.6
        ToolHeight = 0.5
        from matplotlib.patches import Rectangle
        def DrawConnexion(ax, Start, End, Index):
            if SubStreamsColors[Index] is None:
                SubStreamsColors[Index] = ax.plot([Start[1], End[1]], [Start[0], End[0]], label = str(Index))[0].get_color()
            else:
                ax.plot([Start[1], End[1]], [Start[0], End[0]], color = SubStreamsColors[Index])
        for ToolName in self.ToolsList:
            Tool = self.Tools[ToolName]
            if not Tool.__SubStreamInputIndexes__:
                Col = list(Tool.__SubStreamOutputIndexes__)[0]
                Level = 0
            else:
                Col = np.mean(list(Tool.__SubStreamInputIndexes__))
                Level = max([SubStreamsToolLevels[Index] for Index in Tool.__SubStreamInputIndexes__])
            R = Rectangle((Col-ToolWidth/2, (Level-ToolHeight/2)), ToolWidth, ToolHeight, color = 'k', fill = False)
            ax.add_artist(R)
            ax.text(Col, Level, ToolName, color = 'k', va= 'center', ha = 'center')
            for nIndex, Index in enumerate(np.sort(list(Tool.__SubStreamInputIndexes__))):
                DrawConnexion(ax, SubStreamsToolsOutput[Index], ((Level-ToolHeight/2), Col - ToolWidth / 2 + (nIndex + 1) * ToolWidth / (len(Tool.__SubStreamInputIndexes__)+1)), Index)
                SubStreamsToolsOutput[Index] = ((Level+ToolHeight/2), Col - ToolWidth / 2 + (nIndex + 1) * ToolWidth / (len(Tool.__SubStreamOutputIndexes__)+1))
            for nIndex, Index in enumerate(np.sort(list(Tool.__SubStreamOutputIndexes__))):
                SubStreamsToolLevels[Index] = Level+1
                if SubStreamsToolsOutput[Index] is None:
                    SubStreamsToolsOutput[Index] = ((Level+ToolHeight/2), Col - ToolWidth / 2 + (nIndex + 1) * ToolWidth / (len(Tool.__SubStreamOutputIndexes__)+1))
        ax.set_xlim(min(list(self._SubStreamIndexes)) - ToolWidth, max(list(self._SubStreamIndexes)) + ToolWidth)
        ax.set_ylim(max(list(SubStreamsToolLevels.values()))-ToolHeight, -ToolHeight)
        ax.legend()

    def _Log(self, Message, MessageType = 0, Module = None, Raw = False, AutoSendIfPaused = True):
        if self._LogType == 'columns' and not Raw:
            if '\n' in Message:
                for Line in Message.split('\n'):
                    self._Log(Line, MessageType, Module, Raw, AutoSendIfPaused)
                return
            if Module is None:
                ModuleName = 'Framework'
            elif not Module._NeedsLogColumn:
                ModuleName = 'Framework'
                Message = Module.__Name__ + ": " + Message
            else:
                ModuleName = Module.__Name__
            if self._LogT is None and self.Running:
                if not self.PropagatedContainer is None:
                    self._LogT = self.PropagatedContainer.timestamp
                    self._Log('t = {0:.3f}s'.format(self._LogT))

            while len(Message) > self._MaxColumnWith:
                FirstPart, Message = Message[:self._MaxColumnWith], Message[self._MaxColumnWith:]
                self._EventLogs[ModuleName] += [self._LogColors[MessageType] + FirstPart]
            if Message:
                self._EventLogs[ModuleName] += [self._LogColors[MessageType] + Message + (self._MaxColumnWith-len(Message))*' ']
            self._HasLogs = max(self._HasLogs, len(self._EventLogs[ModuleName]))
            if (not self.Running or self.Paused) and AutoSendIfPaused and not self._Initializing:
                self._SendLog()
        elif self._LogType == 'raw' or Raw:
            if Module is None:
                ModuleName = 'Framework'
            else:
                ModuleName = Module.__Name__
            Message = self._LogColors[MessageType] + int(bool(Message))*(ModuleName + ': ') + Message
            for Log in self._SessionLogs:
                Log.write(Message + self._LogColors[0] + "\n")
    def _SendLog(self):
        if self._LogType == 'raw' or not self._HasLogs:
            return
        for nLine in range(self._HasLogs):
            CurrentLine = self._Default_Color
            if self._EventLogs['Framework']:
                CurrentLine += self._EventLogs['Framework'].pop(0)
            else:
                CurrentLine += self._MaxColumnWith*' '
            for ToolName in self._LogsColumns[1:]:
                if self._EventLogs[ToolName]:
                    CurrentLine += self._Default_Color + ' | ' + self._EventLogs[ToolName].pop(0)
                else:
                    CurrentLine += self._Default_Color + ' | ' + self._MaxColumnWith*' '
            for Log in self._SessionLogs:
                Log.write(CurrentLine + "\n")
        self._HasLogs = 0
        self._LogT = None
    def _LogInit(self, Resume = False):
        self._HasLogs = 2
        self._LogT = None
        self._LogsColumns = [ToolName for ToolName in ['Framework'] + self.ToolsList if (ToolName == 'Framework' or self.Tools[ToolName]._NeedsLogColumn)]
        self._MaxColumnWith = int((self._Terminal_Width - len(self._LogsColumns)*3 ) / len(self._LogsColumns))
        if not Resume:
            self._EventLogs = {ToolName:[' '*((self._MaxColumnWith - len(ToolName))//2) + self._LogColors[1] + ToolName + (self._MaxColumnWith - len(ToolName) - (self._MaxColumnWith - len(ToolName))//2)*' ', self._MaxColumnWith*' '] for ToolName in self._LogsColumns}
            self._SendLog()
        else:
            self._EventLogs = {ToolName:[] for ToolName in self._LogsColumns}
    def _LogReset(self):
        self._EventLogs = {ToolName:[] for ToolName in self._EventLogs.keys()}
        self._HasLogs = 0

class Module:
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Default module class.
        Each module in the Framework should inherit this class, whose 3 main methods and main parameters are required annd defined here.
        Type should be set manually.
        '''
        self.__Framework__ = Framework
        self.__Name__ = Name

        self.Log("Generating module")
        self.__ReferencesAsked__ = []
        self.__CreationReferences__ = dict(argsCreationReferences)
        self.__Type__ = None
        self.__Initialized__ = False
        self.__SavedValues__ = {}
        self.__SubStreamInputIndexes__ = []
        self.__SubStreamOutputIndexes__ = []
        
        self._MonitoredVariables = []
        self._MonitorDt = 0
        self._ProposesTau = ('EventTau' in self.__class__.__dict__) # We set this as a variable, for user to change it at runtime. It can be accessed as a public variable through 'self.ProposesTau'
        self._NeedsLogColumn = True
        self.__LastMonitoredTimestamp = -np.inf
        
        try:
            self.__ToolIndex__ = self.__Framework__.ToolsOrder[self.__Name__]
        except:
            None

    def _GetParameters(self):
        InputDict = {}
        for Key, Value in self.__dict__.items():
            if type(Value) == types.MethodType:
                continue
            if len(Key) > 1 and Key[0] == '_' and Key[1] != '_' and Key[:7] != '_Module':
                InputDict[Key] = Value
        return InputDict

    @property
    def StreamName(self):
        '''
        Method to recover the name of the stream fed to this module.
        Looks for the closest 'Input' module generated a corresponding Camera Index Restriction
        '''
        if self.__Type__ == 'Input':
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
    def FrameworkTau(self):
        '''
        Method to retreive the highest level information Tau from the framework.
        If no Tau is proposed by any other module, will return None, so default value has to be Module specific
        '''
        return self.__Framework__._GetLowerLevelTau(self._RunningEvent, self)

    def __Initialize__(self, Parameters):
        # First restore all previous values
        self.Log(" > Initializing module")
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
                if type(Value) != type(self.__dict__[Key]):
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
        
        # Initialize the stuff corresponding to this specific module
        if not self._InitializeModule():
            return False

        # Finalize Module initialization
        if self.__Type__ != 'Input':
            OnEventMethodUsed = self.__OnEventRestricted__
        else:
            OnEventMethodUsed = self.__OnEventInput__

        if self._MonitorDt and self._ProposesTau: # We check if that method was overloaded
            if not self._MonitoredVariables:
                self.LogWarning("Enabling monitoring for Tau value")
            self._MonitoredVariables = [('AverageTau', float)] + self._MonitoredVariables

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

    @property
    def ProposesTau(self):
        return self._ProposesTau

    def _SetInputModuleSubStreamIndexes(self, Indexes):
        '''
        Specific method for input modules.
        Upon calling when framework is created, will allow the module to set its variables accordingly to what was specified in the project file.
        Cannot create a framework with an input module that has not its proper method.
        Be careful to have variables that wont be overwritten during initialization of the framework
        '''
        return False

    def __UpdateParameter__(self, Key, Value):
        self.__SavedValues__[Key] = copy.copy(self.__dict__[Key])
        self.__dict__[Key] = Value
        self.Log("Changed specific value {0} from {1} to {2}".format(Key, self.__SavedValues__[Key], self.__dict__[Key]))

    def GetSnapIndexAt(self, t):
        return (abs(np.array(self.History['t']) - t)).argmin()
    def _Restart(self):
        # Template method for restarting modules, for instant display handler. Quite specific for now
        pass
    def _InitializeModule(self):
        # Template for user-filled module initialization
        return True
    def _OnEventModule(self, event):
        # Template for user-filled module event running method
        pass
    def EventTau(self, event = None):
        # User-filled method to propose a tau for the whole framework. Ideally, thet further in the framework, the higher the information and the more acurate Tau information is
        # Return None for default, or 0 for non-defined tau yet
        pass
    @property
    def AverageTau(self):
        # For monitoring purposes, a module that proposes a Tau should propose an average tau value, that does not depend on the event. Thus, EventTau(None) will be called upon monitoring
        return self.EventTau(None)
    @property
    def MapTau(self):
        if not self._ProposesTau:
            return
        Map = np.zeros(self.Geometry)
        for x in range(Map.shape[0]):
            for y in range(Map.shape[1]):
                Map[x,y] = self.EventTau(CameraEvent(timestamp = self.__Framework__.t, location = [x,y], polarity = None, SubStreamIndex = None))
        return Map
    def _OnSnapModule(self):
        # Template for user-filled module preparation for taking a snapshot. 
        pass
    def _Pause(self, PauseOrigin):
        # Template method for module implications when the framework is paused. The origin of the pause is given to void the origin itself to apply consequences a second time
        # Useful especially for threaded modules
        pass
    def _Resume(self):
        # Template method for module implications when the framework is resumed after pause
        # Useful especially for threaded modules
        pass
    def _SaveAdditionalData(self, ExternalDataDict):
        # Template method to save additional data from module, when Framework 'SaveData' is called.
        # Data 'History' from automatic monitoring is already saved. 
        # Insert freely DataDict[Key] = Data as references as much as possible. Each Data chunk is assumed to be easily pickled (reasonable size).
        pass
    def _RecoverAdditionalData(self, ExternalDataDict):
        # Template method to recover additional data from module, when Framework 'RecoverData' is called.
        # Data 'History' from automatic monitoring is already recovered. 
        pass
    def _OnClosing(self):
        # Template method to be called when python closes. 
        # Used for closing any files or connexions that could have been made. Use that rather than another atexit call
        pass

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
        OnEventMethodUsed(eventContainer)
        if eventContainer.timestamp - self.__LastMonitoredTimestamp > self._MonitorDt:
            self.__LastMonitoredTimestamp = eventContainer.timestamp
            self._OnSnapModule()
            self.History['t'] += [eventContainer.timestamp]
            for VarName, RetreiveMethod in self.__MonitorRetreiveMethods.items():
                self.History[VarName] += [RetreiveMethod()]
        return eventContainer.IsFilled

    def _IsParent(self, ChildModule):
        if ChildModule.__ToolIndex__ < self.__ToolIndex__:
            return False
        for SubStreamIndex in self.__SubStreamOutputIndexes__:
            if SubStreamIndex in ChildModule.__SubStreamInputIndexes__:
                return True
        return False

    def _IsChile(self, ParentModule):
        if ParentModule.__ToolIndex__ > self.__ToolIndex__:
            return False
        for SubStreamIndex in self.__SubStreamInputIndexes__:
            if SubStreamIndex in ParentModule.__SubStreamOutputIndexes__:
                return True
        return False

    @property
    def PicturesFolder(self):
        return self.__Framework__.PicturesFolder

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

    def PlotHistoryData(self, MonitoredVariable, fax = None):
        t = np.array(self.History['t'])
        Data = np.array(self.History[MonitoredVariable])
        if len(Data.shape) == 0:
            raise Exception("No data saved yet")
        if len(Data.shape) == 1:
            if fax is None:
                f, ax = plt.subplots(1,1)
                ax.set_title(self.__Name__ + ' : ' + MonitoredVariable)
                ax.plot(t, Data)
            else:
                f, ax = fax
                ax.plot(t, Data, label = self.__Name__ + ' : ' + MonitoredVariable)
                ax.legend()
            return f, ax
        if len(Data.shape) == 2:
            if fax is None:
                f, ax = plt.subplots(1,1)
                ax.set_title(self.__Name__ + ' : ' + MonitoredVariable)
            else:
                f, ax = fax
            for nCol in range(Data.shape[1]):
                ax.plot(t, Data[:,nCol], label = str(nCol))
            ax.legend()
            return f, ax
        if len(Data.shape) > 2:
            raise Exception("Matrices unfit to be plotted")

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

        ExternalDataDict = {}
        self._SaveAdditionalData(ExternalDataDict)
        if ExternalDataDict:
            cPickle.dump(('.'.join([self.__Name__, 'External']), ExternalDataDict), BinDataFile)

    def SaveCSVData(self, FileName, Variables = [], FloatPrecision = 6, Default2DIndexes = 'xy', Default3DIndexes = 'xyz', Separator = ' '):
        if not Variables:
            if not self._MonitoredVariables:
                self.LogWarning("No CSV export as no data was being monitored")
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
            TemplateVar = self.History[Key][0]
            if Type == np.array:
                Size = TemplateVar.flatten().shape[0]
                if Size > 9: # We remove here anything that would be too big for CSV files, like frames, ST-contexts, ...
                    self.LogWarning("Avoiding monitored variable {0} as its shapes is not fitted for CSV files".format(Key))
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
                for FormatFunction, (Key, Index) in zip(FormatFunctions, CSVDataAccess):
                    if Index is None:
                        Data += [FormatFunction(self.History[Key][nLine])]
                    else:
                        Data += [FormatFunction(self.History[Key][nLine][Index])]
                fCSV.write(Separator.join(Data)+'\n')
            self.LogSuccess("Saved {0} data in {1}".format(self.__Name__, FileName))

    def _RecoverData(self, Identifier, Data):
        if Identifier == 'External':
            self._RecoverAdditionalData(Data)
            self.LogSuccess("Recovered additional data")
            return
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

    def Log(self, Message, MessageType = 0):
        '''
        Log system to be used for verbose. for more clear information.
        Message :  str, message specific to the module
        MessageType : int. 0 for simple information, 1 for warning, 2 for error, stopping the stream, 3 for green highlight
        '''
        self.__Framework__._Log(Message, MessageType, self)
    def LogWarning(self, Message):
        self.Log(Message, 1)
    def LogError(self, Message):
        self.Log(Message, 2)
        self.__Framework__.Paused = self.__Name__
    def LogSuccess(self, Message):
        self.Log(Message, 3)

class _EventContainerClass: # Holds the timestamp and manages the subStreamIndexes and extensions. Should in theory never be used as such.
    def __init__(self, timestamp = None, FirstEvent = None, FirstSubStreamIndex = None, Bare = False):
        if not Bare:
            self.timestamp = timestamp
            self.Events = {FirstSubStreamIndex: [FirstEvent]}
        else:
            self.BareEvent = _BareEventClass(self)
            self.timestamp = None
            self.Events = {}
    def _AddEvent(self, Extension, SubStreamIndex, **kwargs):
        if not SubStreamIndex in self.Events:
            self.Events[SubStreamIndex] = []
        self.Events[SubStreamIndex].append(Extension(Container = self, SubStreamIndex = SubStreamIndex, **kwargs))
    def GetEvents(self, SubStreamRestriction = {}):
        RequestedEvents = []
        for SubStreamIndex, Events in self.Events.items():
            if not SubStreamRestriction or SubStreamIndex in SubStreamRestriction:
                RequestedEvents += Events
        return RequestedEvents
    def Filter(self, event):
        self.Events[event.SubStreamIndex].remove(event)
        if len(self.Events[event.SubStreamIndex]) == 0:
            del self.Events[event.SubStreamIndex]
    @property
    def IsEmpty(self):
        return len(self.Events) == 0
    @property
    def IsFilled(self):
        return len(self.Events) != 0
    def __eq__(self, rhs):
        return self.timestamp == rhs.timestamp
    def __lt__(self, rhs):
        return self.timestamp < rhs.timestamp
    def __le__(self, rhs):
        return self.timestamp <= rhs.timestamp
    def __gt__(self, rhs):
        return self.timestamp > rhs.timestamp
    def __ge__(self, rhs):
        return self.timestamp >= rhs.timestamp

class _BareEventClass: # First basic event given to an input module. That input module is expected to join another event to this one, restructuring internally the event packet
    def __init__(self, Container):
        self._Container = Container
    def Join(self, Extension, **kwargs):
        if 'SubStreamIndex' in kwargs:
            SubStreamIndex = kwargs['SubStreamIndex']
            del kwargs['SubStreamIndex']
        else:
            raise Exception("No SubStreamIndex specified during first event creation")
        if 'timestamp' in kwargs:
            self._Container.timestamp = kwargs['timestamp']
            del kwargs['timestamp']
        else:
            raise Exception("No timestamp specified during first event creation")
        self._Container._AddEvent(Extension, SubStreamIndex, **kwargs)
        del self._Container.__dict__['BareEvent']
        return self._Container.Events[SubStreamIndex][-1]

class _EventClass:
    def __init__(self, **kwargs):
        if not 'Container' in kwargs:
            self._Container = _EventContainerClass(kwargs['timestamp'], self, kwargs['SubStreamIndex'])
        else:
            self._Container = kwargs['Container']
            del kwargs['Container']
        self.SubStreamIndex = kwargs['SubStreamIndex']
        del kwargs['SubStreamIndex']
        self._Extensions = set()
        if 'Extensions' in kwargs:
            for Extension in kwargs['Extensions']:
                self.Attach(Extension, **kwargs)
    def Attach(self, Extension, **kwargs):
        if not Extension._CanAttach or Extension in self._Extensions:
            self.Join(Extension, **kwargs) # For now, its better to join when instance is already there (ex: multiple TrackerEvents)
            return
        self._Extensions.add(Extension)
        for Field in Extension._Fields:
            self.__dict__[Field] = kwargs[Field]
    def Join(self, Extension, **kwargs):
        if 'SubStreamIndex' in kwargs:
            SubStreamIndex = kwargs['SubStreamIndex']
            del kwargs['SubStreamIndex']
        else:
            SubStreamIndex = self.SubStreamIndex
        self._Container._AddEvent(Extension, SubStreamIndex, **kwargs)
        return self._Container.Events[SubStreamIndex][-1]
    def Filter(self):
        self._Container.Filter(self)
    def AsList(self, Keys = ()):
        Output = [self.timestamp]
        for Extension in self._Extensions:
            if Keys and Extension._Key not in Keys:
                continue
            Output += [[Extension._Key]+[self.__dict__[Field]for Field in Extension._Fields]]
        return Output
    def AsDict(self, Keys = ()):
        Output = {0:self.timestamp}
        for Extension in self._Extensions:
            if Keys and Extension._Key not in Keys:
                continue
            Output[Extension._Key] = [self.__dict__[Field] for Field in Extension._Fields]
        return Output
    def Copy(self):
        kwargs = {'timestamp':self.timestamp, 'SubStreamIndex':self.SubStreamIndex, 'Extensions':self._Extensions}
        for Extension in self._Extensions:
            for Field in Extension._Fields:
                if type(self.__dict__[Field]) != np.ndarray:
                    kwargs[Field] = type(self.__dict__[Field])(self.__dict__[Field])
                else:
                    kwargs[Field] = np.array(self.__dict__[Field])
        return self.__class__(**kwargs)
    @property
    def timestamp(self):
        return self._Container.timestamp
    def Has(self, Extension):
        return (Extension in self._Extensions)
    def __eq__(self, rhs):
        return self.timestamp == rhs.timestamp
    def __lt__(self, rhs):
        return self.timestamp < rhs.timestamp
    def __le__(self, rhs):
        return self.timestamp <= rhs.timestamp
    def __gt__(self, rhs):
        return self.timestamp > rhs.timestamp
    def __ge__(self, rhs):
        return self.timestamp >= rhs.timestamp
    def __repr__(self):
        return "{0:.3f}s".format(self.timestamp)

# Listing all the events existing

class _EventExtensionClass:
    _Key = -1 # identifier for this type of events
    _CanAttach = True
    _Fields = ()
    def __new__(cls, *args, **kwargs):
        Event = _EventClass(**kwargs)
        Event.Attach(cls, **kwargs)
        return Event

class CameraEvent(_EventExtensionClass):
    _Key = 1
    _Fields = ('location', 'polarity')
class TrackerEvent(_EventExtensionClass):
    _Key = 2
    _CanAttach = False # From experience, many trackers can be updated upon a single event. For equity, all trackers are joined, not attached
    _Fields = ('TrackerLocation', 'TrackerID', 'TrackerAngle', 'TrackerScaling', 'TrackerColor', 'TrackerMarker')
class DisparityEvent(_EventExtensionClass):
    _Key = 3
    _Fields = ('disparity', 'sign')
class PoseEvent(_EventExtensionClass):
    _Key = 4
    _Fields = ('poseHomography', 'worldHomography', 'reprojectionError')
class TauEvent(_EventExtensionClass):
    _Key = 5
    _Fields = ('tau',)
class FlowEvent(_EventExtensionClass):
    _Key = 6
    _Fields = ('flow',)
class OdometryEvent(_EventExtensionClass):
    _Key = 7
    _Fields = ('omega', 'v')

class ParametersDictClass(dict):
    def __init__(self, *args):
        super().__init__(*args)
    def __setitem__(self, key, value):
        if key[0] == '*':
            key = key[1:]
            if key[0] == '_':
                Found = False
                for subKey in self.keys():
                    if key in self[subKey].keys():
                        Found = True
                        self[subKey][key] = value
                if not Found:
                    raise KeyError(key)
            else:
                Pattern, key = key.split('_')
                key = '_'+key
                Found = False
                for subKey in self.keys():
                    if Pattern in subKey:
                        if key in self[subKey].keys():
                            Found = True
                            self[subKey][key] = value
                if not Found:
                    raise KeyError(key)
        else:
            super().__setitem__(key, value)
