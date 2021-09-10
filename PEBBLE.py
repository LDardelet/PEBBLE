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
from datetime import datetime
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
    _RUN_FILE = 'running.lock'

    '''
    Main event-based framework file.
    '''
    def __init__(self, File1 = None, File2 = None, onlyRawData = False, interpreter_in_out = None):
        self._InterpreterInOut = interpreter_in_out
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

        if self._InterpreterInOut is None:
            self._Log("No interpreter container specified. No interactive commands will be saved", 1)

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
        self.InputEvents = {ModuleName: None for ModuleName in self.ModulesList if self.Modules[ModuleName].__IsInput__}
        self.InputModulesTuple = tuple([ModuleName for ModuleName in self.ModulesList if self.Modules[ModuleName].__IsInput__])
        if len(self.InputEvents) == 1:
            self._NextInputEventMethod = self._SingleInputModuleNextInputEventMethod
        else:
            self._NextInputEventMethod = self._MultipleInputModulesNextInputEventMethod

        self._LogType = 'columns'
        self._LogInit()
        for ModuleName in self.ModulesList:
            if not ModuleBase.__Initialize__(self.Modules[ModuleName], self.RunParameters[ModuleName]):
                self._Log("Module {0} failed to initialize. Aborting.".format(ModuleName), 2)
                self._DestroyFolder()
                return False
        self._RunModulesMethodTuple = tuple([self.Modules[ModuleName].__OnEvent__ for ModuleName in self.ModulesList if not self.Modules[ModuleName].__IsInput__]) # Faster way to access tools in the right order, and only not input modules as they are dealt with through _NextInputEvent
        self._Log("Framework initialized", 3, AutoSendIfPaused = False)
        self._Log("")
        self._SendLog()
        self._Initializing = False
        return True

    def _GetCameraIndexChain(self, Index):
        ModulesChain = []
        for ModuleName in self.ModulesList:
            if not self.Modules[ModuleName].__SubStreamOutputIndexes__ or Index in self.Modules[ModuleName].__SubStreamOutputIndexes__:
                ModulesChain += [ModuleName]
        return ModulesChain

    def _GetParentModule(self, Module):
        ModuleEventsRestriction = Module.__SubStreamInputIndexes__
        for InputModuleName in reversed(self.ModulesList[:Module.__ModuleIndex__]):
            InputModule = self.Modules[InputModuleName]
            if not ModuleEventsRestriction:
                return InputModule
            for CameraIndex in InputModule.__SubStreamOutputIndexes__:
                if CameraIndex in ModuleEventsRestriction:
                    return InputModule
        self._Log("{0} was unable to find its parent module".format(Module.__Name__), 1)

    def _GetStreamGeometry(self, Module):
        '''
        Method to retreive the geometry of the events handled by a tool
        '''
        return self._GetParentModule(Module).OutputGeometry

    def _GetStreamFormattedName(self, Module):
        '''
        Method to retreive a formatted name depending on the files providing events to this tool.
        Specifically useful for an Input type tool to get the file it has to process.
        '''
        return self._GetParentModule(Module).StreamName

    def _GetLowerLevelTau(self, EventConcerned, ModuleAsking):
        NameAsking = ModuleAsking.__Name__
        EventTau = None
        for NameProposing in reversed(self.ModulesList[:ModuleAsking.__ModuleIndex__]):
            ModuleProposing = self.Modules[NameProposing]
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
        else:
            ParametersDict = None
        if self._LogType == 'columns':
            self._LogInit(resume)
        if self._RunProcess(StreamName = StreamName, ParametersDict = ParametersDict, start_at = start_at, stop_at = stop_at, resume = resume, AtEventMethod = AtEventMethod):
            self.SaveCSVData()
            self._SetRunFile(False)
        self._LogType = 'raw'

    def _GetCommitValue(self):
        try:
            f = open('.git/FETCH_HEAD', 'r')
            commit_value = f.readline().strip()
            f.close()
        except:
            commit_value = "unknown"
        return commit_value

    def _InitiateFolder(self, suffix = ''):
        self._FolderData = {'home':_RUNS_DIRECTORY + datetime.now().strftime("%Y-%m-%d_%H-%M") + suffix + '/',
                            'history':None,
                            'pictures':None}
        try:
            os.mkdir(self._FolderData['home'])
        except:
            return self._InitiateFolder(suffix + '_bis')
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
            for ModuleName in self.ModulesList:
                self.Modules[ModuleName]._OnClosing()
            self._SessionLogs.pop(1).close()
            return

        if self._FolderData['home'] is None:
            return

        if not self._CSVDataSaved:
            self.SaveCSVData() # By default, we save the data. Files shouldn't be too big anyway
        for ModuleName in self.ModulesList:
            self.Modules[ModuleName]._OnClosing()
        self._SetRunFile(False)
        self._SaveInterpreterData()
        self._SessionLogs.pop(1).close()

    def _DestroyFolder(self):
        shutil.rmtree(self._FolderData['home'])
        self._FolderData['home'] = None

    def _SetRunFile(self, Set):
        if Set:
            with open(self._FolderData['home'] + self._RUN_FILE, 'w') as _:
                pass
            self._RunFileSet = True
        else:
            if not self._RunFileSet:
                return
            else:
                try:
                    os.remove(self._FolderData['home'] + self._RUN_FILE)
                except FileNotFoundError:
                    self._Log("Unable to remove lock", 1)
                self._RunFileSet = False

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

    def _SaveRunStreamData(self, start_at, stop_at, resume):
        if not resume:
            fInputs = open(self.InputsLogFile, 'w')
            fInputs.write("Project {0}\n".format(self.ProjectFile.split("/")[-1].split(".json")[0]))
            for Module, InputFile in self.CurrentInputStreams.items():
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

    def _SaveInterpreterData(self):
        if self._InterpreterInOut is None:
            self._Log("Unable to save interpreter data", 1)
            return

        with open(self.InterpreterLogFile, 'w') as fInterpreter:
            In, Out = self._InterpreterInOut
            for nInput, Input in enumerate(In):
                if not Input:
                    continue
                fInterpreter.write('\n'+10*'#'+ " {0} ".format(nInput) + 10*'#' + '\n\n')
                fInterpreter.write(Input + '\n')
                if nInput in Out:
                    fInterpreter.write("\n -> " + str(Out[nInput]) + '\n')
            self._Log("Saved interpreter commands in {0}".format(self.InterpreterLogFile), 3)

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
    @property 
    def InterpreterLogFile(self):
        # Stores any ipython interpreter I/O into a file. Might be quite messy, but allows to fully save what has been done within the session
        return self._FolderData['home']+'interpreter.txt'

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
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            BinWrite("# Edited on -> "+ dt_string + "\n")
            commit_value = self._GetCommitValue()
            BinWrite("# Framework git commit -> " + commit_value + "\n")
            BinWrite("# Project File -> " + os.path.abspath(self.ProjectFile) + '\n')
            BinWrite("# Project Hash -> " + str(hash(json.dumps(self._ProjectRawData, sort_keys=True))) + "\n")
            BinWrite("# Input Files -> " + str({InputModule : os.path.abspath(InputFile) for InputModule, InputFile in self.CurrentInputStreams.items()}) + "\n")
            BinWrite("# Run arguments -> " + str(self.RunKwargs) + '\n')
            BinWrite("#########\n")

            for ModuleName in self.ModulesList:
                self.Modules[ModuleName]._SaveData(BinDataFile)

    def SaveCSVData(self, Folder = None, FloatPrecision = 6, Default2DIndexes = 'xy', Default3DIndexes = 'xyz', Separator = ' '):
        if Folder is None:
            Folder = self.HistoryFolder
        else:
            if Folder[-1] != '/':
                Folder = Folder + '/'
        for ModuleName in self.ModulesList:
            FileName = Folder + ModuleName + '.csv'
            self.Modules[ModuleName].SaveCSVData(FileName, FloatPrecision = 6, Default2DIndexes = 'xy', Default3DIndexes = 'xyz', Separator = ' ')
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
                    ModuleName = Identifier.split('.')[0]
                    self.Modules[ModuleName]._RecoverData(Identifier[len(ModuleName)+1:], Data)
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
            self.CurrentInputStreams = {ModuleName:None for ModuleName in self.ModulesList if self.Modules[ModuleName].__IsInput__}
            if type(StreamName) == str:
                for ModuleName in self.CurrentInputStreams.keys():
                    self.CurrentInputStreams[ModuleName] = StreamName
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
            
            self._SaveRunStreamData(start_at, stop_at, resume)
            self.StreamHistory += [self.CurrentInputStreams]
            InitializationAnswer = self.Initialize()
            if not InitializationAnswer or BareInit:
                self._SendLog()
                return False
            self._CSVDataSaved = False

        self.Running = True
        self.Paused = ''
        if resume:
            for ModuleName in self.ModulesList:
                self.Modules[ModuleName]._Resume()

        self.t = 0.
        if start_at > 0:
            for ModuleName in self.InputEvents.keys():
                if '_Warp' in self.Modules[ModuleName].__class__.__dict__:
                    self._Log("Warping {0} through module method".format(ModuleName))
                    self._SendLog()
                    if not self.Modules[ModuleName]._Warp(start_at):
                        self._Log("No event remaining in {0} file".format(ModuleName), 2)

        self._SetRunFile(True)

        while not resume and start_at > 0 and self.Running and not self.Paused:
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
        if not self.Running:
            self._Log("Main loop finished without error.")
            return True
        else:
            if self.Paused:
                self._Log("Paused at t = {0:.3f}s by {1}.".format(self.t, self.Paused), 1)
                for ModuleName in self.ModulesList:
                    self.Modules[ModuleName]._Pause(self.Paused)
            return False

    def Resume(self, stop_at = np.inf):
        self._LogType = 'columns'
        self.RunStream(self.StreamHistory[-1], stop_at = stop_at, resume = True)

    def DisplayRestart(self):
        for ModuleName in self.ModulesList:
            self.Modules[ModuleName]._Restart()

    def Next(self, AtEventMethod = None):
        self.PropagatedContainer = self._NextInputEventMethod()
        if self.PropagatedContainer is None:
            return None
        t = self.PropagatedContainer.timestamp
        for RunMethod in self._RunModulesMethodTuple:
            if not AtEventMethod is None:
                AtEventMethod(self.PropagatedContainer)
            if not RunMethod(self.PropagatedContainer):
                break
        if not AtEventMethod is None and not self.PropagatedContainer is None:
            AtEventMethod(self.PropagatedContainer)

        return t

    def _SingleInputModuleNextInputEventMethod(self):
        return self.Modules[self.InputModulesTuple[0]].__OnEvent__(_EventContainerClass(Bare = True))

    def _MultipleInputModulesNextInputEventMethod(self):
        OldestEvent, ModuleSelected = None, None
        for InputName, EventAwaiting in self.InputEvents.items():
            if EventAwaiting is None:
                EventAwaiting = self.Modules[InputName].__OnEvent__(_EventContainerClass(Bare = True))
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
        self.Modules = {}
        self._ModulesModulesLinks = {}
        self._ModulesProjectParameters = {}
        self._ModulesSubStreamsHandled = {}
        self._ModulesSubStreamsCreated = {}
        self._ModulesClasses = {}
        self._SubStreamIndexes = set()

        self.ModulesOrder = {}
        self.ModulesList = []

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

        for ModuleName in self._ProjectRawData.keys():
            fileLoaded = __import__(self._ProjectRawData[ModuleName]['File'])
            self._ModulesClasses[ModuleName] = getattr(fileLoaded, self._ProjectRawData[ModuleName]['Class'])

            self._ModulesModulesLinks[ModuleName] = self._ProjectRawData[ModuleName]['ModulesLinks']
            self._ModulesProjectParameters[ModuleName] = self._ProjectRawData[ModuleName]['ProjectParameters']
            self._ModulesSubStreamsHandled[ModuleName] = self._ProjectRawData[ModuleName]['SubStreamsHandled']
            self._ModulesSubStreamsCreated[ModuleName] = self._ProjectRawData[ModuleName]['SubStreamsCreated']

            self.ModulesOrder[ModuleName] = self._ProjectRawData[ModuleName]['Order']
            self._Log("Imported tool {1} from file {0}.".format(self._ProjectRawData[ModuleName]['File'], self._ProjectRawData[ModuleName]['Class']))

        if len(self.ModulesOrder.keys()) != 0:
            MaxOrder = max(self.ModulesOrder.values()) + 1
            self.ModulesList = [None] * MaxOrder
        else:
            self.ModulesList = []

        for ModuleName in self.ModulesOrder.keys():
            if self.ModulesList[self.ModulesOrder[ModuleName]] is None:
                self.ModulesList[self.ModulesOrder[ModuleName]] = ModuleName
            else:
                self._Log("Double assignement of number {0}. Aborting ProjectFile loading".format(self.ModulesOrder[ModuleName]), 2)
                return None
        while None in self.ModulesList:
            self.ModulesList.remove(None)

        self._Log("Successfully generated tools order", 3)
        self._Log("")
        
        for ModuleName in self.ModulesList:
            self.Modules[ModuleName] = self._ModulesClasses[ModuleName](ModuleName, self, self._ModulesModulesLinks[ModuleName])
            if not self._UpdateModuleParameters(ModuleName):
                raise Exception
            for Index in self.Modules[ModuleName].__SubStreamOutputIndexes__:
                if Index not in self._SubStreamIndexes:
                    self._SubStreamIndexes.add(Index)
                    self._Log("Added SubStream {0}".format(Index), 3)

            if enable_easy_access and ModuleName not in self.__dict__.keys():
                self.__dict__[ModuleName] = self.Modules[ModuleName]
        self._CSVDataSaved = True
        self._Log("Successfully generated Framework", 3)
        self._Log("")

    def _UpdateModuleParameters(self, ModuleName):
        Module = self.Modules[ModuleName]
        for key, value in self._ModulesProjectParameters[ModuleName].items():
            if key in Module.__dict__.keys():
                try:
                    Module.__dict__[key] = type(Module.__dict__[key])(value)
                except:
                    self._Log("Issue with setting the new value of {0} for tool {1}, should be {2}, impossible from {3}".format(key, ModuleName, type(Module.__dict__[key]), value), 1)
            else:
                self._Log("Key {0} for tool {1} doesn't exist. Please check ProjectFile integrity.".format(key, ModuleName), 1)
        if Module.__IsInput__:
            Module.__SubStreamInputIndexes__ = set()
            Module.__SubStreamOutputIndexes__ = set(self._ModulesSubStreamsCreated[ModuleName])
            if '_SetGeneratedSubStreamsIndexes' not in Module.__class__.__dict__:
                self._Log("Missing adequate _SetGeneratedSubStreamsIndexes method for input tool")
                return False
            if not Module._SetGeneratedSubStreamsIndexes(self._ModulesSubStreamsCreated[ModuleName]):
                return False
        else:
            if not self._ModulesSubStreamsHandled[ModuleName]:
                Module.__SubStreamInputIndexes__ = set(self._SubStreamIndexes)
            else:
                Module.__SubStreamInputIndexes__ = set(self._ModulesSubStreamsHandled[ModuleName])
            Module.__SubStreamOutputIndexes__ = set(Module.__SubStreamInputIndexes__)
            if Module.__GeneratesSubStream__:
                for SubStreamIndex in self._ModulesSubStreamsCreated[ModuleName]:
                    Module.__SubStreamOutputIndexes__.add(SubStreamIndex)
                if '_SetGeneratedSubStreamsIndexes' not in Module.__class__.__dict__:
                    self._Log("Missing adequate _SetGeneratedSubStreamsIndexes method for input tool")
                    return False
                if not Module._SetGeneratedSubStreamsIndexes(self._ModulesSubStreamsCreated[ModuleName]):
                    return False
        return True

    def GetModulesParameters(self):
        ParametersDict = {}
        for ModuleName in self.ModulesList:
            ParametersDict[ModuleName] = self.Modules[ModuleName]._GetParameters()
        return ParametersDictClass(ParametersDict)

    def SaveProject(self, ProjectFile):
        GivenExtention = ProjectFile.split('.')[-1]
        if GivenExtention != self._PROJECT_FILE_EXTENSION:
            raise Exception("Enter log file with .{0} extension".format(self._PROJECT_FILE_EXTENSION))
        pickle.dump(self._ProjectRawData, open(ProjectFile, 'wb'))
        self.ProjectFile = ProjectFile
        self.Modified = False
        self._Log("Project saved.", 3)

    def AddModule(self):
        self._Log("Current project :")
        self.Project()
        self._Log("")
        
        Name = input('Enter the name of the new tool : ')                   # Module name 
        if Name == '' or Name in self._ProjectRawData.keys():
            self._Log("Invalid entry (empty or already existing).", 2)
            return None
        self._ProjectRawData[Name] = {'File':None, 'Class':None, 'ModulesLinks':{}, 'ProjectParameters':{}, 'SubStreamsHandled':[], 'Order':None}
        ModuleProjectDict = self._ProjectRawData[Name]
        try:
            entry = ''
            while entry == '' :                                                 # Filename and Class definition
                self._Log("Enter the tool filename :")
                entry = input('')
            if '.py' in entry:
                entry = entry.split('.py')[0]
            ModuleProjectDict['File'] = entry
            fileLoaded = __import__(entry)
            PossibleClasses = []
            if not 'ModuleBase' in fileLoaded.__dict__.keys():
                self._Log("File does not contain any class derived from 'Module'. Aborting entry", 2)
                raise Exception("No ModuleBase-dereived class")
            for key in fileLoaded.__dict__.keys():
                if isinstance(fileLoaded.__dict__[key], type) and key[0] != '_' and fileLoaded.__dict__['ModuleBase'] in fileLoaded.__dict__[key].__bases__:
                    PossibleClasses += [key]
            if len(PossibleClasses) == 0:
                self._Log("No possible Class is available in this file. Aborting.", 2)
                raise Exception("No ModuleBase-dereived class")
            elif len(PossibleClasses) == 1:
                self._Log("Using class {0}".format(PossibleClasses[0]))
                ModuleProjectDict['Class'] = PossibleClasses[0]
            else:
                entry = ''
                while entry == '' :
                    self._Log("Enter the tool class among the following ones :")
                    for Class in PossibleClasses:
                        self._Log(" * {0}".format(Class))
                    entry = input('')
                if entry not in PossibleClasses:
                    self._Log("Invalid class, absent from tool file or not a ClassType.", 2)
                    raise Exception("Wrong entry")
                ModuleProjectDict['Class'] = entry
            self._Log("")
                                                                                  # Loading the class to get the references needed and parameters

            TmpClass = fileLoaded.__dict__[ModuleProjectDict['Class']](Name, self, {})
            ModulesLinksRequested = TmpClass.__ModulesLinksRequested__

            if not TmpClass.__IsInput__:
                self._Log("Enter the tool order number. Void will add that tool at the end of the current framework.")                             # Order definition
                entry = input('')
                if entry:
                    if int(entry) >= len(self._ProjectRawData):
                        self._Log("Excessive tool number, assuming last position")
                        ModuleProjectDict['Order'] = len(self._ProjectRawData)-1
                    else:
                        ModuleProjectDict['Order'] = int(entry)
                else:
                    ModuleProjectDict['Order'] = len(self._ProjectRawData)-1
            else:
                self._Log("Input Type detected. Setting to next default index.")
                ModuleProjectDict['Order'] = 0
                for OtherModuleName in self._ProjectRawData.keys():
                    if OtherModuleName != Name and TmpClass.__IsInput__ :
                        ModuleProjectDict['Order'] = max(ModuleProjectDict['Order'], self._ProjectRawData[OtherModuleName]['Order'] + 1)
            NumberTaken = False
            for OtherModuleName in self._ProjectRawData.keys():
                if OtherModuleName != Name and 'Order' in self._ProjectRawData[OtherModuleName].keys() and ModuleProjectDict['Order'] == self._ProjectRawData[OtherModuleName]['Order']:
                    NumberTaken = True
            if NumberTaken:
                self._Log("Compiling new order.")
                for OtherModuleName in self._ProjectRawData.keys():
                    if OtherModuleName != Name:
                        if self._ProjectRawData[OtherModuleName]['Order'] >= ModuleProjectDict['Order']:
                            self._ProjectRawData[OtherModuleName]['Order'] += 1
                self._Log("Done")
                self._Log("")

            if ModulesLinksRequested:
                self._Log("Fill tool name for the needed links. Currently available tool names:")
                for key in self._ProjectRawData.keys():
                    if key == Name:
                        continue
                    self._Log(" * {0}".format(key))
                self._Log("Void will create no link, may lead to errors")
                for ModuleLinkRequested in ModulesLinksRequested:
                    self._Log("Reference for '" + ModuleLinkRequested + "'")
                    entry = input('-> ')
                    ModuleProjectDict['ModulesLinks'][ModuleLinkRequested] = entry
            else:
                self._Log("No particular reference needed for this tool.")
            self._Log("")
            if not TmpClass.__IsInput__:
                self._Log("Enter streams index(es) handled by this module, coma separated. Void will not create any restriction.")
                entry = input(" -> ")
                if entry:
                    ModuleProjectDict['SubStreamsHandled'] = [int(index.strip()) for index in entry.split(',')]

            if TmpClass.__IsInput__ or TmpClass.__GeneratesSubStream__:
                self._Log("Enter indexes created by this module, coma separated.")
                entry = ''
                while entry == '':
                    entry = input('-> ')
                ModuleProjectDict['SubStreamsCreated'] = [int(index.strip()) for index in entry.split(',')]
            else:
                ModuleProjectDict['SubStreamsCreated'] = []

            PossibleVariables = []
            for var in TmpClass.__dict__.keys():
                if var[0] == '_' and var[1] != '_':
                    PossibleVariables += [var]
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
                                    ModuleProjectDict['ProjectParameters'][entryvar] = [aimedType(value) for value in values]
                                elif type(TmpClass.__dict__[entryvar]) == bool:
                                    if entryvalue == 'False':
                                        ModuleProjectDict['ProjectParameters'][entryvar] = False
                                    elif entryvalue == 'True':
                                        ModuleProjectDict['ProjectParameters'][entryvar] = True
                                    else:
                                        raise ValueError
                                else:
                                    ModuleProjectDict['ProjectParameters'][entryvar] = type(TmpClass.__dict__[entryvar])(entryvalue)
                            except ValueError:
                                self._Log("Could not parse entry into the correct type", 1)
                    elif '=' in entryvar:
                        entryvar, entryvalue = entryvar.split('=')
                        if entryvar.strip() in PossibleVariables:
                            try:
                                ModuleProjectDict['ProjectParameters'][entryvar.strip()] = type(TmpClass.__dict__[entryvar.strip()])(entryvalue.strip())
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
        except Exception as e:
            print(e)
            del self._ProjectRawData[Name]
            return None

        self._Log("AddModule finished. Reloading project.")
        self._LoadProject()
        self._Log("New project : ")
        self._Log("")
        self.Project()
        self.Modified = True

    def DelModule(self, ModuleName):
        ModuleIndex = self._ProjectRawData[ModuleName]['Order']
        del self._ProjectRawData[ModuleName]
        for RemainingModuleName, RemainingRawData in self._ProjectRawData.items():
            if RemainingRawData['Order'] > ModuleIndex:
                RemainingRawData['Order'] -= 1

        self._Log("DelModule finished. Reloading project.")
        self._LoadProject()
        self._Log("New project : ")
        self._Log("")
        self.Project()
        self.Modified = True

    def Project(self):
        self._Log("# Framework\n", 3)
        self._Log("")

        nOrder = 0
        for ModuleName in self.ModulesList:
            self._Log("# {0} : {1}, from class {2} in file {3}.".format(nOrder, ModuleName, str(self.Modules[ModuleName].__class__).split('.')[1][:-2], self._ProjectRawData[ModuleName]['File']), 3)
            if not self.Modules[ModuleName].__IsInput__:
                if self.Modules[ModuleName].__SubStreamInputIndexes__:
                    self._Log("     Uses cameras indexes " + ", ".join([str(CameraIndex) for CameraIndex in self.Modules[ModuleName].__SubStreamInputIndexes__])+".")
                else:
                    self._Log("     Uses all cameras inputs.")
            else:
                self._Log("     Does not consider incomming events.")
            if self.Modules[ModuleName].__IsInput__ or self.Modules[ModuleName].__GeneratesSubStream__:
                self._Log("     Generates SubStream "+", ".join(self.Modules[ModuleName].__GeneratedSubStreams__))
            if self.Modules[ModuleName].__SubStreamOutputIndexes__  and not self.Modules[ModuleName].__SubStreamOutputIndexes__ == self.Modules[ModuleName].__SubStreamInputIndexes__:
                self._Log("     Outputs specific indexes " + ", ".join([str(CameraIndex) for CameraIndex in self.Modules[ModuleName].__SubStreamOutputIndexes__]))
            else:
                self._Log("     Outputs the same camera indexes.")
            SortedAttachments = self._AnalyzeModuleAttachements(ModuleName)
            if not SortedAttachments:
                #self._Log("     Does not add event data.")
                pass
            else:
                if SortedAttachments['DefaultEvent'][1]:
                    self._Log("     Attaches {0} information to the running event.".format(', '.join(SortedAttachments['DefaultEvent'][1])))
                for NewEventName, (SubStream, NewEventData) in SortedAttachments.items():
                    if NewEventName == 'DefaultEvent':
                        continue
                    self._Log("     Adds a new event with {0} for substream {1}.".format(', '.join(SortedAttachments[NewEventName][1]), SubStream))
            if self._ModulesModulesLinks[ModuleName]:
                self._Log("     Modules Linked:")
                for argName, toolLinked in self._ModulesModulesLinks[ModuleName].items():
                    self._Log("         -> Access to {0} from tool {1}".format(argName, toolLinked))
            else:
                self._Log("     No module linked.")
            if self._ModulesProjectParameters[ModuleName]:
                self._Log("     Modified Parameters:")
                for var, value in  self._ModulesProjectParameters[ModuleName].items():
                    self._Log("         -> {0} = {1}".format(var, value))
            else:
                self._Log("     Using default parameters.")
            self._Log("")
            
            nOrder += 1

    def VisualProject(self):
        f, ax = plt.subplots(1,1)
        ax.tick_params('both', left = False, labelleft = False, bottom = False, labelbottom = False)
        SubStreamsModuleLevels = {Index:0 for Index in self._SubStreamIndexes}
        SubStreamsModulesOutput = {Index:None for Index in self._SubStreamIndexes}
        SubStreamsColors = {Index:None for Index in self._SubStreamIndexes}
        ModuleWidth = 0.6
        ModuleHeight = 0.5
        from matplotlib.patches import Rectangle
        def DrawConnexion(ax, Start, End, Index):
            if SubStreamsColors[Index] is None:
                SubStreamsColors[Index] = ax.plot([Start[1], End[1]], [Start[0], End[0]], label = str(Index))[0].get_color()
            else:
                ax.plot([Start[1], End[1]], [Start[0], End[0]], color = SubStreamsColors[Index])
        for ModuleName in self.ModulesList:
            Module = self.Modules[ModuleName]
            if not Module.__SubStreamInputIndexes__:
                Col = list(Module.__SubStreamOutputIndexes__)[0]
                Level = 0
            else:
                Col = np.mean(list(Module.__SubStreamInputIndexes__))
                Level = max([SubStreamsModuleLevels[Index] for Index in Module.__SubStreamInputIndexes__])
            R = Rectangle((Col-ModuleWidth/2, (Level-ModuleHeight/2)), ModuleWidth, ModuleHeight, color = 'k', fill = False)
            ax.add_artist(R)
            ax.text(Col, Level, ModuleName, color = 'k', va= 'center', ha = 'center')
            for nIndex, Index in enumerate(np.sort(list(Module.__SubStreamInputIndexes__))):
                DrawConnexion(ax, SubStreamsModulesOutput[Index], ((Level-ModuleHeight/2), Col - ModuleWidth / 2 + (nIndex + 1) * ModuleWidth / (len(Module.__SubStreamInputIndexes__)+1)), Index)
                SubStreamsModulesOutput[Index] = ((Level+ModuleHeight/2), Col - ModuleWidth / 2 + (nIndex + 1) * ModuleWidth / (len(Module.__SubStreamOutputIndexes__)+1))
            for nIndex, Index in enumerate(np.sort(list(Module.__SubStreamOutputIndexes__))):
                SubStreamsModuleLevels[Index] = Level+1
                if SubStreamsModulesOutput[Index] is None:
                    SubStreamsModulesOutput[Index] = ((Level+ModuleHeight/2), Col - ModuleWidth / 2 + (nIndex + 1) * ModuleWidth / (len(Module.__SubStreamOutputIndexes__)+1))
        ax.set_xlim(min(list(self._SubStreamIndexes)) - ModuleWidth, max(list(self._SubStreamIndexes)) + ModuleWidth)
        ax.set_ylim(max(list(SubStreamsModuleLevels.values()))-ModuleHeight, -ModuleHeight)
        ax.legend()

    def _AnalyzeModuleAttachements(self, ModuleName):
        ModuleClassName, ModuleFile = self._ProjectRawData[ModuleName]['Class'], self._ProjectRawData[ModuleName]['File']
        def AddEventMod(RawArguments, FoundAttachments, KW, nHLine, AttachedEvent, CreatedEvent, Module):
            Arguments = [Argument.strip() for Argument in RawArguments.strip().strip('(').strip(')').split(',')]
            if not Arguments[0] in _AvailableEventsClassesNames:
                return
            SubStreamSpec = 'running'
            if KW == 'Join':
                for RawKwarg in Arguments[1:]:
                    try:
                        Key, Value = [part.strip() for part in RawKwarg.split('=')]
                    except ValueError:
                        pass
                    if Key == 'SubStreamIndex':
                        try:
                            if AttachedEvent in Value and Value != AttachedEvent+'.SubStreamIndex':
                                raise Exception
                            else:
                                if 'self' not in Value:
                                    raise Exception
                                Value = Value.split('self.')[-1]
                                if Value in Module.__dict__:
                                    SubStreamSpec = "{0}".format(Module.__dict__[Value])
                                else:
                                    raise Exception
                        except:
                            SubStreamSpec = '?'
            FoundAttachments[KW].add((Arguments[0], SubStreamSpec, AttachedEvent, CreatedEvent))
            FoundAttachments['Mods'] = True
        with open(ModuleFile+'.py', 'r') as f:
            lines = f.readlines()
        InClass = False
        ModuleIndent = 0
        InEventModification = ''
        FoundAttachments = {'Join':set(), 'Attach':set(), 'Mods':False}
        CreatedEvents = set()
        for nLine, line in enumerate(lines):
            nHLine = nLine + 1
            if (not InClass) and ('class '+ModuleClassName not in line):
                continue
            if not InClass:
                InClass = True
                continue
            LineIndent = line.index(line.strip())
            if ModuleIndent == 0:
                ModuleIndent = LineIndent
            if not line.strip() or line.strip()[0] == '#':
                continue
            if LineIndent < ModuleIndent:
                InClass = False
                break
            for CreatedEvent in set(CreatedEvents):
                if CreatedEvent[1] > LineIndent:
                    CreatedEvents.discard(CreatedEvent)
            if not InEventModification:
                if 'Attach' in line:
                    InEventModification = 'Attach'
                    CreatedEvent = ''
                elif 'Join' in line:
                    InEventModification = 'Join'
                    PreModRawData = line.strip().split(InEventModification)[0]
                    if '=' in PreModRawData:
                        CreatedEvent = PreModRawData.split('=')[0].strip()
                    else:
                        MinValue = 0
                        PrefixDefaultName = 'RandomEvent#'
                        for (_, _, _, AlreadyCreatedEvent) in FoundAttachments['Join']:
                            if PrefixDefaultName in AlreadyCreatedEvent:
                                MinValue = int(AlreadyCreatedEvent.split(PrefixDefaultName)[-1]) + 1
                        CreatedEvent = 'RandomEvent_{0}'.format(MinValue)
                if InEventModification:
                    AttachedEvent = line.strip().split(InEventModification)[0].split('=')[-1].strip()[:-1]
                    RawArguments = line.strip().split(InEventModification)[1]
                    if RawArguments.count('(') == RawArguments.count(')'):
                        AddEventMod(RawArguments, FoundAttachments, InEventModification, nHLine, AttachedEvent, CreatedEvent, self.Modules[ModuleName])
                        InEventModification = ''
                    continue
            else:
                Finished = False
                for char in line.strip():
                    RawArguments = RawArguments + char
                    if RawArguments.count('(') == RawArguments.count(')'):
                        Finished = True
                        break
                if not Finished:
                    continue
                AddEventMod(RawArguments, FoundAttachments, InEventModification, nHLine, AttachedEvent, CreatedEvent, self.Modules[ModuleName])
                InEventModification = ''

        if not FoundAttachments['Mods']:
            return []
        SortedAttachments = {'DefaultEvent':(None, [])}
        for (EventType, SubStreamSpec, AttachedEvent, CreatedEvent) in FoundAttachments['Join']:
            SortedAttachments[CreatedEvent] = (SubStreamSpec, [EventType])
        for (EventType, SubStreamSpec, AttachedEvent, CreatedEvent) in FoundAttachments['Attach']:
            if AttachedEvent not in SortedAttachments:
                Key = 'DefaultEvent'
            else:
                Key = AttachedEvent
            SortedAttachments[Key][1].append(EventType)
        return SortedAttachments

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
            for ModuleName in self._LogsColumns[1:]:
                if self._EventLogs[ModuleName]:
                    CurrentLine += self._Default_Color + ' | ' + self._EventLogs[ModuleName].pop(0)
                else:
                    CurrentLine += self._Default_Color + ' | ' + self._MaxColumnWith*' '
            for Log in self._SessionLogs:
                Log.write(CurrentLine + "\n")
        self._HasLogs = 0
        self._LogT = None
    def _LogInit(self, Resume = False):
        self._HasLogs = 2
        self._LogT = None
        self._LogsColumns = [ModuleName for ModuleName in ['Framework'] + self.ModulesList if (ModuleName == 'Framework' or self.Modules[ModuleName]._NeedsLogColumn)]
        self._MaxColumnWith = int((self._Terminal_Width - len(self._LogsColumns)*3 ) / len(self._LogsColumns))
        if not Resume:
            self._EventLogs = {ModuleName:[' '*((self._MaxColumnWith - len(ModuleName))//2) + self._LogColors[1] + ModuleName + (self._MaxColumnWith - len(ModuleName) - (self._MaxColumnWith - len(ModuleName))//2)*' ', self._MaxColumnWith*' '] for ModuleName in self._LogsColumns}
            self._SendLog()
        else:
            self._EventLogs = {ModuleName:[] for ModuleName in self._LogsColumns}
    def _LogReset(self):
        self._EventLogs = {ModuleName:[] for ModuleName in self._EventLogs.keys()}
        self._HasLogs = 0

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
        
        self.__ProposesTau = ('EventTau' in self.__class__.__dict__) # We set this as a variable, for user to change it at runtime. It can be accessed as a public variable through 'self.ProposesTau'
        
        try:
            self.__ModuleIndex__ = self.__Framework__.ModulesOrder[self.__Name__]
        except:
            None

        self._OnCreation()

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
        If no Tau is proposed by any other module, will return None, so default value has to be Module specific
        '''
        return self.__Framework__._GetLowerLevelTau(self._RunningEvent, self)
    @property
    def FrameworkAverageTau(self):
        '''
        Method to retreive the highest level information Tau from the framework, with no information about the current event.
        If no Tau is proposed by any other module, will return None, so default value has to be Module specific
        '''
        return self.__Framework__._GetLowerLevelTau(None, self)

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
            self._MonitoredVariables = [('AverageTau', float)] + self._MonitoredVariables
            self.__class__.AverageTau = property(lambda self: self.EventTau(None))

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
        return self.__ProposesTau

    def _SetGeneratedSubStreamsIndexes(self, Indexes):
        '''
        Specific method for input modules.
        Upon calling when framework is created, will allow the module to set its variables accordingly to what was specified in the project file.
        Cannot create a framework with an input module that has not its proper method.
        Be careful to have variables that wont be overwritten during initialization of the framework
        '''
        return False

    def __UpdateParameter__(self, Key, Value, Log = True):
        self.__SavedValues__[Key] = copy.copy(self.__dict__[Key])
        self.__dict__[Key] = Value
        if Log:
            self.Log("Changed specific value {0} from {1} to {2}".format(Key, self.__SavedValues__[Key], self.__dict__[Key]))

    def GetSnapIndexAt(self, t):
        return (abs(np.array(self.History['t']) - t)).argmin()
    def _Restart(self):
        # Template method for restarting modules, for instant display handler. Quite specific for now
        pass
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
    def EventTau(self, event = None):
        # User-filled method to propose a tau for the whole framework. Ideally, thet further in the framework, the higher the information and the more acurate Tau information is
        # Return None for default, or 0 for non-defined tau yet
        pass
    @property
    def MapTau(self):
        if not self.__ProposesTau:
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
    def _Warp(self, t):
        # Template method for input modules to accelerate warp process when running an experiments after t=0. 
        # Should improve computation time
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
    def _AddEvent(self, Event):
        Index = Event.SubStreamIndex
        if not Index in self.Events:
            self.Events[Index] = [Event]
        else:
            self.Events[Index].append(Event)
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
        if self._Container.timestamp is None:
            if 'timestamp' in kwargs:
                self._Container.timestamp = kwargs['timestamp']
                del kwargs['timestamp']
            else:
                raise Exception("No timestamp specified during first event creation")
        self._Container._AddEvent(Extension(Container = self._Container, SubStreamIndex = SubStreamIndex, **kwargs))
        del self._Container.__dict__['BareEvent']
        return self._Container.Events[SubStreamIndex][-1]
    def SetTimestamp(self, t):
        self._Container.timestamp = t

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
    def _Attach(self, Extension, **kwargs):
        # Used to assess that the event was just created, so event information have to be attached anyway
        self._Extensions.add(Extension)
        for Field in Extension._Fields:
            self.__dict__[Field] = kwargs[Field]
    def Attach(self, Extension, **kwargs):
        if not Extension._CanAttach or Extension in self._Extensions:
            self.Join(Extension, **kwargs) # For now, its better to join when instance is already there (ex: multiple TrackerEvents)
            return
        self._Extensions.add(Extension)
        for Field in Extension._Fields:
            self.__dict__[Field] = kwargs[Field]
    def Join(self, ExtensionOrEvent, **kwargs):
        if type(ExtensionOrEvent) == _EventClass:
            self._Container._AddEvent(ExtensionOrEvent)
            return ExtensionOrEvent
        if 'SubStreamIndex' in kwargs:
            SubStreamIndex = kwargs['SubStreamIndex']
            del kwargs['SubStreamIndex']
        else:
            SubStreamIndex = self.SubStreamIndex
        self._Container._AddEvent(ExtensionOrEvent(Container = self._Container, SubStreamIndex = SubStreamIndex, **kwargs))
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
    def Copy(self, SubStreamIndex = None):
        if SubStreamIndex is None:
            SubStreamIndex = self.SubStreamIndex
        kwargs = {'timestamp':self.timestamp, 'SubStreamIndex':SubStreamIndex, 'Extensions':self._Extensions}
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
        Event._Attach(cls, **kwargs)
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
class TwistEvent(_EventExtensionClass):
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

_AvailableEventsClassesNames = [EventType.__name__ for EventType in _EventExtensionClass.__subclasses__()]

