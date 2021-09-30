import numpy as np
import pickle
import sys
import select
import inspect
import atexit
import types
import os
import shutil
import pathlib
import _pickle as cPickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
from ast import literal_eval

from Events import _AvailableEventsClassesNames, _EventContainerClass
from ModuleBase import ModuleBase
from DisplayHandler import DisplayHandler as _DisplayHandlerClass

_RUNS_DIRECTORY = os.path.expanduser('~/Runs/')
        
_TYPE_TO_STR = {np.array:'np.array', float:'float', int:'int', str:'str', tuple:'tuple', list:'list'}
_STR_TO_TYPE = {value:key for key, value in _TYPE_TO_STR.items()}
_SPECIAL_NPARRAY_CAST_MESSAGE = '$ASNPARRAY$'

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
        self._RunFileSet = False
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
                EventTau = ModuleProposing._OnTauRequest(EventConcerned)
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
            self._SetLockFile(False)
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
            os.mkdir(self.RunFolder)
        except:
            return self._InitiateFolder(suffix + '_bis')
        self._Log("Created output folder {0}".format(self.RunFolder), 1)
        self._SessionLogs = [sys.stdout, open(self.RunFolder+'log.txt', 'w')]

    def _OnClosing(self):
        if self.Modified:
            ans = 'void'
            while self._PROJECT_FILE_EXTENSION not in ans and ans != '':
                ans = input('Unsaved changes. Please enter a file name with extension .{0}, or leave blank to discard : '.format(self._PROJECT_FILE_EXTENSION))
            if ans != '':
                self.SaveProject(ans)

        if not self._FolderData['home'] is None:
            if not ('_LastStartWarning' in self.__dict__):
                if not self._CSVDataSaved:
                    self.SaveCSVData() # By default, we save the data. Files shouldn't be too big anyway
            else:
                self._Log("Stopped mid warp, nothing saved")

        for ModuleName in self.ModulesList:
            self.Modules[ModuleName]._OnClosing()
        self._SetLockFile(False)
        self._Log("Closing done", 3)
        self._SendLog()
        self._SaveInterpreterData()
        self._SessionLogs.pop(1).close()

    def _DestroyFolder(self):
        shutil.rmtree(self._FolderData['home'])
        self._FolderData['home'] = None

    def _SetLockFile(self, Set):
        if Set:
            try:
                with open(self._FolderData['home'] + self._RUN_FILE, 'r') as _:
                    self._Log("Found a lock file in current folder : should not happend", 1)
            except FileNotFoundError:
                pass
            with open(self._FolderData['home'] + self._RUN_FILE, 'w') as _:
                pass
            self._RunFileSet = True
            self._Log("Placed lock file")
        else:
            if not self._RunFileSet:
                self._Log("No lockfile to remove")
                return
            else:
                try:
                    os.remove(self._FolderData['home'] + self._RUN_FILE)
                    self._Log("Lock file removed")
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
    def RunFolder(self):
        return self._FolderData['home']
    @property
    def HistoryFolder(self):
        if self._FolderData['history'] is None:
            self._FolderData['history'] = self.RunFolder+'History/'
            os.mkdir(self._FolderData['history'])
        return self._FolderData['history']
    @property
    def PicturesFolder(self):
        if self._FolderData['pictures'] is None:
            self._FolderData['pictures'] = self.RunFolder+'Pictures/'
            os.mkdir(self._FolderData['pictures'])
        return self._FolderData['pictures']
    @property
    def ParamsLogFile(self):
        return self.RunFolder+'params.json'
    @property
    def InputsLogFile(self):
        return self.RunFolder+'inputs.txt'
    @property 
    def InterpreterLogFile(self):
        # Stores any ipython interpreter I/O into a file. Might be quite messy, but allows to fully save what has been done within the session
        return self.RunFolder+'interpreter.txt'

    def SaveData(self, Filename, forceOverwrite = False):
        '''
        Interactive data saving method. 
        Should save the data in such a format that it can be recovered later on in a recreated framework.

        Might fail, so default data saving method is through CSV files right now.
        '''
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
            try:
                self.Modules[ModuleName].SaveCSVData(FileName, FloatPrecision = 6, Default2DIndexes = 'xy', Default3DIndexes = 'xyz', Separator = ' ')
            except:
                self.Modules[ModuleName].LogWarning(f"Failed saving {ModuleName} data to CSV file : " + str(sys.exc_info()[0]))
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
                    InputFiles = literal_eval(Value)
                    self._Log(Value)
                elif Key == 'Run arguments':
                    RunArgs = literal_eval(Value)
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
            if start_at > 0:
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

        self.t = 0.
        if start_at > 0:
            for ModuleName in self.InputEvents.keys():
                if '_OnWarp' in self.Modules[ModuleName].__class__.__dict__:
                    self._Log("Warping {0} through module method".format(ModuleName))
                    self._SendLog()
                    if not self.Modules[ModuleName]._OnWarp(start_at):
                        self._Log("No event remaining in {0} file".format(ModuleName), 2)

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
        
        if not resume:
            self._SetLockFile(True)

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
            return False

    def Resume(self, stop_at = np.inf):
        self._LogType = 'columns'
        self.RunStream(self.StreamHistory[-1], stop_at = stop_at, resume = True)

    def DisplayRestart(self):
        for ModuleName in self.ModulesList:
            if self.Modules[ModuleName].__class__ == _DisplayHandlerClass:
                self.Modules[ModuleName].Restart()

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
            if '_OnInputIndexesSet' not in Module.__class__.__dict__:
                self._Log("Missing adequate _OnInputIndexesSet method for input tool")
                return False
            if not Module._OnInputIndexesSet(self._ModulesSubStreamsCreated[ModuleName]):
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
                if '_OnInputIndexesSet' not in Module.__class__.__dict__:
                    self._Log("Missing adequate _OnInputIndexesSet method for input tool")
                    return False
                if not Module._OnInputIndexesSet(self._ModulesSubStreamsCreated[ModuleName]):
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
        '''
        Message types:
            0 : Common log (default)
            1 : Warning log
            2 : Error log, should stop the framework from running (to be checked)
            3 : Success log
        '''
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
                    raise KeyError(Pattern + " -> " + key)
        else:
            super().__setitem__(key, value)

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
