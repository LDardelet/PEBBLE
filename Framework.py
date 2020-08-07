import numpy as np
import pickle
import sys
import select
import inspect
import atexit
import types
import copy
import os
import _pickle as cPickle
import json
from datetime import datetime as dtClass
import ast

from Plotting_methods import *
from pydoc import locate

class Framework:
    _Terminal_Width = 250
    _Default_Color = '\033[0m'
    _LogColors = {0:'\033[0m', 1: "\033[1;33;40m", 2: "\033[1;31;40m", 3: "\033[1;32;40m"}
    _LOG_FILE_EXTENSION = 'log'
    _PROJECT_FILE_EXTENSION = 'json'
    _DATA_FILE_EXTENSION = 'data'
    '''
    Main event-based framework file.
    Each tool is in a  different file, and caan be added through 'Project files', that create the sequence of events processing.
    Each tool must contain a simple  __init__ method with 'self' and 'argCreationDict' only, the second one containing all necessary variables from tools above itself.
    It also must contain a '_Initialization' method, with 'self' and 'argInitializationDict' only, the second one containing all necessary information about the current stream processed.

    Finally, each tool contains an '_OnEvent' method processing each event. It must only have 'event' as argument - apart from 'self' - and all variables must have be stored inside, prior to this.
    '_OnEvent' can be seen as a filter, thus it must return the event incase one wants the processing of the current event to go on with the following tools.

    Finally, each tool must declare a '__Type__' variable, to disclaim to the framework what kind to job is processed.
    For the special type 'Input', no '_OnEvent' method is currently needed.

    Currently implemented tools :
        Input tools :
            -> StreamReader : Basic input method
        Display and visualization tools :
            -> DisplayHandler : allows to use the StreamDisplay Program
        Memory tools :
            -> Memory : Basic storage tool
        Filters :
            -> RefractoryPeriod : Filters events from spike trains for a certain period.
        Computation tools :
            -> SpeedProjector : Recover the primitive generator of an events sequence by projecting the 3D ST-Context along several speeds, until finding the correct one
            -> FlowComputer : Computes the optical flow of a stream of events

    '''
    def __init__(self, File = None, onlyRawData = False, FullSessionLogFile = None):
        self.__Type__ = 'Framework'
        self._LogType = 'raw'
        if FullSessionLogFile is None:
            self._SessionLog = sys.stdout
        else:
            self._SessionLog = open(FullSessionLogFile, 'w')
            atexit.register(self._SessionLog.close)
        self._LogOut = self._SessionLog
        self._LastLogOut = self._SessionLog
        
        self.Modified = False
        self.StreamHistory = []

        self.PropagatedEvent = None
        self.Running = False
        self._Initializing = False
        self.Paused = ''

        if not File is None and File.split('.')[-1] == self._DATA_FILE_EXTENSION:
            self._LoadFromDataFile(File, onlyRawData = onlyRawData)
        else:
            self._LoadProject(File, onlyRawData = onlyRawData)

        atexit.register(self._OnClosing)


    def Initialize(self, **ArgsDict):
        self._Initializing = True
        self.PropagatedEvent = None
        self._LogType = 'columns'
        self._LogInit()
        for tool_name in self.ToolsList:
            ToolArgsDict = {}
            for key, value in ArgsDict.items():
                if tool_name in key or key[0] == '_':
                    ToolArgsDict['_'+'_'.join(key.split('_')[1:])] = value
            InitializationAnswer = Module.__Initialize__(self.Tools[tool_name], **ToolArgsDict)
            if not InitializationAnswer:
                self._Log("Tool {0} failed to initialize. Aborting.".format(tool_name), 2)
                return False
        self._RunToolsTuple = tuple(self.ToolsList) # Faster way to access tools in the right order
        self._Log("Framework initialized", 3, AutoSendIfPaused = False)
        self._Log("")
        self._SendLog()
        self._Initializing = False
        return True

    def _OnClosing(self):
        if self.Modified:
            ans = 'void'
            while self._PROJECT_FILE_EXTENSION not in ans and ans != '':
                ans = input('Unsaved changes. Please enter a file name with extension .{0}, or leave blank to discard : '.format(self._PROJECT_FILE_EXTENSION))
            if ans != '':
                self.SaveProject(ans)

    def _GetStreamGeometry(self, Tool):
        '''
        Method to retreive the geometry of the events handled by a tool
        '''
        ToolEventsRestriction = Tool.__CameraIndexRestriction__
        Geometry = np.array([0, 0, 0])
        for InputToolName in self.ToolsList:
            InputTool = self.Tools[InputToolName]
            if InputTool.__Type__ == 'Input':
                if InputTool.__Initialized__:
                    if not ToolEventsRestriction or InputTool.__CameraIndexRestriction__[0] in ToolEventsRestriction:
                        Geometry = np.maximum(Geometry, InputTool.Geometry)
                else:
                    self._Log("Unable to retreive geometry from input tool {0}, possibly due to wrong tool order.".format(InputToolName), 1)
        return Geometry

    def _GetStreamFormattedName(self, Tool):
        '''
        Method to retreive a formatted name depending on the files providing events to this tool.
        Specifically useful for an Input type tool to get the file it has to process.
        '''
        if Tool.__Type__ == 'Input':
            return self.CurrentInputStreams[Tool.__Name__]

        ToolEventsRestriction = Tool.__CameraIndexRestriction__
        StreamsNames = []
        for InputToolName in self.ToolsList:
            InputTool = self.Tools[InputToolName]
            if InputTool.__Type__ == 'Input':
                if InputTool.__Initialized__:
                    if not ToolEventsRestriction or InputTool.__CameraIndexRestriction__[0] in ToolEventsRestriction:
                        StreamsNames += [InputTool.StreamName]
                else:
                    self._Log("Unable to retreive stream name from input tool {0}, possibly due to wrong tool order.".format(InputToolName), 1)
        return '\n'.join(StreamsNames)

    def ReRun(self, stop_at = np.inf):
        self.RunStream(self.StreamHistory[-1], stop_at = stop_at)

    def RunStream(self, StreamName = None, start_at = 0., stop_at = np.inf, resume = False, AtEventMethod = None, LogFile = None, **kwargs):
        if resume:
            if self._LastLogOut is self._SessionLog:
                pass
            else:
                self._LogOut = open(self._LastLogOut.name, 'a')
        else:
            if LogFile is None:
                self._LogOut = self._SessionLog
            else:
                GivenExtention = LogFile.split('.')[-1]
                if GivenExtention != self._LOG_FILE_EXTENSION:
                    raise Exception("Enter log file with .{0} extension".format(self._LOG_FILE_EXTENSION))
                self._LogOut = open(LogFile, 'w')
            self._LastLogOut = self._LogOut
        if self._LogType == 'columns':
            if len(self.ToolsList) > 6:
                self._LogType = 'raw'
            else:
                self._LogInit(resume)
        self._RunProcess(StreamName = StreamName, start_at = start_at, stop_at = stop_at, resume = resume, AtEventMethod = AtEventMethod, **kwargs)
        if not self._LogOut is self._SessionLog:
            self._LogOut.close()
            self._LogOut = self._SessionLog
        self._LogType = 'raw'

    def _GetCommitValue(self):
        try:
            f = open('.git/FETCH_HEAD', 'r')
            commit_value = f.readline().strip()
            f.close()
        except:
            commit_value = "unknown"
        return commit_value

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
            now = dtClass.now()
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

    def _LoadFromDataFile(self, DataFile, onlyRawData = False):
        self._Log('Loading data from saved file')
        with open(DataFile, 'rb') as BinDataFile:
            def BinRead():
                Line = BinDataFile.readline().decode('utf-8')
                return (Line[:2] == '##'), Line[2:].split('->')[0].strip(), Line[2:].split('->')[-1].strip()
            ProjectFile = None
            ProjectFileHash = None
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
                    ProjectFile = Value
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

    def _RunProcess(self, StreamName = None, start_at = 0., stop_at = np.inf, resume = False, AtEventMethod = None, BareInit = False, **kwargs):
        if StreamName is None:
            N = 0
            StreamName = "DefaultStream_{0}".format(N)
            while StreamName in self.StreamHistory:
                N += 1
                StreamName = "DefaultStream_{0}".format(N)
        if not resume:
            self.RunKwargs = dict(kwargs)
            self.CurrentInputStreams = {ToolName:None for ToolName in self.ToolsList if self.Tools[ToolName].__Type__ == 'Input'}
            if type(StreamName) == str:
                for ToolName in self.CurrentInputStreams.keys():
                    self.CurrentInputStreams[ToolName] = StreamName
            elif type(StreamName) == dict:
                if len(StreamName) != len(self.CurrentInputStreams):
                    self._Log("Wrong number of stream names specified :", 2)
                    self._Log("Framework contains {0} input tools, while {1} tool(s) has been given a file".format(len(self.CurrentInputStreams), len(StreamName)), 2)
                    return None
                for key in StreamName.keys():
                    if key not in self.CurrentInputStreams.keys():
                        self._Log("Wrong input tool key specified in stream names : {0}".format(key), 2)
                        return None
                    self.CurrentInputStreams[key] = StreamName[key]
            else:
                self._Log("Wrong StreamName type. It can be :", 2)
                self._Log(" - None : Default name is then placed, for None file specific input tools.", 2)
                self._Log(" - str : The same file is then used for all input tools.", 2)
                self._Log(" - dict : Dictionnary with input tools names as keys and specified filenames as values", 2)
                return None
            
            self.StreamHistory += [self.CurrentInputStreams]
            InitializationAnswer = self.Initialize(**kwargs)
            if not InitializationAnswer or BareInit:
                return None

        self.PropagatedEvent = None
        self.Running = True
        self.Paused = ''
        if resume:
            for tool_name in self._RunToolsTuple:
                self.Tools[tool_name]._Resume()

        while self.Running and not self.Paused:
            t = self.NextEvent(start_at, AtEventMethod)

            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                promptedLine = sys.stdin.readline()
                if 'a' in promptedLine or 'q' in promptedLine:
                    self.Paused = 'user'
            if t is None or t > stop_at:
                self.Paused = 'Framework'
        if not self.Running:
            self._Log("Main loop finished without error.")
        else:
            if self.Paused:
                self._Log("Paused at t = {0:.3f}s by {1}.".format(t, self.Paused), 1)
                for tool_name in self._RunToolsTuple:
                    self.Tools[tool_name]._Pause(self.Paused)

    def Resume(self, stop_at = np.inf):
        self.RunStream(self.StreamHistory[-1], stop_at = stop_at, resume = True)

    def NextEvent(self, start_at, AtEventMethod = None):
        self.PropagatedEvent = None
        t = None
        for tool_name in self._RunToolsTuple:
            self.PropagatedEvent = self.Tools[tool_name].__OnEvent__(self.PropagatedEvent)
            if self.PropagatedEvent is None:
                break
            else:
                if t is None:
                    t = self.PropagatedEvent.timestamp
                if t < start_at:
                    break
            if not self.PropagatedEvent is None and not AtEventMethod is None:
                AtEventMethod(self.PropagatedEvent)
        self._SendLog()
        return t

    def Rewind(self, t):
        ForbiddingModules = []
        for tool_name in self.ToolsList:
            if self.Tools[tool_name].__RewindForbidden__:
                ForbiddingModules += [tool_name]
        if ForbiddingModules:
            return ForbiddingModules
        for tool_name in reversed(self.ToolsList):
            self.Tools[tool_name]._Rewind(t)
        self._Log("Framework : rewinded to {0:.3f}".format(t), 1)

#### Project Management ####

    def _GenerateEmptyProject(self):
        self.Tools = {}
        self._ToolsCreationReferences = {}
        self._ToolsExternalParameters = {}
        self._ToolsCamerasRestrictions = {}
        self._ToolsClasses = {}

        self.ToolsOrder = {}
        self.ToolsList = []

    def _LoadProject(self, ProjectFile = None, enable_easy_access = True, onlyRawData = False):
        self.ProjectFile = ProjectFile
        self._ProjectRawData = {}
        self._GenerateEmptyProject()
        if self.ProjectFile is None:
            return
        self._LogType = 'raw'

        if ProjectFile is None:
            data = self._ProjectRawData
        else:
            data = pickle.load(open(ProjectFile, 'rb'))
            self._ProjectRawData = data

        if onlyRawData:
            return None

        for tool_name in data.keys():
            fileLoaded = __import__(data[tool_name]['File'])
            self._ToolsClasses[tool_name] = getattr(fileLoaded, data[tool_name]['Class'])

            self._ToolsCreationReferences[tool_name] = data[tool_name]['CreationReferences']
            self._ToolsExternalParameters[tool_name] = data[tool_name]['ExternalParameters']
            if 'CamerasHandled' in data[tool_name].keys(): # For previous version support
                self._ToolsCamerasRestrictions[tool_name] = data[tool_name]['CamerasHandled']
            else:
                self._ToolsCamerasRestrictions[tool_name] = []

            self.ToolsOrder[tool_name] = data[tool_name]['Order']
            self._Log("Imported tool {1} from file {0}.".format(data[tool_name]['File'], data[tool_name]['Class']))

        if len(self.ToolsOrder.keys()) != 0:
            MaxOrder = max(self.ToolsOrder.values()) + 1
            self.ToolsList = [None] * MaxOrder
        else:
            self.ToolsList = []

        for tool_name in self.ToolsOrder.keys():
            if self.ToolsList[self.ToolsOrder[tool_name]] is None:
                self.ToolsList[self.ToolsOrder[tool_name]] = tool_name
            else:
                self._Log("Double assignement of number {0}. Aborting ProjectFile loading".format(self.ToolsOrder[tool_name]), 2)
                return None
        if None in self.ToolsList:
            while None in self.ToolsList:
                self.ToolsList.remove(None)

        self._Log("Successfully generated tools order", 3)
        self._Log("")
        
        for tool_name in self.ToolsList:
            self.Tools[tool_name] = self._ToolsClasses[tool_name](tool_name, self, self._ToolsCreationReferences[tool_name])
            self._UpdateToolsParameters(tool_name)

            if enable_easy_access and tool_name not in self.__dict__.keys():
                self.__dict__[tool_name] = self.Tools[tool_name]
        self._Log("Successfully generated Framework", 3)
        self._Log("")

    def _UpdateToolsParameters(self, tool_name):
        for key, value in self._ToolsExternalParameters[tool_name].items():
            if key in self.Tools[tool_name].__dict__.keys():
                try:
                    self.Tools[tool_name].__dict__[key] = type(self.Tools[tool_name].__dict__[key])(value)
                except:
                    self._Log("Issue with setting the new value of {0} for tool {1}, should be {2}, impossible from {3}".format(key, tool_name, type(self.Tools[tool_name].__dict__[key]), value), 1)
            else:
                self._Log("Key {0} for tool {1} doesn't exist. Please check ProjectFile integrity.".format(key, tool_name), 1)
        self.Tools[tool_name].__CameraIndexRestriction__ = self._ToolsCamerasRestrictions[tool_name]

    def SaveProject(self, ProjectFile):
        GivenExtention = ProjectFile.split('.')[-1]
        if GivenExtention:
            raise Exception("Enter log file with .{0} extension".format(self._PROJECT_FILE_EXTENSION))
        pickle.dump(self._ProjectRawData, open(ProjectFile, 'wb'))
        self.ProjectFile = ProjectFile
        self.Modified = False

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
                self._ProjectRawData[Name]['Order'] = int(entry)
            else:
                self._Log("Input Type detected. Setting to next default index.")
                self._ProjectRawData[Name]['Order'] = 0
                for tool_name in self._ProjectRawData.keys():
                    if tool_name != Name and self._ProjectRawData[tool_name]['IsInput']:
                        self._ProjectRawData[Name]['Order'] = max(self._ProjectRawData[Name]['Order'], self._ProjectRawData[tool_name]['Order'] + 1)
            NumberTaken = False
            for tool_name in self._ProjectRawData.keys():
                if tool_name != Name and 'Order' in self._ProjectRawData[tool_name].keys() and self._ProjectRawData[Name]['Order'] == self._ProjectRawData[tool_name]['Order']:
                    NumberTaken = True
            if NumberTaken:
                self._Log("Compiling new order.")
                for tool_name in self._ProjectRawData.keys():
                    if tool_name != Name and 'Order' in self._ProjectRawData[tool_name].keys():
                        if self._ProjectRawData[tool_name]['Order'] >= self._ProjectRawData[Name]['Order']:
                            self._ProjectRawData[tool_name]['Order'] += 1
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
                self._Log("Enter camera index for this input module, if necessary.")
            else:
                self._Log("Enter camera index(es) handled by this module, coma separated. Void will not create any restriction.")
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

    def DisplayCurrentProject(self):
        self._Log("# Framework\n", 3)
        self._Log("")

        nOrder = 0
        for tool_name in self.ToolsList:
            filename = inspect.getfile(self.Tools[tool_name].__class__)
            self._Log("# {0} : {1}, from class {2} in file {3}.".format(nOrder, tool_name, str(self.Tools[tool_name].__class__).split('.')[1][:-2], filename), 3)
            self._Log("     Type : {0}".format(self.Tools[tool_name].__Type__))
            if self.Tools[tool_name].__CameraIndexRestriction__:
                self._Log("     Uses cameras indexes " + ", ".join([str(CameraIndex) for CameraIndex in self.Tools[tool_name].__CameraIndexRestriction__]))
            else:
                self._Log("     Uses all cameras inputs.")
            if self._ToolsCreationReferences[tool_name]:
                self._Log("     Creation References:")
                for argName, toolReference in self._ToolsCreationReferences[tool_name].items():
                    self._Log("         -> Access to {0} from tool {1}".format(argName, toolReference))
            else:
                self._Log("     No creation reference.")
            if self._ToolsExternalParameters[tool_name]:
                self._Log("     Modified Parameters:")
                for var, value in  self._ToolsExternalParameters[tool_name].items():
                    self._Log("         -> {0} = {1}".format(var, value))
            else:
                self._Log("     Using default parameters.")
            self._Log("")
            
            nOrder += 1

    def _Log(self, Message, MessageType = 0, ModuleName = 'Framework', Raw = False, AutoSendIfPaused = True):
        if self._LogType == 'columns' and not Raw:
            if self._LogT is None:
                if not self.PropagatedEvent is None:
                    self._LogT = self.PropagatedEvent.timestamp
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
            Message = self._LogColors[MessageType] + int(bool(Message))*(ModuleName + ': ') + Message
            self._LogOut.write(Message + self._LogColors[0] + "\n")
    def _SendLog(self):
        if self._LogType == 'raw' or not self._HasLogs:
            return
        for nLine in range(self._HasLogs):
            CurrentLine = self._Default_Color
            if self._EventLogs['Framework']:
                CurrentLine += self._EventLogs['Framework'].pop(0)
            else:
                CurrentLine += self._MaxColumnWith*' '
            for ToolName in self.ToolsList:
                if self._EventLogs[ToolName]:
                    CurrentLine += self._Default_Color + ' | ' + self._EventLogs[ToolName].pop(0)
                else:
                    CurrentLine += self._Default_Color + ' | ' + self._MaxColumnWith*' '
            self._LogOut.write(CurrentLine + "\n")
        self._HasLogs = 0
        self._LogT = None
    def _LogInit(self, Resume = False):
        self._HasLogs = 2
        self._LogT = None
        self._MaxColumnWith = int((self._Terminal_Width - len(self.ToolsList)*3 ) / (len(self.ToolsList) + 1))
        if not Resume:
            self._EventLogs = {ToolName:[' '*((self._MaxColumnWith - len(ToolName))//2) + self._LogColors[1] + ToolName + (self._MaxColumnWith - len(ToolName) - (self._MaxColumnWith - len(ToolName))//2)*' ', self._MaxColumnWith*' '] for ToolName in ['Framework'] + self.ToolsList}
            self._SendLog()
        else:
            self._EventLogs = {ToolName:[] for ToolName in ['Framework'] + self.ToolsList}
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
        self.__RewindForbidden__ = False
        self.__SavedValues__ = {}
        self.__CameraIndexRestriction__ = []
        
        self._MonitoredVariables = []
        self._MonitorDt = 0
        self.__LastMonitoredTimestamp = -np.inf
        
        try:
            self.__ToolIndex__ = self.__Framework__.ToolsOrder[self.__Name__]
        except:
            None

    def __Initialize__(self, **kwargs):
        # First restore all prevous values
        self.Log(" > Initializing module")
        if self.__SavedValues__:
            for key, value in self.__SavedValues__.items():
                self.__dict__[key] = value
        self.__SavedValues__ = {}
        # Now change specific values for this initializing module
        for key, value in kwargs.items():
            if key[0] != '_':
                key = '_' + key
            if key not in self.__dict__:
                pass
            else:
                self.Log("Changed specific value {0} from {1} to {2}".format(key, self.__dict__[key], value))
                self.__SavedValues__[key] = copy.copy(self.__dict__[key])
                self.__dict__[key] = value
        
        # Initialize the stuff corresponding to this specific module
        if not self._InitializeModule(**kwargs):
            return False

        # Finalize Module initialization
        if self.__CameraIndexRestriction__ and self.__Type__ != 'Input':
            OnEventMethodUsed = self.__OnEventRestricted__
        else:
            OnEventMethodUsed = self._OnEventModule

        if self._MonitorDt and self._MonitoredVariables:
            self.History = {'t':[]}
            self.__MonitorRetreiveMethods = {'t': lambda event: event.timestamp}
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

            self.__OnEvent__ = lambda event: self.__OnEventMonitor__(OnEventMethodUsed(event))
        else:
            self.__OnEvent__ = OnEventMethodUsed

        self.__Initialized__ = True
        return True

    def GetSnapIndexAt(self, t):
        return (abs(np.array(self.History['t']) - t)).argmin()

    def _InitializeModule(self, **kwargs):
        # Template for user-filled module initialization
        return True
    def _OnEventModule(self, event):
        # Template for user-filled module event running method
        return event
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

    def __OnEventRestricted__(self, event):
        if event.cameraIndex in self.__CameraIndexRestriction__:
            return self._OnEventModule(event)
        else:
            return event

    def __OnEventMonitor__(self, event):
        if event.timestamp - self.__LastMonitoredTimestamp > self._MonitorDt:
            self.__LastMonitoredTimestamp = event.timestamp
            self._OnSnapModule()
            for VarName, RetreiveMethod in self.__MonitorRetreiveMethods.items():
                self.History[VarName] += [RetreiveMethod(event)]
        return event

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
                return lambda event: [UsedType(SubRetreiveMethod(Instance)) for Instance in self.__dict__[Container]]
            elif type(self.__dict__[Container]) == dict:
                if UsedType is None:
                    ExampleVar = SubRetreiveMethod(self.__dict__[Container].values()[0])
                    UsedType = type(ExampleVar)
                return lambda event: [(LocalDictKey, UsedType(SubRetreiveMethod(Instance))) for LocalDictKey, Instance in self.__dict__[Container].items()]
            else:
                if UsedType is None:
                    ExampleVar = SubRetreiveMethod(self.__dict__[Container])
                    UsedType = type(ExampleVar)
                print(UsedType)
                return lambda event: UsedType(SubRetreiveMethod(self.__dict__[Container]))
        else:
            if UsedType is None:
                UsedType = type(self.__dict__[VarName])
            return lambda event: UsedType(self.__dict__[VarName])

    def _Rewind(self, t):
        pass

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
        self.__Framework__._Log(Message, MessageType, self.__Name__)
    def LogWarning(self, Message):
        self.Log(Message, 1)
    def LogError(self, Message):
        self.Log(Message, 2)
        self.__Framework__.Paused = self.__Name__
    def LogSuccess(self, Message):
        self.Log(Message, 3)

# Listing all the events existing

class BareEvent:
    _Key = 0
    _Fields = ['timestamp']
    def __init__(self, **kwargs):
        for Field in self.__class__._Fields:
            try:
                self.__dict__[Field] = kwargs[Field]
            except:
                self.__dict__[Field] = self._Defaults[Field]
        if self.__class__ == BareEvent:
            self._Extensions = []
            self._ExtensionKeys = set()
            self._RunningExtensions = None
    def _AsList(self):
        if self._Key < 0: #Ill defined Key
            return None
        List = [self._Key]
        for Field in self.__class__._Fields:
            Value = getattr(self, Field)
            if type(Value) == np.ndarray:
                List += [Value.tolist()]
            else:
                List += [Value]
        if not self.__class__ == BareEvent:
            return List
        for Extension in self._Extensions:
            ExList = Extension._AsList()
            if not ExList is None:
                List += [ExList]
        return List
    def Attach(self, Extension, **kwargs):
        self._Extensions += [Extension(bare = True, **kwargs)]
        self._ExtensionKeys.add(Extension._Key)
        if Extension._AutoPublic:
            self._Extensions[-1]._Publicize(self)

    def Has(self, ExtensionRequired):
        if not isinstance(self, BareEvent):
            return False
        return ExtensionRequired._Key in self._ExtensionKeys
    def Get(self, ExtensionRequired):
        self._RunningExtensions = [Extension for Extension in self._Extensions if ExtensionRequired._Key == Extension._Key]
        return self.__iter__()
    def __iter__(self):
        if self._RunningExtensions is None:
            raise Exception("No subevent type selected. Use Get() method first")
        self._IterVar = 0
        return self
    def __next__(self):
        if self._IterVar < len(self._RunningExtensions):
            self._IterVar += 1
            self._RunningExtensions[self._IterVar-1]._Publicize(self)
            return self
        else:
            if not self._RunningExtensions[-1]._AutoPublic:
                self._RunningExtensions[-1]._Hide(self)
            self._RunningExtensions = None
            raise StopIteration

    def Copy(self):
        SelfDict = {Key: getattr(self, Key) for Key in self._Fields}
        NewInstance = self.__class__(**SelfDict)
        for Extension in self._Extensions:
            NewInstance.Attach(Extension.__class__, **{Key: getattr(Extension, Key) for Key in Extension._Fields})
        return NewInstance

class _EventExtension(BareEvent):
    _Key = -1
    _AutoPublic = False # Careful to check that this extention _Fields are not conflicting with another publishable extension.
    def __new__(cls, *args, **kw):
        if not 'bare' in kw:
            BaseInstance = BareEvent(**kw)
            BaseInstance.Attach(cls, **kw)
            return BaseInstance
        else:
            del kw['bare']
            ExtensionInstance = object.__new__(cls)
            BareEvent.__init__(ExtensionInstance, *args, **kw)
            return ExtensionInstance
    def __init__(self, *args, **kwargs):
        BareEvent.__init__(self, **kwargs)
    def _Publicize(self, BaseEvent):
        for Field in self._Fields:
            BaseEvent.__dict__[Field] = getattr(self, Field)
    def _Hide(self, BaseEvent):
        for Field in self._Fields:
            del BaseEvent.__dict__[Field]

class Event(_EventExtension):
    _Key = 1
    _Fields = ['location', 'polarity', 'cameraIndex']
    _Defaults = {'cameraIndex':0}
    _AutoPublic = True

class TrackerEvent(_EventExtension):
    _Key = 2
    _Fields = ['TrackerLocation', 'TrackerID', 'TrackerAngle', 'TrackerScaling', 'TrackerColor', 'TrackerMarker']
    _Defaults = {'TrackerAngle':0, 'TrackerScaling':1, 'TrackerColor':'b', 'TrackerMarker': 'o'}

class EventOld:
    def __init__(self, timestamp=None, location=None, polarity=None, cameraIndex = 0, original = None):
        if original == None:
            self.timestamp = timestamp
            self.location = np.array(location, dtype = np.int16)
            self.polarity = polarity
            self.cameraIndex = cameraIndex # Used for indexing cameras incase of multiple cameras, for stereoscopy in particular
        else:
            for key, value in original.__dict__.items():
                self.__dict__[key] = value

    def _AsList(self):
        return self.location.tolist() + [self.timestamp, self.polarity]

class TrackerEventOld(Event):
    def __init__(self, original, TrackerLocation = None, TrackerID = None, TrackerAngle = 0., TrackerScaling = 1., TrackerColor = 'b', TrackerMarker = 'o'): # Some fields are added for dev purposes.
        super().__init__(original = original)
        self.TrackerLocation = np.array(TrackerLocation)
        self.TrackerID = TrackerID
        self.TrackerAngle = TrackerAngle
        self.TrackerScaling = TrackerScaling
        self.TrackerColor = TrackerColor
        self.TrackerMarker = TrackerMarker

    def _AsList(self):
        return super()._AsList() + [1, self.TrackerID] + self.TrackerLocation.tolist() + [self.TrackerAngle, self.TrackerScaling, self.TrackerColor, self.TrackerMarker]

