import numpy as np
import pickle
import sys
import select
import inspect
import atexit
import types
import copy

from Plotting_methods import *

TypesLimits = {'Input':1}
NonRunningTools = ['Input']

class Framework:
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
    def __init__(self, ProjectFile = None, onlyRawData = False, verboseRatio = 10000):
        if 'F' in globals().keys(): # Self protection line for mpython overwriting errors
            ans = input('F Framework found in currect session. Override ? (y/N)')
            if not ans.lower() == 'y':
                raise Exception()

        self.ProjectFile = ProjectFile
        self.Modified = False

        self.__Type__ = 'Framework'

        self.StreamsGeometries = {}
        self.StreamHistory = []

        self.VerboseRatio = verboseRatio

        atexit.register(self._OnClosing)

        if not ProjectFile is None:
            self.LoadProject(ProjectFile, onlyRawData = onlyRawData)
        else:
            self._ProjectRawData = {}
            self._GenerateEmptyProject()

    def Initialize(self, **ArgsDict):
        for tool_name in self.ToolsList:
            ToolArgsDict = {}
            for key, value in ArgsDict.items():
                if tool_name in key or key[0] == '_':
                    ToolArgsDict['_'+'_'.join(key.split('_')[1:])] = value
            InitializationAnswer = Module.__Initialize__(self.Tools[tool_name], **ToolArgsDict)
            if not InitializationAnswer:
                print("Tool {0} failed to initialize. Aborting.".format(tool_name))
                return False
        return True

    def _OnClosing(self):
        if self.Modified:
            ans = 'void'
            while '.json' not in ans and ans != '':
                ans = input('Unsaved changes. Please enter a file name, or leave blank to discard : ')
            if ans != '':
                self.SaveProject(ans)

    def ReRun(self, stop_at = np.inf):
        self.RunStream(self.StreamHistory[-1], stop_at = stop_at)

    def RunStream(self, StreamName = None, start_at = 0., stop_at = np.inf, resume = False, **kwargs):
        if StreamName is None:
            N = 0
            StreamName = "DefaultStream_{0}".format(N)
            while StreamName in self.StreamHistory:
                N += 1
                StreamName = "DefaultStream_{0}".format(N)
        if not resume:
            self.StreamHistory += [StreamName]
            InitializationAnswer = self.Initialize(**kwargs)
            if not InitializationAnswer:
                return None

        self.Running = True
        self.Paused = ''
        while self.Running and not self.Paused:
            t = self.NextEvent(start_at)

            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                promptedLine = sys.stdin.readline()
                if 'a' in promptedLine or 'q' in promptedLine:
                    self.Paused = 'user'
            if t is None or t > stop_at:
                self.Paused = 'Framework'
        if not self.Running:
            print("Main loop finished without error.")
        else:
            if self.Paused:
                print("Paused at t = {0:.3f} by {1}.".format(t, self.Paused))

    def Resume(self, stop_at = np.inf):
        self.RunStream(self.StreamHistory[-1], stop_at = stop_at, resume = True)

    def NextEvent(self, start_at):
        PropagatedEvent = None
        t = None
        for tool_name in self.ToolsList:
            PropagatedEvent = self.Tools[tool_name].__OnEvent__(PropagatedEvent)
            if PropagatedEvent is None:
                break
            else:
                if t is None:
                    t = PropagatedEvent.timestamp
                if t < start_at:
                    break
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
        print("Framework : rewinded to {0:.3f}".format(t))

#### Project Management ####

    def _GenerateEmptyProject(self):
        self.Tools = {}
        self.Types = {}
        self._ToolsCreationReferences = {}
        self._ToolsExternalParameters = {}
        self._ToolsCamerasRestrictions = {}
        self._ToolsClasses = {}

        self.ToolsOrder = {}
        self.ToolsList = []

    def LoadProject(self, ProjectFile = None, enable_easy_access = True, onlyRawData = False):
        self._GenerateEmptyProject()

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
            print("Imported tool {1} from file {0}.".format(data[tool_name]['File'], data[tool_name]['Class']))

        if len(self.ToolsOrder.keys()) != 0:
            MaxOrder = max(self.ToolsOrder.values()) + 1
            self.ToolsList = [None] * MaxOrder
        else:
            self.ToolsList = []

        for tool_name in self.ToolsOrder.keys():
            if self.ToolsList[self.ToolsOrder[tool_name]] is None:
                self.ToolsList[self.ToolsOrder[tool_name]] = tool_name
            else:
                print("Double assignement of number {0}. Aborting ProjectFile loading".format(self.ToolsOrder[tool_name]))
                return None
        if None in self.ToolsList:
            while None in self.ToolsList:
                self.ToolsList.remove(None)

        print("")
        print("Successfully generated tools order")
        print("")
        
        for tool_name in self.ToolsList:
            self.Tools[tool_name] = self._ToolsClasses[tool_name](tool_name, self, self._ToolsCreationReferences[tool_name])
            self._UpdateToolsParameters(tool_name)

            if enable_easy_access and tool_name not in self.__dict__.keys():
                self.__dict__[tool_name] = self.Tools[tool_name]

            NewType = self.Tools[tool_name].__Type__
            if NewType not in self.Types.keys():
                self.Types[NewType] = 0
            self.Types[NewType] += 1
            if NewType in TypesLimits.keys() and self.Types[NewType] > TypesLimits[NewType]:
                print("Project contains too many {0} types, aborting Projectfile loading.".format(NewType))
                continue

    def _UpdateToolsParameters(self, tool_name):
        for key, value in self._ToolsExternalParameters[tool_name].items():
            if key in self.Tools[tool_name].__dict__.keys():
                try:
                    self.Tools[tool_name].__dict__[key] = type(self.Tools[tool_name].__dict__[key])(value)
                except:
                    print("Issue with setting the new value of {0} for tool {1}, should be {2}, impossible from {3}".format(key, tool_name, type(self.Tools[tool_name].__dict__[key]), value))
            #elif key[0] == '_':
            #    print("Trying to modify protected variable {0} for tool {1}, ignoring.".format(key, tool_name))
            else:
                print("Key {0} for tool {1} doesn't exist. Please check ProjectFile integrity.".format(key, tool_name))
        self.Tools[tool_name].__CameraIndexRestriction__ = self._ToolsCamerasRestrictions[tool_name]

    def SaveProject(self, ProjectFile):
        pickle.dump(self._ProjectRawData, open(ProjectFile, 'wb'))
        self.ProjectFile = ProjectFile
        self.Modified = False

    def AddTool(self):
        print("Current project :")
        self.DisplayCurrentProject()
        print("")
        FieldList = [('File', str, False), ('Class', str, False), ('Order', int, True), ('CreationReferences', str, False), ('ExternalParameters', list, True)]
        
        Name = input('Enter the name of the new tool : ')                   # Tool name 
        if Name == '' or Name in self._ProjectRawData.keys():
            print("Invalid entry (empty or already existing).")
            return None
        self._ProjectRawData[Name] = {}
        try:
            entry = ''
            while entry == '' :                                                 # Filename and Class definition
                print("Enter the tool filename :")
                entry = input('')
            if '.py' in entry:
                entry = entry.split('.py')[0]
            self._ProjectRawData[Name]['File'] = entry
            fileLoaded = __import__(entry)
            classFound = False
            PossibleClasses = []
            if not 'Module' in fileLoaded.__dict__.keys():
                print("File does not contain any class derived from 'Module'. Aborting entry")
                del self._ProjectRawData[Name]
                return None
            for key in fileLoaded.__dict__.keys():
                if isinstance(fileLoaded.__dict__[key], type) and key[0] != '_' and fileLoaded.__dict__['Module'] in fileLoaded.__dict__[key].__bases__:
                    PossibleClasses += [key]
#                    if key == entry:
#                        classFound = True
#                        self._ProjectRawData[Name]['Class'] = entry
#                        print("Found the corresponding class in the file.")
#                        break
            if not classFound:
                if len(PossibleClasses) == 0:
                    print("No possible Class is available in this file. Aborting.")
                    del self._ProjectRawData[Name]
                    return None
                elif len(PossibleClasses) == 1:
                    print("Using class {0}".format(PossibleClasses[0]))
                    self._ProjectRawData[Name]['Class'] = PossibleClasses[0]
                else:
                    entry = ''
                    while entry == '' :
                        print("Enter the tool class among the following ones :")
                        for Class in PossibleClasses:
                            print(" * {0}".format(Class))
                        entry = input('')
                    if entry not in PossibleClasses:
                        print("Invalid class, absent from tool file or not a ClassType.")
                        del self._ProjectRawData[Name]
                        return None
                    self._ProjectRawData[Name]['Class'] = entry
            print("")
                                                                                  # Loading the class to get the references needed and parameters

            TmpClass = fileLoaded.__dict__[self._ProjectRawData[Name]['Class']](Name, self, {})
            ReferencesAsked = TmpClass.__ReferencesAsked__

            PossibleVariables = []
            for var in TmpClass.__dict__.keys():
                if var[0] == '_' and var[1] != '_':
                    PossibleVariables += [var]
            if TmpClass.__Type__ != 'Input':
                print("Enter the tool order number :")                             # Order definition
                entry = ''
                while entry == '':
                    entry = input('')
                self._ProjectRawData[Name]['Order'] = int(entry)
            else:
                print("Input Type detected. Setting default order to 0.")
                self._ProjectRawData[Name]['Order'] = 0
            NumberTaken = False
            for tool_name in self._ProjectRawData.keys():
                if tool_name != Name and 'Order' in self._ProjectRawData[tool_name].keys() and self._ProjectRawData[Name]['Order'] == self._ProjectRawData[tool_name]['Order']:
                    NumberTaken = True
            if NumberTaken:
                print("Compiling new order.")
                for tool_name in self._ProjectRawData.keys():
                    if tool_name != Name and 'Order' in self._ProjectRawData[tool_name].keys():
                        if self._ProjectRawData[tool_name]['Order'] >= self._ProjectRawData[Name]['Order']:
                            self._ProjectRawData[tool_name]['Order'] += 1
                print("Done")
                print("")

            self._ProjectRawData[Name]['CreationReferences'] = {}
            if ReferencesAsked:
                print("Fill tool name for the needed references. Currently available tool names:")
                for key in self._ProjectRawData.keys():
                    if key == Name:
                        continue
                    print(" * {0}".format(key))
                for Reference in ReferencesAsked:
                    print("Reference for '" + Reference + "'")
                    entry = ''
                    while entry == '':
                        entry = input('-> ')
                    self._ProjectRawData[Name]['CreationReferences'][Reference] = entry
            else:
                print("No particular reference needed for this tool.")
            print("")
            if TmpClass.__Type__ == 'Input':
                print("Enter camera index for this input module, if necessary.")
            else:
                print("Enter camera index(es) handled by this module, coma separated. Void will not create any restriction.")
            entry = input(" -> ")
            self._ProjectRawData[Name]['CamerasHandled'] = []
            if entry:
                for index in entry.split(','):
                    self._ProjectRawData[Name]['CamerasHandled'] += [int(index.strip())]

            self._ProjectRawData[Name]['ExternalParameters'] = {}
            if PossibleVariables:
                print("Current tool parameters :")
                for var in PossibleVariables:
                    print(" * {0} : {1}".format(var, TmpClass.__dict__[var]))
                entryvar = 'nothing'
                while entryvar != '':
                    print("Enter variable to change :")
                    entryvar = input('-> ')
                    if entryvar != '' and entryvar in PossibleVariables:
                        print("Enter new value :")
                        entryvalue = input('-> ')
                        if entryvalue != '':
                            try:
                                self._ProjectRawData[Name]['ExternalParameters'][entryvar] = type(TmpClass.__dict__[entryvar])(entryvalue)
                            except ValueError:
                                print("Could not parse entry into the correct type")
                    elif '=' in entryvar:
                        entryvar, entryvalue = entryvar.split('=')
                        if entryvar.strip() in PossibleVariables:
                            try:
                                self._ProjectRawData[Name]['ExternalParameters'][entryvar.strip()] = type(TmpClass.__dict__[entryvar.strip()])(entryvalue.strip())
                            except ValueError:
                                print("Could not parse entry into the correct type")
                    elif entryvar != '':
                        print("Wrong variable name.")
            print("")

        except KeyboardInterrupt:
            print("Canceling entries.")
            del self._ProjectRawData[Name]
            return None
        except ImportError:
            print("No such file found. Canceling entries")
            del self._ProjectRawData[Name]
            return None

        print("AddTool finished. Reloading project.")
        self.LoadProject()
        print("New project : ")
        print("")
        self.DisplayCurrentProject()
        self.Modified = True

    def DisplayCurrentProject(self):
        print("# Framework")
        print("")

        nOrder = 0
        for tool_name in self.ToolsList:
            filename = inspect.getfile(self.Tools[tool_name].__class__)
            print("# {0} : {1}, from class {2} in file {3}.".format(nOrder, tool_name, str(self.Tools[tool_name].__class__).split('.')[1], filename))
            print("     Type : {0}".format(self.Tools[tool_name].__Type__))
            if self.Tools[tool_name].__CameraIndexRestriction__:
                print("     Uses cameras indexes " + ", ".join([str(CameraIndex) for CameraIndex in self.Tools[tool_name].__CameraIndexRestriction__]))
            else:
                print("     Uses all cameras inputs.")
            if self._ToolsCreationReferences[tool_name]:
                print("     Creation References:")
                for argName, toolReference in self._ToolsCreationReferences[tool_name].items():
                    print("         -> Access to {0} from tool {1}".format(argName, toolReference))
            else:
                print("     No creation reference.")
            if self._ToolsExternalParameters[tool_name]:
                print("     Modified Parameters:")
                for var, value in  self._ToolsExternalParameters[tool_name].items():
                    print("         -> {0} = {1}".format(var, value))
            else:
                print("     Using default parameters.")
            print("")
            
            nOrder += 1

class Module:
    def __init__(self, Name, Framework, argsCreationReferences):
        '''
        Default module class.
        Each module in the Framework should inherit this class, whose 3 main methods and main parameters are required annd defined here.
        Type should be set manually.
        '''
        print("Generation of module {0}".format(Name))
        self.__ReferencesAsked__ = []
        self.__Name__ = Name
        self.__Framework__ = Framework
        self.__CreationReferences__ = dict(argsCreationReferences)
        self.__Type__ = None
        self.__Initialized__ = False
        self.__RewindForbidden__ = False
        self.__SavedValues__ = {}
        self.__CameraIndexRestriction__ = []

    def __Initialize__(self, **kwargs):
        # First restore all prevous values
        print(" > Initializing module {0}".format(self.__Name__))
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
                #print("Specified value {0} not previously defined in {1} of module {2}".format(key, self.__Name__, self.__class__.__name__))
            else:
                print("Changed specific value {0} from {1} to {2} for {3}".format(key, self.__dict__[key], value, self.__Name__))
                self.__SavedValues__[key] = copy.copy(self.__dict__[key])
                self.__dict__[key] = value
        
        # Initialize the stuff corresponding to this specific module
        if not self._InitializeModule(**kwargs):
            return False

        # Finalize Module initialization
        if self.__CameraIndexRestriction__ and self.__Type__ != 'Input':
            self.__OnEvent__ = self.__OnEventRestricted__
        else:
            self.__OnEvent__ = self._OnEventModule
        self.__Initialized__ = True
        return True

    def _InitializeModule(self, **kwargs):
        return True

    def _OnEventModule(self, event):
        return event
    def __OnEventRestricted__(self, event):
        if event.cameraIndex in self.__CameraIndexRestriction__:
            return self._OnEventModule(event)
        else:
            return event

    def _Rewind(self, t):
        None

class Event:
    def __init__(self, timestamp=None, location=None, polarity=None, cameraIndex = 0, original = None):
        if original == None:
            self.timestamp = timestamp
            self.location = np.array(location, dtype = np.int16)
            self.polarity = polarity
            self.cameraIndex = cameraIndex # Used for indexing cameras incase of multiple cameras, for stereoscopy in particular
        else:
            self.timestamp = original.timestamp
            self.location = original.location
            self.polarity = original.polarity
            self.cameraIndex = original.cameraIndex

    def __le__(self, rhs):
        if type(rhs) == float or type(rhs) == np.float64 or type(rhs) == int:
            return self.timestamp <= rhs
        elif type(rhs) == type(self) and rhs.__class__.__name__ == self.__class__.__name__:
            return self.timestamp <= rhs.timestamp
        else:
            print("Event .__le__ type not implemented : {0}".format(type(rhs)))
            return NotImplemented

    def __ge__(self, rhs):
        if type(rhs) == float or type(rhs) == np.float64 or type(rhs) == int:
            return self.timestamp >= rhs
        elif type(rhs) == type(self) and rhs.__class__.__name__ == self.__class__.__name__:
            return self.timestamp >= rhs.timestamp
        else:
            print("Event .__le__ type not implemented : {0}".format(type(rhs)))
            return NotImplemented

    def __lt__(self, rhs):
        if type(rhs) == float or type(rhs) == np.float64 or type(rhs) == int:
            return self.timestamp < rhs
        elif type(rhs) == type(self) and rhs.__class__.__name__ == self.__class__.__name__:
            return self.timestamp < rhs.timestamp
        else:
            print("Event .__le__ type not implemented : {0}".format(type(rhs)))
            return NotImplemented

    def __gt__(self, rhs):
        if type(rhs) == float or type(rhs) == np.float64 or type(rhs) == int:
            return self.timestamp > rhs
        elif type(rhs) == type(self) and rhs.__class__.__name__ == self.__class__.__name__:
            return self.timestamp > rhs.timestamp
        else:
            print("Event .__le__ type not implemented : {0}".format(type(rhs)))
            return NotImplemented
