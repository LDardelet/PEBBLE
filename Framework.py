import numpy as np
import pickle
import sys
import select
import inspect
import atexit
import types

from event import Event
import tools

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
        self.ProjectFile = ProjectFile
        self.Modified = False

        self.__Type__ = 'Framework'

        self.StreamsGeometries = {}
        self.StreamHistory = []

        self.VerboseRatio = verboseRatio

        atexit.register(self._OnClosing)

        if not ProjectFile is None:
            try:
                self.LoadProject(ProjectFile, onlyRawData = onlyRawData)
            except e:
                print e
                print "Unable to load project, check self._ProjectRawData for file integrity"
        else:
            self._ProjectRawData = {}
            self._GenerateEmptyProject()

    def Initialize(self):
        for tool_name in self.ToolsList:
            self.Tools[tool_name]._Initialize()

    def _OnClosing(self):
        if self.Modified:
            ans = 'void'
            while '.json' not in ans and ans != '':
                ans = raw_input('Unsaved changes. Please enter a file name, or leave blank to discard : ')
            if ans != '':
                self.SaveProject(ans)

    def ReRun(self, stop_at = np.inf):
        self.RunStream(self.StreamHistory[-1], stop_at = stop_at)

    def RunStream(self, StreamName, stop_at = np.inf, resume = False):
        if not resume:
            self.StreamHistory += [StreamName]
            self.Initialize()

        self.Running = True
        self.Paused = False
        while self.Running and not self.Paused:
            t = self.NextEvent()

            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                promptedLine = sys.stdin.readline()
                if 'a' in promptedLine or 'q' in promptedLine:
                    self.Paused = True
            if t > stop_at:
                self.Paused = True
        if not self.Running:
            print "Main loop finished without error."
        if self.Paused:
            print "Paused at t = {0:.2f}.".format(t)

    def Resume(self, stop_at = np.inf):
        self.RunStream(self.StreamHistory[-1], stop_at = stop_at, resume = True)

    def NextEvent(self):
        PropagatedEvent = None
        t = None
        for tool_name in self.ToolsList:
            PropagatedEvent = self.Tools[tool_name]._OnEvent(PropagatedEvent)
            if PropagatedEvent is None:
                break
            else:
                if t is None:
                    t = PropagatedEvent.timestamp
        return t


#### Project Management ####

    def _GenerateEmptyProject(self):
        self.Tools = {}
        self.Types = {}
        self._ToolsCreationReferences = {}
        self._ToolsExternalParameters = {}
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

            self.ToolsOrder[tool_name] = data[tool_name]['Order']
            print "Imported tool {1} from file {0}.".format(data[tool_name]['File'], data[tool_name]['Class'])

        if len(self.ToolsOrder.keys()) != 0:
            MaxOrder = max(self.ToolsOrder.values()) + 1
            self.ToolsList = [None] * MaxOrder
        else:
            self.ToolsList = []

        for tool_name in self.ToolsOrder.keys():
            if self.ToolsList[self.ToolsOrder[tool_name]] is None:
                self.ToolsList[self.ToolsOrder[tool_name]] = tool_name
            else:
                print "Double assignement of number {0}. Aborting ProjectFile loading".format(self.ToolsOrder[tool_name])
                return None
        if None in self.ToolsList:
            while None in self.ToolsList:
                self.ToolsList.remove(None)

        print ""
        print "Successfully generated tools order"
        
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
                print "Project contains too many {0} types, aborting Projectfile loading.".format(NewType)
                continue
            print "Created tool {0}.".format(tool_name)

    def _UpdateToolsParameters(self, tool_name):
        for key, value in self._ToolsExternalParameters[tool_name].items():
            if key in self.Tools[tool_name].__dict__.keys() and key[0] != '_':
                try:
                    self.Tools[tool_name].__dict__[key] = type(self.Tools[tool_name].__dict__[key])(value)
                except:
                    print "Issue with setting the new value of {0} for tool {1}, should be {2}, impossible from {3}".format(key, tool_name, type(self.Tools[tool_name].__dict__[key]), value)
            elif key[0] == '_':
                print "Trying to modify protected variable {0} for tool {1}, ignoring.".format(key, tool_name)
            else:
                print "Key {0} for tool {1} doesn't exist. Please ProjectFile integrity.".format(key, tool_name)

    def SaveProject(self, ProjectFile):
        pickle.dump(self._ProjectRawData, open(ProjectFile, 'wb'))
        self.ProjectFile = ProjectFile
        self.Modified = False

    def AddTool(self):
        print "Current project :"
        self.DisplayCurrentProject()
        print ""
        FieldList = [('File', str, False), ('Class', str, False), ('Order', int, True), ('CreationReferences', str, False), ('ExternalParameters', list, True)]
        
        Name = raw_input('Enter the name of the new tool : ')                   # Tool name 
        if Name == '' or Name in self._ProjectRawData.keys():
            print "Invalid entry (empty or already existing)."
            return None
        self._ProjectRawData[Name] = {}
        try:
            entry = ''
            while entry == '' :                                                 # Filename and Class definition
                print "Enter the tool filename :"
                entry = raw_input('')
            if '.py' in entry:
                entry = entry.split('.py')[0]
            self._ProjectRawData[Name]['File'] = entry
            fileLoaded = __import__(entry)
            classFound = False
            PossibleClasses = []
            for key in fileLoaded.__dict__.keys():
                if type(fileLoaded.__dict__[key]) is types.ClassType:
                    PossibleClasses += [key]
                    if key == entry:
                        classFound = True
                        self._ProjectRawData[Name]['Class'] = entry
                        print "Found the corresponding class in the file."
                        break
            if not classFound:
                if len(PossibleClasses) == 0:
                    print "No possible Class is available in this file. Aborting."
                    del self._ProjectRawData[Name]
                    return None
                elif len(PossibleClasses) == 1:
                    print "Using class {0}".format(PossibleClasses[0])
                    self._ProjectRawData[Name]['Class'] = PossibleClasses[0]
                else:
                    entry = ''
                    while entry == '' :
                        print "Enter the tool class among the following ones :"
                        for Class in PossibleClasses:
                            print " * {0}".format(Class)
                        entry = raw_input('')
                    if entry not in PossibleClasses:
                        print "Invalid class, absent from tool file or not a ClassType."
                        del self._ProjectRawData[Name]
                        return None
                    self._ProjectRawData[Name]['Class'] = entry
            print ""
                                                                                  # Loading the class to get the references needed and parameters

            TmpClass = fileLoaded.__dict__[self._ProjectRawData[Name]['Class']](Name, self, {})
            ReferencesAsked = TmpClass._ReferencesAsked

            PossibleVariables = []
            for var in TmpClass.__dict__.keys():
                if var[0] != '__':
                    PossibleVariables += [var]
            if TmpClass.__Type__ != 'Input':
                print "Enter the tool order number :"                             # Order definition
                entry = ''
                while entry == '':
                    entry = raw_input('')
                self._ProjectRawData[Name]['Order'] = int(entry)
            else:
                print "Input Type detected. Setting default order to 0."
                self._ProjectRawData[Name]['Order'] = 0
            NumberTaken = False
            for tool_name in self._ProjectRawData.keys():
                if tool_name != Name and 'Order' in self._ProjectRawData[tool_name].keys() and self._ProjectRawData[Name]['Order'] == self._ProjectRawData[tool_name]['Order']:
                    NumberTaken = True
            if NumberTaken:
                print "Compiling new order."
                for tool_name in self._ProjectRawData.keys():
                    if tool_name != Name and 'Order' in self._ProjectRawData[tool_name].keys():
                        if self._ProjectRawData[tool_name]['Order'] >= self._ProjectRawData[Name]['Order']:
                            self._ProjectRawData[tool_name]['Order'] += 1
                print "Done"
                print ""

            self._ProjectRawData[Name]['CreationReferences'] = {}
            if ReferencesAsked:
                print "Fill tool name for the needed references. Currently available tool names:"
                for key in self._ProjectRawData.keys():
                    print " * {0}".format(key)
                for Reference in ReferencesAsked:
                    print Reference
                    entry = ''
                    while entry == '':
                        entry = raw_input('-> ')
                    self._ProjectRawData[Name]['CreationReferences'][Reference] = entry
            else:
                print "No particular reference needed for this tool."
            print ""
            self._ProjectRawData[Name]['ExternalParameters'] = {}
            if PossibleVariables:
                print "Current tool parameters :"
                for var in PossibleVariables:
                    print " * {0} : {1}".format(var[1:], TmpClass.__dict__[var])
                entryvar = 'nothing'
                while entryvar != '':
                    print "Enter variable to change :"
                    entryvar = raw_input('-> ')
                    if entryvar != '' and entryvar in PossibleVariables:
                        print "Enter new value :"
                        entryvalue = raw_input('-> ')
                        if entryvalue != '':
                            try:
                                self._ProjectRawData[Name]['ExternalParameters'][entryvar] = type(TmpClass.__dict__[entryvar])(entryvalue)
                            except ValueError:
                                print "Could not parse entry into the correct type"
                    elif '=' in entryvar:
                        entryvar, entryvalue = entryvar.split('=')
                        if entryvar.strip() in PossibleVariables:
                            try:
                                self._ProjectRawData[Name]['ExternalParameters'][entryvar.strip()] = type(TmpClass.__dict__[entryvar.strip()])(entryvalue.strip())
                            except ValueError:
                                print "Could not parse entry into the correct type"
                    elif entryvar != '':
                        print "Wrong variable name."
            print ""

        except KeyboardInterrupt:
            print "Canceling entries."
            del self._ProjectRawData[Name]
            return None
        except ImportError:
            print "No such file found. Canceling entries"
            del self._ProjectRawData[Name]
            return None

        print "AddTool finished. Reloading project."
        self.LoadProject()
        print "New project : "
        print ""
        self.DisplayCurrentProject()
        self.Modified = True

    def DisplayCurrentProject(self):
        print "# Framework"
        print ""

        nOrder = 0
        for tool_name in self.ToolsList:
            filename = inspect.getfile(self.Tools[tool_name].__class__)
            print "# {0} : {1}, from class {2} in file {3}.".format(nOrder, tool_name, str(self.Tools[tool_name].__class__).split('.')[1], filename)
            print "     Type : {0}".format(self.Tools[tool_name].__Type__)
            if self._ToolsCreationReferences[tool_name]:
                print "     Creation References:"
                for argName, toolReference in self._ToolsCreationReferences[tool_name].items():
                    print "         -> Access to {0} from tool {1}".format(argName, toolReference)
            else:
                print "     No creation reference."
            if self._ToolsExternalParameters[tool_name]:
                print "     Modified Parameters:"
                for var, value in  self._ToolsExternalParameters[tool_name].items():
                    print "         -> {0} = {1}".format(var, value)
            else:
                print "     Using default parameters."
            print ""
            
            nOrder += 1
