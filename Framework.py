import numpy as np
import pickle
import sys
import select
import inspect

from event import Event
import tools

TypesLimits = {'Input':1}
NonRunningTools = ['Input', 'Framework']

class Framework:
    '''
    Main event-based framework file.
    Each tool is in a  different file, and caan be added through 'Project files', that create the sequence of events processing.
    Each tool must contain a simple  __init__ method with 'self' and 'argCreationDict' only, the second one containing all necessary variables from tools above itself.
    It also must contain a '_Initialization' method, with 'self' and 'argInitializationDict' only, the second one containing all necessary information about the current stream processed.

    Finally, each tool contains an '_OnEvent' method processing each event. It must only have 'event' as argument - apart from 'self' - and all variables must have be stored inside, prior to this.
    '_OnEvent' can be seen as a filter, thus it must return the event incase one wants the processing of the current event to go on with the following tools.

    Finally, each tool must declare a '_Type' variable, to disclaim to the framework what kind to job is processed.
    For the special type 'Input', no '_OnEvent' method is currently needed.

    Currently implemented tools :
        -> DisplayHandler : allows to use the StreamDisplay Program
        -> StreamReader : Basic input method
        -> Memory : Basic storage tool
    '''
    def __init__(self, ProjectFile = None, verboseRatio = 10000):
        self.ProjectFile = ProjectFile

        self.Self = self

        self._Type = 'Framework'

        self.Streams = {}
        self.StreamsGeometries = {}
        self.nEvents = {}
        self.StreamHistory = []

        self.VerboseRatio = verboseRatio

        if not ProjectFile is None:
            try:
                self.LoadProject(ProjectFile)
            except e:
                print e
                print "Unable to load project, check self._ProjectRawData for file integrity"

    def Initialize(self):
        for tool_name in [self.InputName] + self.ToolsOrder:
            CurrentDict = {}
            for ArgName in self._ToolsInitializationParameters[tool_name]:
                argClass, argName = ArgName.split('.')
                CurrentDict[ArgName] = self.Tools[argClass].__dict__[argName]
            self.Tools[tool_name]._Initialize(CurrentDict)

    def RunStream(self, StreamName, resume = False):
        if not resume:
            self.StreamHistory += [StreamName]
            self.Initialize()

        for event in self.Streams[StreamName][self.nEvents[StreamName]:]:
            self.nEvents[StreamName] += 1
            PropagatedEvent = Event(original = event)
            for tool_name in self.ToolsOrder:
                PropagatedEvent = self.Tools[tool_name]._OnEvent(PropagatedEvent)
                if PropagatedEvent is None:
                    break
            
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                promptedLine = sys.stdin.readline()
                if 'a' in promptedLine or 'q' in promptedLine:
                    print "Closed main loop at event {0}".format(self.nEvents[StreamName])
                    break
        if self.nEvents[StreamName] == len(self.Streams[StreamName]):
            print "Main loop finished without error."

    def Resume(self):
        self.RunStream(self.StreamHistory[-1], resume = True)

#### Project Management ####

    def LoadProject(self, ProjectFile = None, enable_easy_access = True):
        self.Tools = {}
        self.Types = {}
        self.InputName = None
        self._ToolsCreationParameters = {}
        self._ToolsInitializationParameters = {}
        self._ToolsClasses = {}

        self.Tools['Framework'] = self
        self._ToolsCreationParameters['Framework'] = {}
        self._ToolsInitializationParameters['Framework'] = {}
        self._ToolsClasses['Framework'] = self.__class__

        ToolsOrder = {}
        FirstTools = []

        if ProjectFile is None:
            data = self._ProjectRawData
        else:
            data = pickle.load(open(ProjectFile, 'rb'))
            self._ProjectRawData = data

        for tool_name in data.keys():
            fileLoaded = __import__(data[tool_name]['File'])
            self._ToolsClasses[tool_name] = getattr(fileLoaded, data[tool_name]['Class'])

            self._ToolsCreationParameters[tool_name] = {}
            for argClass, argName, argAlias in data[tool_name]['CreationArgs']:
                self._ToolsCreationParameters[tool_name][argClass+'.'+argName+'.'+argAlias] = self.Tools[argClass].__dict__[argName]
            self._ToolsInitializationParameters[tool_name] = []
            for argClass, argName in data[tool_name]['InitializationArgs']:
                self._ToolsInitializationParameters[tool_name] += [argClass+'.'+argName]

            if 'Order' in data[tool_name].keys():
                ToolsOrder[tool_name] = data[tool_name]['Order']
            else:
                FirstTools += [tool_name]

            print "Imported tool {1} from file {0}.".format(data[tool_name]['File'], data[tool_name]['Class'])

        MaxOrder = max(ToolsOrder.values()) + 1
        self.ToolsOrder = [None] * MaxOrder

        for tool_name in ToolsOrder.keys():
            if self.ToolsOrder[ToolsOrder[tool_name]] is None:
                self.ToolsOrder[ToolsOrder[tool_name]] = tool_name
            else:
                print "Double assignement of number {0}. Aborting ProjectFile loading".format(ToolsOrder[tool_name])
                return None
        if None in self.ToolsOrder:
            print "Removing remaining 'None' in self.ToolsOrder, please check file integrity."
            while None in self.ToolsOrder:
                self.ToolsOrder.remove(None)

        print ""
        print "Successfully generated tools order"
        
        for tool_name in FirstTools:
            self.Tools[tool_name] = self._ToolsClasses[tool_name](self._ToolsCreationParameters[tool_name])
            if enable_easy_access and tool_name not in self.__dict__.keys():
                self.__dict__[tool_name] = self.Tools[tool_name]

            NewType = self.Tools[tool_name]._Type
            if NewType not in self.Types.keys():
                self.Types[NewType] = 0
            self.Types[NewType] += 1
            if NewType in TypesLimits.keys() and self.Types[NewType] > TypesLimits[NewType]:
                print "Project contains too many {0} types, aborting Projectfile loading.".format(NewType)
            if NewType == 'Input':
                self.InputName = str(tool_name)
            print "Created tool {0} (among FirstTools).".format(tool_name)
        for tool_name in self.ToolsOrder:
            self.Tools[tool_name] = self._ToolsClasses[tool_name](self._ToolsCreationParameters[tool_name])
            if enable_easy_access and tool_name not in self.__dict__.keys():
                self.__dict__[tool_name] = self.Tools[tool_name]

            NewType = self.Tools[tool_name]._Type
            if NewType not in self.Types.keys():
                self.Types[NewType] = 0
            self.Types[NewType] += 1
            if NewType in TypesLimits.keys() and self.Types[NewType] > TypesLimits[NewType]:
                print "Project contains too many {0} types, aborting Projectfile loading.".format(NewType)
            if NewType == 'Input':
                self.InputName = str(tool_name)
            print "Created tool {0} (among ToolsOrder).".format(tool_name)
        
    def SaveProject(self, ProjectFile):
        pickle.dump(self._ProjectRawData, open(ProjectFile, 'wb'))

    def AddTool(self):
        print "Current project :"
        self.DisplayCurrentProject()
        print ""
        FieldList = [('File', str, False), ('Class', str, False), ('Order', int, True), ('CreationArgs', list, True), ('InitializationArgs', list, True)]
        
        Name = raw_input('Enter the name of the new tool : ')
        if Name == '' or Name in self._ProjectRawData.keys():
            print "Invalid entry (empty or already existing)."
            return None
        self._ProjectRawData[Name] = {}
        try:
            for field in FieldList:
                if field[1] != list:
                    entry = None
                    while entry is None or (entry == '' and not field[2]):
                        print "Enter the value for field {0} : ".format(field[0])
                        entry = raw_input('')
                    if entry != '':
                        self._ProjectRawData[Name][field[0]] = field[1](entry)
                else:
                    entry = None
                    self._ProjectRawData[Name][field[0]] = []
                    while entry != '':
                        print "Enter new item for field {0} - blank stops adding items to the list, create tuples by commas separation : ".format(field[0])
                        entry = raw_input('')
                        if entry != '':
                            self._ProjectRawData[Name][field[0]] += [tuple([item.strip() for item in entry.split(',')])]
            print ""
            
            if 'Order' in self._ProjectRawData[Name].keys():
                self._ProjectRawData[Name]['Order'] -= 1
                print "Compiling new order."
                print ""
                for tool_name in self._ProjectRawData.keys():
                    if tool_name != Name and 'Order' in self._ProjectRawData[tool_name].keys():
                        if self._ProjectRawData[tool_name]['Order'] >= self._ProjectRawData[Name]['Order']:
                            self._ProjectRawData[tool_name]['Order'] += 1
        except KeyboardInterrupt:
            print "Canceling entries."
            del self._ProjectRawData[Name]
            return None

        self.LoadProject()
        print "New project : "
        self.DisplayCurrentProject()

    def DisplayCurrentProject(self):
        print "# Framework"
        print ""
        if self.InputName != None:
            tool_name = self.InputName
            filename = inspect.getfile(self.Tools[tool_name].__class__)
            print "# 0 : {1}, from class {1} in file {2}.".format(None, tool_name, str(self.Tools[tool_name].__class__).split('.')[1], filename)
            print "     Type : {0}".format(self.Tools[tool_name]._Type)
            print "     Parameters used for Creation:"
            for Parameter in self._ToolsCreationParameters[tool_name].keys():
                print "         -> {0} from {1}, aliased {2}".format(Parameter.split('.')[1], Parameter.split('.')[0], Parameter.split('.')[2])
            print "     Parameters used for Initialization:"
            for Parameter in self._ToolsInitializationParameters[tool_name].keys():
                print "         -> {0} from {1}".format(Parameter.split('.')[1], Parameter.split('.')[0])

            print ""

        nOrder = 1
        for tool_name in self.ToolsOrder:
            filename = inspect.getfile(self.Tools[tool_name].__class__)
            print "# {0} : {1}, from class {1} in file {2}.".format(nOrder, tool_name, str(self.Tools[tool_name].__class__).split('.')[1], filename)
            print "     Type : {0}".format(self.Tools[tool_name]._Type)
            print "     Parameters used for Creation:"
            for Parameter in self._ToolsCreationParameters[tool_name].keys():
                print "         -> {0} from {1}, aliased {2}".format(Parameter.split('.')[1], Parameter.split('.')[0], Parameter.split('.')[2])
            print "     Parameters used for Initialization:"
            for Parameter in self._ToolsInitializationParameters[tool_name].keys():
                print "         -> {0} from {1}".format(Parameter.split('.')[1], Parameter.split('.')[0])
            print ""
            
            nOrder += 1
