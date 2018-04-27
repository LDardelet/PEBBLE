import numpy as np
import pickle
import sys
import select

import inspect
from event import Event

TypesLimits = {'Input':1}
NonRunningTools = ['Input', 'Framework']

class Framework:
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
            except:
                print "Unable to load project, check self._ProjectRawData for file integrity"

    def _Initialize(self, argsInitializationDict):
        None

    def SaveProject(self, ProjectFile):
        pickle.dump(self._ProjectRawData, open(ProjectFile, 'wb'))

    def LoadProject(self, ProjectFile):
        self.Tools = {}
        self.Types = {}
        self.InputName = None
        self.ToolsCreationParameters = {}
        self.ToolsInitializationParameters = {}

        self.Tools['Framework'] = self
        self.ToolsCreationParameters['Framework'] = {}
        self.ToolsInitializationParameters['Framework'] = {}

        ToolsOrder = {}

        data = pickle.load(open(ProjectFile, 'rb'))

        self._ProjectRawData = data

        for tool_name in data.keys():
            fileLoaded = __import__(data[tool_name]['File'])
            classLoaded = getattr(fileLoaded, data[tool_name]['Class'])

            self.ToolsCreationParameters[tool_name] = {}
            for argClass, argName, argAlias in data[tool_name]['CreationArgs']:
                self.ToolsCreationParameters[tool_name][argClass+'.'+argName+'.'+argAlias] = self.Tools[argClass].__dict__[argName]
            self.ToolsInitializationParameters[tool_name] = {}
            for argClass, argName in data[tool_name]['InitializationArgs']:
                self.ToolsInitializationParameters[tool_name][argClass+'.'+argName] = self.Tools[argClass].__dict__[argName]

            self.Tools[tool_name] = classLoaded(self.ToolsCreationParameters[tool_name])

            NewType = self.Tools[tool_name]._Type
            if NewType not in self.Types.keys():
                self.Types[NewType] = 0
            self.Types[NewType] += 1
            if NewType in TypesLimits.keys() and self.Types[NewType] > TypesLimits[NewType]:
                print "Project contains too many {0} types, aborting Projectfile loading.".format(NewType)
            if NewType == 'Input':
                self.InputName = str(tool_name)

            if NewType not in NonRunningTools:
                ToolsOrder[tool_name] = data[tool_name]['Order']

            print "Imported tool {0} from file {1}.".format(data[tool_name]['File'], data[tool_name]['Class'])

        MaxOrder = max(ToolsOrder.values()) + 1
        self.ToolsOrder = [None] * MaxOrder

        for tool_name in ToolsOrder.keys():
            if self.ToolsOrder[ToolsOrder[tool_name]] is None:
                self.ToolsOrder[ToolsOrder[tool_name]] = tool_name
            else:
                print "Double assignement of number {0}. Aborting ProjectFile loading"
                return None
        if None in self.ToolsOrder:
            print "Removing remaining 'None' in self.ToolsOrder, please check file integrity."
            while None in self.ToolsOrder:
                self.ToolsOrder.remove(None)

    def Initialize(self):
        for tool_name in [self.InputName] + self.ToolsOrder:
            self.Tools[tool_name]._Initialize(self.ToolsInitializationParameters[tool_name])

    def RunStream(self, StreamName):
        self.StreamHistory += [StreamName]
        self.Initialize()

        for event in self.Streams[StreamName]:
            self.nEvents[StreamName] += 1
            for tool_name in self.ToolsOrder:
                self.Tools[tool_name].OnEvent(event)
            
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                promptedLine = sys.stdin.readline()
                if 'a' in promptedLine or 'q' in promptedLine:
                    print "Closed main loop at event {0}".format(self.nEvents[StreamName])
                    break
        print "Main loop finished without error."

    def DisplayCurrentProject(self):
        print "# Framework"
        print ""
        if self.InputName != None:
            tool_name = self.InputName
            filename = inspect.getfile(self.Tools[tool_name].__class__)
            print "# 0 : {1}, from class {1} in file {2}.".format(None, tool_name, str(self.Tools[tool_name].__class__).split('.')[1], filename)
        print "     Type : {0}".format(self.Tools[tool_name]._Type)
        print "     Parameters used for Creation:"
        for Parameter in self.ToolsCreationParameters[tool_name].keys():
            print "         -> {0} from {1}, aliased {2}".format(Parameter.split('.')[1], Parameter.split('.')[0], Parameter.split('.')[2])
        print "     Parameters used for Initialization:"
        for Parameter in self.ToolsInitializationParameters[tool_name].keys():
            print "         -> {0} from {1}".format(Parameter.split('.')[1], Parameter.split('.')[0])

        print ""

        nOrder = 1
        for tool_name in self.ToolsOrder:
            filename = inspect.getfile(self.Tools[tool_name].__class__)
            print "# {0} : {1}, from class {1} in file {2}.".format(nOrder, tool_name, str(self.Tools[tool_name].__class__).split('.')[1], filename)
            print "     Type : {0}".format(self.Tools[tool_name]._Type)
            print "     Parameters used for Creation:"
            for Parameter in self.ToolsCreationParameters[tool_name].keys():
                print "         -> {0} from {1}, aliased {2}".format(Parameter.split('.')[1], Parameter.split('.')[0], Parameter.split('.')[2])
            print "     Parameters used for Initialization:"
            for Parameter in self.ToolsInitializationParameters[tool_name].keys():
                print "         -> {0} from {1}".format(Parameter.split('.')[1], Parameter.split('.')[0])
            print ""
            
            nOrder += 1
