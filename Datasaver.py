import numpy as np
import pickle
import atexit
import copy

from BaseObjects import TypedList
from Framework import Module

_PERIOD_STR = '_dt'
_LASTSAVE_STR = '_ls'
_TOOLREF_STR = '_tr'
_TSLIST_STR = 'Timestamps'
_VARIABLESLIST_STR = '_vl'

_TYPEDLIST_SEPARATOR = '__with__'
_TYPEDLIST_STAGE_SEPARATOR = '    '

_INI_STARTED_STR = '_is'
_RUNNING_STARTED_STR = '_rs_'
_STARTED_VARIABLE = '__Started__'

_MONITOR_EXTENSION = '.mntr'

def ExtractDeeperObject(Object, VariableSteps):
    if len(VariableSteps) == 1:
        return Object.__dict__[VariableSteps[0]]
    else:
        return [ExtractDeeperObject(item, VariableSteps[1:]) for item in Object.__dict__[VariableSteps[0]]]

class Saver(Module):
    def __init__(self, Name, Framework, argsCreationReferences):
	'''
        Class to filter events from spike trains.
        Expects nothing.
        '''
        Module.__init__(self, Name, Framework, argsCreationReferences)
        self.__Type__ = 'Analysis'

        self.__MonitorableTypes__ = ['Computation', 'Filter']


    def _Initialize(self, **kwargs):
        Module._Initialize(self, **kwargs)

        self._UnsavedParameters = False
        atexit.register(self._OnClosing)

        try:
            PF = self.__Framework__.ProjectFile
            MF = PF.split('.json')[0] + _MONITOR_EXTENSION
            self._MonitorFileName = MF
        except:
            print "Unable to parse ProjectFile name"
            self._DataDict = {}
            self._MonitorFileName = None
        else:
            try:
                SavedData = pickle.load(open(MF, 'rb'))
                self._DataDict = SavedData
                print "Recovered parameters from default monitor file"
            except:
                print "No default monitor file found for {0}".format(PF)
                self._DataDict = {}

        self.PreviousPeriod = None
        self._AddMonitoredVariables()

        self._Start()

        return True

    def _OnEvent(self, event):
        for ToolName in self._DataDict.keys():
            if not self._DataDict[ToolName][_RUNNING_STARTED_STR]:
                if self._DataDict[ToolName][_TOOLREF_STR].__dict__[_STARTED_VARIABLE]:
                    self._DataDict[ToolName][_RUNNING_STARTED_STR] = True
                else:
                    continue
            if event >= self._DataDict[ToolName][_LASTSAVE_STR] + self._DataDict[ToolName][_PERIOD_STR]:
                self._DataDict[ToolName][_LASTSAVE_STR] = event.timestamp
                for VariableName in self._DataDict[ToolName][_VARIABLESLIST_STR]:
                    self._DataDict[ToolName][VariableName] += [copy.copy(ExtractDeeperObject(self._DataDict[ToolName][_TOOLREF_STR], VariableName.split(_TYPEDLIST_SEPARATOR)))]
                self._DataDict[ToolName][_TSLIST_STR] += [event.timestamp]
        return event

    def _Start(self):
        print " > Initiating monitor lists"
        for ToolName in self._DataDict.keys():
            self._DataDict[ToolName][_RUNNING_STARTED_STR] = self._DataDict[ToolName][_INI_STARTED_STR]
            self._DataDict[ToolName][_TSLIST_STR] = []
            self._DataDict[ToolName][_LASTSAVE_STR] = -np.inf
            self._DataDict[ToolName][_TOOLREF_STR] = self.__Framework__.Tools[ToolName]
            self.__dict__[ToolName] = _Ghost()
            self.__dict__[ToolName].__dict__[_TSLIST_STR] = self._DataDict[ToolName][_TSLIST_STR]
            for VariableName in self._DataDict[ToolName][_VARIABLESLIST_STR]:
                self._DataDict[ToolName][VariableName] = []
                self.__dict__[ToolName].__dict__[VariableName] = self._DataDict[ToolName][VariableName]

    def _CleanMonitor(self):
        for ToolName in self._DataDict.keys():
            del self.__dict__[ToolName]
            del self._DataDict[ToolName][_RUNNING_STARTED_STR]
            del self._DataDict[ToolName][_TSLIST_STR]
            del self._DataDict[ToolName][_LASTSAVE_STR]
            del self._DataDict[ToolName][_TOOLREF_STR]

            for VariableName in self._DataDict[ToolName][_VARIABLESLIST_STR]:
                del self._DataDict[ToolName][VariableName]

    def _AddMonitoredVariables(self):
        print ""
        print " # "*10
        AvailableTools = [ToolName for ToolName, Tool in self.__Framework__.Tools.items() if Tool.__Type__ in self.__MonitorableTypes__]
        ans_tool = None
        while ans_tool != '':
            print "Enter new tool to monitor : "
            for nTool, ToolName in enumerate(AvailableTools):
                print " - {0} : {1} instance of class {2}.".format(nTool, ToolName, self.__Framework__.Tools[ToolName].__class__.__name__) + " (m)"*(ToolName in self._DataDict.keys())
            ans_tool = raw_input(" -> ")
            try:
                nToolChoosen = int(ans_tool)
                ToolNameChoosen = AvailableTools[nToolChoosen]
            except:
                continue
            ToolChoosen = self.__Framework__.Tools[ToolNameChoosen]

            if ToolNameChoosen not in self._DataDict.keys():
                self._DataDict[ToolNameChoosen] = {}
                Entry = "Enter a monitoring period for this tool "
                if self.PreviousPeriod:
                    Entry = Entry + "(blank for previous value of {0:.3f}) ".format(self.PreviousPeriod)
                Entry = Entry + ": "
                ans_dt = raw_input(Entry)
                if ans_dt == '' and not self.PreviousPeriod is None:
                    ans_dt = str(self.PreviousPeriod)
                try:
                    _dt = float(ans_dt)
                except:
                    print "Invalide period value"
                    del self._DataDict[ToolNameChoosen]
                    continue
                self._DataDict[ToolNameChoosen][_LASTSAVE_STR] = -np.inf
                self._DataDict[ToolNameChoosen][_TOOLREF_STR] = ToolChoosen
                self._DataDict[ToolNameChoosen][_PERIOD_STR] = _dt
                self._DataDict[ToolNameChoosen][_TSLIST_STR] = []
                self._DataDict[ToolNameChoosen][_VARIABLESLIST_STR] = []
                if _STARTED_VARIABLE in ToolChoosen.__dict__.keys():
                    self._DataDict[ToolNameChoosen][_INI_STARTED_STR] = ToolChoosen.__dict__[_STARTED_VARIABLE]
                else:
                    self._DataDict[ToolNameChoosen][_INI_STARTED_STR] = True
                self.PreviousPeriod = _dt
                print "" 
            
            def GetContents(Object, ParentVariableName):
                ObjectAvailableVariables = []
                for VariableName in Object.__dict__.keys():
                    if VariableName[0] != '_':
                        if type(Object.__dict__[VariableName]) != TypedList:
                            ObjectAvailableVariables += [ParentVariableName + VariableName]
                        else:
                            InstanceTestClass = Object.__dict__[VariableName].__elems_type__
                            InstanceTestObject = InstanceTestClass(fake = True)
                            ObjectAvailableVariables += GetContents(InstanceTestObject, ParentVariableName + VariableName + _TYPEDLIST_SEPARATOR)
                return ObjectAvailableVariables

            AvailableVariables = GetContents(ToolChoosen, '')

            ans_variable = None
            while ans_variable != '':
                print "Enter new variable to monitor : "
                PreviousPath = []
                for nVariable, VariableName in enumerate(AvailableVariables):
                    VariableDepth = VariableName.count(_TYPEDLIST_SEPARATOR)
                    CurrentPath = VariableName.split(_TYPEDLIST_SEPARATOR)[:-1]
                    for nStage, Stage in enumerate(CurrentPath):
                        if nStage >= len(PreviousPath) or Stage != PreviousPath[nStage]:
                            print nStage * _TYPEDLIST_STAGE_SEPARATOR + " |-> " + Stage
                    PreviousPath = CurrentPath
                    print VariableDepth * _TYPEDLIST_STAGE_SEPARATOR + " |-> {0} : {1}".format(nVariable, VariableName.split(_TYPEDLIST_SEPARATOR)[-1]) + " (m)"*(VariableName in self._DataDict[ToolNameChoosen][_VARIABLESLIST_STR])
                ans_variable = raw_input(" -> ")
                try:
                    nVariableChoosen = int(ans_variable)
                    VariableNameChoosen = AvailableVariables[nVariableChoosen]
                except:
                    continue
                if VariableNameChoosen in self._DataDict[ToolNameChoosen][_VARIABLESLIST_STR]:
                    print "Variable {0} removed from being monitored.".format(VariableNameChoosen)
                    self._DataDict[ToolNameChoosen][_VARIABLESLIST_STR].remove(VariableNameChoosen)
                else:
                    print "Monitoring variable {0}.".format(VariableNameChoosen)
                    self._DataDict[ToolNameChoosen][_VARIABLESLIST_STR] += [VariableNameChoosen]
                self._UnsavedParameters = True
                print "" 
            print ""
            if not self._DataDict[ToolNameChoosen][_VARIABLESLIST_STR]:
                del self._DataDict[ToolNameChoosen]

    def _OnClosing(self):
        if not self._UnsavedParameters:
            return None

        print "Unsaved monitor variables."
        ans = 'void'
        while ans != 'D' and not _MONITOR_EXTENSION in ans:
            if not self._MonitorFileName is None:
                ans = raw_input("Leave blank to save to default file {0}, enter new filename ({1}) or (D)iscard : ".format(self._MonitorFileName, _MONITOR_EXTENSION))
                if ans == '':
                    ans = self._MonitorFileName
            else:
                ans = raw_input("Enter monitor filename ({1}) or (D)iscard : ".format(_MONITOR_EXTENSION))
        if ans != 'D':
            self.SaveProject(ans, from_closing = True)
        else:
            print " > Discarding monitor parameters."

    def SaveProject(self, FileName, from_closing = False):
        print " > Cleaning up saved variables"
        self._CleanMonitor()
        print " > Saving monitor parameters in {0}".format(FileName)
        pickle.dump(self._DataDict, open(FileName, 'wb'))
        self._MonitorFileName = FileName
        self._UnsavedParameters = False
        print " > Done !"

        if not from_closing:
            self._Start()

class _Ghost:
    def __init__(self):
        return None
