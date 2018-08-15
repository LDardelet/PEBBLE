import numpy as np

_PERIOD_STR = '_dt'
_LASTSAVE_STR = '_ls'
_TOOLREF_STR = '_tr'
_TSLIST_STR = '_ts'
_VARIABLESLIST_STR = '_vl'

class Saver:
    def __init__(self, Name, Framework, argsCreationReferences):
	'''
        Class to filter events from spike trains.
        Expects nothing.
        '''
        self.__ReferencesAsked__ = []
        self.__Name__ = Name
        self.__Framework__ = Framework
        self.__Type__ = 'Analysis'
        self.__CreationReferences__ = dict(argsCreationReferences)

        self.__MonitorableTypes__ = ['Computation', 'Filter']

    def _Initialize(self):
        self._UnsavedParameters = False
        self.DataDict = {}

        self.PreviousPeriod = None

        print ""
        print " # "*10
        AvailableTools = [ToolName for ToolName, Tool in self.__Framework__.Tools.items() if Tool.__Type__ in self.__MonitorableTypes__]
        ans_tool = None
        while ans_tool != '':
            print "Enter new tool to monitor : "
            for nTool, ToolName in enumerate(AvailableTools):
                print " - {0} : {1} instance of class {2}.".format(nTool, ToolName, self.__Framework__.Tools[ToolName].__class__.__name__) + " (m)"*(ToolName in self.DataDict.keys())
            ans_tool = raw_input(" -> ")
            try:
                nToolChoosen = int(ans_tool)
                ToolNameChoosen = AvailableTools[nToolChoosen]
            except:
                continue
            ToolChoosen = self.__Framework__.Tools[ToolNameChoosen]

            if ToolNameChoosen not in self.DataDict.keys():
                self.DataDict[ToolNameChoosen] = {}
                self.DataDict[ToolNameChoosen][_LASTSAVE_STR] = -np.inf
                self.DataDict[ToolNameChoosen][_TOOLREF_STR] = ToolChoosen
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
                    del self.DataDict[ToolNameChoosen]
                    continue
                self.DataDict[ToolNameChoosen][_PERIOD_STR] = _dt
                self.DataDict[ToolNameChoosen][_TSLIST_STR] = []
                self.DataDict[ToolNameChoosen][_VARIABLESLIST_STR] = []
                self.PreviousPeriod = _dt
                print "" 
            
            AvailableVariables = [VariableName for VariableName in ToolChoosen.__dict__.keys() if VariableName[0] != '_']
            ans_variable = None
            while ans_variable != '':
                print "Enter new variable to monitor : "
                for nVariable, VariableName in enumerate(AvailableVariables):
                    print " - {0} : {1}".format(nVariable, VariableName) + " (m)"*(VariableName in self.DataDict[ToolNameChoosen].keys())
                ans_variable = raw_input(" -> ")
                try:
                    nVariableChoosen = int(ans_variable)
                    VariableNameChoosen = AvailableVariables[nVariableChoosen]
                except:
                    continue
                if VariableNameChoosen in self.DataDict[ToolNameChoosen].keys():
                    print "Variable {0} already being monitored.".format(VariableNameChoosen)
                    continue
                self.DataDict[ToolNameChoosen][VariableNameChoosen] = []
                self.DataDict[ToolNameChoosen][_VARIABLESLIST_STR] += [VariableNameChoosen]

                print "Monitoring variable {0}.".format(VariableNameChoosen)
                print "" 
            print ""
            if not self.DataDict[ToolNameChoosen][_VARIABLESLIST_STR]:
                del self.DataDict[ToolNameChoosen]

    def _OnEvent(self, event):
        None
        # TODO
