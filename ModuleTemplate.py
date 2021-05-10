from PEBBLE import ModuleBase

class ModuleTemplate(ModuleBase):
    def _OnCreation(self):
        '''
        Method called upon creation of the framework.
        This first six elements defined here describe framework level that can be change for that module.
        Any parameter P needed for that module can be added as self._P = DefaultValue
        After that, those parameters are recovered by the framework, and can be changed at runtime, through a dictionnary, or be project-specific, and changed by default.
        '''
        self.__ModulesLinksRequested__ = []     # Does that module need a reference to another module, to recove some module-specific data ? The entry is a list of strings, and upon framework definition, links are user-defined
                                                # The Linked modules are automatically as class variables with the specific name:
                                                # Example : self.__ModulesLinksRequested__ = ['Memory']. Upon framework definition, we link 'Memory' to another module M, with M.__Name__ = 'LeftCameraMemory'
                                                # Upon initialization, self.Memory = M

        self._NeedsLogColumn = False            # Do we want logs to show a column dedicated to the outputs of that module ?

        self.__IsInput__ = False                # Is that module a primary input of the project ? If so, no incoming event will be considered, and it will define some of the first events that start the computation
        self.__GeneratesSubStream__ = False     # Does that module generated another substream among the framework ? It would be some additional data that we don't want mixed to the rest of the events

        self._MonitoredVariables = []           # Do we want to store any data over the computation ? Each element of the list must be the tuple (str Var, type Type), in order for the data to be properly copied
        self._MonitorDt = 0                     # If data is stored, what is the maximum waiting time between two data logs ?

        # self._Parameter1 = 0.1
        # self._Parameter2 = "ExampleParameter that can be set upon creation or at runtime"

    def _OnInitialization(self):
        '''
        Method called at runtime to initialize the module.
        All parameters have been set though the default value, the project values or the runtime values, and the links between modules have been made.
        This is where runtime variables have to be created. This method should not be called after run started.
        Return True if all is set and done.
        '''
        return True

    def _OnEventModule(self, event):
        '''
        Main method loop. All argument events are within constraints of substreams handled by this module. 
        However, no filter is made upon the data carried by each event. 
        Return nothing, as events are handled by the framework.
        Should the event be filtered and not propagated though the rest of the framework, call the method event.Filter()
        '''
        return

    def _SetGeneratedSubStreamsIndexes(self, Indexes):
        '''
        Method necessary for all modules that are at the base of a SubStream (self.__IsInput__ = True or self.__GeneratesSubStream__ = True).
        Allows to set the variables according to the specified data of the project file.
        Return True if all is set and done. Leave unchanged for unconcerned modules.
        '''
        return False

    def _OnClosing(self):
        '''
        Method to close any file or connexions that would have been created by the framework, upon frameork destruction - typically when interpreter exits.
        Use that instead of an atexit instance.
        '''
