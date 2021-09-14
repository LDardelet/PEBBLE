from ModuleBase import ModuleBase

class StreamSpliter(ModuleBase):
    def _OnCreation(self):
        '''
        Creates a second stream, copying events from one stream to the new one
        '''
        self.__GeneratesSubStream__ = True

    def _OnInputIndexesSet(self, Indexes):
        self.NewStreamIndexes = Indexes
        return True

    def _OnInitialization(self):
        return True

    def _OnEventModule(self, event):
        for NewIndex in self.NewStreamIndexes:
            event.Join(event.Copy(NewIndex))
        return
