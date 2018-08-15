import numpy as np

class Event:
    def __init__(self, timestamp=None, location=None, polarity=None, original = None):
        if original == None:
            self.timestamp = timestamp
            self.location = location
            self.polarity = polarity
        else:
            self.timestamp = original.timestamp
            self.location = original.location
            self.polarity = original.polarity

    def __le__(self, rhs):
        if type(rhs) == float or type(rhs) == np.float64 or type(rhs) == int:
            return self.timestamp <= rhs
        elif type(rhs) == type(self) and rhs.__class__.__name__ == self.__class__.__name__:
            return self.timestamp <= rhs.timestamp
        else:
            print "Event .__le__ type not implemented : {0}".format(type(rhs))
            return NotImplemented

    def __ge__(self, rhs):
        if type(rhs) == float or type(rhs) == np.float64 or type(rhs) == int:
            return self.timestamp >= rhs
        elif type(rhs) == type(self) and rhs.__class__.__name__ == self.__class__.__name__:
            return self.timestamp >= rhs.timestamp
        else:
            print "Event .__le__ type not implemented : {0}".format(type(rhs))
            return NotImplemented

    def __lt__(self, rhs):
        if type(rhs) == float or type(rhs) == np.float64 or type(rhs) == int:
            return self.timestamp < rhs
        elif type(rhs) == type(self) and rhs.__class__.__name__ == self.__class__.__name__:
            return self.timestamp < rhs.timestamp
        else:
            print "Event .__le__ type not implemented : {0}".format(type(rhs))
            return NotImplemented

    def __gt__(self, rhs):
        if type(rhs) == float or type(rhs) == np.float64 or type(rhs) == int:
            return self.timestamp > rhs
        elif type(rhs) == type(self) and rhs.__class__.__name__ == self.__class__.__name__:
            return self.timestamp > rhs.timestamp
        else:
            print "Event .__le__ type not implemented : {0}".format(type(rhs))
            return NotImplemented
