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
