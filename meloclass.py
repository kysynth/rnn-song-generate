class NoteEvent:
    """ Almost identical to standard Midi Message but with "note_on"
    and "note_off" mode only, and using absolute time instead
    of relative time in Midi file. 
    Also, NoteEvent uses args.resolution as the unit of time, 
    while MidiFile uses ticks as the unit of time.
    """
    def __init__(self, type, pitch, time):
        self.type = type
        self.pitch = int(pitch)
        self.time = int(time)

class Track:
    """ A list of NoteEvent
    """
    def __init__(self):
        self.notes = []
        
class Melo:
    """ A list of Track. This class is a lot easier to handle than
    vanilla MidiFile because Melo uses absolute time and has only
    note information.
    """
    def __init__(self):
        self.tracks = []
        self.ts_numerator = 4 # Time signature numerator
        self.ts_denominator = 4 # Time signature denominator