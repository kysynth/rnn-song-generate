import meloclass

from mido import Message, MidiFile, MidiTrack, MetaMessage

DRUM_CHANNEL = 10

class MidiMeloConverter:
    """ Class that is responsible for converting 
    midi to melo and melo to midi.
    """
    
    def __init__(self, args):
        """ Following arg will be used:
        args.resolution
        """
        self.args = args
        
    def midi_to_melo(self, midifile):
        """ Convert Midi to Melo.
        Args:
            midifile (MidiFile)
        Output:
            output_melo (Melo)
        """
        output_melo = meloclass.Melo()
        
        # Meaning of resolution: (1/(4*resolution))-th note.
        # For example, if resolution = 1, the smallest unit of Melo
        # will be quarter note. If resolution = 4, the smallest unit of 
        # Melo will be 16-th note.
        ticks_per_smallest_unit = midifile.ticks_per_beat // self.args.prepare_resolution  
        
        for midi_track in midifile.tracks:
            abs_time = 0
            melo_track = meloclass.Track()
            for msg in midi_track:
                abs_time += msg.time # It is necessary to add MetaMessage's time too even if it is not added to Melo
                if msg.type == "time_signature":
                    output_melo.ts_numerator = msg.numerator
                    output_melo.ts_denominator = msg.denominator
                if not msg.is_meta and msg.channel + 1 == DRUM_CHANNEL: # ignore drums. Add 1 because mido's channel range starts from 0 while Midi standard starts from 1.
                    # Notice that because meta message has no attribute "channel"
                    # The short-circuit evaluation is needed to prevent AttributeError
                    continue  
                if msg.type == "note_on" or msg.type == "note_off":
                    melo_track.notes.append(meloclass.NoteEvent(str(msg.type), msg.note, abs_time//ticks_per_smallest_unit))
            output_melo.tracks.append(melo_track)    
        return output_melo        
                
    def melo_to_midi(self, input_melo):
        """ Convert Melo to Midi.
        Melo need to be sorted by time within each track,
        or negative time will occur.
        Melo can have multiple tracks though.
        
        Args:
            input_melo (Melo)
        Output:
            midifile (MidiFile)
        """
        midifile = MidiFile()
        ticks_per_beat = midifile.ticks_per_beat
        rel_melo = self._convert_melo_to_relative_time(input_melo)
        
        for melo_track in rel_melo.tracks:
            midi_track = MidiTrack()
            midi_track.append(Message('program_change', program=0, time=0))
            for note in melo_track.notes:
                msg = Message(note.type, note=note.pitch, velocity=127, time=note.time * ticks_per_beat // self.args.prepare_resolution)
                midi_track.append(msg)
            midifile.tracks.append(midi_track)
        return midifile
            
    def _convert_melo_to_relative_time(self, abs_melo):
        """ Normally, Melo only uses absolute time.
        However this is inconvenient to convert to MidiFile
        that uses relative time. This function changes all the 
        NoteEvent's time to relative time (but still use smallest
        unit defined by resolution instead of tick as unit of time)
        
        NOTE that a Melo that uses relative time instead of 
        absolute time should never ever appear outside midimeloconverter!
        Args:
            abs_melo (Melo)
        Output:
            rel_melo (Melo) DO NOT USE THIS MELO OUTSIDE THIS CLASS.
        """
        rel_melo = meloclass.Melo()
        
        for abs_melo_track in abs_melo.tracks:
            prev_time = 0
            rel_melo_track = meloclass.Track()
            for abs_note in abs_melo_track.notes:
                rel_melo_track.notes.append(meloclass.NoteEvent(abs_note.type,abs_note.pitch,abs_note.time-prev_time))
                prev_time = abs_note.time
            rel_melo.tracks.append(rel_melo_track)
        return rel_melo
    
    
        
        