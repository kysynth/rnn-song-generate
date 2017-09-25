import meloclass

class MeloProcessor:
    """Process Melo
    """
    
    def __init__(self, args):
        """ Following arg will be used:
        args.prepare_melody_bound_mode
        args.prepare_melody_mono
        args.prepare_normalize_lowest
        args.prepare_normalize_range
        """
        self.args = args
        
    class MelodyBoundMode:
        """ Container for different modes of extracting
        melody from raw midi.
        """
        NONE = 'none' # No notes will be removed
        NOTE_RANGE_KILL_TRACK = 'note_range_kill_track' # Exclude tracks that have notes outside of the note_range
        NOTE_RANGE_KILL_NOTE = 'note_range_kill_note'# Exclude notes outside of the note_range
        
        @staticmethod
        def get_modes():
            return [
                MeloProcessor.MelodyBoundMode.NONE,
                MeloProcessor.MelodyBoundMode.NOTE_RANGE_KILL_TRACK,
                MeloProcessor.MelodyBoundMode.NOTE_RANGE_KILL_NOTE
            ]
        
    def raw_melo_to_melody_melo(self, raw_melo):
        """ Convert multi-track melo to one-track melo.
        Args:
            raw_melo (Melo)
        Return:
            (Melo)
        """

        mode = self.args.prepare_melody_bound_mode[0]
        
        bound_melo = None
        
        if mode == MeloProcessor.MelodyBoundMode.NONE:
            bound_melo = self._no_bound_combine(raw_melo)
        elif mode == MeloProcessor.MelodyBoundMode.NOTE_RANGE_KILL_TRACK:
            bound_melo = self._kill_track_combine(raw_melo)
        elif mode == MeloProcessor.MelodyBoundMode.NOTE_RANGE_KILL_NOTE:
            bound_melo = self._kill_note_combine(raw_melo)
        else:
            raise ValueError("Invalid melody_bound_mode: %s. Please choose from %s" % (mode,str(MidiProcessor.MelodyBoundMode.get_modes())))
        
        if self.args.prepare_melody_mono:
            return self._turn_to_mono(bound_melo)
        return bound_melo
    
    def _no_bound_combine(self, raw_melo):
        """ Convert multi-track melo to one-track melo sorted by time
        without removing anything.
        Args:
            raw_melo (Melo)
        Return:
            processed_melo (Melo)
        """
        processed_melo = meloclass.Melo()
        processed_track = meloclass.Track()
        processed_melo.tracks.append(processed_track)
        for raw_track in raw_melo.tracks:
            for note in raw_track.notes:
                processed_track.notes.append(note)
        processed_track.notes.sort(key=lambda x:x.time)
        return processed_melo
    
    def _kill_track_combine(self, raw_melo):
        """ Convert multi-track melo to one-track melo sorted by time.
        However, if a track has at least one note that goes 
        out of the range defined by args.melody_bound_mode,
        that track will be ignored.
        Args:
            raw_melo (Melo)
        Return:
            processed_melo (Melo)
        """
        processed_melo = meloclass.Melo()
        for raw_track in raw_melo.tracks:
            processed_track = meloclass.Track()
            track_in_range = True
            for note in raw_track.notes:
                if not self._is_in_range(note):
                    track_in_range = False
                processed_track.notes.append(note)
            if track_in_range:
                processed_melo.tracks.append(processed_track)
        
        return self._no_bound_combine(processed_melo)
    
    def _kill_note_combine(self, raw_melo):
        """ Convert multi-track melo to one-track melo sorted by time.
        However, if a note goes out of the range 
        defined by args.melody_bound_mode, that note will be ignored.
        Args:
            raw_melo (Melo)
        Return:
            processed_melo (Melo)
        """
        processed_melo = meloclass.Melo()
        processed_track = meloclass.Track()
        processed_melo.tracks.append(processed_track)
        for raw_track in raw_melo.tracks:      
            for note in raw_track.notes:
                if self._is_in_range(note):
                    processed_track.notes.append(note)
        processed_track.notes.sort(key=lambda x:x.time)
        return processed_melo

    def _is_in_range(self, note_event):
        """ Check if NoteEvent is in the range of args.melody_bound_mode.
        Args:
            note_event (NoteEvent)
        Return:
            (bool)
        """
        if len(self.args.prepare_melody_bound_mode) != 3:
            raise ValueError("Wrong number of arguments for melody_bound_mode. You should provide 3 arguments: mode, lower_bound, upper_bound")
        return self.args.prepare_melody_bound_mode[1] < note_event.pitch < self.args.prepare_melody_bound_mode[2]
        
    def _turn_to_mono(self, bound_melo):
        """ Turn the Melo to be monotone. 
        bound_melo has to be sorted by time at this time or 
        this code will fail.
        Currently it will simply choose the highest pitch when 
        several notes have same note_on time. 
        Args:
            bound_melo (Melo)
        Return:
            (Melo)
        """
        
        # Construct a list of note_on to delete
        del_on_list = [] # note_on Note that should be removed. List of (time, pitch)
        prev_time = 0
        same_time_notes = []
        for note in bound_melo.tracks[0].notes:
            if note.type == "note_on":
                if note.time > prev_time:
                    # Some notes have to be deleted before
                    # going on to the next time if more than
                    # one note happened at the last time
                    if len(same_time_notes) > 1:
                        highest_pitch = max(same_time_notes)
                        same_time_notes.remove(highest_pitch)
                        for pitch in same_time_notes:
                            del_note_on = (prev_time, pitch)
                            del_on_list.append(del_note_on)
                    # Notes to be deleted have been selected,
                    # go on to the next time
                    prev_time = note.time
                    same_time_notes = [note.pitch]
                    continue
                elif note.time == prev_time: 
                    # Same note_on time so some notes have to be cut
                    same_time_notes.append(note.pitch)
        
        # Last time is not handled yet
        if len(same_time_notes) > 1:
            highest_pitch = max(same_time_notes)
            same_time_notes.remove(highest_pitch)
            for pitch in same_time_notes:
                del_note_on = (prev_time, pitch)
                del_on_list.append(del_note_on)
        
        # Now find corresponding note_off to delete for each note_on in del_on_list
        del_off_list = [] # note_off Note that should be removed. List of (time, pitch)
        for time, pitch in del_on_list:
            off_note_found = False
            for note in bound_melo.tracks[0].notes:
                if note.type == "note_off":
                    if note.time > time and note.pitch == pitch and not off_note_found: 
                        # closet note_off found
                        del_note_off = (note.time, pitch)
                        del_off_list.append(del_note_off)
                        off_note_found = True
        
        # Delete the note_on
        for del_note_on_time, del_note_on_pitch in del_on_list:
            for note in bound_melo.tracks[0].notes:
                if note.type == "note_on":
                    if note.time == del_note_on_time \
                       and note.pitch == del_note_on_pitch:
                        bound_melo.tracks[0].notes.remove(note)
                       
        # Delete the note off
        for del_note_off_time, del_note_off_pitch in del_off_list:
            for note in bound_melo.tracks[0].notes:
                if note.type == "note_off":
                    if note.time == del_note_off_time \
                       and note.pitch == del_note_off_pitch:
                        bound_melo.tracks[0].notes.remove(note)
        
        return bound_melo
        
    def normalize_melo(self, bound_melo):
        """ Transpose one-track Melo so that its lowest note
        will be around args.normalize_lowest. The transpose
        will try to make the new Melo as close to C major or 
        A minor as possible by minimizing number of black keys.
        Any note higher than args.normalize_lowest + args.normalize_range
        will be removed (not to extract a melody, which was done
        by raw_melo_to_melody_melo() already, but to restrict
        the number of features of input of neural network)
        Args:
            bound_melo (Melo)
        Return:
            normalized_melo (Melo)
        
       
        args.normalize_lowest
        args.normalize_range
        """
        # TODO: This is not a good transposing algorithm
        # Using advanced key detection algorithm can 
        # help to perform better normalization.
        
        if len(bound_melo.tracks) != 1:
            raise ValueError("Do not call normalize_melo on Melo that does not have exactly one track. This Melo has %i tracks." % len(bound_melo.tracks) )
        
        original_pitch_set = set([n.pitch for n in bound_melo.tracks[0].notes])
        start_trans = min(original_pitch_set) - self.args.prepare_normalize_lowest
        start_pitch_set = set([p - start_trans for p in original_pitch_set])
        min_black_keys = self._count_black_keys(start_pitch_set)
        min_pitch_set = start_pitch_set
        
        for i in range(1,12):
            pitch_set = set([p + i for p in start_pitch_set])
            black_keys = self._count_black_keys(pitch_set)
            if black_keys < min_black_keys:
                min_black_keys = black_keys
                min_pitch_set = pitch_set
        final_trans = min(original_pitch_set) - min(min_pitch_set)
        
        normalized_melo = meloclass.Melo()
        normalized_track = meloclass.Track()
        normalized_melo.tracks.append(normalized_track)
        normalized_track.notes = [meloclass.NoteEvent(n.type, n.pitch-final_trans, n.time) 
                                  for n in bound_melo.tracks[0].notes 
                                  if self.args.prepare_normalize_lowest <= n.pitch-final_trans <= self.args.prepare_normalize_lowest + self.args.prepare_normalize_range]
        
        return normalized_melo
        
    def _count_black_keys(self, pitch_set):
        """ Helper function for nromalize_melo.
        Counts how many black keys are in this start_pitch_set.
        Args:
            pitch_set (Set(int)) 
        REturn:
            (int)
        """
        BLACK_KEYS = [1,3,6,8,10]
        mod_pitch_set = set([p % 12 for p in pitch_set])
        return len(mod_pitch_set.intersection(BLACK_KEYS))
    
    def sort_melo(self, unsorted_melo):
        """ Sort notes in melo with the following precedence:
        time(increasing)->type(note_off)
        """
        for track in unsorted_melo.tracks:
            # Sort by time first, then type first (note_off < note_on)
            track.notes = sorted(track.notes, key=lambda x: (x.time, x.type))
        return unsorted_melo