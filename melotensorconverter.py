import meloclass
from meloprocessor import MeloProcessor

import numpy as np

class MeloTensorConverter:
    """ Class that is responsible for converting
    melo to tensor and tensor to melo.
    """
    
    def __init__(self, args):
        """ Following arg will be used:
        args.tensor_mode
        args.prepare_normalize_lowest
        args.prepare_normalize_range
        """
        self.args = args
        self.melo_processor = MeloProcessor(args) # needed for sorting Melo
    
    class TensorMode:
        """ Container for different modes of converting
        melo to tensor.
        Tensor of one Melo will always have the shape of (num_of_time, num_features)
        The only thing that TensorMode affect is the input features.
        
        ON_OFF:
            Note_on event will be represeted by a value of 1 at (2*relative_pitch)-th features.
            Note_off event will be represented by a value of 1 at (2*relative_pitch+1)-th features.
            All other feature values are 0 if note_on or note_off do not occur at that time step.
        HOLD_ON:
            Note_on event will be represented by a value of 1 at (2*relative_pitch)
            If (2*relative_pitch+1)-th feature is 1, it means that relative_pitch was played/held at last time
            and for this time step, this relative_pitch will still hold, until it is released,
            which is represented by (2*relative_pitch+1)-th feature being 0.
        """
        ON_OFF = 'on_off' 
        HOLD_ON = 'hold_on'
        
                
        @staticmethod
        def get_modes():
            return [
                MeloTensorConverter.TensorMode.ON_OFF,
                MeloTensorConverter.TensorMode.HOLD_ON
            ]
    
    def melo_to_tensor(self, sorted_melo):
        """ Convert one Melo to one tensor of shape
        (num_of_time, num_of_features)
        
        Melo has to be sorted by time or this function
        will not work as intended.
        
        Args:
            sorted_melo (Melo)
        Output:
            output_tensor (np.ndarray) - shape: num_of_time_steps, num_features
        """
        
        if len(sorted_melo.tracks) != 1:
            raise ValueError("Do not call normalize_melo on Melo that does not have exactly one track. This Melo has %i tracks." % len(bound_melo.tracks) )
         
        output_tensor = None
        if self.args.tensor_mode == MeloTensorConverter.TensorMode.ON_OFF:
            output_tensor = self._on_off_tensor(sorted_melo)
        elif self.args.tensor_mode == MeloTensorConverter.TensorMode.HOLD_ON:
            output_tensor = self._hold_on_tensor(sorted_melo)
        else:
            raise ValueError("Invalid tensor_mode: %s. Please choose from %s" % (self.args.tensor_mode,str(MeloTensorConverter.TensorMode.get_modes())))

        return output_tensor    
        
    def _on_off_tensor(self, sorted_melo):
        """ Convert one Melo to on-off mode tensor.
        
        Melo has to be sorted by time or this function 
        will not work as intended.
        """
        
        melo_length = sorted_melo.tracks[0].notes[-1].time
        num_features = 2 * (self.args.prepare_normalize_range + 1)
        output_tensor = np.zeros((melo_length+1,num_features))
        
        for note in sorted_melo.tracks[0].notes:
            if note.type == "note_on":
                output_tensor[note.time][2*(note.pitch-self.args.prepare_normalize_lowest)] = 1.0
            elif note.type == "note_off":
                output_tensor[note.time][2*(note.pitch-self.args.prepare_normalize_lowest)+1] = 1.0
        
        return output_tensor
    
    def _hold_on_tensor(self, sorted_melo):
        """ Convert one Melo to on-off mode tensor.
        
        Melo has to be sorted by time or this function 
        will not work as intended.
        """
        
        melo_length = sorted_melo.tracks[0].notes[-1].time 
        num_features = 2 * (self.args.prepare_normalize_range + 1)
        output_tensor = np.zeros((melo_length+1,num_features))
        
        for i, on_note in enumerate(sorted_melo.tracks[0].notes):
            if on_note.type == "note_on":
                output_tensor[on_note.time][2*(on_note.pitch-self.args.prepare_normalize_lowest)] = 1.0
                
                # Find when it no longer holds
                off_note_found = False
                off_time = None
                j = i
                
                while not off_note_found and j < len(sorted_melo.tracks[0].notes):
                    if sorted_melo.tracks[0].notes[j].type == "note_off" \
                       and sorted_melo.tracks[0].notes[j].pitch == on_note.pitch:
                        off_time = sorted_melo.tracks[0].notes[j].time
                        off_note_found = True
                    j += 1
                
                if j == len(sorted_melo.tracks[0].notes):
                    j = len(sorted_melo.tracks[0].notes) - 1
                    off_time = j
                
                for hold_time in range(on_note.time+1, off_time):
                    if hold_time < output_tensor.shape[0]:
                        output_tensor[hold_time][2*(on_note.pitch-self.args.prepare_normalize_lowest)+1] = 1.0
        return output_tensor
        
    def tensor_to_melo(self, tensor_01):
        """ Convert one-song tensor to one Melo.
        Tensor has to have only 0 or 1 only. 
        Melo returned is sorted.
        
        Args:
            tensor_01 (np.ndarray) - shape (num_of_time, num_of_features)
        Output:
            output_melo (Melo)
        """         
        output_melo = None
        
        if self.args.tensor_mode == MeloTensorConverter.TensorMode.ON_OFF:
            output_melo = self._on_off_melo(tensor_01)
        elif self.args.tensor_mode == MeloTensorConverter.TensorMode.HOLD_ON:
            output_melo = self._hold_on_melo(tensor_01)
        else:
            raise ValueError("Invalid tensor_mode: %s. Please choose from %s" % (self.args.tensor_mode,str(MeloTensorConverter.TensorMode.get_modes())))

        return output_melo
    
    def _on_off_melo(self, tensor_01):
        """ Convert one-song tensor to one Melo.
        Tensor has to have only 0 or 1 only.
        Melo returned is sorted.
        
        Args:
            tensor_01 (np.ndarray) - shape (num_of_time, num_of_features)
        Output:
            output_melo (Melo)
        """
        output_melo = meloclass.Melo()
        output_track = meloclass.Track()
        output_melo.tracks.append(output_track)
        
        for time in range(tensor_01.shape[0]):
            for pitch in range(tensor_01.shape[1]):
                if tensor_01[time][pitch] == 1:
                    if pitch % 2 == 0: # Note on event
                        output_track.notes.append(meloclass.NoteEvent("note_on",pitch/2+self.args.prepare_normalize_lowest,time))
                    else: # Note off event
                        output_track.notes.append(meloclass.NoteEvent("note_off",(pitch-1)/2+self.args.prepare_normalize_lowest,time))
                elif tensor_01[time][pitch] != 0:
                    print(tensor_01[time][pitch])
                    raise ValueError("The tensor of tensor_of_melo cannot contain value other than 0 or 1.")
        
        return self.melo_processor.sort_melo(output_melo)
    
    def _hold_on_melo(self, tensor_01):
        """ Convert one-song tensor to one Melo.
        Tensor has to ahve only 0 or 1 only.
        Melo returned is sorted.
        
        Args:
            tensor_01 (np.ndarray) - shape (num_of_time, num_of_features)
        Output:
            output_melo (Melo)
        
        """
        output_melo = meloclass.Melo()
        output_track = meloclass.Track()
        output_melo.tracks.append(output_track)
        
        for time in range(tensor_01.shape[0]):
            for pitch in range(tensor_01.shape[1]):
                if pitch % 2 == 0 and tensor_01[time][pitch] == 1:
                    on_note = meloclass.NoteEvent("note_on",pitch/2+self.args.prepare_normalize_lowest,time)
                    output_track.notes.append(on_note)                    
                    # Find note_off time
                    off_note_found = False
                    off_time = None
                    j = time + 1
                    while not off_note_found and j < tensor_01.shape[0]:
                        if tensor_01[j][pitch+1] == 0:
                            off_time = j
                            off_note_found = True
                        j += 1
                    # when examining a subsong, it is possile that 
                    # an off_note cannot be found within range
                    # in that case, simply set j to be the last time
                    if j == tensor_01.shape[0]:
                        j = tensor_01.shape[0] - 1
                        off_time = j
                    
                    off_note = meloclass.NoteEvent("note_off",pitch//2+self.args.prepare_normalize_lowest,off_time)
                    output_track.notes.append(off_note)
                elif tensor_01[time][pitch] != 0 and tensor_01[time][pitch] != 1:
                    raise ValueError("The tensor of tensor_of_melo cannot contain value other than 0 or 1.")
        return self.melo_processor.sort_melo(output_melo)
    