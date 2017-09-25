
from melotensorconverter import MeloTensorConverter

import math
import random
import numpy as np

import melotensorconverter

class DatasetPreparor:
    """ Class that prepares dataset of tensors from
    a list of Melo objects.
    """
    
    class MiniBatch:
        """ Container for inputs for RNN and corresponding
        outputs.
        
        shape of inputs: [mini_batch_size, time_steps, num_features]
        shape of outputs: [mini_batch_size, time_steps, num_features]
        
        outputs is simply a one-unit-time-later version of inputs.
        """
        def __init__(self):
            self.inputs = None
            self.targets = None
    
    def __init__(self, args):
        """ Following arg will be used:
        args.train_rnn_timesteps
        args.prepare_subsongs_ratio
        args.train_mini_batch_size
        args.prepare_seed
        """
        self.args = args
        self.melo_tensor_converter = MeloTensorConverter(args)
        
    def get_dataset(self, melos):
        """ Create dataset from list of Melos
        """
        if self.args.prepare_seed != None:
            random.seed(args.seed)
            
        all_subsongs = []
        for i, curr_melo in enumerate(melos):
            print("Converting song %i" % i)
            song_end_pos = curr_melo.tracks[0].notes[-1].time            
            if song_end_pos < self.args.train_rnn_timesteps:
                sub_song = self.melo_tensor_converter.melo_to_tensor(curr_melo)
                all_subsongs.append(sub_song)
                continue
            else:
                max_start_position = song_end_pos - self.args.train_rnn_timesteps
                print("Converting melo %i to tensor..." % i)
                whole_tensor = self.melo_tensor_converter.melo_to_tensor(curr_melo)
                num_sub_songs = int(math.ceil(self.args.prepare_subsongs_ratio * (song_end_pos+1) / self.args.train_rnn_timesteps))
                print("Tensor has been obtained for melo %i, dividing them..." % i)
                print(num_sub_songs)
                for i in range(num_sub_songs):
                    start_position = random.randint(0,max_start_position)
                    sub_song_tensor_input = whole_tensor[start_position:start_position+self.args.train_rnn_timesteps]
                    sub_song_tensor_output = whole_tensor[start_position+1:start_position+1+self.args.train_rnn_timesteps]
                    all_subsongs.append((sub_song_tensor_input,sub_song_tensor_output))
                        

        print("Shuffling subsongs...")
        random.shuffle(all_subsongs)
        
        print(len(all_subsongs))
        
        training_subsongs = all_subsongs
        
        print("Converting songs to MiniBatch...")
        training_set = self._create_mini_batch(training_subsongs)
        
        print("Dataset prepared.")
        print(len(training_set))
        return training_set
        
    def _create_mini_batch(self, subsongs):
        """ Create MiniBatch from the list of subsongs.
        Args:
            subsongs (list of tuple: (input_tensor, target_tensor))
        Output:
            mini_batches (list of MiniBatch)
        """
        # Get matrix with shape [num_instances, time_steps, num_features]
        input_matrix = np.asarray([input_tensor for input_tensor, target_tensor in subsongs])
        target_matrix = np.asarray([target_tensor for input_tensor, target_tensor in subsongs])
        
        # Divide them into list of MiniBatch 
        num_mini_batch = int(math.ceil(len(input_matrix) / self.args.train_mini_batch_size))
        mini_batches = []
        for batch_i in range(num_mini_batch):
            mini_batch = DatasetPreparor.MiniBatch()
            start_i = batch_i * self.args.train_mini_batch_size
            end_i = (batch_i + 1) * self.args.train_mini_batch_size
            if end_i > len(input_matrix):
                end_i = len(input_matrix)
            mini_batch.inputs = input_matrix[start_i:end_i]
            mini_batch.targets = target_matrix[start_i:end_i]
            mini_batches.append(mini_batch)
        return mini_batches
            