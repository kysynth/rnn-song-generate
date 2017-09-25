from datasetpreparor import *
from midimeloconverter import *
from meloprocessor import *
from melotensorconverter import *
from rnnmusicmodel import Model
import meloclass

from mido import MidiFile
from os import walk
from os import path
import argparse
import datetime
import math
import pickle
import tensorflow as tf

DATE = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
DEFAULT_DATASET_NAME = '%s-dataset.pkl' % DATE
DEFAULT_MODEL_PREFIX = '%s-model' % DATE
 
def main():
    """ The main part of the program that will prepare dataset,
    prepare dataset then train model, or compose new music.
    """
    args = get_args()
    tools = get_tools(args)
 
    if args.prepare: # Prepare
        prepare_dataset(args, tools)
    elif not args.compose: # Train model
        if args.train_dataset_name == None: 
            training_set = prepare_dataset(args, tools)
        else:
            training_set = load_dataset(args)
        train_model(args, training_set)  
    else: # Compose song
        compose_music(args, tools)

def get_args():
    """ Add arguments and return parsed arguments.
    """
    # TODO: add --verbose
    # TODO: use threshold of loss instead of epoch  
    # TODO: add --train_restore to restore from previous model
    # TODO: need better way to store model and parameter
    
    parser = argparse.ArgumentParser()

    # What do you want the main to do
    parser.add_argument('--prepare',action='store_true',help='If this is True, this program will simply prepare dataset from the Midi folder and save it to a pickle file, without training.')
    parser.add_argument('--prepare_dataset_name',type=str,default=DEFAULT_DATASET_NAME,help='Name of the pickle of the dataset that will be saved to the disk.')
    parser.add_argument('--compose',action='store_true',help='The model, specified by \'--compose_model\', will compose a song based on Midi file specified by \'--start_with\' if this is True. Otherwise, a new model will be trained.')
    parser.add_argument('--compose_model_name',type=str,help='File path of the model that will be used to compose a song.')
    parser.add_argument('--compose_start_with',type=str,help='File path of the Midi file that the model will start with when it is composing')
    parser.add_argument('--compose_num_bars',type=int,default=4,help='Number of bars that RNN will predict during one prediction.')
    parser.add_argument('--compose_song_name',type=str,help='Name of the song composed by the model.')
    parser.add_argument('--compose_total_bars',type=int,default=32,help='Number of bars that RNN will compose.')
    
    # How to train the model
    parser.add_argument('--train_model_prefix',type=str,default=DEFAULT_MODEL_PREFIX,help='Model being trained will be saved to this path.')
    parser.add_argument('--train_dataset_name',type=str,help='The dataset pickle the model will use.')
    parser.add_argument('--train_num_neurons',type=int,default=512,help='Number of neurons this single-layer RNN will use.')
    parser.add_argument('--train_learning_rate',type=float,default=0.001,help='Learning rate that RNN will use during training.')
    parser.add_argument('--train_epoch',type=int,default=10000,help='Number of epochs that the model will be trained.')
    
    # How to prepare dataset from Midi files
    parser.add_argument('--train_num_bars',type=int,default=4,help='Number of bars that RNN will look at during one training.')
    parser.add_argument('--prepare_midi_folder',type=str,default='midi',help='All Midi files inside this folder will be used to prepare the dataset.')
    parser.add_argument('--prepare_subsongs_ratio',type=int,default=2,help='Number of subsongs that will be extracted from each Midi file will be prepare_subsongs_ratio * song-length(unit:timestep) // train_rnn_timesteps.')
    parser.add_argument('--train_mini_batch_size',type=int,default=64,help='Number of training examples in a single mini-batch.')
    parser.add_argument('--prepare_seed',type=int,help='Random seed that will be used to randomly select starting positions of songs when generating subsongs.')
    parser.add_argument('--prepare_save_dataset',type=str,help='Save the dataset obtained from Midi files into the path specified by this option.')
    
    # How you want the melody extracted or normalization happened
    parser.add_argument('--prepare_melody_bound_mode',nargs='+',default=[MeloProcessor.MelodyBoundMode.NONE],help='The mode of extracting melody. ')
    parser.add_argument('--prepare_melody_mono',action='store_true',help='If this is True, only highest node will be extracted from Midi file to make the Melo have a monotone version of Midi file.')
    parser.add_argument('--prepare_normalize',action='store_true',help='If this is turned on, this program will try to transpose all Midi files from the folder to as close as A minor/C major by minimizing number of black keys, and centralize the song so that the lowest note will be not lower than the note specified by \'----normalize_lowest\'.')
    parser.add_argument('--prepare_normalize_lowest',type=int,default=48,help='The lowest possible note that normalized songs will have.')
    parser.add_argument('--prepare_normalize_range',type=int,default=36,help='Maximum range of normalized songs. Any note in normalized songs with pitch higher than normalize_lowest + normalize_range will be clipped.')
    parser.add_argument('--prepare_resolution',type=int,default=4,help='Maximum resolution of Melo. The smallest time unit of Melo will be (resolution*4)-th note. Example: if resolution = 1, then Melo can only represent quarter note. If resolution = 2, then Melo can represent eighth note. If you want to support triplet, you should set it to an integer that can be divided by 3.')
    
    # How you want midi represented and with what precision
    parser.add_argument('--tensor_mode',type=str,default=MeloTensorConverter.TensorMode.HOLD_ON,help='The way how tensor represents Melo. \'on_off\' means unless a note_on event or note_off occur at this time step, the value will be 0. \'hold_on\' means when a note is pressed for the first time, 1 will be set for that pitches\' first position, otherwise 0. If a note that was pressed before will still be played, 1 will be set for that note\'s second position, otherwise 0.')
        
    args = parser.parse_args()
    
    # TODO: it assumes 4/4 only, maybe support 3/4 as well
    args.train_rnn_timesteps = 4 * args.train_num_bars * args.prepare_resolution
    args.compose_rnn_timesteps = 4 * args.compose_num_bars * args.prepare_resolution
    args.compose_total_steps = 4 * args.compose_total_bars * args.prepare_resolution
    
    return args    

def get_tools(args):
    """ Tools that are used across methods in main.py
    """
    class Tools: pass
    tools = Tools()
    tools.midi_melo_converter = MidiMeloConverter(args)
    tools.melo_processor = MeloProcessor(args)
    tools.melo_tensor_converter = MeloTensorConverter(args)
    tools.dataset_preparor = DatasetPreparor(args) 
    return tools    
  
def prepare_dataset(args, tools):
    """ Prepare the dataset by converting Midi files from the Midi folder
    to a class I defined as Melo, then convert Melo to matrix (tensor).
    This dataset will be used to train the RNN model.
    """
    melos = []
    for (dirpath, dirnames, filenames) in walk(args.prepare_midi_folder):
        for filename in filenames:
            true_name = dirpath+"\\"+filename
            print("Converting", true_name, "into melo...")
            melo = _midi_to_processed_milo(MidiFile(true_name), tools)
            print("Done.")
            melos.append(melo)

    print("Preparing training_set...")
    training_set = tools.dataset_preparor.get_dataset(melos)
    
    print("Dumping training_set...")
    pickle.dump(training_set, open(args.prepare_dataset_name, "wb"))   
    print("Done.")
    
    if not args.prepare:
        return training_set

def load_dataset(args):
    """ Load dataset that have been dumped before by prepare_dataset
    """
    training_set = pickle.load(open(args.train_dataset_name,'rb'))
    return training_set
        
def train_model(args, training_set):
    """ Train the RNN model from training_set.
    """
    model_path_prefix = path.join('.','model',args.train_model_prefix)
    rnn_model = Model(args)
    rnn_model.construct_graph()
    
    print("Trying to run session...")
    with tf.Session() as sess:
        print("Trying to initialize...")
        rnn_model.init.run()
        num_epoch = args.train_epoch
        for epoch in range(num_epoch):
            for mini_batch in training_set:
                ops, feed_dict = rnn_model.get_train_ops(mini_batch)
                optimizing_op, loss = ops
                sess.run(optimizing_op,feed_dict=feed_dict)
            
            # This loss is the loss of last batch, not the average loss acorss all batches
            loss_value = loss.eval(feed_dict=feed_dict)
            
            if epoch % 1 == 0:
                print(epoch,"Loss:",loss_value)
            if epoch % 100 == 0:
                save_path = rnn_model.saver.save(sess, "%s-%i" % (model_path_prefix, epoch))
        
        save_path = rnn_model.saver.save(sess, "%s-%i" % (model_path_prefix, num_epoch-1))
        print("Model saved in file: %s" % save_path)
       
def compose_music(args, tools):
    """ Compose a new song by using a previously-trained RNN
    model, and save it to the disk.
    """
    n_steps = args.compose_rnn_timesteps
    num_features = 2 * (args.prepare_normalize_range + 1)
    
    rnn_model = Model(args)
    rnn_model.construct_graph()
    
    with tf.Session() as sess:
        rnn_model.saver.restore(sess,r".\model\%s" % args.compose_model_name)

        starting_phrase = tools.melo_tensor_converter.melo_to_tensor(_midi_to_processed_milo(MidiFile(args.compose_start_with),tools))
        final_song = np.copy(starting_phrase[:n_steps])
        
        for iteration in range(args.compose_total_steps):
            gen_matrix = final_song[iteration:iteration+n_steps].reshape(1,n_steps,num_features)
            newest_note = sess.run(rnn_model.newest_note, feed_dict={rnn_model.inputs:gen_matrix})
            final_song = np.concatenate((final_song, newest_note))
        
        new_midi = tools.midi_melo_converter.melo_to_midi(tools.melo_tensor_converter.tensor_to_melo(final_song))

        _save_midi(new_midi,args.compose_song_name)

def _save_midi(midifile, filename):
    """ Save midi.
    """
    midifile.save(filename + ".mid")        
        
def _midi_to_processed_milo(midifile, tools):
    """ Convert Midi to Melo that has been normalized (transposed).
    """
    t_melo = tools.midi_melo_converter.midi_to_melo(midifile)
    t_melo = tools.melo_processor.raw_melo_to_melody_melo(t_melo)
    t_melo = tools.melo_processor.normalize_melo(t_melo)
    return t_melo
    
if __name__ == "__main__":
    main()