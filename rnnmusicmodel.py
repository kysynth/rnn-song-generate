import tensorflow as tf
import numpy as np

class Model:
    """ Class that performs construction phase of 
    the model. Execution phase is hanlded by main.py.
    """
    
    def __init__(self, args):
        """ Following arg will be used:
        args.rnn_max_step # TODOL chain this to train_max_step
        args.normalize_range
        args.rnn_num_neurons
        args.rnn_num_layers   # TODO: Not implemented yet
        args.rnn_learning_rate
        args.compose
        """
        self.args = args
        self.inputs = None
        self.targets = None
        self.cell = None
        self.rnn_outputs = None
        self.logits = None
        self.cross_entropy = None
        self.loss = None
        self.optimizer = None
        self.optimize = None
        self.init = None
        self.saver = None
        
    def construct_graph(self):
        """ Construct the RNN model.
        """
        
        num_features = 2 * (self.args.prepare_normalize_range + 1)
    
        with tf.name_scope('placeholder'):
            if self.args.compose:
                self.inputs = tf.placeholder(tf.float32,[None,self.args.compose_rnn_timesteps,
                            num_features],name="inputs")               
                self.targets = tf.placeholder(tf.float32,[None,self.args.compose_rnn_timesteps,
                            num_features],name="targets")
            else:
                self.inputs = tf.placeholder(tf.float32,[None,self.args.train_rnn_timesteps,
                            num_features],name="inputs")               
                self.targets = tf.placeholder(tf.float32,[None,self.args.train_rnn_timesteps,
                            num_features],name="targets") 
        with tf.name_scope('rnn'):
            self.cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.args.train_num_neurons)
            self.rnn_outputs, _ = tf.nn.dynamic_rnn(self.cell, self.inputs, dtype=tf.float32)
            # Note that activation_fn has to be None to produce logits
            self.logits = tf.contrib.layers.fully_connected(self.rnn_outputs,num_outputs=2 * (self.args.prepare_normalize_range + 1),activation_fn=None)
        
        with tf.name_scope('loss'):
            # Use sigmoid instead of softmax because each class is not mutually exclusive
            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.targets)
            self.loss = tf.reduce_mean(self.cross_entropy)
            
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.train_learning_rate)
            self.optimize = self.optimizer.minimize(self.loss)
        
        with tf.name_scope('compose'):
            # TODO: maybe should allow a threshold, such as > 0.4 then round up, can be achieved by
            # +0.1 element-wise for example
            self.newest_note = tf.round(tf.sigmoid(tf.reshape(self.logits[0][-1],(1,-1))))
        
        with tf.name_scope('misc'): # Miscellaneous
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
    
    def get_train_ops(self, mini_batch):
        """ Return operations that execution phase
        needs to actually train the model.
        feed_dict is also needed so that placeholder can be filled.
        """
        ops = ()
        feed_dict = {}
        feed_dict[self.inputs] = mini_batch.inputs
        feed_dict[self.targets] = mini_batch.targets
        ops += (self.optimize, self.loss)
        return ops, feed_dict
            