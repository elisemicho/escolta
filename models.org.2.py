"""Model definitions for simple speech recognition. 
Inspiration sources:
Code organisation: https://www.tensorflow.org/versions/master/tutorials/audio_recognition
Convolutions: https://www.tensorflow.org/tutorials/layers

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np


def create_model(feats2d, shapes, model_settings, model_architecture,
                                 is_training, runtime_settings=None):
    """Builds a model of the requested architecture compatible with the settings.

    There are many possible ways of deriving predictions from a spectrogram
    input, so this function provides an abstract interface for creating different
    kinds of models in a black-box way. You need to pass in a TensorFlow node as
    the 'feats2D' input, and this should output a batch of 2D features that
    describe the audio. Typically this will be derived from a spectrogram.

    The function will build the graph it needs in the current TensorFlow graph,
    and return the tensorflow output that will contain the 'logits' input to the
    softmax prediction process. If training flag is on, it will also return a
    placeholder node that can be used to control the dropout amount.

    See the implementations below for the possible model architectures that can be
    requested.

    Args:
        feats2d: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        model_architecture: String specifying which kind of model to create.
        is_training: Whether the model is going to be used for training.
        runtime_settings: Dictionary of information about the runtime.

    Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.

    Raises:
        Exception: If the architecture type isn't recognized.
    """
    if model_architecture == 'single_fc':
        return create_single_fc_model(feats2d, shapes, model_settings, is_training)
    elif model_architecture == 'simple_conv2D':
        return create_simple_conv2D_model(feats2d, shapes, model_settings, is_training)
    elif model_architecture == 'simple_conv1D':
        return create_simple_conv1D_model(feats2d, shapes, model_settings, is_training)
    elif model_architecture == 'lstm':
        return create_bidirectionnal_dynamic_rnn_model(feats2d, shapes, model_settings, is_training)
    elif model_architecture == 'cnn_lstm':
        return create_CNN_LSTM_model(feats2d, shapes, model_settings, is_training)
    elif model_architecture == 'dcnn_lstm':
        return create_DCNN_LSTM_model(feats2d, shapes, model_settings, is_training)
    elif model_architecture == '2D_dcnn_lstm':
        return create_2D_DCNN_LSTM_model(feats2d, shapes, model_settings, is_training)
    elif model_architecture == '2D_vdcnn_lstm':
        return create_2D_VDCNN_LSTM_model(feats2d, shapes, model_settings, is_training)
    elif model_architecture == 'organizers':
        return organizers_model(feats2d, shapes, model_settings, is_training)
    else:
        raise Exception('model_architecture argument "' + model_architecture +
                                        '" not recognized, should be one of "single_fc", "simple_conv2D",' +
                                        ' "simple_conv1D", or "lstm"')


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.

    Args:
        sess: TensorFlow session.
        start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)

# Fully-connected layer model does not work because expects fixed size
def create_single_fc_model(feats2d, shapes, model_settings, is_training):
    """Builds a model with a single hidden fully-connected layer.

    This is a very simple model with just one matmul and bias layer. As you'd
    expect, it doesn't produce very accurate results, but it is very fast and
    simple, so it's useful for sanity testing.

    Here's the layout of the graph:

                (feats2d)
                    v
            [MatMul]<-(weights)
                    v
            [BiasAdd]<-(bias)
                    v

    Args:
        feat2d: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.

    Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.
    """
    # Inputs
    if is_training:
        dropout_prob = model_settings['dropout_prob']  

    # Reshape into 1D
    print(feats2d.get_shape())                 
    assert len(feats2d.get_shape()) == 3          # a tuple (batch_size, seq_length, n_mel) - static shape
    #shape = feats2d.get_shape().as_list()        # a list: [batch_size, seq_length, n_mel] - static shape
    shape = tf.shape(feats2d)                     # a tensor - dynamic shape
    dim = shape[1]*shape[2]                  # dim = prod(9,2) = 18

    #dim = reduce(lambda x, y: x*y, shape[1:])

    #feats1d = tf.reshape(feats2d, [-1, dim])   
    #batch_size = tf.shape(feats2d)[0]             # a strided slice
    print(shape)
    #print(dim)
    #print(batch_size)
    feats1d = tf.reshape(feats2d, [shape[0], -1]) 

    # Get dimensions
    feat1d_size = tf.shape(feats1d)[1]         # a strided slice
    #feat1d_size = feats1d.get_shape().as_list()[1]
    num_classes = model_settings['num_classes']

    print(feats1d)
    print(feat1d_size)
    print(num_classes)
    # Weights and biases
    # weights = tf.Variable(
    #          tf.truncated_normal([feat1d_size, num_classes], stddev=0.001))
    weights = tf.get_variable(name="weights",shape=[dim, num_classes], initializer=tf.truncated_normal_initializer(), validate_shape=False)
    bias = tf.Variable(tf.zeros([num_classes]))

    # Fully-connected layer
    logits = tf.matmul(feats1d, weights) + bias
    if is_training:
        return logits, dropout_prob
    else:
        return logits


def create_simple_conv2D_model(feats2d, shapes, model_settings, is_training):
    """Builds a standard convolutional model.

                (feats2d)
                    v
            [Conv2D]
                    v
            [BiasAdd]
                    v
                [Relu]
                    v
            [MaxPool]
                    v
            [Conv2D]
                    v
            [BiasAdd]
                    v
                [Relu]
                    v
            [MaxPool]
                    v
            [Dense]
                    v

    Args:
        feats2d: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.

    Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.
    """
    if is_training:
        dropout_prob = model_settings['dropout_prob']     
    else:
        dropout_prob = 0
    # Input Layer
    shape = tf.shape(feats2d) # features are of shape [max seq length for batch, 40]
    input_layer = tf.reshape(feats2d,tf.stack([-1, shape[1], shape[2], 1])) # [batch_size, seq_length, 40, 1]

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[20, 8],
      padding="same",
      activation=tf.nn.relu)

    dropout1 = tf.layers.dropout(
      inputs=conv1, rate=dropout_prob, training=is_training)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=dropout1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[10, 4],
      padding="same",
      activation=tf.nn.relu)

    dropout2 = tf.layers.dropout(
      inputs=conv2, rate=dropout_prob, training=is_training)

    pool2 = tf.layers.max_pooling2d(inputs=dropout2, pool_size=[2, 2], strides=2) # [batch_size, pool2_shape[1], pool2_shape[2], 64]

    # in case we want to use a flat output layer from convolutions
    # pool2_flat = tf.layers.flatten(pool2)             # [batch_size, pool2_shape[1] * pool2_shape[2] * 64]
    # idem as: 
    # pool2_shape = tf.shape(pool2) 
    # pool2_flat = tf.reshape(pool2, [-1, pool2_shape[1] * pool2_shape[2] * 64]) 

    # Average of the result of convolutions over 2 axes: max sequence length in the batch and dimension of sepctrogram
    pool_sum = tf.reduce_sum(pool2,[1,2],keep_dims=True)    # [batch_size, 1, 1, 64]
    mean = pool_sum/tf.cast(shape[1] * shape[2], tf.float32)# [batch_size, 1, 1, 64]
    res1=tf.squeeze(mean, axis=[1,2])                                   # [batch_size, 64]

    # Logits Layer
    num_classes = model_settings['num_classes']
    logits = tf.layers.dense(inputs=res1, units=num_classes)

    if is_training:
        return logits, dropout_prob
    else:
        return logits

def create_simple_conv1D_model(feats2d, shapes, model_settings, is_training):
    """Builds a standard convolutional model.

                (feats2d)
                    v
            [Conv1D]
                    v
            [BiasAdd]
                    v
                [Relu]
                    v
            [MaxPool]
                    v
            [Conv1D]
                    v
            [BiasAdd]
                    v
                [Relu]
                    v
            [MaxPool]
                    v
            [Dense]
                    v

    Args:
        feats2d: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.

    Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.
    """
    if is_training:
        dropout_prob = model_settings['dropout_prob']
    else:
        dropout_prob = 0
    # Input Layer
    shape = tf.shape(feats2d) # features are of shape [max seq length for batch, 40]
    input_layer = tf.reshape(feats2d,[-1, shape[1], model_settings['feature_width']]) # [batch_size, seq_length, 40]

    # Convolutional Layer #1 (Dropout #1) and Pooling Layer #1
    conv1 = tf.layers.conv1d(
      inputs=input_layer,
      filters=model_settings['conv1_num_filters'],
      kernel_size=model_settings['conv1_kernel_size'],
      padding="same",
      activation=tf.nn.relu)

    dropout1 = tf.layers.dropout(
      inputs=conv1, rate=dropout_prob, training=is_training)

    pool1 = tf.layers.max_pooling1d(inputs=dropout1, pool_size=model_settings['pool1_pool_size'], strides=model_settings['pool1_strides']) 

    # Convolutional Layer #2 (Dropout #2) and Pooling Layer #2
    conv2 = tf.layers.conv1d(
      inputs=pool1,
      filters=model_settings['conv2_num_filters'],
      kernel_size=model_settings['conv2_kernel_size'],
      padding="same",
      activation=tf.nn.relu)

    dropout2 = tf.layers.dropout(
      inputs=conv2, rate=dropout_prob, training=is_training)

    pool2 = tf.layers.max_pooling1d(inputs=dropout2, pool_size=model_settings['pool2_pool_size'], strides=model_settings['pool2_strides']) # [batch_size, pool2_shape[1], 64]

    # in case we want to use a flat output layer from convolutions
    # pool2_flat = tf.layers.flatten(pool2)             # [batch_size, pool2_shape[1] * 64]
    # idem as: 
    # pool2_shape = tf.shape(pool2)   
    # pool2_flat = tf.reshape(pool2, [-1, pool2_shape[1] * 64]) 

    # Average of the result of convolutions over max sequence length in the batch
    pool_sum = tf.reduce_sum(pool2,1,keep_dims=True)    # [batch_size, 1, 64]
    mean = pool_sum/tf.cast(shape[1], tf.float32)       # [batch_size, 1, 64]
    res1=tf.squeeze(mean,axis=1)                        # [batch_size, 64]

    # Logits Layer
    num_classes = model_settings['num_classes']
    logits = tf.layers.dense(inputs=res1, units=num_classes)
    
    if is_training:
        return logits, dropout_prob
    else:
        return logits

def create_bidirectionnal_dynamic_rnn_model(feats2d, shapes, model_settings, is_training):
    """Builds a standard convolutional model.
    
    Here's the layout of the graph:

                (feats2d)
                    v
            [BiLSTM]<-(cell_fw, cell_bw)
                    v
            [MatMul]<-(weights)
                    v
            [BiasAdd]<-(bias)
                    v

    Args:
        feats2d: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.

    Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.
    """
    if is_training:
        dropout_prob = model_settings['dropout_prob']  
    else:
        dropout_prob = 0
    # Get dimensions
    lstm_size = model_settings['lstm_size']

    batch_size = tf.shape(feats2d)[0] 
    feats2d = tf.reshape(feats2d, shape=[batch_size,-1,model_settings['feature_width']]) # features are of shape [max seq length for batch, 40]
    seq_lengths = shapes[:,0] # all shapes are [seq_length, 40], we extract seq_length

    # seq_lengths = tf.slice(shapes, [0, 0], [batch_size, 1])
    # print(seq_lengths)

    # LSTM cells
    cell_fw = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)
    cell_bw = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)
    # ini_fw = cell_fw.zero_state(batch_size,dtype=tf.float32)
    # ini_bw = cell_bw.zero_state(batch_size,dtype=tf.float32)

    # Bi-directional RNN (+ Dropout)
    (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, feats2d, 
                                                                sequence_length=seq_lengths, 
                                                                dtype=tf.float32)

    # initial_state_fw = ini_fw, initial_state_bw = ini_bw, 
    # if state_is_tuple, state is a tuple (cell_state, memory_state)
    concat_rnn = tf.concat([state_fw[0], state_bw[0]], axis=1)

    if is_training:
        first_dropout = tf.nn.dropout(concat_rnn, dropout_prob)
    else:
        first_dropout = concat_rnn

    # Logits Layer
    num_classes = model_settings['num_classes']
    logits = tf.layers.dense(inputs=first_dropout, units=num_classes)
    
    if is_training:
        return logits, dropout_prob
    else:
        return logits

def create_CNN_LSTM_model(feats2d, shapes, model_settings, is_training):
    """Builds a standard convolutional model.

                (feats2d)
                    v
            [Conv1D]
                    v
            [BiasAdd]
                    v
                [Relu]
                    v
            [MaxPool]
                    v
            [Conv1D]
                    v
            [BiasAdd]
                    v
                [Relu]
                    v
            [MaxPool]
                    v
            [BiLSTM]<-(cell_fw, cell_bw)
                    v
                [Dense]
                    v

    Args:
        feats2d: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.

    Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.
    """
    if is_training:
        dropout_prob = model_settings['dropout_prob']
    else:
        dropout_prob = 0

    # Input Layer
    shape = tf.shape(feats2d) # features are of shape [max seq length for batch, 40]
    input_layer = tf.reshape(feats2d,[-1, shape[1], model_settings['feature_width']]) # [batch_size, seq_length, 40]

    # Convolutional Layer #1 (Dropout #1) and Pooling Layer #1
    conv1 = tf.layers.conv1d(
      inputs=input_layer,
      filters=model_settings['conv1_num_filters'],
      kernel_size=model_settings['conv1_kernel_size'],
      padding="same")

    if 'is_batch_norm' in model_settings.keys():
        if model_settings['is_batch_norm']:
            batch_n1 = tf.layers.batch_normalization(inputs=conv1,  training=is_training)
        else:
            batch_n1 = conv1
    else:
        batch_n1 = conv1

    if "activation" in model_settings.keys():
        if model_settings['activation'] == 'sigmoid':
            ac1 = tf.nn.sigmoid(batch_n1)
        else:
            ac1 = tf.nn.relu(batch_n1)
    else:
        ac1 = tf.nn.relu(batch_n1)

    dropout1 = tf.layers.dropout(
      inputs=ac1, rate=dropout_prob, training=is_training)

    pool1 = tf.layers.max_pooling1d(inputs=dropout1, pool_size=model_settings['pool1_pool_size'], strides=model_settings['pool1_strides']) 

    # Convolutional Layer #2 (Dropout #2) and Pooling Layer #2
    conv2 = tf.layers.conv1d(
      inputs=pool1,
      filters=model_settings['conv2_num_filters'],
      kernel_size=model_settings['conv2_kernel_size'],
      padding="same")

    if "is_batch_norm"in model_settings.keys():
        if model_settings['is_batch_norm']:
            batch_n2 = tf.layers.batch_normalization(inputs=conv2,  training=is_training)
        else:
            batch_n2 = conv2
    else:
        batch_n2 = conv2
        
    if "activation" in model_settings.keys():
        if model_settings['activation'] == 'sigmoid':
            ac2 = tf.nn.sigmoid(batch_n2)
        else:
            ac2 = tf.nn.relu(batch_n2)
    else:
        ac2 = tf.nn.relu(batch_n2)

    dropout2 = tf.layers.dropout(
      inputs=ac2, rate=dropout_prob, training=is_training)

    pool2 = tf.layers.max_pooling1d(inputs=dropout2, pool_size=model_settings['pool2_pool_size'], strides=model_settings['pool2_strides']) # [batch_size, pool2_shape[1], 64]

    # in case we want to use a flat output layer from convolutions
    # pool2_flat = tf.layers.flatten(pool2)             # [batch_size, pool2_shape[1] * 64]
    # idem as: 
    # pool2_shape = tf.shape(pool2)   
    # pool2_flat = tf.reshape(pool2, [-1, pool2_shape[1] * 64]) 

    # Get dimensions
    lstm_size = model_settings['lstm_size']

    # batch_size = tf.shape(feats2d)[0] 
    # feats2d = tf.reshape(feats2d, shape=[batch_size,-1,40]) # features are of shape [max seq length for batch, 40]
    # seq_lengths = shapes[:,0] # all shapes are [seq_length, 40], we extract seq_length
    # seq_lengths = tf.shape(pool2)[1]
    # seq_lengths = tf.slice(shapes, [0, 0], [batch_size, 1])
    # print(seq_lengths)

    # LSTM cells
    cell_fw = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)
    cell_bw = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)
    # ini_fw = cell_fw.zero_state(batch_size,dtype=tf.float32)
    # ini_bw = cell_bw.zero_state(batch_size,dtype=tf.float32)

    # Bi-directional RNN (+ Dropout)
    (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, pool2,  
                                                                dtype=tf.float32)

    # initial_state_fw = ini_fw, initial_state_bw = ini_bw, 
    # if state_is_tuple, state is a tuple (cell_state, memory_state)
    concat_rnn = tf.concat([state_fw[0], state_bw[0]], axis=1)

    if is_training:
        first_dropout = tf.nn.dropout(concat_rnn, dropout_prob)
    else:
        first_dropout = concat_rnn

    # Logits Layer
    num_classes = model_settings['num_classes']
    logits = tf.layers.dense(inputs=first_dropout, units=num_classes)
    
    if is_training:
        return logits, dropout_prob
    else:
        return logits

def create_LSTM_LSTM_model(feats2d, shapes, model_settings, is_training):
    """Builds a standard convolutional model.
    
    Here's the layout of the graph:

                (feats2d)
                    v
            [BiDirRNN]<-(cell_fw, cell_bw)
                    v
            [MatMul]<-(weights)
                    v
            [BiasAdd]<-(bias)
                    v

    Args:
        feats2d: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.

    Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.

    TODO: look at https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/stack_bidirectional_dynamic_rnn
                    https://github.com/google/seq2seq/blob/master/seq2seq/encoders/rnn_encoder.py
    """
    if is_training:
        dropout_prob = model_settings['dropout_prob']  
    else:
        dropout_prob = 0
    # Get dimensions
    lstm_size = model_settings['lstm_size']

    batch_size = tf.shape(feats2d)[0] 
    feats2d = tf.reshape(feats2d, shape=[batch_size,-1,model_settings['feature_width']]) # features are of shape [max seq length for batch, 40]
    seq_lengths = shapes[:,0] # all shapes are [seq_length, 40], we extract seq_length

    # First LSTM 

    # LSTM cells
    cell_fw = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)
    cell_bw = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)

    # Bi-directional RNN (+ Dropout)
    (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, feats2d, 
                                                                sequence_length=seq_lengths, 
                                                                dtype=tf.float32)

    # TODO: make predictions after every 64 time slices

    concat_rnn = tf.concat([state_fw[0], state_bw[0]], axis=1)

    if is_training:
        first_dropout = tf.nn.dropout(concat_rnn, dropout_prob)
    else:
        first_dropout = concat_rnn

    # Second LSTM 
    # TODO

    # Logits Layer
    num_classes = model_settings['num_classes']
    logits = tf.layers.dense(inputs=first_dropout, units=num_classes)
    
    if is_training:
        return logits, dropout_prob
    else:
        return logits

def create_DCNN_LSTM_model(feats2d, shapes, model_settings, is_training):
    """Builds a standard convolutional model.

                (feats2d)
                    v
            [Conv1D]
                    v
            [BiasAdd]
                    v
                [Relu]
                    v
            [MaxPool]
                    v
            [Conv1D]
                    v
            [BiasAdd]
                    v
                [Relu]
                    v
            [MaxPool]
                    v
            [BiLSTM]<-(cell_fw, cell_bw)
                    v
                [Dense]
                    v

    Args:
        feats2d: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.

    Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.
    """
    if is_training:
        dropout_prob = model_settings['dropout_prob']
    else:
        dropout_prob = 0
    # Input Layer
    shape = tf.shape(feats2d) # features are of shape [max seq length for batch, 40]
    input_layer = tf.reshape(feats2d,[-1, shape[1], model_settings['feature_width']]) # [batch_size, seq_length, 40]

    # Convolutional Layer #1 (Dropout #1) and Pooling Layer #1
    conv1 = tf.layers.conv1d(
      inputs=input_layer,
      filters=model_settings['conv1_num_filters'],
      kernel_size=model_settings['conv1_kernel_size'],
      padding="same",
      activation=tf.nn.relu)

    dropout1 = tf.layers.dropout(
      inputs=conv1, rate=dropout_prob, training=is_training)

    pool1 = tf.layers.max_pooling1d(inputs=dropout1, pool_size=model_settings['pool1_pool_size'], strides=model_settings['pool1_strides']) 

    # Convolutional Layer #2 (Dropout #2) and Pooling Layer #2
    conv2 = tf.layers.conv1d(
      inputs=pool1,
      filters=model_settings['conv2_num_filters'],
      kernel_size=model_settings['conv2_kernel_size'],
      padding="same",
      activation=tf.nn.relu)

    dropout2 = tf.layers.dropout(
      inputs=conv2, rate=dropout_prob, training=is_training)

    pool2 = tf.layers.max_pooling1d(inputs=dropout2, pool_size=model_settings['pool2_pool_size'], strides=model_settings['pool2_strides']) # [batch_size, pool2_shape[1], 64]

    # Convolutional Layer #3 (Dropout #3) and Pooling Layer #3
    conv3 = tf.layers.conv1d(
      inputs=pool2,
      filters=model_settings['conv3_num_filters'],
      kernel_size=model_settings['conv3_kernel_size'],
      padding="same",
      activation=tf.nn.relu)

    dropout3 = tf.layers.dropout(
      inputs=conv3, rate=dropout_prob, training=is_training)

    pool3 = tf.layers.max_pooling1d(inputs=dropout3, pool_size=model_settings['pool3_pool_size'], strides=model_settings['pool3_strides']) 

    # Convolutional Layer #4 (Dropout #4) and Pooling Layer #4
    conv4 = tf.layers.conv1d(
      inputs=pool3,
      filters=model_settings['conv4_num_filters'],
      kernel_size=model_settings['conv4_kernel_size'],
      padding="same",
      activation=tf.nn.relu)

    dropout4 = tf.layers.dropout(
      inputs=conv4, rate=dropout_prob, training=is_training)

    pool4 = tf.layers.max_pooling1d(inputs=dropout4, pool_size=model_settings['pool4_pool_size'], strides=model_settings['pool4_strides']) # [batch_size, pool4_shape[1], 64]

    # Get dimensions
    lstm_size = model_settings['lstm_size']
    
    # LSTM cells
    cell_fw = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)
    cell_bw = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)


    # Bi-directional RNN (+ Dropout)
    (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, pool4,  
                                                                dtype=tf.float32)

    # initial_state_fw = ini_fw, initial_state_bw = ini_bw, 
    # if state_is_tuple, state is a tuple (cell_state, memory_state)
    concat_rnn = tf.concat([state_fw[0], state_bw[0]], axis=1)

    if is_training:
        first_dropout = tf.nn.dropout(concat_rnn, dropout_prob)
    else:
        first_dropout = concat_rnn

    # Logits Layer
    num_classes = model_settings['num_classes']
    logits = tf.layers.dense(inputs=first_dropout, units=num_classes)
    
    if is_training:
        return logits, dropout_prob
    else:
        return logits

def create_2D_DCNN_LSTM_model(feats2d, shapes, model_settings, is_training):
    """Builds a standard convolutional model.

                (feats2d)
                    v
            [Conv2D]
                    v
                [Relu]
                    v
            [MaxPool]
                    v
            [Conv2D]
                    v
                [Relu]
                    v
            [MaxPool]
                    v
            [Conv2D]
                    v
                [Relu]
                    v
            [MaxPool]
                    v
            [Conv2D]
                    v
                [Relu]
                    v
            [MaxPool]
                    v
            [BiLSTM]<-(cell_fw, cell_bw)
                    v
                [Dense]
                    v

    Args:
        feats2d: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.

    Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.
    """
    if is_training:
        dropout_prob = model_settings['dropout_prob']     
    else:
        dropout_prob = 0
    # Input Layer
    shape = tf.shape(feats2d) # features are of shape [max seq length for batch, 40]
    input_layer = tf.reshape(feats2d,tf.stack([-1, shape[1], model_settings['feature_width'], 1])) # [batch_size, seq_length, feature_width, 1]

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=model_settings['conv1_num_filters'],
      kernel_size=model_settings['conv1_kernel_size'],
      padding="same")

    batch_n1 = tf.layers.batch_normalization(inputs=conv1, training=is_training)

    if model_settings['activation'] == 'sigmoid':
        ac1 = tf.nn.sigmoid(batch_n1)
    else:
        ac1 = tf.nn.relu(batch_n1)

    dropout1 = tf.layers.dropout(
      inputs=ac1, rate=dropout_prob, training=is_training)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=dropout1, pool_size=model_settings['pool1_pool_size'], strides=model_settings['pool1_strides']) # [batch_size, seq_length/2, 40/2, conv2_num_filters] 
    #print(pool1)

    # Convolutional Layer #2 (Dropout #2) and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=model_settings['conv2_num_filters'],
      kernel_size=model_settings['conv2_kernel_size'],
      padding="same")

    batch_n2 = tf.layers.batch_normalization(inputs=conv2, training=is_training)

    if model_settings['activation'] == 'sigmoid':
        ac2 = tf.nn.sigmoid(batch_n2)
    else:
        ac2 = tf.nn.relu(batch_n2)

    dropout2 = tf.layers.dropout(
      inputs=ac2, rate=dropout_prob, training=is_training)

    pool2 = tf.layers.max_pooling2d(inputs=dropout2, pool_size=model_settings['pool2_pool_size'], strides=model_settings['pool2_strides']) # [batch_size, seq_length/2/2, 20/2, conv2_num_filters]
    #print(pool2)

    # Convolutional Layer #3 (Dropout #3) and Pooling Layer #3
    conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=model_settings['conv3_num_filters'],
      kernel_size=model_settings['conv3_kernel_size'],
      padding="same")

    batch_n3 = tf.layers.batch_normalization(inputs=conv3, training=is_training)

    if model_settings['activation'] == 'sigmoid':
        ac3 = tf.nn.sigmoid(batch_n3)
    else:
        ac3 = tf.nn.relu(batch_n3)

    dropout3 = tf.layers.dropout(
      inputs=ac3, rate=dropout_prob, training=is_training)

    pool3 = tf.layers.max_pooling2d(inputs=dropout3, pool_size=model_settings['pool3_pool_size'], strides=model_settings['pool3_strides']) # [batch_size, seq_length/2/2/3, 10/3, conv3_num_filters]
    #print(pool3)

    # Convolutional Layer #4 (Dropout #4) and Pooling Layer #4
    conv4 = tf.layers.conv2d(
      inputs=pool3,
      filters=model_settings['conv4_num_filters'],
      kernel_size=model_settings['conv4_kernel_size'],
      padding="same")

    batch_n4 = tf.layers.batch_normalization(inputs=conv4,  training=is_training)

    if model_settings['activation'] == 'sigmoid':
        ac4 = tf.nn.sigmoid(batch_n4)
    else:
        ac4 = tf.nn.relu(batch_n4)

    dropout4 = tf.layers.dropout(
      inputs=ac4, rate=dropout_prob, training=is_training)

    pool4 = tf.layers.max_pooling2d(inputs=dropout4, pool_size=model_settings['pool4_pool_size'], strides=model_settings['pool4_strides']) # [batch_size, seq_length/2/2/3/3, 3/3, conv4_num_filters]
    #print(pool4)

    pool4_squeezed = tf.squeeze(pool4, axis=[2])
    # Get dimensions
    lstm_size = model_settings['lstm_size']
    
    # LSTM cells
    cell_fw = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)
    cell_bw = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)

    # Bi-directional RNN (+ Dropout)
    (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, pool4_squeezed,  
                                                                dtype=tf.float32)

    # initial_state_fw = ini_fw, initial_state_bw = ini_bw, 
    # if state_is_tuple, state is a tuple (cell_state, memory_state)
    concat_rnn = tf.concat([state_fw[0], state_bw[0]], axis=1)

    if is_training:
        first_dropout = tf.nn.dropout(concat_rnn, dropout_prob)
    else:
        first_dropout = concat_rnn

    # Logits Layer
    num_classes = model_settings['num_classes']
    logits = tf.layers.dense(inputs=first_dropout, units=num_classes)
    
    if is_training:
        return logits, dropout_prob
    else:
        return logits

def create_2D_VDCNN_LSTM_model(feats2d, shapes, model_settings, is_training):
    """Builds a standard convolutional model.

                (feats2d)
                    v
            [Conv2D]
                    v
                [Relu]
                    v
            [MaxPool]
                    v
            [Conv2D]
                    v
                [Relu]
                    v
            [MaxPool]
                    v
            [Conv2D]
                    v
                [Relu]
                    v
            [MaxPool]
                    v
            [Conv2D]
                    v
                [Relu]
                    v
            [MaxPool]
                    v
            [Conv2D]
                    v
                [Relu]
                    v
            [MaxPool]
                    v
            [BiLSTM]<-(cell_fw, cell_bw)
                    v
                [Dense]
                    v

    Args:
        feats2d: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.

    Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.
    """

    if is_training:
        dropout_prob = model_settings['dropout_prob']     
    else:
        dropout_prob = 0
    # Input Layer
    shape = tf.shape(feats2d) # features are of shape [max seq length for batch, 40]
    input_layer = tf.reshape(feats2d,tf.stack([-1, shape[1], model_settings['feature_width'], 1])) # [batch_size, seq_length, feature_width, 1]

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=model_settings['conv1_num_filters'],
      kernel_size=model_settings['conv1_kernel_size'],
      padding="same")

    batch_n1 = tf.layers.batch_normalization(inputs=conv1,  training=is_training)

    if model_settings['activation'] == 'sigmoid':
        ac1 = tf.nn.sigmoid(batch_n1)
    else:
        ac1 = tf.nn.relu(batch_n1)

    dropout1 = tf.layers.dropout(
      inputs=ac1, rate=dropout_prob, training=is_training)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=dropout1, pool_size=model_settings['pool1_pool_size'], strides=model_settings['pool1_strides']) # [batch_size, seq_length/2, 40/2, conv2_num_filters] 
    print(pool1)

    # Convolutional Layer #2 (Dropout #2) and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=model_settings['conv2_num_filters'],
      kernel_size=model_settings['conv2_kernel_size'],
      padding="same")

    batch_n2 = tf.layers.batch_normalization(inputs=conv2,  training=is_training)

    if model_settings['activation'] == 'sigmoid':
        ac2 = tf.nn.sigmoid(batch_n2)
    else:
        ac2 = tf.nn.relu(batch_n2)

    dropout2 = tf.layers.dropout(
      inputs=ac2, rate=dropout_prob, training=is_training)

    pool2 = tf.layers.max_pooling2d(inputs=dropout2, pool_size=model_settings['pool2_pool_size'], strides=model_settings['pool2_strides']) # [batch_size, seq_length/2/2, 20/2, conv2_num_filters]
    print(pool2)

    # Convolutional Layer #3 (Dropout #3) and Pooling Layer #3
    conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=model_settings['conv3_num_filters'],
      kernel_size=model_settings['conv3_kernel_size'],
      padding="same")

    batch_n3 = tf.layers.batch_normalization(inputs=conv3,  training=is_training)

    if model_settings['activation'] == 'sigmoid':
        ac3 = tf.nn.sigmoid(batch_n3)
    else:
        ac3 = tf.nn.relu(batch_n3)

    dropout3 = tf.layers.dropout(
      inputs=ac3, rate=dropout_prob, training=is_training)

    pool3 = tf.layers.max_pooling2d(inputs=dropout3, pool_size=model_settings['pool3_pool_size'], strides=model_settings['pool3_strides']) # [batch_size, seq_length/2/2/2, 10/2, conv3_num_filters]
    print(pool3)

    # Convolutional Layer #4 (Dropout #4) and Pooling Layer #4
    conv4 = tf.layers.conv2d(
      inputs=pool3,
      filters=model_settings['conv4_num_filters'],
      kernel_size=model_settings['conv4_kernel_size'],
      padding="same")

    batch_n4 = tf.layers.batch_normalization(inputs=conv4,  training=is_training)

    if model_settings['activation'] == 'sigmoid':
        ac4 = tf.nn.sigmoid(batch_n4)
    else:
        ac4 = tf.nn.relu(batch_n4)

    dropout4 = tf.layers.dropout(
      inputs=ac4, rate=dropout_prob, training=is_training)

    pool4 = tf.layers.max_pooling2d(inputs=dropout4, pool_size=model_settings['pool4_pool_size'], strides=model_settings['pool4_strides']) # [batch_size, seq_length/2/2/2/2, 5/2, conv4_num_filters]
    print(pool4)

    # Convolutional Layer #5 (Dropout #5) and Pooling Layer #5
    conv5 = tf.layers.conv2d(
      inputs=pool4,
      filters=model_settings['conv5_num_filters'],
      kernel_size=model_settings['conv5_kernel_size'],
      padding="same")

    batch_n5 = tf.layers.batch_normalization(inputs=conv5,  training=is_training)

    if model_settings['activation'] == 'sigmoid':
        ac5 = tf.nn.sigmoid(batch_n5)
    else:
        ac5 = tf.nn.relu(batch_n5)

    dropout5 = tf.layers.dropout(
      inputs=ac5, rate=dropout_prob, training=is_training)

    pool5 = tf.layers.max_pooling2d(inputs=dropout5, pool_size=model_settings['pool5_pool_size'], strides=model_settings['pool5_strides']) # [batch_size, seq_length/2/2/2/2, 2/2, conv4_num_filters]
    
    #print(pool4)

    pool5_squeezed = tf.squeeze(pool5, axis=[2])
    # Get dimensions
    lstm_size = model_settings['lstm_size']
    
    # LSTM cells
    cell_fw = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)
    cell_bw = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)

    # Bi-directional RNN (+ Dropout)
    (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, pool5_squeezed,  
                                                                dtype=tf.float32)

    # initial_state_fw = ini_fw, initial_state_bw = ini_bw, 
    # if state_is_tuple, state is a tuple (cell_state, memory_state)
    concat_rnn = tf.concat([state_fw[0], state_bw[0]], axis=1)

    if is_training:
        first_dropout = tf.nn.dropout(concat_rnn, dropout_prob)
    else:
        first_dropout = concat_rnn

    # Logits Layer
    num_classes = model_settings['num_classes']
    logits = tf.layers.dense(inputs=first_dropout, units=num_classes)
    
    if is_training:
        return logits, dropout_prob
    else:
        return logits

def organizers_model(feats2d, shapes, model_settings, is_training):

    if is_training:
        dropout_prob = model_settings['dropout_prob']
    else:
        dropout_prob = 0
        
    shape = tf.shape(feats2d) # features are of shape [max seq length for batch, 40]
    x = tf.reshape(feats2d,[-1, shape[1], model_settings['feature_width']]) # [batch_size, seq_length, 40]
    shape_list = shapes[:,0] # all shapes are [seq_length, 40], we extract seq_length
    is_batchnorm = True

    featdim = model_settings['feature_width'] #channel
    weights = []
    kernel_size = model_settings['conv1_kernel_size']
    stride = model_settings['conv1_strides']
    depth = model_settings['conv1_depth']
            
    shape_list = shape_list/stride
    conv1 = conv_layer(x,kernel_size,featdim,stride,depth,'conv1',shape_list)
    conv1_bn = batch_norm_wrapper_1dcnn(conv1, is_training,'bn1',shape_list,is_batchnorm)
    conv1r= tf.nn.relu(conv1_bn)

    featdim = depth #channel
    weights = []
    kernel_size = model_settings['conv2_kernel_size']
    stride = model_settings['conv2_strides']
    depth = model_settings['conv2_depth']
            
    shape_list = shape_list/stride
    conv2 = conv_layer(conv1r,kernel_size,featdim,stride,depth,'conv2',shape_list)
    conv2_bn = batch_norm_wrapper_1dcnn(conv2, is_training,'bn2',shape_list,is_batchnorm)
    conv2r= tf.nn.relu(conv2_bn)
   
    featdim = depth #channel
    weights = []
    kernel_size = model_settings['conv3_kernel_size']
    stride = model_settings['conv3_strides']
    depth = model_settings['conv3_depth']
            
    shape_list = shape_list/stride
    conv3 = conv_layer(conv2r,kernel_size,featdim,stride,depth,'conv3',shape_list)
    conv3_bn = batch_norm_wrapper_1dcnn(conv3, is_training,'bn3',shape_list,is_batchnorm)
    conv3r= tf.nn.relu(conv3_bn)
   
    featdim = depth #channel
    weights = []
    kernel_size = model_settings['conv4_kernel_size']
    stride = model_settings['conv4_strides']
    depth = model_settings['conv4_depth']
            
    shape_list = shape_list/stride
    conv4 = conv_layer(conv3r,kernel_size,featdim,stride,depth,'conv4',shape_list)
    conv4_bn = batch_norm_wrapper_1dcnn(conv4, is_training,'bn4',shape_list,is_batchnorm)
    conv4r= tf.nn.relu(conv4_bn)
    
    print(conv1) # [batch_size, ?, 500]
    print(conv2) # [batch_size, ?, 500]
    print(conv3) # [batch_size, ?, 500]
    print(conv4) # [batch_size, ?, 3000]

    if model_settings['before_softmax'] == 'lstm':
        # Get dimensions
        lstm_size = model_settings['lstm_size']
        
        # LSTM cells
        cell_fw = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)

        # Bi-directional RNN (+ Dropout)
        (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, conv4r,  
                                                                    dtype=tf.float32)

        # if state_is_tuple, state is a tuple (cell_state, memory_state)
        concat_rnn = tf.concat([state_fw[0], state_bw[0]], axis=1)

        if is_training:
            first_dropout = tf.nn.dropout(concat_rnn, dropout_prob)
        else:
            first_dropout = concat_rnn

        # Logits Layer
        num_classes = model_settings['num_classes']
        logits = tf.layers.dense(inputs=first_dropout, units=num_classes)

    else:

        shape_list = tf.cast(shape_list, tf.float32)
        shape_list = tf.reshape(shape_list,[-1,1,1])
        mean = tf.reduce_sum(conv4r,1,keep_dims=True)/shape_list
        res1=tf.squeeze(mean,axis=1)
        print(res1) # [batch_size, 3000]

        fc1 = fc_layer(res1,model_settings['fc1_size'],"fc1")
        fc1_bn = batch_norm_wrapper_fc(fc1, is_training,'bn5',is_batchnorm)
        ac1 = tf.nn.relu(fc1_bn)
        fc2 = fc_layer(ac1,model_settings['fc2_size'],"fc2")
        fc2_bn = batch_norm_wrapper_fc(fc2, is_training,'bn6',is_batchnorm)
        ac2 = tf.nn.relu(fc2_bn)    
        logits = fc_layer(ac2,model_settings['num_classes'],"fc3")

        print(fc1) # [batch_size, 1500]
        print(fc2) # [batch_size, 600]
        print(logits) # [batch_size, 5]

    if is_training:
        return logits, dropout_prob
    else:
        return logits
    
def xavier_init(n_inputs, n_outputs, uniform=True):
  if uniform:
    init_range = np.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)
  else:
    stddev = np.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)

def fc_layer( bottom, n_weight, name):
    print( bottom.get_shape())
    assert len(bottom.get_shape()) == 2
    n_prev_weight = bottom.get_shape()[1]

    initer = xavier_init(int(n_prev_weight),n_weight)
    W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
    b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.random_uniform([n_weight],-0.001,0.001, dtype=tf.float32))
    fc = tf.nn.bias_add(tf.matmul(bottom, W), b, name= name+"layer")
    return fc


def conv_layer( bottom, kernel_size,num_channels, stride, depth, name, shape_list):   # n_prev_weight = int(bottom.get_shape()[1])
    n_prev_weight = tf.shape(bottom)[1]

    inputlayer=bottom
    initer = tf.truncated_normal_initializer(stddev=0.1)

    W = tf.get_variable(name+'W', dtype=tf.float32, shape=[kernel_size, num_channels, depth], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.001, shape=[depth], dtype=tf.float32))
    
    conv =  ( tf.nn.bias_add( tf.nn.conv1d(inputlayer, W, stride, padding='SAME'), b))
    mask = tf.sequence_mask(shape_list,tf.shape(conv)[1]) # make mask with batch x frame size
    mask = tf.where(mask, tf.ones_like(mask,dtype=tf.float32), tf.zeros_like(mask,dtype=tf.float32))
    mask=tf.tile(mask, tf.stack([tf.shape(conv)[2],1])) #replicate make with depth size
    mask=tf.reshape(mask,[tf.shape(conv)[2], tf.shape(conv)[0], -1])
    mask = tf.transpose(mask,[1, 2, 0])
    print(mask)
    conv=tf.multiply(conv,mask)
    return conv

def batch_norm_wrapper_1dcnn( inputs, is_training, name, shape_list, is_batchnorm,decay = 0.999 ):
    if is_batchnorm:
        shape_list = tf.cast(shape_list, tf.float32)
        epsilon = 1e-3
        scale = tf.get_variable(name+'scale',dtype=tf.float32,initializer=tf.ones([inputs.get_shape()[-1]]) )
        beta = tf.get_variable(name+'beta',dtype=tf.float32,initializer= tf.zeros([inputs.get_shape()[-1]]) )
        pop_mean = tf.get_variable(name+'pop_mean',dtype=tf.float32,initializer = tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.get_variable(name+'pop_var',dtype=tf.float32,initializer = tf.ones([inputs.get_shape()[-1]]), trainable=False)
        if is_training:
            #batch_mean, batch_var = tf.nn.moments(inputs,[0,1])
            batch_mean = tf.reduce_sum(inputs,[0,1])/tf.reduce_sum(shape_list) # for variable length input
            batch_var = tf.reduce_sum(tf.square(inputs-batch_mean), [0,1])/tf.reduce_sum(shape_list) # for variable length input
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)
    else:
        return inputs
    
                 
def batch_norm_wrapper_fc( inputs, is_training, name, is_batchnorm, decay = 0.999 ):
    if is_batchnorm:
            epsilon = 1e-3
            scale = tf.get_variable(name+'scale',dtype=tf.float32,initializer=tf.ones([inputs.get_shape()[-1]]) )
            beta = tf.get_variable(name+'beta',dtype=tf.float32,initializer= tf.zeros([inputs.get_shape()[-1]]) )
            pop_mean = tf.get_variable(name+'pop_mean',dtype=tf.float32,initializer = tf.zeros([inputs.get_shape()[-1]]), trainable=False)
            pop_var = tf.get_variable(name+'pop_var',dtype=tf.float32,initializer = tf.ones([inputs.get_shape()[-1]]), trainable=False)
            if is_training:
                batch_mean, batch_var = tf.nn.moments(inputs,[0])
                train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, epsilon)
            else:
                return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)
    else:
        return inputs
