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

def prepare_model_settings(num_classes):
    """Calculates common settings needed for all models.

    Args:
        num_classes: How many classes are to be recognized.

    Returns:
        Dictionary containing common settings.
    """
    return {
            'num_classes': num_classes,
    }


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
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

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
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')    

    # Input Layer
    shape = tf.shape(feats2d)
    input_layer = tf.reshape(feats2d,[-1, shape[1], shape[2], 1])

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

    pool2 = tf.layers.max_pooling2d(inputs=dropout2, pool_size=[2, 2], strides=2)

    #pool2_shape = tf.shape(pool2)
    #pool2_flat = tf.reshape(pool2, [-1, pool2_shape[1] * pool2_shape[2] * 64])
    # pool2_flat = tf.layers.flatten(pool2)
    shape_list = shapes[:,0]
    shape_list = tf.cast(shape_list, tf.float32)
    shape_list = tf.reshape(shape_list,[-1,1,1])
    mean = tf.reduce_sum(pool2,1,keep_dims=True)/shape_list
    res1=tf.squeeze(mean,axis=1)

    # Logits Layer
    num_classes = model_settings['num_classes']
    logits = tf.layers.dense(inputs=res1, units=num_classes)
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
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')    

    # Input Layer
    shape = tf.shape(feats2d)
    input_layer = feats2d

    # Convolutional Layer #1 (Dropout #1) and Pooling Layer #1
    conv1 = tf.layers.conv1d(
      inputs=input_layer,
      filters=64,
      kernel_size=8,
      padding="same",
      activation=tf.nn.relu)

    dropout1 = tf.layers.dropout(
      inputs=conv1, rate=dropout_prob, training=is_training)

    pool1 = tf.layers.max_pooling2d(inputs=dropout1, pool_size=2, strides=2)

    # Convolutional Layer #2 (Dropout #2) and Pooling Layer #2
    conv2 = tf.layers.conv1d(
      inputs=pool1,
      filters=64,
      kernel_size=4,
      padding="same",
      activation=tf.nn.relu)

    dropout2 = tf.layers.dropout(
      inputs=conv2, rate=dropout_prob, training=is_training)

    pool2 = tf.layers.max_pooling2d(inputs=dropout2, pool_size=2, strides=2)

    pool2_shape = tf.shape(pool2)
    # pool2_flat = tf.reshape(pool2, [-1, pool2_shape[1] * pool2_shape[2] * 64])
    pool2_flat = tf.layers.flatten(pool2)

    # Logits Layer
    num_classes = model_settings['num_classes']
    logits = tf.layers.dense(inputs=pool2_flat, units=num_classes)
    return logits


def create_bidirectionnal_dynamic_rnn_model(feats2d, shapes, model_settings, is_training):
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
    """
    # Get dimensions
    num_classes = model_settings['num_classes']
    batch_size = tf.shape(feats2d)[0] 
    lstm_size = 128

    # Do we need reshaping here?

    # LSTM cells
    cell_fw = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)
    cell_bw = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)
    ini_fw = cell_fw.zero_state(batch_size,dtype=tf.float32)
    ini_bw = cell_bw.zero_state(batch_size,dtype=tf.float32)

    # Bi-directional RNN (+ Dropout)
    (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, feats2d, initial_state_fw = ini_fw, initial_state_bw = ini_bw, 
                                                                sequence_length=shapes[:,0], dtype=tf.float32)
    concat_rnn = tf.concat([state_fw, state_bw], axis=1)

    if is_training:
        first_dropout = tf.nn.dropout(concat_rnn, dropout_prob)
    else:
        first_dropout = concat_rnn

    # Get dimensions
    # Do we need reshaping here?
    first_dropout_size = first_dropout.get_shape()[1] 
    num_classes = model_settings['num_classes']

    # Weigths and biases
    final_fc_weights = tf.Variable(
            tf.truncated_normal(
                    [first_dropout_size, num_classes], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([num_classes]))

    # Fully-connected layer
    final_fc = tf.matmul(first_dropout, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc

