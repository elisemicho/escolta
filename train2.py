"""Model definitions for simple speech recognition. 
Inspiration sources:
Reading: https://www.tensorflow.org/programmers_guide/datasets
Decoding: https://github.com/swshon/dialectID_e2e
Padding: https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/utils/data.py
Training: https://www.tensorflow.org/versions/master/tutorials/audio_recognition
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

import tensorflow as tf

import models

# Basic model parameters as external flags.
FLAGS = None

parser = argparse.ArgumentParser()

# Input parameters
parser.add_argument(
        '--train_dir',
        type=str,
        default='../../data/tfrecords',
        help='Directory with the training data.')
parser.add_argument(
        '--num_classes',
        type=int,
        default=5,
        help='Number of classes.')

# Model parameters
parser.add_argument(
        '--model_architecture',
        type=str,
        default='simple_conv1D',
        help='What model architecture to use.')

# Training parameters
parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.')
parser.add_argument(
        '--num_epochs',
        type=int,
        default=2,
        help='Number of epochs to run trainer.')
parser.add_argument(
        '--batch_size', 
        type=int, 
        default=10, 
        help='Batch size.')
parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=400,
        help='How often to evaluate the training results.')

# Output parameters
parser.add_argument(
        '--summaries_dir',
        type=int,
        default=5,
        help='Directory for Tensorflow summaries')
parser.add_argument(
        '--save_step_interval',
        type=int,
        default=100,
        help='Save model checkpoint every save_steps.')

FLAGS, unparsed = parser.parse_known_args()

# Constants used for dealing with the tfrecords files.
TRAIN_FILE = 'train_shuffle.tfrecords'
VALIDATION_FILE = 'dev_shuffle.tfrecords'
TEST_FILE = 'test_shuffle.tfrecords'

def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                        'labels': tf.FixedLenFeature([], tf.int64),
                        'shapes': tf.FixedLenFeature([2], tf.int64),
                        'features': tf.VarLenFeature( tf.float32)
            })

    labels = features['labels']
    shapes = features['shapes']
    feats = features['features']
    print(feats)
    shapes = tf.cast(shapes, tf.int32)
    feats2d = tf.reshape(feats.values, shapes)
    print(feats2d)
    return labels, feats2d, shapes


def get_padded_shapes(dataset):
    """Returns the padded shapes for ``tf.data.Dataset.padded_batch``.
    Args:
    dataset: The dataset that will be batched with padding.
    Returns:
    The same structure as ``dataset.output_shapes`` containing the padded
    shapes.
    """
    return tf.contrib.framework.nest.map_structure(lambda shape: shape.as_list(), dataset.output_shapes)


def get_dataset_shape(train): 
    c = 0
    filename = os.path.join(FLAGS.train_dir, TRAIN_FILE if train else VALIDATION_FILE)
    for record in tf.python_io.tf_record_iterator(filename):
        c += 1
    return c


def inputs(train, batch_size, num_epochs):
    """Reads input data num_epochs times.
    Args:
        train: Selects between the training (True) and validation (False) data.
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to
             train forever.
    Returns:
        A tuple (feats2d, labels).
        This function creates a one_shot_iterator, meaning that it will only iterate
        over the dataset once. On the other hand there is no special initialization
        required.
    """
    if not num_epochs:
        num_epochs = None
    filename = os.path.join(FLAGS.train_dir, TRAIN_FILE
                                                    if train else VALIDATION_FILE)

    with tf.name_scope('input'):
        # TFRecordDataset opens a binary file and reads one record at a time.
        # `filename` could also be a list of filenames, which will be read in order.
        dataset = tf.data.TFRecordDataset(filename)

        # The map transformation takes a function and applies it to every element
        # of the dataset.
        dataset = dataset.map(decode)
        #print(dataset)
        dataset = dataset.repeat(num_epochs)
        
        #dataset = dataset.batch(batch_size)
        dataset = dataset.padded_batch(batch_size, padded_shapes=get_padded_shapes(dataset))

        iterator = dataset.make_one_shot_iterator()
        #iterator = dataset.make_initializable_iterator()

    return iterator.get_next()


# Tell TensorFlow that the model will be built into the default Graph.
with tf.Graph().as_default():
    
    # Set parameters to convey to the model
    model_settings = models.prepare_model_settings(FLAGS.num_classes)
    
    # Input images and labels
    label_batch, feat2d_batch, shape_batch = inputs(
            TRAIN_FILE, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
    
    # Build a Graph that computes predictions from the model
    logits, dropout_prob = models.create_model(
            feat2d_batch, shape_batch,
            model_settings,
            FLAGS.model_architecture,
            is_training=True)

    # Define loss
    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
                labels=label_batch, logits=logits)   
    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    
    # Define optimizer
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(
                FLAGS.learning_rate).minimize(cross_entropy_mean)
        
    # Define evaluation metrics
    predicted_indices = tf.argmax(logits, 1)
    correct_prediction = tf.equal(predicted_indices, label_batch)
    confusion_matrix = tf.confusion_matrix(
            label_batch, predicted_indices, num_classes=FLAGS.num_classes)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)

    saver = tf.train.Saver(tf.global_variables())

    # Merge all the summaries and write them out to directory
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')
    
    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                                         tf.local_variables_initializer())

    
    # Create a session for running operations in the Graph.
    with tf.Session() as sess:
        # Initialize the variables (the trained variables and the
        # epoch counter).
        sess.run(init_op)
        
        # Training loop.
        start_step = 1
        training_steps_max = FLAGS.batch_size * FLAGS.num_epochs
        for training_step in xrange(start_step, training_steps_max + 1):
                
            #print(sess.run(label_batch))
            """
            print(sess.run(tf.shape(feat2d_batch)))
            print(sess.run(tf.shape(label_batch)))
            print(sess.run(tf.shape(shape_batch)))
            
            """
            # Run one training step of the model:
            # Write summary, compute accuracy, compute loss, train model, increment step
            train_summary, train_accuracy, cross_entropy_value, _ = sess.run(
            [merged_summaries, evaluation_step, cross_entropy_mean, train_step])

            # Report
            train_writer.add_summary(train_summary, training_step)
            tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                                            (training_step, learning_rate_value, train_accuracy * 100,
                                             cross_entropy_value))

            # Evaluate on validation data  
            if (training_step % FLAGS.eval_step_interval) == 0:
                valid_set_size = get_dataset_shape(VALIDATION_FILE)
                total_accuracy = 0
                total_conf_matrix = None
                for i in xrange(0, valid_set_size, FLAGS.batch_size):
                    # Input images and labels.
                    valid_label_batch, valid_feat2d_batch = inputs(
                            VALIDATION_FILE, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)

                    # Run evaluation step and capture training summaries for TensorBoard
                    # with the `merged` op.
                    validation_summary, validation_accuracy, conf_matrix = sess.run(
                            [merged_summaries, evaluation_step, confusion_matrix])

                    # Report
                    validation_writer.add_summary(validation_summary, training_step)
                    batch_size = min(FLAGS.batch_size, valid_set_size - i)
                    total_accuracy += (validation_accuracy * batch_size) / valid_set_size
                    if total_conf_matrix is None:
                        total_conf_matrix = conf_matrix
                    else:
                        total_conf_matrix += conf_matrix
                tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
                tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                                                (training_step, total_accuracy * 100, valid_set_size))

            # Save the model checkpoint periodically.
            if (training_step % FLAGS.save_step_interval == 0 or
                training_step == training_steps_max):
                checkpoint_path = os.path.join(FLAGS.train_dir,FLAGS.model_architecture + '.ckpt')
                tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
                saver.save(sess, checkpoint_path, global_step=training_step)

        print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs,training_steps_max))
        
        # Testing loop
        test_set_size = get_dataset_shape(TEST_FILE)
        tf.logging.info('test_set_size=%d', test_set_size)
        total_accuracy = 0
        total_conf_matrix = None
        for i in xrange(0, test_set_size, FLAGS.batch_size):
            
            # Input images and labels.
            test_label_batch, test_feat2d_batch = inputs(
                    TEST_FILE, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)

            # Run evaluation step
            test_accuracy, conf_matrix = sess.run(
                    [evaluation_step, confusion_matrix])

            # Report 
            batch_size = min(FLAGS.batch_size, test_set_size - i)
            total_accuracy += (test_accuracy * batch_size) / test_set_size
            if total_conf_matrix is None:
                total_conf_matrix = conf_matrix
            else:
                total_conf_matrix += conf_matrix
        tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
        tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100,test_set_size))
        

