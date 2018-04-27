# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

import tensorflow as tf
import numpy as np
import sklearn.metrics as sk

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
        default='cnn_lstm',
        help='What model architecture to use among "simple_conv1D", "simple_conv2D", "lstm" or "cnn_lstm".')

# Training parameters
parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.')
parser.add_argument(
        '--num_epochs',
        type=int,
        default=15,
        help='Number of epochs to run trainer.')
parser.add_argument(
        '--batch_size', 
        type=int, 
        default=10, 
        help='Batch size.')
parser.add_argument(
        '--loss_print_interval',
        type=int,
        default=100,
        help='How often to evaluate the training results.')
parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=500,
        help='How often to evaluate the training results.')

# Output parameters
parser.add_argument(
        '--summaries_dir',
        type=str,
        default='summaries',
        help='Directory for Tensorflow summaries')
parser.add_argument(
        '--save_step_interval',
        type=int,
        default=500,
        help='Save model checkpoint every save_steps.')

FLAGS, unparsed = parser.parse_known_args()

# Constants used for dealing with the tfrecords files.
TRAIN_FILE = os.path.join(FLAGS.train_dir,'mgb3_logmel_fft400_hop160_vad_cmvn_train.0.tfrecords')
VALIDATION_FILE = os.path.join(FLAGS.train_dir,'mgb3_logmel_fft400_hop160_vad_cmvn_dev.0.tfrecords')
TEST_FILE = os.path.join(FLAGS.train_dir,'mgb3_logmel_fft400_hop160_vad_cmvn_test.0.tfrecords')
#TRAIN_FILE = os.path.join(FLAGS.train_dir,'train_completesample.tfrecords')
#VALIDATION_FILE = os.path.join(FLAGS.train_dir,'dev_completesample.tfrecords')
#TEST_FILE = VALIDATION_FILE
# In[2]:


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
    #print(shapes)
    shapes = tf.cast(shapes, tf.int32)
    feats2d = tf.reshape(feats.values, shapes)
    #print(shapes)
    return labels, feats2d, shapes


# In[3]:


def get_padded_shapes(dataset):
    """Returns the padded shapes for ``tf.data.Dataset.padded_batch``.
    Args:
    dataset: The dataset that will be batched with padding.
    Returns:
    The same structure as ``dataset.output_shapes`` containing the padded
    shapes.
    """
    return tf.contrib.framework.nest.map_structure(lambda shape: shape.as_list(), dataset.output_shapes)


# In[4]:


def get_dataset_shape(filename): 
    c = 0
    for record in tf.python_io.tf_record_iterator(filename):
        c += 1
    return c


# In[5]:


def inputs(filename, batch_size):
    """Reads input data num_epochs times.
    Args:
        train: Selects between the training (True) and validation (False) data.
        batch_size: Number of examples per returned batch.
    Returns:
        A tuple (feats2d, labels).
        This function creates a one_shot_iterator, meaning that it will only iterate
        over the dataset once. On the other hand there is no special initialization
        required.
    """

    with tf.name_scope('input'):
        # TFRecordDataset opens a binary file and reads one record at a time.
        # `filename` could also be a list of filenames, which will be read in order.
        dataset = tf.data.TFRecordDataset(filename)

        # The map transformation takes a function and applies it to every element
        # of the dataset.
        dataset = dataset.map(decode)
        # Would be better if data was shuffled but make the prgoram crash
        dataset = dataset.shuffle(buffer_size=10000)

        #print(dataset)
        dataset = dataset.repeat()
        
        #dataset = dataset.batch(batch_size)
        dataset = dataset.padded_batch(batch_size, padded_shapes=get_padded_shapes(dataset))
        iterator = dataset.make_one_shot_iterator()
        #iterator = dataset.make_initializable_iterator()
        
    return iterator.get_next()


# In[6]:


# Tell TensorFlow that the model will be built into the default Graph.
with tf.Graph().as_default():
    
    # Set parameters to convey to the model
    model_settings = models.prepare_model_settings(FLAGS.num_classes)
    
    # Input images and labels
    label_batch, feat2d_batch, shape_batch = inputs(
            TRAIN_FILE, batch_size=FLAGS.batch_size)
    
    # Build a Graph that computes predictions from the model
    logits, dropout_prob = models.create_model(
            feat2d_batch, shape_batch,
            model_settings,
            FLAGS.model_architecture,
            is_training=True)

    # Define loss
    with tf.name_scope('cross_entropy'):
        loss = tf.losses.sparse_softmax_cross_entropy(
                labels=label_batch, logits=logits)   
    tf.summary.scalar('cross_entropy', loss)
    
    # Define optimizer
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(
                FLAGS.learning_rate).minimize(loss)
        
    # Define evaluation metrics
    predicted_indices = tf.argmax(logits, 1)
    correct_prediction = tf.equal(predicted_indices, label_batch)
    confusion_matrix = tf.confusion_matrix(
            label_batch, predicted_indices, num_classes=FLAGS.num_classes)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)

    saver = tf.train.Saver(tf.global_variables(),max_to_keep=30)

    # Merge all the summaries
    merged_summaries = tf.summary.merge_all()
    
    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                                         tf.local_variables_initializer())

    # Create a session for running operations in the Graph.
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        # Initialize the writers for summaries
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/' + FLAGS.model_architecture + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/' + FLAGS.model_architecture + '/validation')
        
        # Initialize the variables (the trained variables).
        sess.run(init_op)
        
        # Training loop.
        print('TRAIN')
        train_set_size = get_dataset_shape(TRAIN_FILE)
        print('Train set size = ', train_set_size)
        
        for epoch in range(1, FLAGS.num_epochs + 1):
            print('Epoch #%d:' % (epoch))
            total_loss = []
            for training_step in range(1, int(train_set_size / FLAGS.batch_size) + 1):
                                
                # Run one training step of the model:
                # Train model, write summary, compute loss, compute accuracy
                _, train_summary, train_loss, train_accuracy = sess.run(
                [train_step, merged_summaries, loss, evaluation_step])

                # Report
                train_writer.add_summary(train_summary, training_step)
                total_loss.append(train_loss)

                # Evaluate on validation data  
                if (training_step % FLAGS.loss_print_interval) == 0:

                    print('Step #%d: rate %f, loss %f, accuracy %.1f%%' %
                                                (training_step, FLAGS.learning_rate, np.mean(total_loss), 
                                                 train_accuracy * 100))

                if (training_step % FLAGS.eval_step_interval) == 0:

                    valid_set_size = get_dataset_shape(VALIDATION_FILE)
                    #total_accuracy = 0
                    total_predictions = []
                    total_correct_labels = []

                    # Input images and labels.
                    valid_label_batch, valid_feat2d_batch, valid_shape_batch = inputs(VALIDATION_FILE, batch_size=FLAGS.batch_size)
                    
                    for i in range(0, valid_set_size, FLAGS.batch_size):
                        """
                        # Input images and labels.
                        valid_label_batch, valid_feat2d_batch, valid_shape_batch = inputs(
                                VALIDATION_FILE, batch_size=FLAGS.batch_size)
                        """
                        # Run evaluation step and capture training summaries for TensorBoard
                        # with the `merged` op.                        
                        valid_label_batch_, validation_summary, validation_loss, validation_accuracy, conf_matrix, validation_predictions = sess.run(
                            [valid_label_batch, merged_summaries, loss, evaluation_step, confusion_matrix, predicted_indices])
                        
                        # Report
                        validation_writer.add_summary(validation_summary, training_step)
                        #batch_size = min(FLAGS.batch_size, valid_set_size - i)
                        #total_accuracy += (validation_accuracy * batch_size) / valid_set_size 
                    
                        total_predictions.extend(validation_predictions)
                        total_correct_labels.extend(valid_label_batch_)                    
                    
                    #print("predictions",total_predictions)
                    #print("correct labels", total_correct_labels)
                    
                    print("prediction",total_correct_labels)
                    print("ground_truth",total_predictions)
                    
                    
                    print('Step #%d: Validation accuracy %f, f1_score (macro) %f' %
                                                    (training_step, 
                                                     sk.accuracy_score(total_correct_labels, total_predictions),
                                        sk.f1_score(total_correct_labels, total_predictions, average="macro")))
                    print('Confusion Matrix:')
                    print(sk.confusion_matrix(total_correct_labels, total_predictions))
                    print(sk.classification_report(total_correct_labels, total_predictions, 
                                                  target_names=['EGY','GLF','LAV','MSA','NOR'])+'\n')
                          
                          
                # Save the model checkpoint periodically.
                if (training_step % FLAGS.save_step_interval == 0):
                    checkpoint_path = os.path.join(FLAGS.train_dir,FLAGS.model_architecture + '.ckpt')
                    print('Saving to "%s-%d"' %(checkpoint_path, training_step))
                    saver.save(sess, checkpoint_path, global_step=training_step)

        print('Done training for %d epochs, %d steps each.\n' % (epoch,training_step))
        
        # Testing loop
        print('TEST')
        test_set_size = get_dataset_shape(TEST_FILE)
        print('Test set size = ', test_set_size)
        #total_accuracy = 0
        total_predictions = []
        total_correct_labels = []

        # Input images and labels.
        test_label_batch, test_feat2d_batch, test_shape_batch = inputs(TEST_FILE, batch_size=FLAGS.batch_size)
            
        for i in range(0, test_set_size, FLAGS.batch_size):
            """
            # Input images and labels.
            test_label_batch, test_feat2d_batch, test_shape_batch = inputs(
                    TEST_FILE, batch_size=FLAGS.batch_size)
            """
            # Run evaluation step
            test_accuracy, test_predictions = sess.run(
                [evaluation_step, predicted_indices])
            
            # Report 
            #batch_size = min(FLAGS.batch_size, test_set_size - i)
            #total_accuracy += (test_accuracy * batch_size) / test_set_size
            
            total_predictions.extend(test_predictions)
            total_correct_labels.extend(test_label_batch.eval())
            
        print("predictions",total_predictions)
        print("correct labels", total_correct_labels)
            
        print('Test accuracy %f, f1_score (macro) %f (N=%d)' %
                                                    (sk.accuracy_score(total_correct_labels, total_predictions),
                                        sk.f1_score(total_correct_labels, total_predictions, average="macro"),
                                                     test_set_size))
        print('Confusion Matrix:')
        print(sk.confusion_matrix(total_correct_labels, total_predictions))
        print(sk.classification_report(total_correct_labels, total_predictions, 
                                                  target_names=['EGY','GLF','LAV','MSA','NOR'])+'\n')

