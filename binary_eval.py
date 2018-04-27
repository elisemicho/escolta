# coding: utf-8
# In[1]:
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import math
import tensorflow as tf

import numpy as np
import sklearn.metrics as sk

import models
from load_data import *
import json

# Basic model parameters as external flags.
FLAGS = None

parser = argparse.ArgumentParser()

# Input parameters
parser.add_argument(
        '--config_dir',
        type=str,
        default='models/1/config.json',
        help='Directory with config file')

parser.add_argument(
        '--checkpoint',
        type=str,
        default='model.ckpt-50',
        help='checkpoint')

FLAGS, unparsed = parser.parse_known_args()

# Set parameters to convey to the model
model_settings = json.load(open(FLAGS.config_dir,"r"))

# Constants used for dealing with the tfrecords files.
VALID_FILE = os.path.join(model_settings["data_dir"],model_settings["dev_file"])

# Tell TensorFlow that the model will be built into the default Graph.
with tf.Graph().as_default():    
        
    # Input images and labels
    label_batch, feat2d_batch, shape_batch = balanced_binary_inputs(
            VALID_FILE, batch_size= 1, class_ = model_settings["target_class"])
    
    # Build a Graph that computes predictions from the model
    logits = models.create_model(
            feat2d_batch, shape_batch,
            model_settings,
            model_settings["model_architecture"],
            is_training=False)

    # Define loss
    with tf.name_scope('cross_entropy'):
        loss = tf.losses.sparse_softmax_cross_entropy(
                labels=label_batch, logits=logits)   
    tf.summary.scalar('cross_entropy', loss)    
    
    # Define evaluation metrics       
    predicted_indices = tf.argmax(logits, 1)
    correct_prediction = tf.equal(predicted_indices, label_batch)
    confusion_matrix = tf.confusion_matrix(
            label_batch, predicted_indices, num_classes=model_settings["num_classes"])
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
   
    # Merge all the summaries
    merged_summaries = tf.summary.merge_all()    
    
    # Create a session for running operations in the Graph.
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
 
        # restore the graph
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=30)
        ckpt_path = os.path.dirname(FLAGS.config_dir)
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        #print(ckpt_path)
        #print(ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            print("restoring model: %s"%(os.path.join(ckpt_path, FLAGS.checkpoint)))
            saver.restore(sess, os.path.join(ckpt_path, FLAGS.checkpoint))
        else:
            print("None checkpoint")
                  
        print('VALID')
        valid_set_size = get_dataset_shape(VALID_FILE)
        print('Test set size = ', valid_set_size)
        #total_accuracy = 0
        total_predictions = []
        total_correct_labels = []
                
        # Run evaluation step
        # Input images and labels        

        for i in range(0, valid_set_size):
            # Run evaluation step and capture training summaries for TensorBoard
            # with the `merged` op.                        
            valid_label_batch_, validation_summary, validation_loss, validation_predictions, accuracy, confusion_matrix_ = sess.run(
            [label_batch, merged_summaries, loss, predicted_indices, evaluation_step, confusion_matrix])
            
            total_predictions.extend(validation_predictions)
            total_correct_labels.extend(valid_label_batch_)
                             
        #print("prediction",total_correct_labels)
        #print("ground_truth",total_predictions)

        print('Validation accuracy %f, f1_score (macro) %f' %
                                                   (sk.accuracy_score(total_correct_labels, total_predictions),
                                        sk.f1_score(total_correct_labels, total_predictions, average="macro")))
        print('Confusion Matrix:')
        print(sk.confusion_matrix(total_correct_labels, total_predictions))
        print(sk.classification_report(total_correct_labels, total_predictions))
      
