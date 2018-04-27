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
TEST_FILE = os.path.join(model_settings["data_dir"],model_settings["test_file"])

# Tell TensorFlow that the model will be built into the default Graph.
with tf.Graph().as_default():    
        
    # Input images and labels
    label_batch, feat2d_batch, shape_batch = inputs(
            TEST_FILE, batch_size=model_settings["batch_size"])
    
    # Build a Graph that computes predictions from the model
    logits, dropout_prob = models.create_model(
            feat2d_batch, shape_batch,
            model_settings,
            model_settings["model_architecture"],
            is_training=True)

    # Define loss
    with tf.name_scope('cross_entropy'):
        loss = tf.losses.sparse_softmax_cross_entropy(
                labels=label_batch, logits=logits)   
    tf.summary.scalar('cross_entropy', loss)    
    
    # Define evaluation metrics
    predicted_indices = tf.argmax(logits, 1)   
   
    # Merge all the summaries
    merged_summaries = tf.summary.merge_all()    
    
    # Create a session for running operations in the Graph.
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
 
        # restore the graph
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=30)
        ckpt_path = os.path.dirname(FLAGS.config_dir)
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        print(ckpt_path)
        print(ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            print("restoring model: %s"%(os.path.join(ckpt_path, FLAGS.checkpoint)))
            saver.restore(sess, os.path.join(ckpt_path, FLAGS.checkpoint))
        else:
            print("None checkpoint")
                  
        # prediction loop
        print('prediction')
        test_set_size = get_dataset_shape(TEST_FILE)
        print('Test set size = ', test_set_size)
        #total_accuracy = 0
        total_predictions = []        

        # Input images and labels.
        test_label_batch, test_feat2d_batch, test_shape_batch = inputs(TEST_FILE, batch_size= 1, shuffle=False)
            
        for i in range(0, test_set_size):
            # Run evaluation step
            test_predictions = sess.run(predicted_indices)            
                        
            total_predictions.extend(test_predictions)            
            
        print("predictions",total_predictions)
        print("number of samples:", len(total_predictions))
        
