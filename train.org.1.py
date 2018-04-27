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
from load_data import *
import json
from shutil import copyfile

# Basic model parameters as external flags.
FLAGS = None

parser = argparse.ArgumentParser()

# Input parameters
parser.add_argument(
        '--config_dir',
        type=str,
        default='models/1/config.json',
        help='Directory with config file')

FLAGS, unparsed = parser.parse_known_args()

# Set parameters to convey to the model
model_settings = json.load(open(FLAGS.config_dir,"r"))

# Constants used for dealing with the tfrecords files.
TRAIN_FILE = os.path.join(model_settings["data_dir"],model_settings["train_file"])
VALIDATION_FILE = os.path.join(model_settings["data_dir"],model_settings["dev_file"])
TEST_FILE = os.path.join(model_settings["data_dir"],model_settings["test_file"])

# Tell TensorFlow that the model will be built into the default Graph.
with tf.Graph().as_default():    
        
    # Input images and labels
    label_batch, feat2d_batch, shape_batch = inputs(
            TRAIN_FILE, batch_size=model_settings["batch_size"])
    
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
    
    # Define optimizer
    with tf.name_scope('train_optimizer'):
        #self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        if "grad_clip" in model_settings.keys():
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),model_settings["grad_clip"])
        else:
            grads = tf.gradients(loss, tvars)

        if model_settings["optimizer"] == "Adadelta":
            optimizer = tf.train.AdadeltaOptimizer(model_settings["learning_rate"])
        elif model_settings["optimizer"] == "Adam":
            optimizer = tf.train.AdamOptimizer(model_settings["learning_rate"], epsilon = model_settings["ADAM_epsilon"])
        elif model_settings["optimizer"] == "GradientDescent":
            optimizer = tf.train.GradientDescentOptimizer(model_settings["learning_rate"])
        elif model_settings["optimizer"] == "Adagrad":
            optimizer = tf.train.AdagradOptimizer(model_settings["learning_rate"], initial_accumulator_value = model_settings["initial_accumulator_value"])

        train_step = optimizer.apply_gradients(zip(grads, tvars))

    # Define evaluation metrics
    predicted_indices = tf.argmax(logits, 1)
    correct_prediction = tf.equal(predicted_indices, label_batch)
    confusion_matrix = tf.confusion_matrix(
            label_batch, predicted_indices, num_classes=model_settings["num_classes"])
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)

    saver = tf.train.Saver(tf.global_variables(),max_to_keep=180)

    # Merge all the summaries
    merged_summaries = tf.summary.merge_all()
    
    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                                         tf.local_variables_initializer())

    # Create a session for running operations in the Graph.
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        # Initialize the writers for summaries
        train_writer = tf.summary.FileWriter(model_settings["summaries_dir"] + '/' + model_settings["model_architecture"] + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(model_settings["summaries_dir"] + '/' + model_settings["model_architecture"] + '/validation')
        
        # Initialize the variables (the trained variables).
        sess.run(init_op)
        checkpoint_dir = os.path.join(model_settings["model_dir"], model_settings["model_ID"])
        if tf.train.latest_checkpoint(checkpoint_dir):
            training_step = int(tf.train.latest_checkpoint(checkpoint_dir).split("-")[-1])
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            print("Training in progression")
            print("Model %s loaded"%tf.train.latest_checkpoint(checkpoint_dir))
        else:
            print("Training from scratch")
            training_step = 0
       
        # Training loop.
        print('TRAIN')
        train_set_size = get_dataset_shape(TRAIN_FILE)
        print('Train set size = ', train_set_size)
        
        for epoch in range(1, model_settings["num_epochs"] + 1):
            print('Epoch #%d:' % (epoch))
            total_loss = []
            for _step in range(1, int(train_set_size / model_settings["batch_size"]) + 1):
                training_step = training_step + 1 # global training step                
                # Run one training step of the model:
                # Train model, write summary, compute loss, compute accuracy
                _, train_summary, train_loss, train_accuracy = sess.run(
                [train_step, merged_summaries, loss, evaluation_step])

                # Report
                train_writer.add_summary(train_summary, training_step)
                total_loss.append(train_loss)

                # Evaluate on validation data  
                if (training_step % model_settings["loss_print_interval"]) == 0:

                    print('Step #%d: rate %f, loss %f, accuracy %.1f%%' %
                                                (training_step, model_settings["learning_rate"], np.mean(total_loss), 
                                                 train_accuracy * 100))                    

                if (training_step % model_settings["eval_step_interval"]) == 0:

                    valid_set_size = get_dataset_shape(VALIDATION_FILE)
                    #total_accuracy = 0
                    total_predictions = []
                    total_correct_labels = []

                    # Input images and labels.
                    valid_label_batch, valid_feat2d_batch, valid_shape_batch = inputs(VALIDATION_FILE, batch_size=model_settings["batch_size"])
                    
                    for i in range(0, valid_set_size, model_settings["batch_size"]):
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
                if (training_step % model_settings["save_step_interval"] == 0):
                    checkpoint_path = os.path.join(model_settings["model_dir"], model_settings["model_ID"], 'model.ckpt')
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
        test_label_batch, test_feat2d_batch, test_shape_batch = inputs(TEST_FILE, batch_size=model_settings["batch_size"])
            
        for i in range(0, test_set_size, model_settings["batch_size"]):
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

