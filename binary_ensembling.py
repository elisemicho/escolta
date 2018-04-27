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

class ImportGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, config_dir, checkpoint):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph,config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
        model_settings = json.load(open(config_dir,"r"))
        # restore the graph
        
        with self.graph.as_default():
            VALID_FILE = os.path.join(model_settings["data_dir"], "mgb3_logmel_fft400_hop160_vad_cmvn_adi-test.0.tfrecords")
            self.valid_set_size = get_dataset_shape(VALID_FILE)
            print('Test set size = ', self.valid_set_size)
            # Input images and labels
            self.label_batch, feat2d_batch, shape_batch = binary_inputs(VALID_FILE, batch_size=1, class_ = model_settings['target_class'], shuffle=False)
            # Build a Graph that computes predictions from the model
            logits = models.create_model(feat2d_batch, shape_batch, model_settings, model_settings["model_architecture"], is_training=False)
            saver = tf.train.Saver(tf.global_variables())
            ckpt_path = os.path.dirname(config_dir)
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            print(ckpt_path)
            #print(ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                print("restoring model: %s"%(os.path.join(ckpt_path, checkpoint)))
                saver.restore(self.sess, os.path.join(ckpt_path, checkpoint))
            else:
                print("None checkpoint")
                 
            self.predicted_prob = tf.nn.softmax(logits)[:,1]
    
    def run(self):
        """ Running the activation operation previously imported """
        # The 'x' corresponds to name of input placeholder
        predicted_prob_ = []
        labels = []
        for i in range(self.valid_set_size):            
            prob_,label_ = self.sess.run([self.predicted_prob,self.label_batch])
            predicted_prob_.extend(prob_)
            labels.extend(label_)
        return predicted_prob_, labels

predict_probs = []
checkpoints = ["model.ckpt-84000","model.ckpt-98500","model.ckpt-21000","model.ckpt-48500","model.ckpt-113500"]
config_dirs = ["models/b_1/config.json","models/b_2/config.json","models/b_3/config.json","models/b_4/config.json","models/b_5/config.json"]

for i in range(4):
    model_ = ImportGraph(config_dirs[i], checkpoints[i])
    predicted_prob , label_ = model_.run()
    #predict_probs.append(predicted_prob)
    #total_correct_labels = label_
    statistic = {"probs": predicted_prob, "truth_labels": label_ }
    np.savez("prediction/statistics.%d.npz"%i, **statistic)

"""
predict_probs = np.array(predict_probs)
total_predictions = []
for j in range(predict_probs.shape[1]):
    probs = []
    for i in range(5):
        probs.append(predict_probs[i,j])
    total_predictions.append(max(predict_probs))

print('Validation accuracy %f, f1_score (macro) %f' % (sk.accuracy_score(total_correct_labels, total_predictions),
                                        sk.f1_score(total_correct_labels, total_predictions, average="macro")))
print('Confusion Matrix:')
print(sk.confusion_matrix(total_correct_labels, total_predictions))
print(sk.classification_report(total_correct_labels, total_predictions,
                                                  target_names=['EGY','GLF','LAV','MSA','NOR'])+'\n')
"""
        
