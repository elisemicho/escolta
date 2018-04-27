import json
import os
import argparse
FLAGS = None

parser = argparse.ArgumentParser()

# Input parameters
parser.add_argument(
        '--model_dir',
        type=str,
        default='models',
        help='directory containing models')
parser.add_argument(
        '--model_ID',
        type=str,
        default='25',
        help='ID of the training')

parser.add_argument(
        '--train_dir',
        type=str,
        default='../../data/tfrecords',
        help='Directory with the training data.')
FLAGS, unparsed = parser.parse_known_args()
config = {
            # dir
            "data_dir": FLAGS.train_dir,
            "model_dir" : FLAGS.model_dir,
            "model_ID" : FLAGS.model_ID,
            "train_file":"mgb3_logmel_fft400_hop160_vad_cmvn_train.0.tfrecords",
            "dev_file":"mgb3_logmel_fft400_hop160_vad_cmvn_dev.0.tfrecords",
            "test_file":"mgb3_logmel_fft400_hop160_vad_cmvn_test.0.tfrecords",
            # "train_file":"train_completesample.tfrecords",
            # "dev_file":"dev_completesample.tfrecords",
            # "test_file":"dev_completesample.tfrecords",
            # architecture
            "model_architecture": "cnn_lstm",  #What model architecture to use among "simple_conv1D", "simple_conv2D", "lstm" or "cnn_lstm"
            #optimizer
            "optimizer": "GradientDescent", # options = {"Adadelta", "Adam", "GradientDescent", "Adagrad"}
            "learning_rate": 0.001,
            "ADAM_epsilon": 1e-8,
            "num_epochs": 90,  #Number of epochs to run trainer
            "batch_size": 10,
            "loss_print_interval": 100, # How often to evaluate the training results
            "eval_step_interval": 500, # How often to evaluate the training results
            "summaries_dir": "summaries", # Directory for Tensorflow summaries
            "save_step_interval": 500, # Save model checkpoint every save_steps
            # Parameters common to all models
            'num_classes': 2,
            'target_class': 2,
            'feature_width': 40,
            'dropout_prob': 0.5,
            'activation':'relu',
            'is_batch_norm': False,
            'is_mask': True,

            # Parameters for simple_conv1D
            'conv1_num_filters':200,
            'conv1_kernel_size': 8,
            'pool1_pool_size': 2,
            'pool1_strides': 2,
            'conv2_num_filters':400,
            'conv2_kernel_size': 4,
            'pool2_pool_size': 2,
            'pool2_strides': 2,

            # Parameters for LSTM
            'lstm_size': 400,
    }

if not os.path.exists(os.path.join(FLAGS.model_dir,FLAGS.model_ID)):
    os.makedirs(os.path.join(FLAGS.model_dir,FLAGS.model_ID))

json.dump(config,open(os.path.join(FLAGS.model_dir,FLAGS.model_ID,"config.json"),"w"),indent=2)

print(os.path.join(FLAGS.model_dir,FLAGS.model_ID,"config.json"))
