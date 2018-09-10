# ----------------------------------------------------------------------------------------------------------------------
# Overview: ABSTRACTS OUT TENSORFLOW RNN FUNCTIONALITY
# @author: ANISH AGARWAL
# ----------------------------------------------------------------------------------------------------------------------

# GLOBAL IMPORTS
# ----------------------------------------------------------------------------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import time
import tensorflow as tf
from tensorflow.contrib import rnn
from datetime import datetime
import matplotlib.pyplot as plt
import tkinter as tk


# LOCAL IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

from logger import *
from plotter import *
from data_feeder import *


# RNN CLASS
# ----------------------------------------------------------------------------------------------------------------------
class recurrent_neural_network:

    def __init__(self,
                 batch_size,
                 one_hot_encoding_config,
                 rnn_size,
                 learning_rate,
                 model_directory,
                 plot=False):

        # Setting configuration variables
        # --------------------------------------------------------------------------------------------------------------
        self.batch_size = batch_size
        self.one_hot_max = one_hot_encoding_config[0]
        self.one_hot_min = one_hot_encoding_config[1]
        self.one_hot_res = one_hot_encoding_config[2]
        self.one_hot_size = int((self.one_hot_max-self.one_hot_min)/self.one_hot_res)
        self.rnn_size = rnn_size
        self.model_directory = model_directory
        self.learning_rate = learning_rate
        logger.info("Batch Size: {}".format(self.batch_size))
        logger.info("One Hot Max: {}".format(self.one_hot_max))
        logger.info("One Hot Min: {}".format(self.one_hot_min))
        logger.info("One Hot Res: {}".format(self.one_hot_res))
        logger.info("One Hot Size: {}".format(self.one_hot_size))
        logger.info("RNN Size: {}".format(self.rnn_size))
        logger.info("Model Directory: {}".format(self.model_directory))
        logger.info("Learning Rate: {}".format(self.learning_rate))

        # Setting up the TensorFlow Graph
        # --------------------------------------------------------------------------------------------------------------
        logger.info("Initializing Neural Network")

        # x is the input matrix
        self.x_m = batch_size  # num rows -> equals the number of points in data batch (e.g. last 10 values in signal)
        self.x_n = 1  # num columns -> equals the number of signals (not tested with multiple signals)

        # y is the label matrix (used to correct the prediction)
        self.y_m = self.one_hot_size  # num rows

        # w is the weight matrix used in the softmax
        self.w_m = rnn_size  # num rows
        self.w_n = self.one_hot_size  # num cols

        # b is the bias matrix used in the softmax
        self.b_m = self.one_hot_size  # num rows

        # Setting up the inputs to the graph
        self.x = tf.placeholder("float", [None, self.x_m, self.x_n], name='x')
        self.y = tf.placeholder("float", [None, self.y_m], name='y')

        # Setting up the weights and biases of the graph
        self.w = tf.Variable(tf.random_normal([self.w_m, self.w_n]), name='w')
        self.b = tf.Variable(tf.random_normal([self.b_m]), name='b')

        # Reshaping x from an x_m-cols by x_n-rows matrix to a 1-col by x_m-rows matrix
        self.x_r = tf.reshape(self.x, [-1, batch_size])  # Flattening the input matrix into a single row

        # Then splitting x into x_m number of separate arrays/matrices
        self.x_s = tf.split(self.x_r, batch_size, 1)

        # Setting up the rnn cell
        self.rnn_cell = rnn.BasicLSTMCell(rnn_size)

        # Outputs contains the output from every rnn cell
        # States contains the state from every rnn cell
        self.outputs , self.states = rnn.static_rnn(self.rnn_cell, self.x_s, dtype=tf.float32)

        # Pred is the prediction after softmax
        self.pred = tf.matmul(self.outputs[-1], self.w) + self.b

        # Checking whether the prediction was correct
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))

        # Running backpropagation
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Evaluation of the model on the current iteration
        self.prediction_index = tf.argmax(self.pred, 1)
        self.pred_correct = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.pred_correct, tf.float32))

        # Initializing the saver
        self.saver = tf.train.Saver()

        logger.info("Successfully initialized Neural Network")

    def train(self,
              data,
              mode = None,
              num_epochs=None,
              accuracy=None,
              model_name=None,
              load_model=False,
              save_model=True):

        # Error checking
        # --------------------------------------------------------------------------------------------------------------
        if mode != 'accuracy' and mode != 'epochs':
            logger.critical("One of num_epochs or accuracy must be defined for mode")
        else:

            # Setting up the graph
            graph = tf.global_variables_initializer()
            with tf.Session() as session:
                session.run(graph)

                # Loading the model
                # ------------------------------------------------------------------------------------------------------
                if load_model == True:

                    # Getting model file
                    checkpoint = tf.train.get_checkpoint_state(self.model_directory)
                    logger.info("Model File: {}".format(checkpoint.model_checkpoint_path))

                    # Getting epoch number
                    epoch = str(checkpoint.model_checkpoint_path).split('\\')
                    epoch = epoch[len(epoch) - 1]
                    epoch = epoch.split('-')
                    epoch = int(epoch[len(epoch) - 1])
                    logger.info("Epoch: {}".format(epoch))

                    # Loading checkpoint
                    if checkpoint and checkpoint.model_checkpoint_path:
                        checkpoint_path = str(checkpoint.model_checkpoint_path)  # convert the unicode to string
                        self.saver.restore(session, os.path.join(os.getcwd(), checkpoint_path))
                        print("Loaded model")
                    else:
                        print("Model loading failed")
                else:
                    epoch = 0
                # ------------------------------------------------------------------------------------------------------

                # Epoch Mode
                # ------------------------------------------------------------------------------------------------------
                if mode == 'epochs':

                    batch_num = 0
                    num_epochs_total = epoch + num_epochs

                    while epoch < num_epochs_total:

                        # Getting the next batch
                        batches_remaining, data_batch, one_hot_vector, current_value = data.get_next_batch(False)
                        batch_num += 1
                        if batches_remaining == False: # If all of the batches have already been processed
                            epoch += 1
                            batch_num = 0
                            if save_model == True:
                                if os.path.exists(model_directory) == False:
                                    os.makedirs(model_directory)
                                self.saver.save(session, os.path.join(self.model_directory, model_name), global_step=epoch)
                                logger.info("Saved model at epoch: {}".format(epoch))
                                batches_remaining, data_batch, one_hot_vector, current_value = data.get_next_batch(True)
                        data_batch = [[[x] for x in data_batch]]

                        # Running the RNN Network
                        w, b, opt, pred_index, acc, pred_corr, cst = session.run(
                                                                                    [self.w,
                                                                                     self.b,
                                                                                     self.optimizer,
                                                                                     self.prediction_index,
                                                                                     self.accuracy,
                                                                                     self.pred_correct,
                                                                                     self.cost],
                                                                                    feed_dict={
                                                                                        self.x: data_batch,
                                                                                        self.y: [one_hot_vector[0]]
                                                                                    }
                                                                                )

                        if batch_num % 25 == 0:
                            print("Batch Number: {} | Actual: {} | Pred: {} | Error: {} | Learning Rate: {}"
                                  .format(batch_num, "%.2f" % current_value,
                                          "%.2f" % (pred_index[0] * self.one_hot_res + self.one_hot_min),
                                          "%.2f" % cst, self.learning_rate))
                # ------------------------------------------------------------------------------------------------------


                # Epoch Mode
                # ------------------------------------------------------------------------------------------------------
                if mode == 'accuracy':

                    print("Accuracy mode")
                # ------------------------------------------------------------------------------------------------------

    def run(self,
            data,
            load_model=True,
            plot_flag=False,
            save_flag=False,
            save_directory=None):

        # Setting up the graph
        graph = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(graph)

            # Loading the model
            # ----------------------------------------------------------------------------------------------------------
            if load_model == True:

                # Getting model file
                checkpoint = tf.train.get_checkpoint_state(self.model_directory)
                logger.info("Model File: {}".format(checkpoint.model_checkpoint_path))

                # Getting epoch number
                epoch = str(checkpoint.model_checkpoint_path).split('\\')
                epoch = epoch[len(epoch) - 1]
                epoch = epoch.split('-')
                epoch = int(epoch[len(epoch) - 1])
                logger.info("Epochs: {}".format(epoch))

                # Loading checkpoint
                if checkpoint and checkpoint.model_checkpoint_path:
                    checkpoint_path = str(checkpoint.model_checkpoint_path)  # convert the unicode to string
                    self.saver.restore(session, os.path.join(os.getcwd(), checkpoint_path))
                    print("Loaded model")
                else:
                    print("Model loading failed")
            # ----------------------------------------------------------------------------------------------------------

            # Setting up the plot
            # ----------------------------------------------------------------------------------------------------------
            plot_data = pd.DataFrame()

            batch_num = 0
            batches_remaining = True
            initial_batch = True
            while batches_remaining == True:

                # Getting the next batch
                if initial_batch == True:
                    batches_remaining, data_batch, one_hot_vector, current_value = data.get_next_batch(True)
                    initial_batch = False
                else:
                    batches_remaining, data_batch, one_hot_vector, current_value = data.get_next_batch(False)

                if batches_remaining == True:

                    batch_num += 1
                    data_batch = [[[x] for x in data_batch]]

                    # Running one iteration of the network
                    pred_index = session.run(
                        [self.prediction_index],
                        feed_dict={
                            self.x: data_batch,
                            self.y: [one_hot_vector[0]]
                        }
                    )

                    if batch_num % 25 == 0:
                        print("Batch Number: {} | Actual: {} | Pred: {}"
                              .format(batch_num, "%.2f" % current_value,
                                      pred_index[0] * self.one_hot_res + self.one_hot_min))

                    # Updating plotting data
                    data_point = pd.DataFrame({"x": batch_num,
                                               "Actual": current_value,
                                               "Pred": pred_index[0] * self.one_hot_res + self.one_hot_min})
                    plot_data = plot_data.append(data_point, ignore_index=True)

                elif batches_remaining == False:

                    logger.info('Completed running RNN')

                    # Plotting predictions vs actual
                    # --------------------------------------------------------------------------------------------------
                    if (plot_flag == True):

                        logger.info("Plotting graph")

                        logger.info("Plot data columns: {}".format(plot_data.columns.values))

                        p = plot(plot_data)
                        p.create_plot(x_signal='x', y_signals=['Actual', 'Pred'], show_plot=False)
                        p.set_plot_info(title='Fourier Series Model', x_label='X', x_unit='N/A',
                                        y_label='F(X)', y_unit='N/A')

                        if save_flag == True:
                            if os.path.exists(save_directory) == False:
                                os.makedirs(save_directory)
                            p.save_plot(save_directory, r'epoch_{}'.format(epoch))
                        p.close_plot()

                        logger.info("Done plotting")

                    if save_flag == True:
                        if os.path.exists(save_directory) == False:
                            os.makedirs(save_directory)
                        plot_data.to_csv(save_directory + r'\epoch_{}.csv'.format(epoch))
                        logger.info("Saved prediction to {}".format(save_directory + r'\epoch_{}.csv'.format(epoch)))


# SAMPLE CODE
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    batch_size_list = [10,20,30,40,50]
    rnn_size_list = [50,100,150,200,250]
    num_epochs_list = [5,10,15,20,25,50]
    downsample_factor_list = [2,4,6,8,10]
    # initial_learning_rate = 0.01

    for batch_size in batch_size_list:
        for rnn_size in rnn_size_list:
            for num_epochs in num_epochs_list:
                for downsample_factor in downsample_factor_list:

                    # Initializing Parameters
                    # ==================================================================================================================
                    # batch_size = 50
                    one_hot_encoding_config = [3, -3, 0.01]
                    # rnn_size = 250
                    initial_learning_rate = 0.01 #0.0001
                    # num_epochs = 5
                    # downsample_factor = 10 # 3

                    # signal = ['BMS_packVoltage']
                    signal = ['y']#['BMS_packVoltage']



                    # Specifying the dataset which is to be processed
                    # ==================================================================================================================

                    data_path = r'C:\Users\anagarwal\Desktop\RNN\Data'
                    function = "2sin(08x)+02cos(2x)+sin(02x+2)+sin(001x)"#"PTCE - 2018-08-15 - 07;07;23"
                    file_extension = ".csv"
                    csv_paths = [data_path + r'\{}{}'.format(function,file_extension)]

                    model_directory = r'C:\Users\anagarwal\Desktop\RNN\Models\{}\RNNSize-{}_BatchSize-{}'.format(function,rnn_size, batch_size)

                    # Saving the training configuration
                    # ==================================================================================================================
                    train_config = dict()
                    train_config['batch_size'] = batch_size
                    train_config['one_hot_encoding'] = ",".join(str(x) for x in one_hot_encoding_config)
                    train_config['rnn_size'] = rnn_size
                    train_config['initial_learning_rate'] = initial_learning_rate
                    train_config['num_epochs'] = num_epochs
                    train_config['signal'] = signal
                    train_config['downsample_factor'] = downsample_factor
                    train_config['training_data'] = ",".join(str(x) for x in csv_paths)

                    # TODO: Saving the training config
                    # ==================================================================================================================
                    # train_config_df = pd.DataFrame(train_config, index=[0])
                    # logger.info(train_config_df)

                    # Initializing the data feeder
                    # ==================================================================================================================
                    data = data_feeder(csv_path_list=csv_paths,
                                       signal_list=signal,
                                       batch_size=batch_size,
                                       one_hot_encoding_config=one_hot_encoding_config,
                                       downsample_factor=downsample_factor)


                    # Initializing the RNN
                    # ==================================================================================================================
                    r_nn = recurrent_neural_network(batch_size=batch_size,
                                                    one_hot_encoding_config=one_hot_encoding_config,
                                                    rnn_size=rnn_size,
                                                    learning_rate=initial_learning_rate,
                                                    model_directory=model_directory)

                    # Training the RNN
                    # ==================================================================================================================
                    load_model_flag = False

                    while r_nn.learning_rate > 0.0000000001:

                        if load_model_flag == False:
                            load_model = False
                            load_model_flag = True
                        else:
                            load_model = True

                        r_nn.train(data=data, mode='epochs', num_epochs=num_epochs, load_model=load_model,
                                  save_model=True, model_name='signal-{}-'
                                                              'batchsize-{}-'
                                                              'rnnsize-{}'
                                                              .format(signal[0],
                                                                      batch_size,
                                                                      rnn_size))

                        r_nn.run(data, load_model=True, plot_flag=True, save_flag=True,
                                 save_directory=r'C:\Users\anagarwal\Desktop\RNN\Plots\{}\BS{}-RS{}-NE{}-DF{}'.format(
                                     function,batch_size,rnn_size,num_epochs,downsample_factor))

                        # num_epochs = num_epochs*2
                        r_nn.learning_rate = r_nn.learning_rate / 10
                        r_nn.optimizer = tf.train.AdamOptimizer(learning_rate=r_nn.learning_rate).minimize(r_nn.cost)

                    # Running the RNN
                    # ==================================================================================================================
                    r_nn.run(data,load_model=True,plot_flag=True,save_flag=True,
                             save_directory=r'C:\Users\anagarwal\Desktop\RNN\Plots\{}\BS{}-RS{}-NE{}-DF{}'.format(
                                 function,batch_size, rnn_size, num_epochs, downsample_factor))

                    tf.reset_default_graph()
                    del r_nn

