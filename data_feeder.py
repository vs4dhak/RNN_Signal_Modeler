# ----------------------------------------------------------------------------------------------------------------------
# Overview: DATA FEEDER TAKES IN A NUMBER OF CSVs, SPLITS THE DATA INTO BATCHES AND THEN FEEDS THE BATCHES INTO THE RNN
# @author: ANISH AGARWAL
# ----------------------------------------------------------------------------------------------------------------------


# GLOBAL IMPORTS
# ----------------------------------------------------------------------------------------------------------------------
import os
import sys
import numpy as np


# LOCAL IMPORTS
# ----------------------------------------------------------------------------------------------------------------------
import core_csv
from logger import *
from plotter import *


# MAIN CLASS
# ----------------------------------------------------------------------------------------------------------------------

class data_feeder:

    def __init__(self,
                 csv_path_list,  # List contains the various paths of the CSVs
                 signal_list,  # Contains which signals are of interest
                 batch_size,  # Contains how many data points are in a single batch
                 one_hot_encoding_config, # Matrix containing 3 values that define the one-hot
                                          # encoding which is generated -> [max,min,resolution]
                 downsample_factor):

        # Initializing data_feeder configuration
        self.csv_path_list = csv_path_list
        self.num_csv_files = len(self.csv_path_list)
        self.signals = signal_list
        self.num_signals = len(self.signals)
        self.batch_size = batch_size

        self.one_hot_max = one_hot_encoding_config[0]
        self.one_hot_min = one_hot_encoding_config[1]
        self.one_hot_res = one_hot_encoding_config[2]

        logger.info('CSV PATH LIST: {}'.format(self.csv_path_list))
        logger.info('NUM CSV FILES: {}'.format(self.num_csv_files))
        logger.info('SIGNALS: {}'.format(self.signals))
        logger.info('BATCH SIZE: {}'.format(self.batch_size))

        self.read_csv_files(downsample_factor)

        self.data_feed_complete = False
        self.current_csv_index = 0
        self.current_batch_num = 0
        # self.num_batches = int(self.csv_data[self.current_csv_index].shape[0] - self.batch_size) + 1
        self.num_batches = int(self.downsampled_csv_data[self.current_csv_index].shape[0] - self.batch_size) + 1
        logger.info('NUM BATCHES: {}'.format(self.num_batches))

    def read_csv_files(self, downsample_factor):

        # Reading CSV
        # ==============================================================================================================
        self.csv_data = []

        for csv_path in self.csv_path_list:
            logger.info("Reading CSV {} ...".format(csv_path))
            self.csv_data.append(core_csv.filter_df_columns(df=core_csv.read_csv(csv_path),
                                                       columns_to_keep=self.signals))
            logger.info("Done reading CSV.")

        # Downsampling data
        # ==============================================================================================================
        self.downsampled_csv_data = []

        for data in self.csv_data:
            cnt = 0
            self.downsampled_data = pd.DataFrame()
            for index,row in data.iterrows():
                if cnt % downsample_factor == 0:
                    self.downsampled_data = self.downsampled_data.append({self.signals[0] : row[self.signals[0]]}, ignore_index=True)

                cnt+=1

            self.downsampled_csv_data.append(self.downsampled_data)

        # Normalizing Data
        # ==============================================================================================================
        self.normalized_csv_data = self.downsampled_csv_data  # Comment everything out except this line to use un-normalized data for processing.

        # self.normalized_csv_data = []
        #
        # for data in self.downsampled_csv_data:
        #     mean = self.get_mean(data)
        #     variance = self.get_variance(data)
        #
        #     normalized_data = data - mean
        #     normalized_data = normalized_data / variance
        #
        #     self.normalized_csv_data.append(normalized_data)

    def get_next_batch(self,start_from_beginning = False):

        if start_from_beginning == True:
            self.data_feed_complete = False
            self.current_csv_index = 0
            self.current_batch_num = 0

        if self.data_feed_complete == False:  # If there is still data remaining to be processed

            # Getting the data corresponding to the next batch
            # ==========================================================================================================
            data_batch = []

            for i in range(self.current_batch_num,(self.current_batch_num+self.batch_size)):

                current_row = []
                for signal in self.signals:
                    current_row.append(self.normalized_csv_data[self.current_csv_index][signal].iloc[i])
                data_batch.append(current_row[0])

                last_index = i # required for one hot encoding

            # Generating the inputs to the RNN
            # ==========================================================================================================
            one_hot_vector = self.one_hot_encode(self.normalized_csv_data[self.current_csv_index][signal].iloc[last_index+1])

            next_value = self.normalized_csv_data[self.current_csv_index][signal].iloc[last_index+1]

            # Updating the cycle num and batch num
            # ==========================================================================================================
            self.current_batch_num += 1

            if self.current_batch_num == self.num_batches - 1:
                self.current_csv_index += 1

                if self.current_csv_index >= self.num_csv_files:  # Checking if all CSVs have finished processing
                    self.data_feed_complete = True
                else:
                    self.num_batches = int(self.normalized_csv_data[self.current_csv_index].shape[0] - self.batch_size) + 1
                    logger.info('NUM BATCHES: {}'.format(self.num_batches))
                    self.current_batch_num = 0

            return True, data_batch, one_hot_vector, next_value

        elif self.data_feed_complete == True:   # If all of the data has been processed

            logger.info("No more batches to process")
            return False, None, None, None

    def one_hot_encode(self,value):

        if value > self.one_hot_max:
            value = self.one_hot_max
        elif value < self.one_hot_min:
            value = self.one_hot_min

        vector_size = int((self.one_hot_max-self.one_hot_min)/self.one_hot_res)

        one_hot_vector = np.empty([vector_size])

        for index in range(0,vector_size):
            one_hot_vector[index] = np.zeros([1])

        one_position = int((value - self.one_hot_min) / self.one_hot_res)
        if one_position >= vector_size:
            one_position = vector_size-1
        one_hot_vector[one_position] = np.ones([1])

        return one_hot_vector,one_position

    def one_hot_decode(self,one_hot_pos):

        return one_hot_pos*self.one_hot_res+self.one_hot_min

    def get_mean(self,data):
        return data.mean(axis=0)

    def get_variance(self,data):
        return data.var(axis=0)


# SAMPLE CODE
# ----------------------------------------------------------------------------------------------------------------------

# if __name__ == '__main__':




