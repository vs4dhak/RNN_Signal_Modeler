# ----------------------------------------------------------------------------------------------------------------------
# Overview: GENERATES DATA INTO A CSV FILE DEPENDING ON THE FUNCTION SPECIFIED
# @author: ANISH AGARWAL
# ----------------------------------------------------------------------------------------------------------------------


# GLOBAL IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

import os
import math
import pandas as pd


# FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------

def generate_data(file_name):

    if not os.path.exists(r'/home/veda/git/RNN_Signal_Modeler/data'):
        os.makedirs(r'/home/veda/git/RNN_Signal_Modeler/data')

    data = pd.DataFrame()

    for step in range(0,20000):

        x = 0.001*step
        y = 2*math.sin(0.08*x) + 0.2*math.cos(2*x) + math.sin(0.02*x+2) + math.sin(0.001*x)

        data_point = pd.DataFrame({ "x": [x], "y": [y] })
        data = data.append(data_point, ignore_index=True)

        print("x: {} | y: {}".format(x,y))

    data.to_csv(os.path.join('/home/veda/git/RNN_Signal_Modeler','data/{}.csv'.format(file_name)))


# MAIN
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    generate_data("2sin(08x)+02cos(2x)+sin(02x+2)+sin(001x)")


