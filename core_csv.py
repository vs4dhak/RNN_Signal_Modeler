# ----------------------------------------------------------------------------------------------------------------------
# Overview: READS A CSV FILE AND RETURNS A PANDAS DATAFRAME
# @author: ANISH AGARWAL
# ----------------------------------------------------------------------------------------------------------------------


# GLOBAL IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from datetime import datetime
from logger import *


# FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------

def read_csv(csv_path,add_time_colums=False):

    csv_data = pd.read_csv(csv_path, sep=',')
    # logger.debug("CSV Data Shape (raw): {}".format(csv_data.shape))
    # csv_data['Time Sec'] = np.floor(csv_data['Time Sec']).astype(int)
    # csv_data = csv_data.drop_duplicates(subset=['Time Sec'], keep="first")
    # logger.debug("CSV Data Shape (dropped duplicates): {}".format(csv_data.shape))

    if add_time_colums == True:
        # Adding elapsed_time and elapsed_time_hr columns
        csv_data.loc[:, 'Time Stamp'] = [datetime.strptime(x, '%d-%b-%y %H: %M: %S.%f') for x in csv_data['Time Stamp'].values]
        csv_data.loc[:, 'elapsed_time'] = csv_data['Time Stamp'] - csv_data['Time Stamp'][0]
        csv_data.loc[:, 'elapsed_time_hr'] = csv_data.elapsed_time / np.timedelta64(1, 'h')

    return csv_data

def filter_df_columns(df, columns_to_keep=[]):

    new_df = df
    columns_to_remove = list(df)

    for column in columns_to_keep:
        columns_to_remove.remove(column)

    new_df = new_df.drop(columns_to_remove,axis=1)

    return new_df