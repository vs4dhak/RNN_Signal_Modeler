# ----------------------------------------------------------------------------------------------------------------------
# Overview: GENERAL CLASS WHICH ABSTRACTS OUT PLOTTING FUNCTIONALITY FROM MATPLOTLIB
# @author: ANISH AGARWAL
# ----------------------------------------------------------------------------------------------------------------------


# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------
import os
import sys
import matplotlib.pyplot as plt
import logging
import pandas as pd
import configparser

from logger import *


# MAIN CLASS
# ----------------------------------------------------------------------------------------------------------------------

class plot:

    # This method initializes the data and the plotting object
    def __init__(self,data_frame):

        self.data = data_frame
        self.fig,self.axes = plt.subplots(1,1)

    # This method plots the data
    def create_plot(self,x_signal=None,y_signals=[],dashes=[],legends=[],colors=[],show_plot=False):

        num_signals = len(y_signals)
        logger.debug("Num Signals: {}".format(num_signals))

        if (x_signal == None) or (num_signals <= 0):  # If plotting signals are not specified

            logger.critical("CANNOT CREATE PLOT: No signals were specified.")

        else:

            if len(dashes) == 0:  # No dashes
                for i in range(num_signals):
                    dashes.append([1,1])
            else:
                logger.debug("Dash: {}".format(dashes))

            if len(legends) == 0:  # No legends
                for i in range(num_signals):
                    legends.append(False)
            else:
                logger.debug("Legend: {}".format(legends))

            logger.debug("Num Dashes: {}, Num Legends: {}, Num Colors: {}".format(len(dashes),len(legends),len(colors)))

            for i in range(0, num_signals):  # Iterating through the signals
                if colors:  # If colors are specified
                    self.data.plot(x=x_signal, y=y_signals[i], color=colors[i], dashes=dashes[i],
                                   legend=False, ax=self.axes)
                else:  # If colors are not specified
                    self.data.plot(x=x_signal, y=y_signals[i], dashes=dashes[i], legend=False,
                                   ax=self.axes)

            if legends:  # Setting legend position
                self.axes.legend(bbox_to_anchor=(1, 1), loc=2)

            if show_plot == True:
                self.fig.show()
                print("Press enter to close plot and continue.")
                input()

    # This method sets the title, x_label and y_label of the plot
    def set_plot_info(self,title=None,x_label=None,x_unit=None,y_label=None,y_unit=None):

        logger.debug('title: {} | x_label: {} | x_unit: {} | y_label: {} | y_unit: {}'.format(
                      title,x_label,x_unit,y_label,y_unit))

        # Setting the title
        if title == None:
            logger.warning("NO TITLE SPECIFIED: Will not add a title.")
        else:
            self.axes.set_title('{}\n\n'.format(title))

        # Setting the x-axis label
        if (x_label == None):
            logger.warning("THE X LABEL WAS NOT SPECIFIED: Will not add the x axis label")
        elif (x_unit == None):
            logger.warning("THE X LABEL UNIT WAS NOT SPECIFIED: Will not add the x axis label unit")
            self.axes.set_xlabel('\n{}\n\n'.format(x_label))
        else:
            logger.debug('Set both x_label and x_unit')
            self.axes.set_xlabel('\n{} ({})\n\n'.format(x_label, x_unit))

        # Setting the y-axis label
        if (y_label == None):
            logger.warning("THE Y LABEL WAS NOT SPECIFIED: Will not add the y axis label")
        elif (y_unit == None):
            logger.warning("THE Y LABEL UNIT WAS NOT SPECIFIED: Will not add the y axis label unit")
            self.axes.set_ylabel('\n{}\n'.format(y_label))
        else:
            logger.debug('Set both y_label and y_unit')
            self.axes.set_ylabel('\n{} ({})\n'.format(y_label, y_unit))

    # This method saves the plot with the specified file name and directory
    def save_plot(self,directory=None,file_name=None):

        if directory == None:
            logger.critical('OUTPUT DIRECTORY WAS NOT SPECIFIED: Will not save plot')
        elif file_name == None:
            logger.critical('FILE NAME WAS NOT SPECIFIED: Will not save plot')
        else:
            self.fig.savefig(os.path.join(directory, '{}.png'.format(file_name)), bbox_inches='tight')
            logger.info('Created plot {}\{}.png'.format(directory,file_name))

    # This method closes the plot
    def close_plot(self):
        plt.close(self.fig)


# SAMPLE CODE
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Creating a sample DF
    list = []
    list.append([1, 2])
    list.append([2, 10])
    list.append([3, 7])
    list.append([4, 5])
    list.append([5, 11])
    df = pd.DataFrame(list,columns=['x','y'])

    plot_obj = plot(df)

    plot_obj.create_plot(x_signal='x',y_signals=['y'],show_plot=False)

    plot_obj.set_plot_info(title='Test Graph', x_label='X_Label', x_unit='Second', y_label='Y_label', y_unit='Meters')

    plot_obj.save_plot(directory=r'C:\Users\anagarwal\Desktop',file_name='test')

    plot_obj.close_plot()
