# Recurrent Neural Network Signal Modeler

A general purpose signal modeler built using the Tensorflow Recurrent Neural Network module.

https://medium.com/towards-artificial-intelligence/signal-modelling-using-recurrent-neural-networks-d832f0c50503



![Alt Text](https://github.com/veda-s4dhak/RNN_Signal_Modeler/blob/master/RNN_Signal_Modeler.gif)

# Instructions

1) Define the signal which you would like to model in data_generator.py on line 25
2) Set the directory in which to save the generated data in data_generator.py on line 17
2) Run the data generator (data_generator.py) to generate the signal will be modelled 
4) Set the base directory for the recurrent neural network in rnn_signal_modeller.py on line 30
5) Set the hyper-parameters of the network  in rnn_signal_modeller.py in line 335-339
6) Run the recurrent neural network (rnn_signal_modeller.py) and wait (execution time depends on the hyper-parameters)
7) Experiment!

Note: By default rnn_signal_modeller will generate intermediate plots across the iterations in base_directory/plots/
