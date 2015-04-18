""" Minimal example for the usage of nnet.py
"""

import numpy as np
import mnist_loader
import layers
import nnet 

# Get data in correct format
train_X, train_y, val_X, val_y, test_X, test_y = mnist_loader.load_data_wrapper()

# Specifiy the layers that will form the neural network
# The layers passed to NeuralNetClassifier should be stored in a list,
# of which the last element always is of type NeuralNetworkOutputLayer
layer1 = layers.SigmoidLayer(n_in = 784, n_out = 30)
layer2 = layers.SigmoidOutputLayer(n_in = 30, n_out = 10, cost = layers.CrossEntropy())
nn_layers = [layer1, layer2]

# Initialize the classifier
# The method to estimate the parameters of the network can either be L-BFGS or SGD
net = nnet.NeuralNetClassifier(layers = nn_layers, epochs = 15, method = 'sgd', mini_batch_size = 10, 
                               learning_rate = 1.0, momentum = 0.0, L2 = 3.0/50000, verbose = True)

# Estimate the weights and validate on validation set
net.fit(train_X, train_y, val_X, val_y)
