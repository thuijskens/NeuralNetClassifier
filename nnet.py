# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:44:01 2015

@author: Thomas
"""

import numpy as np
import warnings

from layers import *
from sklearn.utils import shuffle, gen_batches
from sklearn.utils.extmath import safe_sparse_dot
from scipy.optimize import fmin_l_bfgs_b


class NeuralNetClassifier(object):
  def __init__(self, layers, epochs = 50, method = 'sgd', learning_rate = 1.0, 
               adaptive_learning_rate = False, adaptive_learning_rate_decay = 0.0,  momentum = 0.0, 
               mini_batch_size = 50, L1 = 0.0, L2 = 0.0, tol = 1e-6,
               rng_state = None, verbose = True):
    # Initialize class parameters
    self.num_layers = len(layers) + 1 # input layer is not an element of the list layers
    self.layers = layers
    self.cost = layers[-1].cost
    self.epochs = epochs
    self.method = method
    self.learning_rate = learning_rate
    self.adaptive_learning_rate = adaptive_learning_rate
    self.adaptive_learning_rate_decay = adaptive_learning_rate_decay
    self.momentum = momentum
    self.mini_batch_size = mini_batch_size
    self.L1 = L1
    self.L2 = L2
    self.tol = tol
    self.rng_state = rng_state
    self.verbose = verbose
    
    if not isinstance(self.layers[-1], NeuralNetworkOutputLayer):
      raise ValueError("Last element of layers must be of type NeuralNetworkOutputLayer, got type %s" % type(self.layers[-1]))

    for idx, layer in enumerate(self.layers[:-1]):
      if not isinstance(layer, NeuralNetworkLayer):
        raise ValueError("Layer %s must be of type NeuralNetworkLayer, got %s" % (idx, layer))
        
    if self.epochs <= 0:
      raise ValueError("epochs must be positive, got %s" % self.epochs)
    if method not in ['sgd', 'l-bfgs']:
      raise ValueError("method must be 'sgd' or 'l-bfgs', got %s" % self.method)
    if self.learning_rate <= 0:
      raise ValueError("learning_rate must be positive, got %s" % self.learning_rate)
    if not isinstance(self.adaptive_learning_rate, bool):
      raise ValueError("adaptive_learning_rate must be boolean, got %s" % self.rmsprop)
    if self.adaptive_learning_rate_decay < 0:
      raise ValueError("adaptive_learning_rate_decay must be positive, got %s" % self.learning_rate)
    if self.mini_batch_size <= 0 or not isinstance(self.mini_batch_size, int):
      raise ValueError("mini_batch_size must be a positive integer, got %s" % self.mini_batch_size)
    if self.L1 < 0:
      raise ValueError("L1 must be nonnegative, got %s" % self.L1)
    if self.L2 < 0:
      raise ValueError("L2 must be nonnegative, got %s" % self.L2)    
    if not isinstance(self.verbose, bool):
      raise ValueError("verbose must be a boolean, got %s" % (self.verbose))
    
  def pack(self):
    weights_stacked = np.hstack([weights.ravel() for layer in self.layers for weights in layer.weights])
    biases_stacked = np.hstack([biases.ravel() for layer in self.layers for biases in layer.biases])
    
    return np.hstack((weights_stacked, biases_stacked))
    
  def unpack(self, packed_params):
    for idx, layer in enumerate(self.layers):
      start, end, shape = self.weights_idx[idx]
      layer.weights = np.reshape(packed_params[start:end], shape)
      
      start, end = self.biases_idx[idx]
      layer.biases = np.reshape(packed_params[start:end], (1,-1))
    
  def feed_forward(self, X, compute_derivative = False):
    """ Feed forward data from the input layer through the complete network.
        Parameters:
        -----------
        X: (n_samples, n_features)-array containing the input data of the network.
        compute_derivative: Flag that indicates if the derivative of the activations should be computed
    """
    # Process the input layer
    self.layers[0].feed_forward(X, compute_derivative)
    
    # Then the hidden layers and the output layer
    for i in xrange(1, self.num_layers - 1):
      self.layers[i].feed_forward(self.layers[i - 1].output, compute_derivative)
    
    # Alternatively, but probably slower: (?)
    # for layer, layer_prev in zip(self.layers[1:], self.layers[:-1]):
    #   layer.feed_forward(layer_prev.output, compute_derivative)
    
    # return self.layers[-1].output
    
  def back_propagate(self, X, y):
    """ Backpropagate input data through the network to approximate gradients
        Parameters:
        -----------
        X: (n_samples, n_features)-array of input data
        y: (n_samples, 1)-array of target labels/values
    """
    # Initialize variables
    n_samples, n_features = X.shape
    nabla_b = [np.zeros_like(layer.biases) for layer in self.layers]
    nabla_w = [np.zeros_like(layer.weights) for layer in self.layers]
    
    # First step: feed-forward the minibatch through the network and compute the cost
    self.feed_forward(X, compute_derivative = True)
    
    cost = self.layers[-1].compute_cost(y)
    
    # Second step: compute the errors in the final layer
    delta = self.layers[-1].delta(y)    
    nabla_b[-1] = np.mean(delta, axis = 0)    
    nabla_w[-1] = safe_sparse_dot(self.layers[-2].output_dropout.T, delta) / n_samples 
    
    # Third step: backpropagate the deltas through the network
    for i in xrange(self.num_layers - 2, 0, -1):
      delta = safe_sparse_dot(delta, self.layers[i].weights.T) * self.layers[i - 1].output_derivative_dropout
 
      # Compute cost gradient
      nabla_b[i - 1] = np.mean(delta, axis = 0)
      
      if i - 1 != 0: # hidden layers
        nabla_w[i - 1] = safe_sparse_dot(self.layers[i - 2].output_dropout.T, delta) / n_samples
      else:
        nabla_w[i - 1] = safe_sparse_dot(X.T, delta) / n_samples
        
      # Regularization
      # Note: this implementation differs from http://neuralnetworksanddeeplearning.com/chap3.html
      # The factor 1/training_size is absorbed in the regularization parameter here.
      if self.L1 > 0:
        nabla_w[i - 1] += self.L1 * np.sign(self.layers[i - 1].weights)
        cost += self.L1 * np.sum(np.abs(self.layers[i - 1].weights))
      else:
        nabla_w[i - 1] += self.L2 * self.layers[i - 1].weights
        cost += self.L2 * np.sum([w ** 2 for w in self.layers[i - 1].weights]) #np.sum(np.power(self.layers[i - 1].weights, 2))
    
    return (cost, nabla_b, nabla_w)
  
  def SGD(self, train_X, train_y, val_X = None, val_y = None):
    """ Use stochastic gradient descent to estimate the weights and biases of the network
        Parameters:
        -----------
        train_X: (n_samples, n_features)-array of training data.
        train_y: (n_samples, 1)-array of training labels/values
        val_X: (n_val_samples, n_features)-array of validation data.
        val_y: (n_val_samples, n_features)-array of validation labels/values.
    """
    # Initialize parameters
    n_samples, n_features = train_X.shape

    # Start iterating over the network
    for i in xrange(self.epochs):
      prev_cost = np.inf
      cost_increase = 0
      
      # Shuffle the data and generate mini batches
      train_X, train_y = shuffle(train_X, train_y, random_state = self.rng_state) 
      mini_batches = gen_batches(n_samples, self.mini_batch_size)
      
      for mini_batch in mini_batches:
        # Back-propagate the mini batch through the network
        cost, nabla_b, nabla_w = self.back_propagate(train_X[mini_batch], train_y[mini_batch])

        # Update the biases and weights
        for idx, layer in enumerate(self.layers):
          if self.adaptive_learning_rate is True:
            # Update learning rate cache for rmsprop
            layer.update_learning_rate_cache(self.adaptive_learning_rate_decay, nabla_b[idx], nabla_w[idx])
            
            # Update learning rate
            learning_rate_b = self.learning_rate / np.sqrt(layer.cache_b + 1e-8)
            learning_rate_w = self.learning_rate / np.sqrt(layer.cache_w + 1e-8)
          else:
            # If the learning rate is not adapted, just use the original learning rate
            learning_rate_b = self.learning_rate
            learning_rate_w = self.learning_rate
          
          # Perform the gradient update step
          layer.velocity_b = self.momentum * layer.velocity_b - learning_rate_b * nabla_b[idx]  
          layer.biases += layer.velocity_b
          
          layer.velocity_w = self.momentum * layer.velocity_w - learning_rate_w * nabla_w[idx]
          layer.weights += layer.velocity_w
              
      if self.verbose:
        if val_X is not None and val_y is not None:
          print "Epoch {0}: Training cost: {1}. Validation accuracy: {2} / {3}".format(i + 1, cost, self.evaluate(val_X, val_y), len(val_y))
        else:
          print "Epoch {0}: training cost = {1}".format(i + 1, cost)
          
      if cost > prev_cost:
        cost_increase += 1
        if cost_increase >= 0.2*self.epochs:
          warnings.warn('Cost is increasing for more than 20%%'
              ' of the iterations. Consider reducing'
              ' learning_rate_init and preprocessing'
              ' your data with StandardScaler or '
              ' MinMaxScaler.'
              % cost, ConvergenceWarning)
              
      elif np.abs(cost - prev_cost) < self.tol:
        print "Epoch {0}: Algorithm has converged.".format(i + 1)
        break

      prev_cost = cost 
   
  def cost_grad_LBFGS(self, packed_params, X, y):
    """
    """
    self.unpack(packed_params)

    cost, nabla_b, nabla_w = self.back_propagate(X, y)
    nabla_b = np.hstack([b.ravel() for b in nabla_b])
    nabla_w = np.hstack([w.ravel() for w in nabla_w])

    grad = np.hstack((nabla_w, nabla_b))
    return cost, grad
    
  def fit(self, train_X, train_y, val_X = None, val_y = None):
    """ Estimate the biases and weights for the network.
        Parameters:
        -----------
        train_X: (n_samples, n_features)-array of training data.
        train_y: (n_samples, 1)-array of training labels/values
        val_X: (n_val_samples, n_features)-array of validation data.
        val_y: (n_val_samples, n_features)-array of validation labels/values.
    """
    # Ensure relevant vectors have the correct size    
    if train_y.ndim == 1:
      train_y = train_y.reshape((1, -1))
    if train_X.ndim == 1:
      train_X = train_X.reshape((1, -1))
      
    if self.method == 'sgd':
      self.SGD(train_X, train_y, val_X, val_y)
    elif self.method == 'l-bfgs':
      # Initialize lists for indices
      self.weights_idx = []
      self.biases_idx = []
      start = 0
      
      # Safe weight indices for faster packing and unpacking
      for layer in self.layers:
        end = start + layer.n_in*layer.n_out
        self.weights_idx.append((start, end, (layer.n_in, layer.n_out)))
        start = end
      
      # Safe bias indices for faster packing and unpacking
      for layer in self.layers:
        end = start + layer.n_out
        self.biases_idx.append((start, end))
        start = end
        
      # Pack coefficients
      packed_params = self.pack()
      
      # Run L-BFGS algorithm        
      optimal_parameters, cost, d = fmin_l_bfgs_b(
        x0 = packed_params,
        func = self.cost_grad_LBFGS,
        maxfun = self.epochs,
        iprint = 1 if self.verbose is True else 0,
        pgtol = self.tol,
        args = (train_X, train_y))

      # Unpack optimal parameters
      self._unpack(optimal_parameters)
  
  def predict(self, X):
    """ Predicts the outcome for every row in X
    Parameters:
    -----------
    X: (n_samples, n_features)-array of input data
    """
    self.feed_forward(X)
    return np.argmax(self.layers[-1].output, axis = 1)
    
  def evaluate(self, X, y):
    """Return the number of test inputs for which the neural
    network outputs the correct result. Note that the neural
    network's output is assumed to be the index of whichever
    neuron in the final layer has the highest activation."""
    
    # Feed the validation data through the network
    self.feed_forward(X)
    # Compute the predictions (node with maximum activation)
    test_results = [(np.argmax(inp), np.argmax(lab)) for (inp, lab) in zip(self.layers[-1].output, y)]
    
    return sum(int(inp == lab) for (inp, lab) in test_results)
    