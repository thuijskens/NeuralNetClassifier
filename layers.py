# -*- coding: utf-8 -*-
"""
Created on Sat Apr 04 15:23:53 2015

@author: Thomas
"""

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot

class NeuralNetworkLayer(object):
  """ NeuralNetworkLayer class
  Class for the hidden layers of a NeuralNetClassifier-object.
  """
  def __init__(self, n_in, n_out, activation, derivative, biases = None, 
               weights = None, p_dropout = 0.0, rng_state = None):
    self.n_in = n_in
    self.n_out = n_out
    self.activation = activation
    self.derivative = derivative
    self.p_dropout = p_dropout
    self.rng_state = rng_state
    
    if not (self.p_dropout < 1.0 and self.p_dropout >= 0.0):
      raise ValueError("p_dropout must be nonnegative and smaller than 1.0, got %s" % self.p_dropout)
    
    # Initialize random number generator
    rng = check_random_state(self.rng_state)
    
    # Initialize biases and weights
    # Use normalized initialization proposed by Glorot et al. (2010)
    bound = np.sqrt(6.0 / (self.n_in + self.n_out))
    if biases is None:
      self.biases = rng.uniform(low = -bound, high = bound, size = (1, self.n_out))
    else:
      self.biases = biases
    
    if weights is None:
      self.weights = rng.uniform(low = -bound, high = bound, size = (self.n_in, self.n_out))
    else:
      self.weights = weights
      
    # Initialize velocity arrays for momentum optimization
    self.velocity_b = np.zeros_like(self.biases)
    self.velocity_w = np.zeros_like(self.weights)
    
    # Initialize cache for adpative learning rate schemes
    self.cache_b = np.ones_like(self.biases)
    self.cache_w = np.ones_like(self.weights)
    
  def feed_forward(self, X, compute_derivative = False):
    """ Feed forward input data through the network
        Parameters:
        -----------
        X: (n_samples, n_features)-array of input data
        compute_derivative: Flag that indicates if the derivative of the activations should be computed
    """
    # Feedforward input through the layer
    lin_output = safe_sparse_dot(X, self.weights) + self.biases
    self.output = self.activation(lin_output)
     
    # Drop out nodes from the current layer and compute the output of the subnetwork
    rng = check_random_state(self.rng_state)
    mask = rng.binomial(1, 1 - self.p_dropout, self.n_out) / (1 - self.p_dropout)
    self.output_dropout = self.output * mask 
    
    # If necessary, compute the derivative of the activations
    if compute_derivative:
      self.output_derivative = self.derivative(lin_output)
      self.output_derivative_dropout = self.output_derivative * mask

  def update_learning_rate_cache(self, decay_rate, nabla_b, nabla_w):
    """ Update the cache for adaptive learning rate schemes
        Parameters:
        -----------
        decay_rate: The decay rate for rsmprop. decay_rate = 0 corresponds to adagrad
        nabla_b: (1, n_out)-array containing the gradient of the biases
        nabla_w: (n_in, n_out)-array containing the gradient of the weights
    """
    self.cache_b += decay_rate * self.cache_b + (1 - decay_rate) * (nabla_b ** 2)
    self.cache_w += decay_rate * self.cache_w + (1 - decay_rate) * (nabla_w ** 2)
        
class SigmoidLayer(NeuralNetworkLayer):
  """ Sigmoid layer class
  """
  def __init__(self, n_in, n_out, biases = None, weights = None, p_dropout = 0.0, rng_state = None):
    super(SigmoidLayer, self).__init__(n_in, n_out, activation = self.sigmoid, 
                                             derivative = self.sigmoid_derivative, 
                                             biases = biases, weights = weights, p_dropout = p_dropout,
                                             rng_state = rng_state)
  @staticmethod
  def sigmoid(X):
    return 1.0/(1.0 + np.exp(-X, out = X))

  def sigmoid_derivative(self, X):
    #return self.sigmoid(X)*(1.0 - self.sigmoid(X))
    return self.output*(1.0 - self.output)

class TanhLayer(NeuralNetworkLayer):
  """ Tanh layer class
  """   
  def __init__(self, n_in, n_out, biases = None, weights = None, p_dropout = 0.0, rng_state = None):
    super(TanhLayer, self).__init__(n_in, n_out, activation = self.tanh, 
                                             derivative = self.tanh_derivative, 
                                             biases = biases, weights = weights, p_dropout = p_dropout,
                                             rng_state = rng_state)
    # Recommended at http://deeplearning.net/tutorial/mlp.html: 
    # For tanh layers use -4*U[low, high] as the interval
    self.biases = 4*self.biases if biases is None else self.biases
    self.weights = 4*self.weights if weights is None else self.weights
    
  @staticmethod
  def tanh(X):
    return np.tanh(X, out = X)

  @ staticmethod
  def tanh_derivative(X):
    return 1 - X**2

class ReLULayer(NeuralNetworkLayer):
  """ ReLU layer class
  """   
  def __init__(self, n_in, n_out, biases = None, weights = None,  p_dropout = 0.0, rng_state = None):
    super(ReLULayer, self).__init__(n_in, n_out, activation = self.ReLU, 
                                             derivative = self.ReLU_derivative, 
                                             biases = biases, weights = weights, p_dropout = p_dropout,
                                             rng_state = rng_state)
  @staticmethod
  def ReLU(X):
    return np.clip(X, 0, np.finfo(X.dtype).max, out=X)

  @ staticmethod
  def ReLU_derivative(X):
    return (X > 0.0).astype(X.dtype)
    
class IdentityLayer(NeuralNetworkLayer):
  """ ReLU layer class
  """   
  def __init__(self, n_in, n_out, biases = None, weights = None, p_dropout = 0.0, rng_state = None):
    super(IdentityLayer, self).__init__(n_in, n_out, activation = self.identity, 
                                             derivative = self.identity_derivative, 
                                             biases = biases, weights = weights, p_dropout = p_dropout,
                                             rng_state = rng_state)
  @staticmethod
  def identity(X):
    return X
  
  @staticmethod 
  def identity_derivative(X):
    return 1
    
    
""" Neural network cost function classes
    The following combinations are advised, as these avoid the learning slowdown problem:
    
    1. Regression: MSE and IdentityOutputLayer
    2. Binary classification: CrossEntropy and SigmoidOutputLayer
    3. Multiclass classification: LogLikelihood and SoftmaxOutputLayer
    
    Other combinations can also be used, but may lead to numerical instabilities.
"""

class MSE(object):

  @staticmethod
  def fn(a, y):
    """Return the cost associated with an output ``a`` and desired output
    ``y``.
    """
    return 0.5*np.mean(np.linalg.norm(a-y, axis = 1)**2)
    
  @staticmethod
  def derivative(a, y):
    return (a-y)

class CrossEntropy(object):
  @staticmethod
  def fn(a, y):
    return np.sum(-y*np.log(a) - (1 - y)*np.log(1 - a)) / y.shape[0]
    
  @staticmethod
  def derivative(a, y):
    return (a - y) / (a*(1 - a))
  
class LogLikelihood(object):
  
  @staticmethod
  def fn(a, y):
    return -np.sum(y*np.log(a)) / y.shape[0]
  
  @staticmethod
  def derivative(a, y):
    # Not sure if this is right
    return -y / a
  
""" Neural network output layer classes
"""
class NeuralNetworkOutputLayer(NeuralNetworkLayer):
  def __init__(self, n_in, n_out, cost, activation, derivative, biases = None, weights = None,  
               p_dropout = 0.0, rng_state = None):   
    super(NeuralNetworkOutputLayer, self).__init__(n_in, n_out, activation = activation, 
                                             derivative = derivative, biases = biases, 
                                             weights = weights, p_dropout = p_dropout,
                                             rng_state = rng_state)
    self.cost = cost
  
  def compute_cost(self, y):
    return self.cost.fn(self.output, y)
    
  def delta(self, y):
    return self.cost.derivative(self.output_dropout, y) * self.output_derivative_dropout

class SigmoidOutputLayer(NeuralNetworkOutputLayer):
  """ SoftmaxLayer class
  Class for the output layers of a NeuralNetClassifier-object
  """
  def __init__(self, n_in, n_out, cost = MSE, biases = None, weights = None,  p_dropout = 0.0, rng_state = None):   
    super(SigmoidOutputLayer, self).__init__(n_in, n_out, cost = cost, activation = self.sigmoid, 
                                       derivative = self.sigmoid_derivative, biases = biases, 
                                       weights = weights, p_dropout = p_dropout, 
                                       rng_state = rng_state)
    
  def delta(self, y):
    # Add special cases here
    if isinstance(self.cost, CrossEntropy):
      return (self.output_dropout - y) 
    # Otherwise, use the general formula
    else:
      return super(SigmoidOutputLayer, self).delta(y)
      
  @ staticmethod    
  def sigmoid(X):
    return 1.0/(1.0 + np.exp(-X, out = X))

  def sigmoid_derivative(self, X):
    #return self.sigmoid(X)*(1.0 - self.sigmoid(X))
    return self.output*(1.0 - self.output)
    
class SoftmaxOutputLayer(NeuralNetworkOutputLayer):
  """ SoftmaxLayer class
  Class for the output layers of a NeuralNetClassifier-object
  """
  def __init__(self, n_in, n_out, cost, biases = None, weights = None, rng_state = None):   
    super(SoftmaxOutputLayer, self).__init__(n_in, n_out, cost = cost, activation = self.softmax, 
                                       derivative = self.softmax_derivative, biases = biases, 
                                       weights = weights, rng_state = rng_state)
  def delta(self, y):
    if isinstance(self.cost, LogLikelihood):
      return (self.output_dropout - y)
    else:
      return super(SoftmaxOutputLayer, self).delta(y)  
    
  @staticmethod    
  def softmax(X):
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]

    return X

  def softmax_derivative(self, X):
    return self.softmax(X) * (1.0 - self.softmax(X))
  
class IdentityOutputLayer(NeuralNetworkOutputLayer):
  """ Identity Layer class
  Class for the output layers of a NeuralNetClassifier-object
  """
  def __init__(self, n_in, n_out, cost, biases = None, weights = None,  p_dropout = 0.0, rng_state = None):   
    super(IdentityOutputLayer, self).__init__(n_in, n_out, cost = cost, activation = self.identity, 
                                        derivative = self.identity_derivative, 
                                        biases = biases, weights = weights, p_dropout = p_dropout, 
                                        rng_state = rng_state)
  # def delta(self, y):
  #   return super(IdentityOutputLayer, self).delta(y)  

  @staticmethod
  def identity(X):
    return X
  
  @staticmethod 
  def identity_derivative(X):
    return 1

  

                             