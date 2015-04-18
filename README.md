# NeuralNetClassifier
Python implementation of a (flexible) neural network classifier. 

Implementation of the classifier is enspired by code posted on:
1. https://github.com/mnielsen/neural-networks-and-deep-learning
2. https://github.com/IssamLaradji/NeuralNetworks

See the script minimal_example.py for a very small example. This script will be expanded with more examples in the future.

# Guidelines
Each NeuralNetClassifier needs a list containing the different layers making up the neural network. The items of this list need to be of type NeuralNetworkLayer. 

You can make your own layers using this class, or use one of the predefined layer types:
- SigmoidLayer
- TanhLayer
- ReLUlayer
- IdentityLayer

For the output layer, one must use the derived NeuralNetworkOutputLayer class which accepts a (custom) cost function as an extra argument. Each layer (except the input layer) supports drop-out regularization. 

The parameters for the network can be estimated by using the SGD or L-BFGS algorithm. L1 and L2 regularization are implemented, as well as RMSprop but the latter has not been thoroughly tested yet.
