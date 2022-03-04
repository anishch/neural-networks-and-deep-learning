"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

# e.g. net = Network([2, 3, 1])
# -> representing 2 neurons on first layer, 3 second, 1 final.

class Network(object):

    def __init__(self, sizes):

        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes) # number of layers
        self.sizes = sizes # same array as params

        # ----------------------------------------------------------------------------------

        # If positive int_like arguments are provided, randn generates an array of shape (d0, d1, ..., dn),
        # filled with random floats sampled from a univariate “normal” (Gaussian) distribution of mean 0 and variance 1.
        # A single float randomly sampled from the distribution is returned if no argument is provided.
        # Reference on understanding randn: https://www.geeksforgeeks.org/numpy-random-randn-python/

        # 2D array of y values of random numbers each size 1 (redundant) & y representing layer
        # so in essence what's going on is, from 2nd layer onwards, you have
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # array/vector of random biases corresponding to layers 2-n (incides 1 to n-1)
        # bias can be visualized perceptron-wise as threshold.
        # Layer 2 [b_1, b_2, b_3 ...]; Layer 3 [b_a, b_b, b_c, b_d, ...]; Layer 4 [b_alph, b_bet, b_gamm]...

        # ---------------------------------------------------------------------------------------------------------------------

        # 2D array of y values of random numbers each size x, where y represents 1st layer, and x represents second layer
        # combination is basically the "handshake rule" of probability n(n-1) /2 except /2 isn't requried since we're starting array
        # only till y ends.
        # Hence each represents a pair between each perceptron (or actually sigmoid neuron) structurally accessed in the array by layer from which
        # "connection" between sigmoid represents.

        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        # pairing

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
            # in more mathematical terms: a′=σ(wa+b)
            # [σ representing sigmoid function = 1/(1 + e^(-z)]
        return a

    # implements stochastic gradient descent (SGD)
    # @params
    # [training_data : list of tuples (x, y) representing the training inputs and corresponding desired outputs]
    # [epochs : the number of times we want to utilize SGD (i.e. number of sample sizes)
    # [mini_batch_size : size of mini-batches to use while sampling
    # [eta : learning rate, η]
    # [test_data=None]
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        # If the optional argument test_data is supplied,
        # then the program will evaluate the network after each epoch of training,
        # and print out partial progress.
        n = len(training_data)
        for j in xrange(epochs): # in each epoch
            random.shuffle(training_data) # randomly shuffling training data
            mini_batches = [
                training_data[k:k+mini_batch_size] # partitioning into mini_batches
                for k in xrange(0, n, mini_batch_size)] # k is the mini_batch_size, split by n,
                # i.e., k is 0, k+mini_batch_size + 1, etc.
            for mini_batch in mini_batches: # for all mini_batches apply a single step of gradient descent
                self.update_mini_batch(mini_batch, eta)
            if test_data: # once done, evaluating probability?
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)


    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
             # backpropagation algorithm -> fast way of computing the gradient of the cost function.
            # So update_mini_batch works simply by computing these gradients for every training example
            # in the mini_batch, and then updating self.weights and self.biases appropriately.
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
