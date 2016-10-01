import numpy as np

from cs294_129.layers import *
from cs294_129.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
    self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['b2'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    hidden, hidden_cache = affine_relu_forward(X, W1, b1)
    scores, scores_cache = affine_forward(hidden, W2, b2) 
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    loss += .5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    dx, dW2, db2 = affine_backward(dscores, scores_cache)
    _, dW1, db1 = affine_relu_backward(dx, hidden_cache)
    dW1 += self.reg * W1 
    dW2 += self.reg * W2
    grads['W1'] = dW1 
    grads['W2'] = dW2
    grads['b1'] = db1
    grads['b2'] = db2
    return loss, grads

# WORKED W/ KEVIN LI 
class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    full_dims = [input_dim] + hidden_dims + [num_classes] 
    num_layers2 = range(len(full_dims) - 1)
    W = {'W' + str(i + 1): weight_scale * np.random.randn(full_dims[i], full_dims[i + 1]) for i in num_layers2}
    b = {'b' + str(i + 1): np.zeros(full_dims[i + 1]) for i in num_layers2}
    num_layers3 = range(len(full_dims) - 2)
    beta = {'beta' + str(i + 1): np.zeros(full_dims[i + 1]) for i in num_layers3}
    gamma = {'gamma' + str(i + 1):  np.ones(full_dims[i + 1]) for i in num_layers3}
    self.params.update(b)
    self.params.update(W)
    if self.use_batchnorm:
      self.params.update(beta)
      self.params.update(gamma)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
  
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################    
    input_lyr = X
    affine_caches = []
    relu_caches = []
    dropout_caches = []
    batchnorm_caches = []
    for i in range(1, self.num_layers):
  
      # Do affine 
      wi = self.params['W' + str(i)]
      bi = self.params['b' + str(i)]
      affine_lyr, affine_cache = affine_forward(input_lyr, wi, bi)
      affine_caches.append(affine_cache)
    
      # Do batchnorm 
      if self.use_batchnorm:
        beta_i = self.params['beta' + str(i)]
        gamma_i = self.params['gamma' + str(i)]
        bn_param_i = self.bn_params[i - 1]
        batch_norm_lyr, batch_norm_cache = batchnorm_forward(affine_lyr, gamma_i, beta_i, bn_param_i)
        batchnorm_caches.append(batch_norm_cache)
  
        # Do relu from batchnorm layer 
        relu_lyr, relu_cache = relu_forward(batch_norm_lyr)
        relu_caches.append(relu_cache)

      else:
        # Do relu from affine layer 
        relu_lyr, relu_cache = relu_forward(affine_lyr)
        relu_caches.append(relu_cache)

      # Do dropout 
      if self.use_dropout:
        drop_lyr, drop_cache = dropout_forward(relu_lyr, self.dropout_param)
        dropout_caches.append(drop_cache)
        input_lyr = drop_lyr
      else:
        input_lyr = relu_lyr
    
    wl = self.params['W' + str(self.num_layers)]
    bl = self.params['b' + str(self.num_layers)]
    scores, affine_cache = affine_forward(input_lyr, wl, bl)
    affine_caches.append(affine_cache)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    for i in range(1, self.num_layers + 1):
      loss += .5 * self.reg * np.sum(self.params['W' + str(i)] ** 2)

    dout, d_wl, d_bl = affine_backward(dscores, affine_caches[self.num_layers - 1])
    wl = self.params['W' + str(self.num_layers)]
    grads['W' + str(self.num_layers)] = d_wl + self.reg * wl
    grads['b' + str(self.num_layers)] = d_bl

    rev_indcs = list(range(1, self.num_layers))
    rev_indcs = rev_indcs[::-1]

    for i in rev_indcs:
  
      # Do backward dropout pass
      if self.use_dropout:
        drop_cache = dropout_caches[i - 1]
        dout = dropout_backward(dout, drop_cache)

      # Do backward relu pass
      relu_cache = relu_caches[i - 1]
      dout = relu_backward(dout, relu_cache)
  
      # Do backward batchnorm pass
      if self.use_batchnorm:
        batch_norm_cache = batchnorm_caches[i - 1]
        dout, dgamma_i, dbeta_i = batchnorm_backward(dout, batch_norm_cache)
        grads['gamma' + str(i)] = dgamma_i
        grads['beta' + str(i)] = dbeta_i

      # Do backward affine pass
      affine_cache = affine_caches[i - 1]
      dout, d_wi, d_bi = affine_backward(dout, affine_cache)
      wi = self.params['W' + str(i)]
      grads['W' + str(i)] = d_wi + self.reg * wi
      grads['b' + str(i)] = d_bi 
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
