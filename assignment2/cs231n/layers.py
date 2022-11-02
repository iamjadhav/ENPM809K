from builtins import range
from sys import stderr
import numpy as np
import sys


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    n = x.shape[0] # mini-batch size
    X_mod = np.reshape(x, (n, -1))

    # Multiplying with the weights and adding the bias term
    out = X_mod@w + b
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    n = x.shape[0]
    X_mod = np.reshape(x, (n, -1))

    # Derivative wrt x, in out = xw + b, w on LHS with its transpose
    dx = dout@w.T
    dx = dx.reshape(x.shape)
    # Similarly with dw
    dw = X_mod.T@dout
    # Since bias is Mx1 we add up the biases across rows in db 
    db = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # When x>0, the derivative is 1, hence
    out = np.where(x < 0, 0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # The derivatives of negative values removed we take dout grads of positives
    dx = dout * (x > 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # N = number of samples
    num_train = x.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Shifted scores to avoid overflow
    scores_sum = np.max(x, axis=1, keepdims=True)
    scores = x - scores_sum
    # Unnormalized probabilities
    p = np.sum(np.exp(scores), axis=1, keepdims=True)
    # To get Correct class probabities
    log_p = scores - np.log(p)
    # Normalizing the probabilities
    prob_all = np.exp(log_p)
    # Loss over all the dataset
    loss = -np.sum(log_p[np.arange(num_train),y])

    # Gradient of the true classes
    dx = prob_all.copy()
    dx[np.arange(num_train),y] -= 1
    # Increased score resulting in decreased gradient and subsequent loss

    # Loss average
    loss = loss / num_train
    # Average gradient over the dataset is average
    dx = dx / num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Average of each column in X
        mean = np.mean(x, axis=0)
        # Variance for each column in X
        var = np.var(x, axis=0)
        # Standard Deviation with eps which avoids division by 0
        standard_variation = np.sqrt(var + eps)
        # Normalized X over the batch; Scale and Shift factors for the output 
        # to allow the net to squash the range if it wants to
        norm_x = (x - mean) / standard_variation
        out = gamma * norm_x + beta
        # Running averages for mean and variance
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var
        # Intermediates for the backward pass
        cache = (x, mean, norm_x, gamma, var, standard_variation, eps)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Normalizing x using running mean and variance
        norm_x = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * norm_x + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Unpacking cache from the forward pass
    x, mean, norm_x, gamma, var, std, eps = cache
    N, D = x.shape
    # Scale and Shift param grads
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * norm_x, axis=0)
    # Gradients from the backside of the net
    df_dstd = dout * gamma
    dv_dx = (x - mean) * 2 / N
    dstd_dv = - 0.5 * ((var + eps) ** (-1.5)) * (x - mean)
    # Final gradient dL/dX               
    dx = df_dstd * (1 / std) + np.sum(df_dstd * dstd_dv, axis=0) * \
          dv_dx + np.sum(df_dstd * (-1 / std), axis=0) / N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Unpacking cache from the forward pass
    x, mean, norm_x, gamma, var, std, eps = cache
    N, D = x.shape
    # Scale and Shift param grads
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * norm_x, axis=0)
    # Gradients from out the backside of the net
    df_dstd = dout * gamma
    df_dstd_sum = np.sum(df_dstd, axis=0)
    # Final gradient dL/dX 
    dx = df_dstd - df_dstd_sum / N - np.sum(df_dstd * norm_x, axis=0) * norm_x / N
    dx = dx / std
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Average of each row in X
    mean = np.mean(x, axis=1, keepdims=True)
    # Variance for each row in X
    var = np.var(x, axis=1, keepdims=True)
    # Standard Deviation with eps which avoids division by 0
    standard_variation = np.sqrt(var + eps)
    # Normalized X over features; Scale and Shift factors for the output to 
    # allow the net to squash the range if it wants to
    norm_x = (x - mean) / standard_variation
    out = gamma * norm_x + beta
    # Intermediates for the backward pass
    cache = (x, mean, norm_x, gamma, var, standard_variation, eps)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Characteristics of the network
    x, mean, norm_x, gamma, var, std, eps = cache
    N, D = x.shape
    # Scale and Shift param grads
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * norm_x, axis=0)
    # Gradients from out the backside of the net
    df_dstd = dout * gamma
    dv_dx =  2 / D * (x - mean)
    dstd_dv = - 0.5 * ((var + eps) ** (-1.5)) * (x - mean)
    # Final gradient dL/dX                 
    dx = df_dstd * (1 / std) + np.sum(df_dstd * dstd_dv, axis=1).reshape(-1, 1) * \
          dv_dx + np.sum(df_dstd * (-1 / std), axis=1).reshape(-1, 1) / D


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # print("p: ", p)
        # The droput mask
        mask = (np.random.rand(*x.shape) < p) / p
        # Drop lesser than p
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # No change in test mode
        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Regular backpass
        # Downstream grads = local grads * upstream grads
        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param, checkPrint):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # ConvLayer parameters
    stride = conv_param['stride']
    padding = conv_param['pad']
    # Input volume dims
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    # Perform sanity check on the input dims and conv_params
    if (H + 2*padding - HH) % stride == 0:
      if checkPrint==True:
        print(" ConvNet Height Check Clear!")
    else:
      print(" ConvNet H incompatible...Exiting")
      sys.exit() 

    if (W + 2*padding - WW) % stride == 0:
      if checkPrint==True:
        print(" ConvNet Weight Check Clear!")
    else:
      print(" ConvNet W incompatible...Exiting")
      sys.exit() 

    # Output(features) volume size: (N + 2*P - F) / stride + 1
    output_H = int((H + 2*padding - HH) / stride + 1)
    output_W = int((W + 2*padding - WW) / stride + 1)
    # print(" out H", output_H)

    # Output feature tensor
    out = np.zeros((N, F, output_H, output_W))
    # print("output: ", out)

    # Input padding
    x_pad = np.pad(x, ((0,0), (0,0), (padding, padding), (padding,padding)), 'constant')
    pad_H, pad_W = x_pad.shape[2], x_pad.shape[3]

    # Naive convolution loop
    for n in range(N):
      # Accounting for padding on each axis, start at the left-top padded corner
      # HHxWW filters for N channels
      for i in range(0, pad_H - HH + 1, stride):
        for j in range(0, pad_W - WW + 1, stride):
          # For each 4X4 inside the padded input convolved with weights
          x_filter = x_pad[n, :, i: i + HH, j: j + WW]
          # print("filter: ", x_filter)
          # Convolution of filter and the weights of the size of w
          x_convolved = x_filter * w
          # print("conv: ", x_convolved)
          # Sum across all axes for all 4x4 elements
          x_convolved_sum = ((x_convolved.sum(axis=3)).sum(axis=2)).sum(axis=1)
          # print("conv sum: ", x_convolved_sum)
          x_convolved_sum = x_convolved_sum + b
          out[n, :, int(i/stride), int(j/stride)] = x_convolved_sum
          # print("output: ", out)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Extract forward pass params from cache
    x, w, b, conv_param = cache

    # ConvLayer parameters
    stride = conv_param['stride']
    padding = conv_param['pad']

    # Input volume, filter dims
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    # Input padding
    x_pad = np.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)), 'constant')
    pad_H, pad_W = x_pad.shape[2], x_pad.shape[3]

    # Output(features) volume size: (N + 2*P - F) / stride + 1
    output_H = int((H + 2*padding - HH) / stride + 1)
    output_W = int((W + 2*padding - WW) / stride + 1)

    # Gradients wrt input, filters, and the bias
    dx = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    # print("output: ", dx)
    
    # Similar loop for all n data points we get the gradients
    for n in range(N):
      for f in range(F):
        # Accounting for padding on each axis, start at the left-top padded corner
        # HHxWW filters for N channels
        current_x = 0
        for i in range(0, pad_H - HH + 1, stride):
          current_y = 0
          for j in range(0, pad_W - WW + 1, stride):
            # For each 4X4 inside the padded input convolved with weights
            x_filter = x_pad[n, :, i: i + HH, j: j + WW]
            # print(" filt: ", x_filter)
            # Multiply the 3x3 filter with dout for loss grads over all filters
            dw[f] += dout[n, f, current_x, current_y] * x_filter
            # print(" Grad f: ", dw[f])
            # Loss gradient of the input
            dx[n, :, i:i+HH, j:j+WW] += w[f] * dout[n, f, current_x, current_y]
            # print(" ip grad: ", dx[f])
            # Loss gradient of the bias
            db[f] += np.sum(dout[n, f, current_x, current_y])
            # print(" grad b: ", db[f])
            current_y += 1
          current_x += 1

    # Padded gradients
    dx = dx[:, :, padding:padding+H, padding:padding+W]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # PoolLayer parameters
    stride = pool_param['stride']
    pooling_h = pool_param['pool_height']
    pooling_w = pool_param['pool_width']

    # Input volume, filter dims
    N, C, H, W = x.shape
    # F, _, HH, WW = w.shape

    # Output(features) volume size: (N + 2*P - F) / stride + 1
    output_H = int((H - pooling_h) / stride + 1)
    output_W = int((W - pooling_w) / stride + 1)

    # Pooled feature tensor
    out = np.zeros((N, C, output_H, output_W))
    downsampled = out

    # Choosing the max value within the image at each stride
    for n in range(N):
      for f in range(C):
        for h in range(output_H):
          for w in range(output_W):
            # The largest value in the pooling window
            downsampled[n, f, h, w] = np.max(x[n, f, h*stride: h*stride + pooling_h, \
                                    w*stride: w*stride + pooling_w])

    out = downsampled
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Extract forward pass params from cache
    x, pool_param = cache
    N, C, H, W = x.shape

    # PoolLayer parameters
    stride = pool_param['stride']
    pooling_h = pool_param['pool_height']
    pooling_w = pool_param['pool_width']

    # Input volume, filter dims
    N, C, output_H, output_W = dout.shape

    # Gradients wrt input, filters, and the bias
    dx = np.zeros_like(x)

    # Backprop through the max-pooled layer
    for n in range(N):
      for f in range(C):
        for h in range(output_H):
          for w in range(output_W):
            # Index of the maximum value (Non-NaN) in the pooling window
            max_id = np.nanargmax(x[n, f, h*stride: h*stride + pooling_h, \
                                    w*stride: w*stride + pooling_w])
            # print("max: ", x[max_id])
            # Getting the indices back from the flattened max value
            index = np.unravel_index(max_id, (pooling_h, pooling_w))
            # print("unraveled max: ", index)
            # Upstream derivatives from the forward pass having uraveled index 
            # from above to get gradients wrt x
            dx[n, f, h*stride: h*stride + pooling_h, w*stride: w*stride + pooling_w][index] \
                                        = dout[n, f, h, w]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Extract input dims
    N, C, H, W = x.shape

    # For every feature channel's statistics we need to reshape input to 
    # (N*H*W, C) to make it per channel
    xT = x.transpose(0, 2, 3, 1)
    # print("X_t: ", xT)
    # Net statistics for 3 channels for all data
    xT = xT.reshape(N * H * W, C)
    # print("X_t: ", xT)
    # Reusing Batchnorm Forward pass
    out_bn, bn_cache = batchnorm_forward(xT, gamma, beta, bn_param)
    # Output size should match input size
    out = out_bn.reshape(N, H, W, C)
    out = out.transpose(0, 3, 1, 2) # N x C x H x W
    cache = bn_cache

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Extract forward pass values
    N, C, H, W = dout.shape
    # Similar to forward pass we reshape output to get grads per channel
    doutT = dout.transpose(0, 2, 3, 1)
    doutT = doutT.reshape(N * H * W, C)
    # Batchnorm backward pass
    dx, dgamma, dbeta = batchnorm_backward(doutT, cache)
    # Reshaping the grads to match size
    dx = dx.reshape(N, H, W, C)
    dx = dx.transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    # In Groupnorm, D = C//G * H * W
    group_c = C // G 
    # G new groups over the data to get its characteristics
    x = x.reshape((N * G, -1))

    # Average of each row in X
    mean = np.mean(x, axis=1, keepdims=True)
    # Variance for each mini-batch in X
    var = 1 / (group_c * H * W) * np.sum((x - mean) ** 2, axis=1, keepdims=True)
    # Standard Deviation with eps which avoids division by 0
    standard_variation = np.sqrt(var + eps)
    # Per feature shifting and scaling, normalized x reshape to match input x
    norm_x = (x - mean) / standard_variation
    norm_x = norm_x.reshape(N, C, H, W)
    out = gamma * norm_x + beta
    # Intermediates for the backward pass
    cache = (x, mean, norm_x, gamma, beta, var, standard_variation, eps, G, group_c)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Characteristics of the network
    x, mean, norm_x, gamma, beta, var, std, eps, G, group_c = cache
    N, C, H, W = dout.shape
    # In groupnorm, D = C//G * H * W
    d = (group_c * H * W)
    # Scale and Shift param grads
    dbeta = np.sum(dout, axis=(0,2,3), keepdims=True)
    dgamma = np.sum(dout * norm_x, axis=(0,2,3), keepdims=True)
    # Scaled normalized dx 
    df_dstd = dout * gamma
    df_dstd = df_dstd.reshape((N*G, group_c*H*W))
    ones = np.ones((N*G, group_c*H*W))
    # Mean deviation reshaped from 4x60
    p = (x - mean).reshape(df_dstd.shape)
    # print("input mean deviation: ", (p).shape)
    dstd_inv = (-1 / np.square(std)) * np.sum(df_dstd * p, axis=1, keepdims=True)
    dv_dx = 0.5 * (1 / std) * dstd_inv
    # print("dvar: ", dv_dx)
    # Similar to batchnorm_backward_alt
    dmean = df_dstd * (1/std) + 2 * (x-mean) * (1/(d) * ones * dv_dx)
    dx = dmean
    dmean = -1 * np.sum(dmean, axis=1, keepdims=True)
    # print("dmean: ", dmean)
    # Final gradient
    dx = dx + (1/(d)) * ones * dmean
    dx = dx.reshape(N, C, H, W)
    # print("dx: ", (dx).shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
