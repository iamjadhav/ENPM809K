from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # C = numOfClasses = 10 & N = miniBatchSize = 500, Gradient matrix using W
    num_classes = W.shape[1]
    num_train = X.shape[0]
    dW = np.zeros(W.shape)

    # Similar to SVM loss, we iterate over the training batch as the function 
    # mapping stays the same as SVM
    for i in range(num_train):
        scores = X[i].dot(W)
        # Shifting the values of scores so that there is no overflow
        scores = scores - np.max(scores)
        # The shifted true class scores over the summation of the other score exp
        # The unnormalized probablities
        exponent_scores = np.exp(scores)
        # Normalizing the probablities
        p = exponent_scores / np.sum(exponent_scores)
        # Probability of true class and the loss with the terms obtained p & y_p
        y_p = p[y[i]]
        loss = loss - np.log(y_p)

        # Increasing the score of the correct class and with it decreased 
        # gradient leads to decreased loss 
        for j in range(num_classes):
          if j == y[i]:
            dW[:,j] = dW[:,j] + (y_p - 1) * X[i]
          else:
            dW[:,j] = dW[:,j] + (exponent_scores[j] / np.sum(exponent_scores)) * X[i]

    # Loss average
    loss /= num_train
    # Average gradient is over the dataset
    dW = dW / num_train

    # Add regularization to the loss
    loss += reg * np.sum(W * W)
    # Added the regularization to the gradients as well
    dW = dW + (reg * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    # C = numOfClasses = 10 & N = miniBatchSize = 500
    num_classes = W.shape[1]
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X@W
    # Shifted scores to avoid overflow
    scores_sum = np.max(scores, axis=1, keepdims=True)
    scores = scores - scores_sum
    # Unnormalized probabilities
    exponent_scores = np.exp(scores)
    # Normalizing the probabilities
    p = exponent_scores / np.sum(exponent_scores, axis=1, keepdims=True)
    # Correct class probabities
    y_p = p[np.arange(num_train),y]
    # Loss over all the dataset
    loss =  np.sum(-np.log(y_p))

    # Gradient of the true classes
    gradients = np.zeros_like(p)
    gradients[np.arange(num_train),y] = 1
    # Increased score resulting in decreased gradient and subsequent loss
    dW = X.T@(p - gradients)

    # Loss average
    loss /= num_train
    # Average gradient over the dataset is average
    dW = dW / num_train

    # Add regularization to the loss
    loss += reg * np.sum(W * W)
    # Added the regularization to the gradients as well
    dW = dW + (reg * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
