from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero (3073, 10)
    # print(" Gradients :", dW.shape)
    # print("Labels :", y.shape)

    # compute the loss and the gradient
    num_classes = W.shape[1] # Second dimension is C = numOfClasses = 10
    num_train = X.shape[0] # Second dimension is N = miniBatchSize = 500
    # print("N : ", num_train)
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        # print(" Scores : ", scores)
        correct_class_score = scores[y[i]] # correct_class_index score
        for j in range(num_classes):
            if j == y[i]: # correct_class_index
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                # Gradient of all other (j) classes that are incorrect 
                # classifications across all training examples
                dW[:,j] = dW[:,j] + X[i]
                # Gradient of the true class (y[i]) with the correct label
                # y[i] = correct_class_index
                dW[:,y[i]] = dW[:,y[i]] - X[i]
                
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # Average gradient over all minibatch is average
    dW = dW / num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    # Added the regularization to the gradients as well
    dW = dW + (reg * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1] # Second dimension is C = numOfClasses = 10
    num_train = X.shape[0] # Second dimension is N = miniBatchSize = 500
    # print(num_train)
    loss = 0.0

    # Fully vectorized scores with the entire minibatch and the weight matrix
    # where scores mat will contain all scores instead of just one example at
    # a time
    scores = X.dot(W)
    # print("Scores : ",scores.shape)
    # print("y : ",y.shape)
    # Calculating the score of correct classes using broadcasting and scores mat
    correct_scores = scores[np.arange(num_train), y]
    # print("Correct Scores : ",correct_scores.shape)
    # To check if the true score has the margin of 1 we'll have to have the 
    # correct scores as rows to subtract from the original scores
    # First correct scores as rows
    correct_scores_T = correct_scores.reshape(num_train,1)
    # Enabling the broadcast we obtain the other scores to compare
    other_class_scores = scores - correct_scores_T + 1
    # The loss margins computed with comparing the scores with 0
    loss_margins = np.maximum(0, other_class_scores)
    margins = np.where(loss_margins == 1, 0, loss_margins)
    # print("Margins - true :", margins)
    # Sum of the loss count across classes over all training examples
    margin_sum = np.sum(margins, axis=1)
    loss = np.mean(margin_sum)
    # Add regularization to the loss
    loss += reg * np.sum(W * W)

    # Vectorized Gradient using the margins calculated above
    gradients = margins
    gradients = np.where(margins > 0, 1, margins)
    # Gradient sum across classes over the minibatch
    gradients_sum = np.sum(gradients, axis=1)
    gradients[np.arange(num_train),y] -= gradients_sum
    # The vectorized gradient of all classes using the entire minibatch
    dW = X.T@gradients
    # Average gradient over the minibatch & adding the regularization
    dW = dW / num_train
    # print(" dW :", dW)
    dW = dW + reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
