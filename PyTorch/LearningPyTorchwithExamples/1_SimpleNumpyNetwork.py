
'''
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
'''

import numpy as np

N, D_in, H, D_out = 64, 1000, 100, 10

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

def sigmoid(x):
    return np.exp(x)/(np.exp(x)+1)

learning_rate = 1e-6
for t in range(500):
    # Forward propagation
    '''
    My solution:
    
    a1 = sigmoid(np.matmul(x, w1))
    a2 = sigmoid(np.matmul(a1, w2))
    
    grad = sum(2*(y-a2))
    
    w2 += grad * learning_rate
    w1 += grad * learning_rate
    '''

    '''
    Feed forward: X ->(w1)-> h -> h_relu ->(w2)-> y_pred
    '''

    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Take the difference
    loss = np.square(y_pred-y).sum()

    # Backward propagation

    '''
    gradient at the sum of square, partial loss / partial each y
    '''
    grad_y_pred = 2 * (y_pred-y)

    '''
    TODO: need to figure out backpropogation formula
    https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html
    https://www.youtube.com/watch?v=x_Eamf8MHwU
    
    for w2 operation, input is h_relu, output error is grad_y_pred (s.t. h_relu * w2 = y_pred)
    so its gradient is h_relu * pred_y
    '''
    grad_w2 = h_relu.T.dot(grad_y_pred)

    '''
    for h_relu "operation", "input" is w2, output error is grad_y_pred
    so its gradient is w2*grad_y_pred
    '''
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()

    '''
    for the relu operation, input is h, output error is grad_h_relu, prime is 0 or 1 (TODO)
    '''
    grad_h[h < 0] = 0

    '''
    for w1 operation, input is x, output error is grad_h
    '''
    grad_w1 = x.T.dot(grad_h)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

    print loss
