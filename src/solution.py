import numpy as np 
from helper import *
'''
Homework2: logistic regression classifier
'''


def logistic_regression(data, label, max_iter, learning_rate):
    '''
    The logistic regression classifier function.

    Args:
    data: train data with shape (1561, 3), which means 1561 samples and 
		  each sample has 3 features.(1, symmetry, average intensity)
    label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.
    max_iter: max iteration numbers
    learning_rate: learning rate for weight update
	
    Returns:
		w: the seperater with shape (3, 1). You must initilize it with w = np.zeros((d,1))
    '''
    
    w = np.zeros((1,data.shape[1]))
    for i in range(max_iter):
        for row in range(data.shape[0]):
                y = label[row]
                x = data[row,:]
                w = w + y*x*((learning_rate)/(1 + math.exp(y*np.dot(x,np.transpose(w)))))
    print('\nWeights',w)
    return w


def thirdorder(data):
    '''
    This function is used for a 3rd order polynomial transform of the data.
    Args:
    data: input data with shape (:, 3) the first dimension represents 
		  total samples (training: 1561; testing: 424) and the 
		  second dimesion represents total features.

    Return:
      result: A numpy array format new data with shape (:,10), which using 
		a 3rd order polynomial transformation to extend the feature numbers 
		from 3 to 10. 
		The first dimension represents total samples (training: 1561; testing: 424) 
		and the second dimesion represents total features.
    '''
    x = np.ones((data.shape[0],1))
    x1 = np.array(data[:,0])
    x2 = np.array(data[:,1])
    
    
    x1_squared = np.multiply(x1, x1)
    x1_x2 = np.multiply(x1, x2)
    x2_squared = np.multiply(x2, x2)
    
    x1_cubed = np.multiply(x1_squared, x1)
    x1_squared_x2 = np.multiply(x1_squared, x2)
    x2_squared_x1 = np.multiply(x2_squared, x1)
    x2_cubed = np.multiply(x2_squared, x2)
    
    return np.column_stack((x, x1, x2, x1_squared, x1_x2, x2_squared, x1_cubed, x1_squared_x2, x2_squared_x1, x2_cubed))


def accuracy(x, y, w):
    '''
    This function is used to compute accuracy of a logsitic regression model.
    
    Args:
    x: input data with shape (n, d), where n represents total data samples and d represents
        total feature numbers of a certain data sample.
    y: corresponding label of x with shape(n, 1), where n represents total data samples.
    w: the seperator learnt from logistic regression function with shape (d, 1),
        where d represents total feature numbers of a certain data sample.

    Return 
        accuracy: total percents of correctly classified samples. Set the threshold as 0.5,
        which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
    '''
    num_of_data_points = x.shape[0]
    num_classify_correct = 0
    for row in range(num_of_data_points):
        p_of_y_given_x = theta(y[row]*np.dot(x[row,:],np.transpose(w)))
        if p_of_y_given_x > 0.5:
            classification = 1
        else:
            classification = -1
         
        if classification == y[row]:
            num_classify_correct += 1
        
    return num_classify_correct/num_of_data_points*100

def theta(s):
    return 1/(1+math.exp(-s))