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
    
    #for (i in range(0, max_iter))
    # for each data row 
    # for each x(n) in row
    # w = w + label(n)*data(n)*(learning_rate/(1 + exp(label[n]*w*transpose(data[0:]))
    w = np.zeros((1,data.shape[1]))
    for i in range(max_iter):
        for row in range(data.shape[0]):
                y = label[row]
                x = data[row,:]
                w = w + y*x*((learning_rate)/(1 + e**(y*np.dot(x,np.transpose(w)))))
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
	pass


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
     #P(y|x) = theta(ywTx) => theta(label[i]*w*transpose(data[0:]))
     num_of_data_points = x.shape[0]
     num_classify_correct = 0
     for row in range(num_of_data_points):
         p_of_y_given_x = theta(label[row]*np.dot(data[row,:],np.transpose(w))
         if p_of_y_given_x > 0.5:
             classification = 1
         else:
             classification = -1
             
         if classification == label[row]:
             num_classify_correct += 1
         
    return num_classify_correct/num_of_data_points*100

def theta(s):
    return 1/(1+e**-s)