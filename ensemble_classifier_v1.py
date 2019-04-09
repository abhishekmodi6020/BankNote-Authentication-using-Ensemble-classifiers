import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import pprint

def data_pre_processing(filename):
	x = []
	y = []
	data = csv.DictReader(open(filename),fieldnames=['variance','skewness','curtosis','entropy','y'])
	for row in data:
		x.append([float(row['variance']),float(row['skewness']),float(row['curtosis']),float(row['entropy']),float(row['y'])])
		# y.append(row['y'])
	x = np.array(x)
	return x

def initialize_parameters(dim):
	'''Creating parameters for k classifiers each having some dimensions(example: 4 features).
	By default one classifier is considered'''
	param = {}
	param['w'] = np.zeros((dim,1),dtype=float)
	param['b'] = 0
	return param

def sigmoid(z):
	return 1/(1+np.exp(-z))

def fwd_prop(x,param):
    w = param['w']
    b = param['b']
    # no.of examples
    m = x.shape[0]
    # calulated y
    Z = np.dot(w.T,x[:,:-1].T) + b
    A = sigmoid(Z)
#    print('\nsigmoid of z :',A)
    # computing cost
    cost = - (1/m)*(np.sum(np.multiply(x[:,-1],np.log(A))) + np.sum(np.multiply(1- x[:,-1],np.log(1-A))))
    return A,cost

def optimize(A,x,param,learning_rate):
    w = param['w']
    b = param['b']
    # no.of examples
    m = x.shape[0]
    # gradient
    dZ = A - x[:,-1]
    dW = (1/m)*np.sum(np.multiply(x,dZ.T))
    dB = np.sum(dZ)/m
    w = w - learning_rate*dW
    b = b - learning_rate*dB
    param['w'] = w
    param['b'] = b
    return param
    
def predict(x,param):
    A, cost = fwd_prop(x,param)
    A = np.round(A[0])
    incorrect_count = 0
    
    for i,j in zip(A,x[:,-1]):
        if i != j:
            incorrect_count += 1
    print('Accuracy = ',100*(x.shape[0] - incorrect_count)/x.shape[0])
#    A, cost = fwd_prop(xtest,param)
#    A = np.round(A[0])
#    incorrect_count = 0
#    print('\n********************************** Test Prediction ')
#    for i,j in zip(A,xtest[:,-1]):
#        if i != j:
#            incorrect_count += 1
##        print('prediction: ',i)
#    print('Accuracy = ',100*(xtest.shape[0] - incorrect_count)/xtest.shape[0])
    return A

def logisticRegModel(x,iterations,learning_rate):
    param = initialize_parameters(x.shape[1] - 1)
    costs = []
    for i in range(0,iterations):
        A,cost = fwd_prop(x,param)
        param = optimize(A,x,param,learning_rate)
        costs.append(cost)
    print('Last cost: ',cost)
    
    # plotting the cost w.r.t iteration
#    plt.plot(costs)
#    plt.ylabel('cost')
#    plt.xlabel('iterations (per hundreds)')
#    plt.title("Learning rate =" + str(learning_rate))
#    plt.show()
    return param

def logisticReg(x,iterations,learning_rate):    
    param = logisticRegModel(x,iterations,learning_rate)
    
    print('*********************** prediction on test data ***************************')
    # prediction on testing data 
    x = data_pre_processing('test.txt')
    results = predict(x,param)
    print('\n Prediction Results: ',results)
    results = list(results)
    return results

def error_adaboost(data_weights,x,results):
    m = x.shape[0]
    error = 0
    sum_data_weights = 0
    for i in range(m):
        if x[i][-1] != results[i]:
            error += data_weights[i]
        sum_data_weights += data_weights[i]
    error = error/sum_data_weights
    return error

def updating_data_weights(data_weights,x,results,alpha):
    for i in range(len(data_weights)):
        temp = -alpha*x[i][-1]*results[i]
        data_weights[i] = data_weights[i]*np.exp(temp)
    return data_weights

def adaBoost(no_of_classfiers):
    xtrain = data_pre_processing('train.txt')
    m = xtrain.shape[0]
#    print('m (no_of_training_data ): ',m)
    data_weights = [1/xtrain.shape[0]]*xtrain.shape[0]
    classifier_errors = []
    classifier_weights = []
#    print('************************************* Data_weights\n',data_weights)
    print('\n************************* AdaBoost Training\n')
    # Training using adaboost technique
    for i in range(no_of_classfiers):
        param = logisticRegModel(xtrain,1000, 0.01)
        train_results = predict(xtrain,param)
#        print('********Prediction results on Training'+str(i)+'th classifier: ',train_results)
        classifier_errors.append(error_adaboost(data_weights,xtrain,train_results))
        alpha = 0.5*np.log2((1-classifier_errors[i])/classifier_errors[i])
        classifier_weights.append(alpha)
        data_weights = updating_data_weights(data_weights,xtrain,train_results,alpha)
    print('\nAll classifier errors: ',classifier_errors)
    # testing
    print('\n************************* AdaBoost Testing\n')
    xtest = data_pre_processing('test.txt')
    m = xtest.shape[0]
#    adaBoost_output = []
    final_predict = [0]*m
    for i in range(no_of_classfiers):
        param = logisticRegModel(xtest,1000, 0.01)
        test_results = predict(xtest,param)
        print('********Prediction results on testing'+str(i)+'th classifier: ',test_results)
        temp = classifier_weights[i]*test_results
#        print('\ntemp: ',temp)
        final_predict += temp
#        print('\nsum of all the previous predicts: ',final_predict)
    for i in range(m):
        if final_predict[i] < 0.5:
            final_predict[i] = 0
        elif final_predict[i] >= 0.5:
            final_predict[i] = 1
            
    print('\nfinal_predict: ',final_predict)
    incorrect_count = 0
    for i,j in zip(final_predict,xtest[:,-1]):
        if i != j:
            incorrect_count += 1
    print('Final Accuracy = ',100*(xtest.shape[0] - incorrect_count)/xtest.shape[0])
    
        
def bagging(no_of_classfiers,random_samples):
    ensemble_output = []
    x = data_pre_processing('train.txt')
#	print(x)
    k = no_of_classfiers
    random_samples = random_samples
    k_output = []
    for i in range(0,k):
        x_rand = np.zeros((random_samples,x.shape[1]))
        for j in range(0,random_samples):
            x_rand[j] = random.choice(x)
        k_output.append(logisticReg(x_rand, 1000, 0.05))
#    print('k_output: ',k_output)
    
    for row in range(0,len(k_output[0])):
        finalcount = 0
        for col in range(k):
#            print('k_output'+'['+str(row)+']['+str(col)+'] :',k_output[row][col])
            if k_output[col][row] == 0:
#                print('negative')
                finalcount -= 1
            elif k_output[col][row] == 1:
#                print('positive')
                finalcount += 1
        if finalcount >= 0:
            ensemble_output.append(1)
        elif finalcount < 0:
            ensemble_output.append(0)
    print('*****************  ensemble output\n')
    print(ensemble_output)
    xtest = data_pre_processing('test.txt')
    incorrect_count = 0
    print('\n********************************** Test Prediction ')
    for i,j in zip(ensemble_output,xtest[:,-1]):
        print('i: ',i,'| j: ',j)
        if i != j:
            incorrect_count += 1
#        print('prediction: ',i)
    print('Ensemble Classifier Accuracy = ',100*(xtest.shape[0] - incorrect_count)/xtest.shape[0])
#            
            
if __name__ == '__main__':
    no_of_classfiers = int(input('Enter no.of.classifier: '))
    method_ip = int(input('Enter \n 1 --> bagging\n 2 --> adaBoost\n'))
    if method_ip == 1:
        random_samples = int(input('Enter no. of random samples: '))
        bagging(no_of_classfiers,random_samples)
    elif method_ip == 2:
        adaBoost(no_of_classfiers)
            
            