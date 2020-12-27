import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.stats import norm

root = '/home/tuomas/Python/DATA.ML.100/Ex3/'

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def load_whole_trainset():
    trainDataDict = unpickle(root+'cifar-10-batches-py/data_batch_1')
    trainX=trainDataDict["data"].reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("int32")
    trainY=np.array(trainDataDict["labels"])
    for i in range(2,6):
        fn='cifar-10-batches-py/data_batch_%i'%i
        trainDataDict = unpickle(root+fn)
        tx=trainDataDict["data"].reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("int32")
        ty=np.array(trainDataDict["labels"])
        trainX=np.concatenate((trainX,tx))
        trainY=np.concatenate((trainY,ty))
    
    #print("training set size:", trainX.shape)
    return trainX, trainY

def load_testset():
    testDataDict = unpickle(root+'cifar-10-batches-py/test_batch')
    testX = testDataDict["data"].reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("int32")
    testY = np.array(testDataDict["labels"])
    
    #print("testing set size:", testX.shape)
    return testX, testY

# Divide test data into n subsets.
def data_partition(testX, testY, n):
    psize=int(len(testX)/n)
    pdatax=[]
    pdatay=[]
    for i in range(1,n+1):
        pdatax.append(testX[(i-1)*psize : i*psize])
        pdatay.append(testY[(i-1)*psize : i*psize])
        
    return np.array(pdatax), np.array(pdatay)

def cifar10_color(X):
    X_mean=np.zeros((X.shape[0], X.shape[3]))
    for i in range(X.shape[0]):
        img = X[i]
        img_1x1 = resize(img, (1, 1))
        r_vals = img_1x1[:,:,0].reshape(1*1)
        g_vals = img_1x1[:,:,1].reshape(1*1)
        b_vals = img_1x1[:,:,2].reshape(1*1)
        X_mean[i,:] = (r_vals, g_vals, b_vals)
        
    return X_mean

def cifar_10_naivebayes_learn(X, Y):
    x_separated = [[] for i in range(10)]
    for i in range(X.shape[0]): 
        x_separated[Y[i]].append(X[i])
    
    class_rgb_means = []
    class_rgb_stds = []
    priors = []
    for i in range(10):
        class_rgb_means.append(np.mean(x_separated[i],axis=0))
        class_rgb_stds.append(np.std(x_separated[i], axis=0))
        #priors.append(x_separated[i].size/Y.size)
        priors.append(len(x_separated[i])/Y.size)
    
    return np.array(class_rgb_means), np.array(class_rgb_stds), np.array(priors)

def cifar10_classifier_naivebayes(x, mus, stds, priors):
    posteriors = [norm.pdf(x[0],mu[0],sd[0])*norm.pdf(x[1],mu[1],sd[1])*norm.pdf(x[2],mu[2],sd[2])*p for mu,sd,p in zip(mus,stds,priors)]
    
    return np.argmax(posteriors)

# Evaluation function
def class_acc(predicted, actual):
    correct=np.sum([p==a for p,a in zip(predicted,actual)])
    return correct/len(predicted)

def train_model():
    scaledX = cifar10_color(trainX)
    mus, stds, priors = cifar_10_naivebayes_learn(scaledX, trainY)
    
    return mus, stds, priors
    
def main():
    # Train model
    print('Estimating model parameters...')
    mus, stds, priors = train_model()
    # Evaluate model
    print('Evaluating model accuracy...')
    testX, testY = load_testset()
    testX = cifar10_color(testX)
    classified = np.array([cifar10_classifier_naivebayes(x,mus,stds,priors) for x in testX])
    accuracy=class_acc(classified, testY)
    
    print('Gaussian NB-classifier accuracy:', accuracy*100, '%.')
    
    
# Define these as a global variables.
trainX, trainY = load_whole_trainset()

main()

