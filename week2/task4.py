import pickle
import numpy as np
import time
import queue
from multiprocessing import Pool

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

# Evaluation function
def class_acc(pred, gt):
    correct=np.sum([p==g for p,g in zip(pred,gt)])
    return correct/len(pred)

# Difference of two vectors using absolute distance.
def get_diff(x1s, x2s):
    difference_array = np.subtract(x1s, x2s)
    squared_array = np.abs(difference_array)
    return squared_array.sum(axis=1)

# trainX & trainY are globals so they aren't included to function parameters
def cifar10_classifier_1nn(x):
    distances = get_diff(x, trainX)
    return trainY[distances.argmin()]

# Project directory
root='/home/tuomas/Python/DATA.ML.100/Exercise 2/'
def load_whole_trainset():
    trainDataDict = unpickle(root+'cifar-10-batches-py/data_batch_1')
    trainX=trainDataDict["data"].astype("int32")
    trainY=np.array(trainDataDict["labels"])
    for i in range(2,6):
        fn='cifar-10-batches-py/data_batch_%i'%i
        trainDataDict = unpickle(root+fn)
        tx=trainDataDict["data"].astype("int32")
        ty=np.array(trainDataDict["labels"])
        trainX=np.concatenate((trainX,tx))
        trainY=np.concatenate((trainY,ty))
    
    print("training set size:", trainX.shape)
    return trainX, trainY

def load_testset():
    testDataDict = unpickle(root+'cifar-10-batches-py/test_batch')
    testX = testDataDict["data"].astype("int32")
    testY = np.array(testDataDict["labels"])
    
    print("testing set size:", testX.shape)
    return testX, testY

def predict_labels(testX):
    predictedLabels=[]
    print("Starting search...")
    for x in testX:
        predictedLabels.append(cifar10_classifier_1nn(x))
    return predictedLabels

# Divide test data into n subsets.
def data_partition(testX, testY, n):
    psize=int(len(testX)/n)
    pdatax=[]
    pdatay=[]
    for i in range(1,n+1):
        pdatax.append(testX[(i-1)*psize : i*psize])
        pdatay.append(testY[(i-1)*psize : i*psize])
        
    return pdatax, pdatay


def main():

    # Load training data:
    #trainX2, trainY2 = data_partition(trainX, trainY, 1000)
    #trainY3 = np.vstack(trainY2)
    # Load test data:
    testX, testY = load_testset()
    testX, _ = data_partition(testX, testY, 4)
    # Create threadpool with 4 threads
    pool = Pool(4)
    start = time.time()
    predictedLabels = pool.map(predict_labels, [testX[0], testX[1], testX[2], testX[3]])

    accuracy=class_acc(predictedLabels[0]+predictedLabels[1]+predictedLabels[2]+predictedLabels[3], testY)
    print("Execution time:", time.time()-start, "s")
    print("1NN classifier accuracy:", accuracy*100, "%")

# Define these as a global variables.
trainX, trainY = load_whole_trainset()
main()
