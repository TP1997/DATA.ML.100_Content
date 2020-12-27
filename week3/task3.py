import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.stats import norm, multivariate_normal
import time
from multiprocessing import Pool

root = '/home/tuomas/Python/DATA.ML.100/Ex3/'

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def load_whole_trainset():
    trainDataDict = unpickle(root+'cifar-10-batches-py/data_batch_1')
    trainX=trainDataDict["data"].reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("int32")#int32
    trainY=np.array(trainDataDict["labels"])
    for i in range(2,6):
        fn='cifar-10-batches-py/data_batch_%i'%i
        trainDataDict = unpickle(root+fn)
        tx=trainDataDict["data"].reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("int32")
        #print(tx.shape)
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

def cifar10_color(X, n):
    scaledX=np.zeros((X.shape[0], X.shape[3]*n*n))

    for i in range(X.shape[0]):
        img = X[i]
        img_nxn = resize(img, (n, n))
        r_vals = img_nxn[:,:,0].reshape(n*n)
        g_vals = img_nxn[:,:,1].reshape(n*n)
        b_vals = img_nxn[:,:,2].reshape(n*n)
        scaledX[i,:] = np.concatenate((r_vals, g_vals, b_vals))
      
    return scaledX

def cifar10_classifier_bayes(x, mus, covs, priors):
    posteriors = np.array([multivariate_normal.logpdf(x,mu,cov) for mu,cov in zip(mus,covs)])
    
    classified=posteriors.argmax(axis=0)

    return classified #np.argmax(posteriors)

# Evaluation function
def class_acc(predicted, actual):
    correct=np.sum([p==a for p,a in zip(predicted,actual)])
    return correct/len(predicted)

def cifar_10_bayes_learn(X, Y):
    # Rearrange X as x_separated[0] includes all training samples in class 0 etc...
    x_separated = [[] for i in range(10)]
    for i in range(X.shape[0]):
        x_separated[Y[i]].append(X[i])

    class_rgb_means = []
    class_rgb_covs = []
    priors = []
    for i in range(10):
        class_rgb_means.append(np.mean(x_separated[i],axis=0))
        class_rgb_covs.append(np.cov(x_separated[i], rowvar=False, ddof=1)) #!!
        priors.append(len(x_separated[i])/Y.size)
        
    return np.array(class_rgb_means), np.array(class_rgb_covs), np.array(priors)

def train_model(n):
    scaledX = cifar10_color(trainX,n)
    mus, covs, priors = cifar_10_bayes_learn(scaledX ,trainY)
    
    return mus, covs, priors

def bclassify(testX):
    print('Bayes classifier Classifying...')
    classified=cifar10_classifier_bayes(testX,mus,covs,priors)
    return classified

def nbclassify(testX):
    print('Naive Bayes classifier Classifying...')
    covs2=np.array([np.diag(np.array(np.diag(m))) for m in covs])
    classified=cifar10_classifier_bayes(testX,mus,covs2,priors)
    return classified

def plot_a_graph(x,by,nby):
    fig=plt.figure()

    ax = fig.add_subplot(111)
    ax.set_xlim(-2,35)
    ax.set_ylim(15,45)
    
    plt.title('Classifier performance with various image sizes')
    plt.xlabel('Image width')
    plt.ylabel('Accuracy %')
    plt.xticks(x,x)
    
    plt.plot(x,by, marker='o',color='b',label='Gaussian Bayes')
    plt.plot(x,nby, marker='o',linestyle='--',color='r',label='Gaussian Naive Bayes')
    ax.legend(loc='lower right')
    
    for x_,y_ in zip(x,by):
        plt.text(x_-3,y_+0.5,str(y_),color='b')
        
    for x_,y_ in zip(x,nby):
        plt.text(x_-3,y_+0.5,str(y_),color='r')
    
    plt.savefig(root+'accuracy.png')




    
# Define these as a global variables.
start=time.time()
trainX, trainY = load_whole_trainset()
testX1, testY = load_testset()

ns=[1,2,4,8,16,32]
nbaccuracy=[]
baccuracy=[]
for n in ns:
    print('Estimating model parameters for {}x{} images...'.format(n,n))
    mus, covs, priors = train_model(n)
    print('Resizing test images...')
    testX = cifar10_color(testX1,n)
    testX, _ = data_partition(testX,testY,4)
    
    print('Evaluate Gaussian Bayes model...')
    pool = Pool(4)
    classified=pool.map(bclassify, testX)
    classified=np.concatenate(classified)
    bacc=class_acc(classified, testY)
    baccuracy.append(round(bacc*100,2))
    
    print('Evaluate Gaussian Naive Bayes model...')
    pool = Pool(4)
    classified=pool.map(nbclassify, testX)
    classified=np.concatenate(classified)
    nbacc=class_acc(classified, testY)
    nbaccuracy.append(round(nbacc*100,2))
    
    print('{}x{} Gaussian Bayes-classifier accuracy: {}%'.format(n,n,bacc*100))
    print('{}x{} Gaussian Naive Bayes-classifier accuracy: {}%'.format(n,n,nbacc*100))
    
print('Total running time: {}s'.format(time.time()-start))
print('Bayes classifier accuracies:', baccuracy)
print('Naive Bayes classifier accuracies:', nbaccuracy)
plot_a_graph(ns,baccuracy,nbaccuracy)
    
