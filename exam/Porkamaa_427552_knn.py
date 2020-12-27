import numpy as np
import sys
import matplotlib.pyplot as plt

def read_data(fnX, fnY):
    fX = open(fnX, "r")
    linesX = fX.readlines()
    dataX = []
    for l in linesX:
        data = l.split()
        dx = float(data[0])
        dy = float(data[1])
        dz = float(data[2])
        dataX.append(np.array([dx,dy,dz]))
    
    fY = open(fnY, "r")
    linesY = fY.readlines()
    dataY = []
    for l in linesY:
        dataY.append(float(l))
    
    return np.array(dataX), np.array(dataY)
    
# Euclidean distance used
def get_diff(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return np.linalg.norm(x1 - x2)

def classifier_knn(x, trainX, trainY, k):
    mindiffs = [sys.float_info.max for _ in range(k)]
    mindiffs[0]=get_diff(x,trainX[0])
    labels = [None for _ in range(k)]
    labels[0]=trainY[0]
    for i in range(0,len(trainX)):
        diff=get_diff(x,trainX[i])
        idxs = np.where((diff<mindiffs)==True)[0]
        if len(idxs) > 0:
            idx = idxs[0]
            mindiffs = mindiffs[0:idx] + [diff] + mindiffs[idx:-1]
            labels = labels[0:idx] + [trainY[i]] + labels[idx:-1]
      
    
    labels = np.array(labels).astype(int)
    counts = np.bincount(labels)
    return np.argmax(counts)
    
def class_acc(pred, target):
    correct=np.sum([p==t for p,t in zip(pred,target)])
    return correct/len(pred)

def plot_accuracy(Ks, accuracies):
    plt.figure(figsize=(20,10))
    plt.title('Accuracy evolution of Knn-classifier')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    
    plt.plot(Ks,accuracies, marker='o',linestyle='--',color='r',label='XXX')
    plt.xticks(Ks,Ks)
    plt.show()


root = '/home/tuomas/Python/DATA.ML.100/Tentti/'
trainX, trainY = read_data(root+'X_train.txt', root+'Y_train.txt')
testX, testY = read_data(root+'X_test.txt', root+'Y_test.txt')

Ks = [1,2,3,5,10,20]
tot=len(testX)
accuracies = []
for K in Ks:
    print('K = {}'.format(K))
    n=1
    predicted_labels = []
    for x in testX:
        predicted_labels.append(classifier_knn(x, trainX, trainY, K))
        print(n,"/",tot)
        n+=1
    print("Calculating accuracy")
    accuracy=class_acc(predicted_labels, testY)
    print("KNN classifier accuracy (K={}): {} %".format(K, accuracy*100))
    accuracies.append(accuracy)
   
plot_accuracy(Ks, accuracies)

 
