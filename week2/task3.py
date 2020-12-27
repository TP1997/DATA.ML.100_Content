import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random,randint

# Script for 3. CIFAR-10 – Random classifier

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

# 2. CIFAR-10 – Evaluation
def class_acc(pred, gt):
    correct=np.sum([p==g for p,g in zip(pred,gt)])
    return correct/len(pred)

# Asked implementation
def cifar10_classifier_random():
    return randint(0, 9)
                      
root='/home/tuomas/Python/DATA.ML.100/Exercise 2/'
datadict = unpickle(root+'cifar-10-batches-py/test_batch')

X = datadict["data"]
Y = np.array(datadict["labels"])

pred_labels=[]
for i in range(X.shape[0]):
    pred_labels.append(cifar10_classifier_random())
    
accuracy=class_acc(pred_labels, Y)
print("Random classifier accuracy:", accuracy*100, "%")
    
