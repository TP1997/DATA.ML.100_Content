 
import numpy as np

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
    
def get_diff(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return np.linalg.norm(x1 - x2)

def classifier_1nn(x, trainX, trainY):
    mindiff=get_diff(x,trainX[0])
    label=trainY[0]
    for i in range(0,len(trainX)):
        diff=get_diff(x,trainX[i])
        if diff < mindiff:
            mindiff = diff
            label=trainY[i]
        
    return label
    
def class_acc(pred, target):
    correct=np.sum([p==t for p,t in zip(pred,target)])
    return correct/len(pred)
    

root = '/home/tuomas/Python/DATA.ML.100/Tentti/'
trainX, trainY = read_data(root+'X_train.txt', root+'Y_train.txt')
testX, testY = read_data(root+'X_test.txt', root+'Y_test.txt')

predicted_labels = []
n=1
tot=len(testX)
for x in testX:
    predicted_labels.append(classifier_1nn(x, trainX, trainY))
    print(n,"/",tot)
    n+=1
    
print("Calculating accuracy")
accuracy=class_acc(predicted_labels, testY)
print("1NN classifier accuracy:", accuracy*100, "%")
