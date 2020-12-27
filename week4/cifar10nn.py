from tensorflow.keras import datasets, utils
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

''' Prepare the data '''
(trainX, trainY), (testX, testY) = datasets.cifar10.load_data()

trainY = trainY.reshape(50000)
testY = testY.reshape(10000)

trainY = utils.to_categorical(trainY)
testY = utils.to_categorical(testY)

trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainX = trainX / 255.0
testX = testX / 255.0

# Create validation data set
validX = trainX[0:5000]
validY = trainY[0:5000]

trainX = trainX[5000:]
trainY = trainY[5000:]

''' Create the NN-model '''
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

# Output layer of the model
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

''' Set learning rate and compile the model'''
opt = optimizers.SGD(lr = 0.01, momentum=0.9)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=(['accuracy']))

''' Train the model '''
# Data augmentation
idg = ImageDataGenerator(width_shift_range=0.1, 
                         height_shift_range=0.1, 
                         horizontal_flip=True,
                         rotation_range=20,
                         zoom_range=0.1,
                         fill_mode='nearest')
itd = idg.flow(trainX, trainY, batch_size=64)

n_epochs = 100
history = model.fit_generator(itd, 
                              steps_per_epoch=int(trainX.shape[0] / 64),
                              epochs=n_epochs,
                              validation_data=(validX, validY),
                              verbose=1)

''' Plot model's performance statistics '''
plt.figure(figsize=(10,10))
# Plot model's accuracy information based on epochs
plt.subplot(211)
plt.title('Accuracy evolution')
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1.0])
plt.xlim([0.0, n_epochs])
plt.legend(loc = 'lower right')

# Plot model's loss function information based on epochs
plt.subplot(212)
plt.title('Loss function evolution')
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xlim([0.0, n_epochs])
plt.legend(loc = 'lower right')

''' Evaluate model with test data '''
test_loss, test_acc = model.evaluate(testX, testY, verbose=2)

print("Accuracy: {}".format(test_acc))






