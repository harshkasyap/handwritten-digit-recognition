from PIL import Image
import numpy as np
from numpy import array
import sys
import os

x_train = []
y_train = []

def prepareData(img, cls1, iter1):
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    x_train.append(img[0])
    y_train.append(cls1)


data = []

for filename in os.listdir('testChars'):
    if filename.endswith(".png"): 
         data.append({int(filename[0]): filename})
    else:
        continue

from random import shuffle
shuffle(data)

print (data)

for (iter1, item) in enumerate(data):
    im = Image.open('testChars/'+item.items()[0][1])
    prepareData(im, item.items()[0][0], iter1)

x_train = array(x_train)
y_train = array(y_train)

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Create the model
trainedModel = load_model('mnist.h5')
batch_size = 128
num_classes = 10
epochs = 10

y_train = keras.utils.to_categorical(y_train, num_classes)

model = Sequential()
model.add(trainedModel)
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
model.layers[0].trainable = False

hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_train, y_train))
model.save('transfer.h5')

score = model.evaluate(x_train, y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])