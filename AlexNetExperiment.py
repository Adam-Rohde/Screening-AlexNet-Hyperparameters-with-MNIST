'''
AlexNet Type Convolutional Neural Net with Keras using the MNIST
Authors: Adam Rohde & Ashley Chiu
'''

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import time
import random as rn

#########################################################
#Set Seed - this changes for each iteration of our experiment
np.random.seed(1901)
tf.set_random_seed(1901)
rn.seed(1901)

#Import MNIST Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#Plot some examples
plt.imshow(x_train[0])
plt.imshow(x_train[10000])

# Reshaping the array to work with the Keras
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])


#########################################################
#TimeHistory Class: https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
    
#########################################################
#AlexNetRun Function

def AlexNetRun(learning_rate = 0.01,epochs = 1,batch_size = 100,dropout = 1,activation = 'relu',add_layer = 1,norm = 1):
    
    #clear any previous models
    keras.backend.clear_session()
    
    #Instantiate empty model
    model = Sequential()
    
    #1st Layer: Conv2D, MaxPool, BatchNormalization
    model.add(Conv2D(filters=32, input_shape=(28,28,1), kernel_size=(2,2), padding='valid'))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    if norm==1:
        model.add(BatchNormalization())
        
    #2nd Layer: Conv2D, MaxPool, BatchNormalization
    model.add(Conv2D(filters=64, kernel_size=(2,2), padding='valid'))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    if norm==1:
        model.add(BatchNormalization())
        
    #3rd Layer: Conv2D, MaxPool, BatchNormalization
    if add_layer==1:
        model.add(Conv2D(filters=128, kernel_size=(2,2), padding='valid'))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        if norm==1:
            model.add(BatchNormalization())
   
    #Flatten Data for Fully Connected Layers
    model.add(Flatten())
    
    #4th Layer: Fully Connected, Dropout
    if dropout==1:
        model.add(Dropout(0.5))
    model.add(Dense(512, input_shape=(28*28*1,)))
    model.add(Activation(activation))
    
    #5th Layer: Fully Connected, Dropout
    if dropout==1:
        model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation(activation))
    
    #Output Layer: Fully Connected, Softmax
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    #Summarize Model Layers
    #model.summary()
        
    #Set Hyperparameters, Compile and Fit Model
    adam = keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    time_callback = TimeHistory()
    model.fit(x=x_train,y=y_train,epochs=epochs,batch_size=batch_size,callbacks=[time_callback],shuffle=True)
    traintime = time_callback.times
    
    #output test results
    return model.evaluate(x_test, y_test)+[sum(traintime)]



#########################################################
#Create Experiment Matrix: 
#Factor Settings: Learning Rate, Number of Epochs, Batch Size, Dropout, Activation Function,
#                 Additional Convolution Layer, Normalization
#Responses: Loss, Accuracy, Training Time

#Create a 2^(7-2) Resolution IV 1/4 fraction of 7 factors in 32 runs
#Generators F = ABCD and G = ABDE, Defining Relation I = ABCDF = ABDEG = CEFG
CodedSettings = [[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]]    
M = pd.DataFrame(itertools.product(*CodedSettings),columns=['A','B','C','D','E'])
M['F'] = M['A']*M['B']*M['C']*M['D'] 
M['G'] = M['A']*M['B']*M['D']*M['E'] 
M['OrigRunID'] = M.index+1

Ex = pd.DataFrame(M,columns=[
                'Learning Rate','Number of Epochs','Batch Size','Dropout','Activation Function',
                'Additional Convolution Layer','Normalization'])
    
Ex.loc[M['A'] == -1, 'Learning Rate'] = 0.001 #this was reset from 0.01 in the followup experiment 
Ex.loc[M['A'] ==  1, 'Learning Rate'] = 0.0001
Ex.loc[M['B'] == -1, 'Number of Epochs'] = 2
Ex.loc[M['B'] ==  1, 'Number of Epochs'] = 10 
Ex.loc[M['C'] == -1, 'Batch Size'] = 50
Ex.loc[M['C'] ==  1, 'Batch Size'] = 100
Ex.loc[M['D'] == -1, 'Dropout'] = 0
Ex.loc[M['D'] ==  1, 'Dropout'] = 1
Ex.loc[M['E'] == -1, 'Activation Function'] = 'relu'
Ex.loc[M['E'] ==  1, 'Activation Function'] = 'tanh'
Ex.loc[M['F'] == -1, 'Additional Convolution Layer'] = 0
Ex.loc[M['F'] ==  1, 'Additional Convolution Layer'] = 1
Ex.loc[M['G'] == -1, 'Normalization'] = 0
Ex.loc[M['G'] ==  1, 'Normalization'] = 1

Ex['Loss'] = 0
Ex['Accuracy'] = 0
Ex['Training Time'] = 0
Ex['OrigRunID'] = Ex.index+1

Ex = Ex.join(M.set_index('OrigRunID'), on='OrigRunID')

#assign random run order to Experiment runs
Ex = Ex.sample(frac=1).reset_index(drop=True) #this randomizes the run order for each block
Ex['RandomRunID'] = Ex.index+1

#########################################################
#Run Experiment
for i in range(0,Ex.shape[0]):
    print(Ex.loc[i])
    Ex.loc[i,'Loss'],Ex.loc[i,'Accuracy'],Ex.loc[i,'Training Time'] = AlexNetRun(
        learning_rate = float(Ex.loc[i,'Learning Rate']),
        epochs =        int(Ex.loc[i,'Number of Epochs']),
        batch_size =    int(Ex.loc[i,'Batch Size']),
        dropout =       int(Ex.loc[i,'Dropout']),
        activation =    Ex.loc[i,'Activation Function'],
        add_layer =     int(Ex.loc[i,'Additional Convolution Layer']),
        norm =          int(Ex.loc[i,'Normalization']))
    print(Ex.loc[i,'Accuracy'])

#########################################################
#Export Experiment Data
Ex.to_csv(r'AlexNetExperimentData_Iter2_2.csv')






