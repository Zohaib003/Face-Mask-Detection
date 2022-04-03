

import cv2
import os
import numpy as np
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
# getting paths of directories having images of "with_mask" & "without_mask"
path="dataset"
data=os.listdir(path)
data=data[:-1]

target=[i for i in range (len(data))]
dictt=dict(zip(data,Responces))

# Normalizing the data sets, dataset consist of 3 colors RGB, so simply we convertthem into grayscale images which are easy to processible
# diffrenet images have diffrent size so we resize imges.

test_data=[]                     
output=[]             
for i in data:
    dataFolder = os.path.join(path,i)
    images_list = os.listdir(dataFolder)  
    count=0
    for img in images_list:                        
        path_image = os.path.join(dataFolder,img)
        image = cv2.imread(path_image)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)   
        res_image = cv2.resize(gray,(100,100))      
        test_data.append(res_image)
        output.append(Responces_dictioanry[i])    
#using numpy to turn our to arrays and like matrixes so can be used in CNN models 
test_data=np.array(test_data)/255.0     
test_data=np.reshape(test_data,(test_data.shape[0],100,100,1)) #reshapes matrices 
output=np.array(output)

# converting into catagorical representation since at outer-layer have 2 neurons to output                                                               
output=np_utils.to_categorical(output)

#for building MLp model, we are using keros

model= Sequential()
# we will use 2 convolutionary layers fallowed by other layers

# this is first convolutionary layer, we set filters=200, kernal_size=(3,3), activation function 'relu'.

model.add(Conv2D(filters=200,kernel_size=(3,3),input_shape=test_data.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
# first layer is followed by maxpooling layer


#second convolutionary layer, we set filters=200, kernal_size=(3,3), activation function 'relu'.
model.add(Conv2D(filters=100,kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


 
#1 dimonshional list 
model.add(Flatten())

#for overfitting we use dropout
model.add(Dropout(0.5))

model.add(Dense(50,activation='relu'))

#the final output layer with two outputs for two catogaries
model.add(Dense(2,activation='softmax'))

# Now we compile the model using 'categorical_crossentropy' loss, optimizer 'adam' and 'accuracy' as a metric
model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

# You can see an overview of the model you built using .summary()
model.summary()

## We divide our data into test and train sets with 80% training size
from sklearn.model_selection import train_test_split
X_train,Y_train,X_test,Y_test=train_test_split(test_data,output,test_size=0.2)

# We fit the model, and save it to a variable 'history' that can be accessed later to analyze the training profile
# We also set validation_split=0.2 for 20% of training data to be used for validation
# verbose=0 means you will not see the output after every epoch. 
# we set batch_size =16 its means that the numbers of training examples utilized in one iteration
# epochs set 20 mean how many time train data 

history=model.fit(X_train,Y_train,batch_size=16,epochs=20,validation_split=0.2,verbose='auto')

# graphs of training and validation accuracy
fig, ax = plt.subplots(1,2,figsize = (16,4))
ax[0].plot(history.history['loss'],color='#EFAEA4',label = 'Training Loss')
ax[0].plot(history.history['val_loss'],color='#B2D7D0',label = 'Validation Loss')
ax[1].plot(history.history['accuracy'],color='#EFAEA4',label = 'Training Accuracy')
ax[1].plot(history.history['val_accuracy'],color='#B2D7D0',label = 'Validation Accuracy')
ax[0].legend()
ax[1].legend()
ax[0].set_xlabel('Epochs')
ax[1].set_xlabel('Epochs');
ax[0].set_ylabel('Loss')
ax[1].set_ylabel('Accuracy %');
fig.suptitle('MLP Training', fontsize = 24)

#evulating our model
print(model.evaluate(X_train,Y_train))