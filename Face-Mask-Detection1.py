import pytesseract
import cv2
import matplotlib.pyplot as plt
import  os
import numpy as np

loc=r'C:\Users\TARUN\Desktop\New folder (3)\Masks\New Masks Dataset\Train\Mask'
features=[]
labels=[]
f=os.listdir(loc)

for i in f:
    df=cv2.imread(os.path.join(loc,i))
    gray = cv2.cvtColor(df,cv2.COLOR_BGR2GRAY)
    dfr=cv2.resize(gray,(100,100))
    features.append(dfr)
for i in range(len(f)):
    labels.append(1)



loc1=r'C:\Users\TARUN\Desktop\New folder (3)\Masks\New Masks Dataset\Train\Non Mask'
f1=os.listdir(loc1)

for i in f1:
    df1=cv2.imread(os.path.join(loc1,i))
    gray1= cv2.cvtColor(df1,cv2.COLOR_BGR2GRAY)
    dfr1=cv2.resize(gray1,(100,100))
    features.append(dfr1)
for i in range(len(f1)):
    labels.append(0)
print(len(f))
print(len(f1))


set=[]
for i,j in zip(features,labels):
    set.append([i,j])
import random
random.shuffle(set)

Features=[]
Labels=[]
for i in set:
    Features.append(i[0])
    Labels.append(i[1])


Features=np.array(Features)/255

Features=np.reshape(Features,(Features.shape[0],100,100,1))
Labels=np.array(Labels)
from keras.utils import np_utils
Labels=np_utils.to_categorical(Labels)

from keras.models import  Sequential
from keras.layers import Flatten,Dense,Dropout,Conv2D
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint


model=Sequential()

model.add(Conv2D(300,(3,3),input_shape=Features.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(200,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(60,activation='relu'))
#Dense layer of 64 neurons
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Features,Labels,test_size=0.2)
model.fit(Features,Labels,epochs=30,callbacks=[checkpoint],validation_data=(xtest,ytest))



mask_detection=model.to_json()
with open("mask_detection2.json",'w') as json_file:
    json_file.write(mask_detection)
model.save_weights('mask_detection2.h5')