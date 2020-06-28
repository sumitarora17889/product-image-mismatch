import numpy as np
# import pandas as pd
import os
# import matplotlib.pyplot as plt
import cv2
from PIL import Image
import keras
from keras.layers import Dense, Dropout, Activation, Flatten
from glob import glob
from pathlib import Path
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,MaxPooling2D
from keras.models import Sequential
# from google.colab.patches import cv2_imshow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# import warnings
from keras.layers import BatchNormalization
# from IPython.core.display import display, HTML
# import random
import re



basefolder='C:/Users/sumit/PycharmProjects/Product image mismatch/'

X=np.array([])
Y=np.array([])
X=list(Path(basefolder+'google/').glob('*/*'))
Y=list(map(lambda x : x.parts[-2],X))
print(X)
print(Y)
im_folder=Path(basefolder+'IndiaMART Image/')
#
im_files=list(im_folder.glob('*/*/*'))
im_files_labels=list(map(lambda x : x.parts[-3],im_files))
print(im_files)
print(im_files_labels)
img=Image.open(im_files[0])
# # img.show()
# print(np.array(img))
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Y)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y_encoded = onehot_encoder.fit_transform(integer_encoded)
# print(Y_encoded)


scale=(160,160)
i=0


X_images =[]
for imgpath in X:
  i=i+1
  if i%50 == 0:
    print(i)
  img=cv2.imread(str(imgpath),cv2.IMREAD_COLOR)
  imag=cv2.resize(img,scale)
  X_images.append(imag[...,:3])
X_images=np.asarray(X_images)
print(X_images.shape)
x_train,x_test,y_train,y_test = train_test_split(X_images,Y_encoded, test_size=0.30, random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)

# X_images =np.array([])
# for imgpath in X:
#   i=i+1
#   if i%50 == 0:
#     print(i)
#   img=Image.open(imgpath).resize(scale)
#   img_arr=np.array(img)[...,:3]
#   X_images=np.append(X_images,img_arr)
# print(X_images.shape)
# # X_images=np.concatenate( X_image, axis=0 )
# x_train,x_test,y_train,y_test = train_test_split(X_images,Y_encoded, test_size=0.30, random_state=42)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
#
#
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
# model.add(Dense(512))
# model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dense(56))
# model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))
model.summary()


es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=7)
mc = ModelCheckpoint('hand_sanitizer.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.3, shuffle=True, callbacks=[es,mc])
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
#
#
# warnings.filterwarnings("ignore")
# im_files=np.array(im_files)
# im_files=im_files[random.sample(range(len(im_files)), 70)]
# def evaluateIMimages(files):
#   imfiles=[]
#   imfiles_output=[]
#   img_files=[]
#   count_right = {'Lifebouy hand sanitizer':0,'himalaya hand sanitizer':0,'Savlon hand sanitizer':0,'Purell hand sanitizer':0,'dettol hand sanitizer':0}
#   count_wrong = {'Lifebouy hand sanitizer':0,'himalaya hand sanitizer':0,'Savlon hand sanitizer':0,'Purell hand sanitizer':0,'dettol hand sanitizer':0}
#   total_count = {'Lifebouy hand sanitizer':0,'himalaya hand sanitizer':0,'Savlon hand sanitizer':0,'Purell hand sanitizer':0,'dettol hand sanitizer':0}
#   for file in files:
#     label=re.search('/content/drive/My Drive/Machine Learning/IndiaMART Image/(.+?)/primary_image/.*', file).group(1)
#     img = image.load_img(file, target_size=scale)
#     img_tensor = image.img_to_array(img)
#     img_tensor_expanded = np.expand_dims(img_tensor, axis=0)
#     arr=model.predict(img_tensor_expanded)
#     # print(file)
#     display(img)
#     output=label_encoder.inverse_transform(onehot_encoder.inverse_transform(arr.astype(int)))
#     print('Labelled: '+label)
#     print('Calculated' +output[0])
#     if output[0]==label:
#       count_right[label]=count_right[label]+1
#       display(HTML('<b style="color: green">'+str(output[0]==label)+'</b><br/><br/>'))
#     else:
#       count_wrong[label]=count_wrong[label]+1
#       display(HTML('<b style="color: red">'+str(output[0]==label)+'</b><br/><br/>'))
#     total_count[label]=total_count[label]+1
#   # for sanitizer in total_count:
#   #   print(sanitizer+': '+str(count_right[sanitizer]/total_count[sanitizer]))
# evaluateIMimages(im_files)
