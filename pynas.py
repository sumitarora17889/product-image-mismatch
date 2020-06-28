import numpy as np
import os
import cv2
from PIL import Image
import keras
from keras import Input, models
from keras.layers import Dense, Dropout, Activation, Flatten
from pathlib import Path
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# import warnings
from keras.layers import BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.nasnet import NASNetLarge
from keras.applications.vgg16 import preprocess_input
from keras.models import Model, load_model
import gc
from keras import backend as K
import tensorflow as tf

def vgg(x_shape,y_shape):
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=x_shape, pooling='max')
    for layer in vgg.layers:
        # if layer.name in ['block5_conv1', 'block4_conv1']:
        #   layer.trainable = True
        # else:
        layer.trainable = False
    model = Sequential()
    model.add(vgg)
    model.add(Dense(y_shape))
    model.add(Activation('softmax'))
    return model

def vgg_train(x_shape,y_shape):
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=x_shape, pooling='max')
    for layer in vgg.layers:
        # if layer.name in ['block5_conv1', 'block4_conv1']:
        #   layer.trainable = True
        # else:
        layer.trainable = True
    model = Sequential()
    model.add(vgg)
    model.add(Dense(y_shape))
    model.add(Activation('softmax'))
    return model

def vgg_train_2(x_shape,y_shape):
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=x_shape, pooling='max')
    for layer in vgg.layers[:-4]:
        # if layer.name in ['block5_conv1', 'block4_conv1']:
        #   layer.trainable = True
        # else:
        layer.trainable = False
    model = Sequential()
    model.add(vgg)
    model.add(Dense(y_shape))
    model.add(Activation('softmax'))
    return model

def vgg_plus_conv(x_shape,y_shape):
    model_vgg16_conv_2 = VGG16(weights='imagenet', include_top=False)
    for layer in model_vgg16_conv_2.layers:
        layer.trainable = False
    input_2 = Input(shape=x_shape, name='x_train')
    # Use the generated model
    output_vgg16_conv_2 = model_vgg16_conv_2(input_2)
    x2 = Conv2D(64, (3, 3), padding='same')(output_vgg16_conv_2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    # x2= MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Dropout(0.30)(x2)
    x3 = Conv2D(128, (3, 3), padding='same')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    # x3 = MaxPooling2D(pool_size=(2, 2))(x3)
    x3 = Dropout(0.30)(x3)
    x4 = Conv2D(128, (3, 3), padding='same')(x3)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    # x4 = MaxPooling2D(pool_size=(2, 2))(x4)
    x4 = Dropout(0.30)(x4)
    x4 = Flatten(name='flatten_2')(x4)
    x4 = Dense(256, activation='relu', name='fc2')(x4)
    x4 = Dropout(0.30)(x4)
    x4 = Dense(y_shape, activation='softmax', name='fc3')(x4)
    model = Model(inputs=input_2, outputs=x4)
    return model

def custom_easy(x_shape,y_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=x_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=x_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(56))
    model.add(Activation('relu'))
    model.add(Dense(y_shape))
    model.add(Activation('softmax'))
    return model

def custom_advanced(x_shape,y_shape):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", input_shape=x_shape))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(y_shape))
    model.add(Activation('softmax'))
    return model

def resnet(x_shape,y_shape):
    resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=x_shape, pooling='max',classes=y_shape)
    for layer in resnet50.layers:
        # if layer.name in ['block5_conv1', 'block4_conv1']:
        #   layer.trainable = True
        # else:
        layer.trainable = False
    model = Sequential()
    model.add(resnet50)
    model.add(Dense(y_shape))
    model.add(Activation('softmax'))
    return model

def resnet_train(x_shape,y_shape):
    resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=x_shape, pooling='max',classes=y_shape)
    for layer in resnet50.layers:
        # if layer.name in ['block5_conv1', 'block4_conv1']:
        #   layer.trainable = True
        # else:
        layer.trainable = True
    model = Sequential()
    model.add(resnet50)
    model.add(Dense(y_shape))
    model.add(Activation('softmax'))
    return model

def resnet_2(x_shape,y_shape):
    resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=x_shape, pooling='max',classes=y_shape)
    for layer in resnet50.layers[:-4]:
        # if layer.name in ['block5_conv1', 'block4_conv1']:
        #   layer.trainable = True
        # else:
        layer.trainable = False
    model = Sequential()
    model.add(resnet50)
    model.add(Dense(y_shape))
    model.add(Activation('softmax'))
    return model

def resnet_plus_conv(x_shape,y_shape):
    resnet50 = ResNet50(weights='imagenet', include_top=False)
    for layer in resnet50.layers:
        layer.trainable = False
    input_2 = Input(shape=x_shape, name='x_train')
    # Use the generated model
    output_resnet50_conv_2 = resnet50(input_2)
    x2 = Conv2D(64, (3, 3), padding='same')(output_resnet50_conv_2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    # x2= MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Dropout(0.30)(x2)
    x3 = Conv2D(128, (3, 3), padding='same')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    # x3 = MaxPooling2D(pool_size=(2, 2))(x3)
    x3 = Dropout(0.30)(x3)
    x4 = Conv2D(128, (3, 3), padding='same')(x3)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    # x4 = MaxPooling2D(pool_size=(2, 2))(x4)
    x4 = Dropout(0.30)(x4)
    x4 = Flatten(name='flatten_2')(x4)
    x4 = Dense(256, activation='relu', name='fc2')(x4)
    x4 = Dropout(0.30)(x4)
    x4 = Dense(y_shape, activation='softmax', name='fc3')(x4)
    model = Model(inputs=input_2, outputs=x4)
    return model

def nas_plus_conv(x_shape,y_shape):
    nas = NASNetLarge(weights=None, include_top=False)
    for layer in nas.layers:
        layer.trainable = False
    input_2 = Input(shape=x_shape, name='x_train')
    # Use the generated model
    output_resnet50_conv_2 = nas(input_2)
    x2 = Conv2D(64, (3, 3), padding='same')(output_resnet50_conv_2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    # x2= MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Dropout(0.30)(x2)
    x3 = Conv2D(128, (3, 3), padding='same')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    # x3 = MaxPooling2D(pool_size=(2, 2))(x3)
    x3 = Dropout(0.30)(x3)
    x4 = Conv2D(128, (3, 3), padding='same')(x3)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    # x4 = MaxPooling2D(pool_size=(2, 2))(x4)
    x4 = Dropout(0.30)(x4)
    x4 = Flatten(name='flatten_2')(x4)
    x4 = Dense(256, activation='relu', name='fc2')(x4)
    x4 = Dropout(0.30)(x4)
    x4 = Dense(y_shape, activation='softmax', name='fc3')(x4)
    model = Model(inputs=input_2, outputs=x4)
    return model

def nas_2(x_shape,y_shape):
    nas = NASNetLarge( weights=None,include_top=False, input_shape=x_shape, pooling='max',classes=y_shape)
    for layer in nas.layers[:-4]:
        # if layer.name in ['block5_conv1', 'block4_conv1']:
        #   layer.trainable = True
        # else:
        layer.trainable = False
    model = Sequential()
    model.add(nas)
    model.add(Dense(y_shape))
    model.add(Activation('softmax'))
    return model

def getmodel(modelname,x_shape,y_shape):
    if modelname=='vgg':
        model=vgg(x_shape,y_shape)
    elif modelname=='vgg_train':
        model=vgg_train(x_shape,y_shape)
    elif modelname=='vgg_2':
        model=vgg_train_2(x_shape,y_shape)
    elif modelname=='vgg_plus_conv':
        model=vgg_plus_conv(x_shape,y_shape)
    elif modelname=='custom_easy':
        model=custom_easy(x_shape,y_shape)
    elif modelname=='custom_advanced':
        model=custom_advanced(x_shape,y_shape)
    elif modelname=='resnet':
        model=resnet(x_shape,y_shape)
    elif modelname=='resnet_train':
        model=resnet_train(x_shape,y_shape)
    elif modelname=='resnet_2':
        model=resnet_2(x_shape,y_shape)
    elif modelname=='resnet_plus_conv':
        model=resnet_plus_conv(x_shape,y_shape)
    elif modelname == 'nas_2':
        model = nas_2(x_shape, y_shape)
    elif modelname == 'nas_plus_conv':
        model = nas_plus_conv(x_shape, y_shape)
    model.summary()
    return model
def train_model(setname,modelname,X_images,Y_encoded):
    x_train, x_test, y_train, y_test = train_test_split(X_images, Y_encoded, test_size=0.30, random_state=42)
    model=getmodel(modelname,x_train.shape[1:],y_train.shape[1])
    model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])
    # if (Path('hc_'+modelname+'_'+setname+'.h5').is_file()):
    #     model.load_weights(tf.train.latest_checkpoint('hc_'+modelname+'_'+setname+'.h5'))
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
    mc = ModelCheckpoint('hc_'+modelname+'_'+setname+'.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.3, shuffle=True, callbacks=[es, mc])
    scores = model.evaluate(x_test, y_test, verbose=1)
    print(scores)
    model.save(basefolder+'saved_model/hc_'+modelname+'_'+setname)
    del model
    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph()

basefolder = 'C:/Users/sumit/PycharmProjects/Product image mismatch/'
training_folder='training_set/'
scale = (331, 331)
models=['vgg','vgg_2','vgg_plus_conv','custom_easy','custom_advanced','resnet','resnet_2','resnet_plus_conv']
resnets=['resnet_2','resnet_plus_conv']
customs=['custom_easy','custom_advanced']
nas=['nas_2', 'nas_plus_conv']
# models=['resnet_2','resnet_plus_conv']
models_heavy=['vgg_train','resnet_train']
sets=['set 1','set 2', 'set 3']
for set in sets:
    X = list(Path(basefolder + training_folder + set).glob(
        '*/*'))  # Extract all files in Go ogle folder for training dataset
    Y = list(map(lambda x: x.parts[-2], X))  # Extract the mcat name from the set
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(Y)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    Y_encoded = onehot_encoder.fit_transform(integer_encoded)
    X_images = []
    for imgpath in X:
        img = cv2.imread(str(imgpath), cv2.IMREAD_COLOR)
        imag = cv2.resize(img, scale)
        X_images.append(imag[..., :3])
    X_images = np.asarray(X_images)
    for model in nas:
        train_model(set,model,X_images,Y_encoded)


