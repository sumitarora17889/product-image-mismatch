from PIL import Image
from pathlib import Path
import numpy as np
from keras import Input, models
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing import image
import warnings
scale = (150, 150)
basefolder = 'C:/Users/sumit/PycharmProjects/Product image mismatch/'
modelfolder= 'saved_model/'
datasetfolder= 'IndiaMART Image/'
training_folder='training_set/'
respfolder='response/'
im_folder = Path(basefolder + datasetfolder)
im_files= list(im_folder.glob('*/*'))

def getsets(modelname):
    X = list(Path(basefolder + training_folder+'set '+modelname[-1]+'/').glob('*/*'))
    Y = list(map(lambda x: x.parts[-2], X))
    return Y

def evaluatemodelfile(modelname):
    sets=getsets(modelname)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(sets)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    Y_encoded = onehot_encoder.fit_transform(integer_encoded)
    model = models.load_model(basefolder+modelfolder+modelname)
    targetfolder=modelname+'/'
    for file in im_files:
        label = file.parts[-2]
        filename = file.parts[-1]
        img = image.load_img(file, target_size=scale)
        img_tensor = image.img_to_array(img)
        img_tensor_expanded = np.expand_dims(img_tensor, axis=0)
        arr = model.predict(img_tensor_expanded)
        output=label_encoder.inverse_transform(onehot_encoder.inverse_transform(arr.astype(int)))
        print('Labelled: ' + label)
        print('Calculated: ' + output[0])
        print(arr)
        if (label == output[0]):
            out = 'Good'
        else:
            out = 'Bad/' + output[0]
        save_folder = respfolder+targetfolder + label + '/' + out + '/'
        if (os.path.isdir(save_folder) == False):
            os.makedirs(save_folder)
        image.save_img(save_folder + filename, img_tensor)

# for file in os.listdir(basefolder+modelfolder+'*nas*'):
#     if os.path.isfile(basefolder+modelfolder+file):
#         evaluatemodelfile(file)

for file in os.listdir(basefolder+modelfolder+'*nas*'):
    if os.path.isfile(basefolder+modelfolder+file):
        evaluatemodelfile(file)
# im_files = list(im_folder.glob('*/*/*'))
# im_files_labels = list(map(lambda x: x.parts[-3], im_files))




# X = list(Path(basefolder + 'google/').glob('*/*'))
# Y = list(map(lambda x: x.parts[-2], X))
#
# model=models.load_model('saved_model/pretrained_vgg_6cl_extra')
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(Y)
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# Y_encoded = onehot_encoder.fit_transform(integer_encoded)
# # print(Y_encoded)
# outfolder='./response_6cl_extra/'
# warnings.filterwarnings("ignore")
# for file in im_files:
#     label=file.parts[-3]
#     filename=file.parts[-1]
#     img = image.load_img(file, target_size=scale)
#     img_tensor = image.img_to_array(img)
#     img_tensor_expanded = np.expand_dims(img_tensor, axis=0)
#     # print(img_tensor_expanded.shape)
#     arr = model.predict(img_tensor_expanded)
#     output=label_encoder.inverse_transform(onehot_encoder.inverse_transform(arr.astype(int)))
#     print('Labelled: '+label)
#     print('Calculated: ' + output[0])
#     if (label == output[0]):
#         out='Good'
#     else:
#         out='Bad/'+output[0]
#     save_folder=outfolder+label+'/'+out+'/'
#     # print('Calculated' + str(arr))
#     if (os.path.isdir(save_folder)==False):
#         os.makedirs(save_folder)
#     image.save_img(save_folder+filename,img_tensor)


