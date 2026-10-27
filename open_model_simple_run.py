# trying to open the model with the correct versions 
## and it works! 
## now see if it will do this for 2 models
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras import Model

## !! important output labelling to be aware of !! 
# 2DS : ['CA','Co','CC','CBC','CP','HPC','Dif','FA','WD']
# HVPS: ['Co','CBC','CP','FA','HPC','RA','SP']
# ALL : ['CC','Co','CBC','CP','CA','Dif','FA','HPC','SP','WD']

##
model_path = '/gws/nopw/j04/dcmex/users/ezriab/Jaffeux_24/Ezri_CNN_models/'
glob_model = 'modelALL_24042024.h5py' # model2DS_2304.h5py  modelALL_24042024.h5py  modelHVPS_0305.h5py
specif_model = 'model2DS_2304.h5py' # model2DS_2304.h5py  modelALL_24042024.h5py  modelHVPS_0305.h5py

# load the model
ml_model_glob = load_model(model_path+glob_model)
ml_model_specif = load_model(model_path+specif_model)

all_ml_output = ['CC','Co','CBC','CP','CA','Dif','FA','HPC','SP','WD']
if '2DS' in specif_model:
    ml_output = ['CA','Co','CC','CBC','CP','HPC','Dif','FA','WD'] # 2ds predictions
else:
    ml_output = ['Co','CBC','CP','FA','HPC','RA','SP'] # hvps predictions

print('glob model')
print(ml_model_glob.summary())

print('specif model')
print(ml_model_specif.summary())

################# processing images 
#base_path = '/gws/nopw/j04/dcmex/users/ezriab/image_labelling/2ds_10000_sample/'
#one_image = '429121_07ch0.png'
base_path = '/gws/nopw/j04/dcmex/users/ezriab/processed_images/2ds/ch0_v6_all/220730153000_17/'
one_image = '432021_2_30ch0_17.png'
img_path = base_path+one_image

img = image.load_img(img_path, target_size=(200, 200), color_mode='grayscale')

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict using the loaded model
predictions_glob = ml_model_glob.predict(img_array)[0]
print(predictions_glob)
predictions_specif = ml_model_specif.predict(img_array)[0]
print(predictions_specif)


# Find the index of the category with the highest probability
predicted_idx_glob = np.argmax(predictions_glob)
predicted_idx_specif = np.argmax(predictions_specif)

# Get the corresponding category
predicted_cat_glob = all_ml_output[predicted_idx_glob]
print(predicted_cat_glob)
predicted_cat_specif = ml_output[predicted_idx_specif]
print(predicted_cat_specif)





