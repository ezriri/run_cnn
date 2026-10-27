## now doing model predictions for all images!! 
import tensorflow as tf
import numpy as np
import pandas as pd
from glob import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras import Model
import time
import os

## image_location
base_path = '/gws/nopw/j04/dcmex/users/ezriab/processed_images/'
file_group = 'hvps_v6_all/' # '2ds/ch0_v6_all/' '2ds/ch1_v6_all/' 'hvps_v6_all/'
batch_size = 64 # when processing images we can pass multiple at once -> batch size

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
## !! important output labelling to be aware of !! 
# 2DS : ['CA','Co','CC','CBC','CP','HPC','Dif','FA','WD']
# HVPS: ['Co','CBC','CP','FA','HPC','RA','SP']
# ALL : ['CC','Co','CBC','CP','CA','Dif','FA','HPC','SP','WD']

## model set up + save path 
model_path = '/gws/nopw/j04/dcmex/users/ezriab/Jaffeux_24/Ezri_CNN_models/'
base_save = '/gws/nopw/j04/dcmex/users/ezriab/processed_stats/cnn_predictions/'

if '2ds' in file_group:
    specif_model = 'model2DS_2304.h5py'
    cat_specif = ['CA','Co','CC','CBC','CP','HPC','Dif','FA','WD'] # 2ds categories
    if 'ch0' in file_group:
        save_location = base_save + '2ds/ch0/'
    else:
        save_location = base_save + '2ds/ch1/'

else:
    specif_model = 'modelHVPS_0305.h5py'
    cat_specif = ['Co','CBC','CP','FA','HPC','RA','SP'] # hvps categories
    save_location = base_save + 'hvps/'

glob_model = 'modelALL_24042024.h5py'
cat_glob = ['CC','Co','CBC','CP','CA','Dif','FA','HPC','SP','WD']

full_category_list = list(set(cat_specif+cat_glob))

# load the model
ml_model_glob = load_model(model_path+glob_model)
ml_model_specif = load_model(model_path+specif_model)


# folder stuff - looping time
full_path = base_path + file_group
folder_list = os.listdir(full_path) # all sub folders containing images - this will be the new csv name
# got folder for ones already ran (as script stopped for reasons unknown)
if 'cnn_processed' in folder_list:
    folder_list.remove('cnn_processed')

for i in range(len(folder_list)):
#for i in range(3):
    img_path = full_path + folder_list[i] + '/'
    file_list = glob(img_path + '*.png')
    file_names = [os.path.basename(file_path) for file_path in file_list]
    # # # # # # # # processing one file # # # # # # # 
    results = []
    
    # this will write down any problems encountered
    txt_save = save_location + f'txt/{folder_list[i]}_cnn_run_output.txt'
    with open(txt_save, "w") as file:
        total_start_time = time.time()  # Track overall start time
        
        ## going to batch the images together to process faster, instead of looping individually:
        for b in range(0, len(file_list), batch_size):
            start_time = time.time()
            batch_files = file_list[b:b+batch_size]  # This handles the final, smaller batch safely
        
            batch_imgs = []
            batch_names = []
        
            for f in batch_files:
                try: 
                    img = image.load_img(f, target_size=(200, 200), color_mode='grayscale')
                    if img is None:
                        file.write(f'{f} skipped: load issue\n')
                        continue
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                    batch_imgs.append(img_array)
                    batch_names.append(os.path.basename(f)[:-4])
                except Exception as e:
                    print(f"Error processing {f}: {repr(e)}")
                    file.write(f'Error processing {f}: {repr(e)}\n')
            
            batch_array = np.vstack(batch_imgs) # stack images together

            # Predict on the whole batch + + + + + + + + + + + + + + + + + + + + + + + + + + + +
            pred_specif = ml_model_specif.predict(batch_array, verbose=0)
            pred_glob = ml_model_glob.predict(batch_array, verbose=0)
            # + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

            # adding this to be readable
            for j, name in enumerate(batch_names):
                row = {
                    'Name': name,
                    'Category_specif': cat_specif[np.argmax(pred_specif[j])],
                    'Category_glob': cat_glob[np.argmax(pred_glob[j])]
                }
        
                for cat in full_category_list:
                    row[f'{cat}_specif'] = (
                        pred_specif[j][cat_specif.index(cat)] if cat in cat_specif else None)
                    row[f'{cat}_glob'] = (
                        pred_glob[j][cat_glob.index(cat)] if cat in cat_glob else None)
        
                results.append(row)
                
            # Log the time taken for each batch 
            end_time = time.time()
            elapsed_time = end_time - start_time
            file.write(f'batch processing time: {elapsed_time:.2f} seconds\n')
    
        # Log the total processing time
        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
        file.write(f'Total processing time: {total_elapsed_time:.2f} seconds\n')
        print(f'Total processing time: {total_elapsed_time:.2f} seconds')
    
    # Save to DataFrame and CSV
    df = pd.DataFrame(results)
    csv_save = f'{save_location}{folder_list[i]}.csv'
    df.to_csv(csv_save, index=False)
    print(f'Processing {folder_list[i]} finished.')





