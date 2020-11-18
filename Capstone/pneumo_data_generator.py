## Adapted and simplified from my_data_generator.py from the BIMCV-CSUSP repo
## https://github.com/BIMCV-CSUSP/BIMCV-COVID-19

####################################################
#Imports
####################################################

import numpy as np
import os
import keras
from PIL import Image

####################################################
#variables
####################################################

dict_classes={
    1:np.array([1, 0]),
    0:np.array([0, 1])
}

####################################################
#classes
####################################################

class DataGenerator(keras.utils.Sequence):

    ''' Initialization of the generator '''
    def __init__(self, data_frame, y, x, target_channels, indexes_output=None, batch_size=8, path_to_img="./img", shuffle=True):
        # Initialization
        # Tsv data table
        self.df = data_frame
        # Image Y size
        self.y = y
        # Image X size
        self.x = x
        # Channel size
        self.target_channels = target_channels
        # batch size
        self.batch_size = batch_size
        # Boolean that allows shuffling the data at the end of each epoch
        self.shuffle = shuffle
        # Array of positions created from the elements of the table
        self.indexes = np.arange(len(data_frame.index))
        # Path for image files if not in default ./img folder
        self.path_to_img = path_to_img

    def __len__(self):
        ''' Returns the number of batches per epoch '''
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index): #***key function for this class***
        ''' Returns a batch of data (the batches are indexed) '''
        # Take the id's of the batch number "index"
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Batch initialization
        X, Y = [], []
        

        # For each index,the sample and the label is taken. Then the batch is appended
        for idx in indexes:
    
            # Image and idx index tag is get
            x, y = self.get_sample(idx)
            # This image to the batch is added
            X.append(x)
            Y.append(y)

            
    	# The created batch is returned
        #print(X)
        #print(Y)
        return np.array(X), np.array(Y) 
        #X:(batch_size, y, x), y:(batch_size, n_labels_types)

    def on_epoch_end(self):
        ''' Triggered at the end of each epoch '''
        if self.shuffle == True:
            np.random.shuffle(self.indexes) # Shuffles the data

    def get_sample(self, idx):

        '''Returns the sample and the label with the id passed as a parameter'''
        # Get the row from the dataframe corresponding to the index "idx"                                                                       
        df_row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.path_to_img,df_row["ImageID"]))
        #print(os.path.join(self.path_to_img,df_row["ImageID"]))                                                                       
        image = image.resize((self.x,self.x))
        image = np.asarray(image)
        label = dict_classes[df_row["group"]]
        image_resampled = np.reshape(image,image.shape + (self.target_channels,))
        img2=np.array(image_resampled)    
        img2.setflags(write=1)          ##Aligning to prevent data leakage in case images mess up                                                                                                            
        img2 = self.norm(img2)

        # Return the resized image and the label                                                                                                
        return img2, label

    def norm(self, image):
        image = image / 255.0
        return image.astype( np.float32 )
