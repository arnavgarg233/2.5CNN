#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import zipfile
import numpy as np
import tensorflow as tf
import nibabel as nib
from scipy import ndimage

from tensorflow import keras
from tensorflow.keras import layers


# In[2]:


import efficientnet_3D.tfkeras as efn 
from volumentations import *
from math import floor


# In[3]:


Severity_Directory = "/ifs/loni/faculty/dduncan/agarg/data/Severity"
Severity_Path = eval("os.path.abspath(r'{}')".format(Severity_Directory))
home_path = os.path.abspath(r"/ifs/loni/faculty/dduncan/agarg")
processed_scans_path = os.path.join(home_path, 'processed_scans')
ct_0_path = os.path.join(processed_scans_path, 'ct_0.npy')
ct_1_path = os.path.join(processed_scans_path, 'ct_1.npy')
ct_2_path = os.path.join(processed_scans_path, 'ct_2.npy')
ct_3_path = os.path.join(processed_scans_path, 'ct_3.npy')
ct_4_path = os.path.join(processed_scans_path, 'ct_4.npy')


# In[4]:


ct_0 = np.load(ct_0_path)
ct_1 = np.load(ct_1_path)
ct_2 = np.load(ct_2_path)
ct_3 = np.load(ct_3_path)
ct_4 = np.load(ct_4_path)


# In[5]:


floor(2.5)


# In[6]:


TRAIN_FACTOR = 0.7
TEST_FACTOR = 1 - TRAIN_FACTOR

training_images_0 = floor(len(ct_0)*TRAIN_FACTOR)
training_images_1 = floor(len(ct_1)*TRAIN_FACTOR)
training_images_2 = floor(len(ct_2)*TRAIN_FACTOR)
training_images_3 = floor(len(ct_3)*TRAIN_FACTOR)
training_images_4 = floor(len(ct_4)*TRAIN_FACTOR)

ct_train_images_0 = ct_0[:training_images_0]
ct_train_images_1 = ct_1[:training_images_1]
ct_train_images_2 = ct_2[:training_images_2]
ct_train_images_3 = ct_3[:training_images_3]
ct_train_images_4 = ct_4[:training_images_4]

ct_test_images_0 = ct_0[training_images_0:]
ct_test_images_1 = ct_1[training_images_1:]
ct_test_images_2 = ct_2[training_images_2:]
ct_test_images_3 = ct_3[training_images_3:]
ct_test_images_4 = ct_4[training_images_4:]


ct_0_labels = np.array([0 for _ in range(len(ct_0))])
ct_1_labels = np.array([1 for _ in range(len(ct_1))])
ct_2_labels = np.array([2 for _ in range(len(ct_2))])
ct_3_labels = np.array([3 for _ in range(len(ct_3))])
ct_4_labels = np.array([4 for _ in range(len(ct_4))])

ct_train_labels_0 = ct_0_labels[:training_images_0]
ct_train_labels_1 = ct_1_labels[:training_images_1]
ct_train_labels_2 = ct_2_labels[:training_images_2]
ct_train_labels_3 = ct_3_labels[:training_images_3]
ct_train_labels_4 = ct_4_labels[:training_images_4]

ct_test_labels_0 = ct_0_labels[training_images_0:]
ct_test_labels_1 = ct_1_labels[training_images_1:]
ct_test_labels_2 = ct_2_labels[training_images_2:]
ct_test_labels_3 = ct_3_labels[training_images_3:]
ct_test_labels_4 = ct_4_labels[training_images_4:]

x_train = np.concatenate((ct_train_images_0, ct_train_images_1, ct_train_images_2, ct_train_images_3, ct_train_images_4), axis=0)
y_train = np.concatenate((ct_train_labels_0, ct_train_labels_1, ct_train_labels_2, ct_train_labels_3, ct_train_labels_4))
x_val = np.concatenate((ct_test_images_0, ct_test_images_1, ct_test_images_2, ct_test_images_3, ct_test_images_4), axis=0)
y_val = np.concatenate((ct_test_labels_0, ct_test_labels_1, ct_test_labels_2, ct_test_labels_3, ct_test_labels_4))
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)
print(
    "Number of test samples in train and validation are %d and %d."
    % (y_train.shape[0], y_val.shape[0])
)


# In[7]:



import random

from scipy import ndimage


@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    print(".")
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


# In[ ]:


print('1')
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))


# In[ ]:


# setting up data loaders
# train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
print('1')
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
print('2')
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

print('3)')

batch_size = 2

print('4')
train_dataset = (
    train_loader.shuffle(len(x_train)).map(train_preprocessing).batch(batch_size).prefetch(2)    
)

validation_dataset = {
    validation_loader.shuffle(len(x_val)).map(validation_preprocessing).batch(batch_size).prefetch(2)
}


# In[ ]:


# Define data loaders
train_loader = tf.data.Dataset.from_tensor_slices


# In[ ]:


import pandas as pd
pd.read_csv('iris.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:

