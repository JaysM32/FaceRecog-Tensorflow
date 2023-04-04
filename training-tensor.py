import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# Import tensorflow dependencies - Functional API
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# TESTING WITHOUT USING GPUs AS THIS LAPTOP ONLY HAVE INTEGRATED GPUs 
# Avoid OOM errors by setting GPU Memory Consumption Growth
#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus: 
    #tf.config.experimental.set_memory_growth(gpu, True)
    #print(gpus)

# Setup paths
POS_PATH = os.path.join('FaceRecog/tensorflow/data', 'positive')
NEG_PATH = os.path.join('FaceRecog/tensorflow/data', 'negative')
ANC_PATH = os.path.join('FaceRecog/tensorflow/data', 'anchor')

# DONE LOCALLY, RUN ONCE ON NEW PLACES IF NOT MADE
# Make the directories
#os.makedirs(POS_PATH)
#os.makedirs(NEG_PATH)
#os.makedirs(ANC_PATH)

# LFW tar file download link: http://vis-www.cs.umass.edu/lfw/

# move pictures from lfw folder to negative folder 
#for directory in os.listdir('FaceRecog/tensorflow/lfw'):
#    for file in os.listdir(os.path.join('FaceRecog/tensorflow/lfw', directory)):
#        EX_PATH = os.path.join('FaceRecog/tensorflow/lfw', directory, file)
#        NEW_PATH = os.path.join(NEG_PATH, file)
#        os.replace(EX_PATH, NEW_PATH)
