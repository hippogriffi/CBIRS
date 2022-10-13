# imports
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import random
import copy
import pandas as pd
import keras
from keras.utils import image_dataset_from_directory as im
from sklearn.decomposition import PCA
from sklearn import linear_model
from skimage.color import rgb2gray
from scipy.interpolate import interp1d
from PIL import Image


path = "C:/Users/Joe/Desktop/UNI/Yr3/Dissertation/Datasets"

training_data = im(path, label_mode='int',
                   labels='train_labels', shuffle=False, seed=123)

training_data.shape
