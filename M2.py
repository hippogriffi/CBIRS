import numpy as np
import cv2
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from scipy import ndimage as ndi
from skimage.filters import gabor_kernel

from skimage.feature import hog

import mahotas as mt
import operator

import global_functions as gf


# ==================== COLOUR  ==================== #

def colour_features(img):
    features = [] 
    # convert to HSV colour space
    pp_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # get clr channels
    channels = cv2.split(pp_img)
    channel_names = ('h', 's', 'v')
    for (channel, channel_name) in zip(channels, channel_names):
        # calculate histogram for each clr channel 
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        # flatten hist channels
        features.extend(hist.flatten())
    return features
# ==================== GABOR ==================== #

# create kernals at different orientations
def build_gabor_kernals(theta, freq, sigma):
    kern_l = []
    for t in range(theta):
        t = t / float(theta) * np.pi
        for f in freq:
                for s in sigma:
                    kern = gabor_kernel(f, theta = t, sigma_x=s, sigma_y=s)
                    kern_l.append(kern)
    return kern_l

# convolve given image and filters 
def convolve_filters(img, kern):
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    conv = np.sqrt(ndi.convolve(img, np.real(kern), mode='wrap')**2 +
        ndi.convolve(img, np.imag(kern), mode = 'wrap')**2)
    
    features = np.zeros(2, dtype = np.double)
    features[0] = conv.mean()
    features[1] = conv.var()
    return features

# calculate gabor feature vector for given image
def gabor_features(img, g_kernals):
    # convert img to grayscale
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = []
    #convolve all filters on img
    for kern in g_kernals:
        results.append(convolve_filters(g_img, kern))
    
    histogram = np.array(results)
    return histogram.T.flatten()

# ==================== Haralick ==================== #

def haralick_features(img):
    # convert img to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # calculate haralick features for img
    histogram = mt.features.haralick(img).mean(axis=0)      
    return histogram

# ==================== HoG ==================== #
def HOG_features(img):
     # convert img to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # calculate hog hist for img
    fd = hog(img, orientations=8, pixels_per_cell=(2,2), cells_per_block=(1,1))
    b = np.linspace(0, np.max(fd), 10+1)
    histogram, _ = np.histogram(fd, bins=b)
    
    return histogram


# ==================== Feature Functions ==================== #

# create img reigions and feature extraction based on passed function
def reigon_based(img, bin_n, func, g_kernals = None):
    # no of img reigions
    slice_n = 2
    # create blank hist for each reigion
    histogram = np.zeros((slice_n, slice_n, bin_n))
    slice_h =  np.around(np.linspace(0, img.shape[0], slice_n + 1)).astype(int)
    slice_w = np.around(np.linspace(0, img.shape[1], slice_n + 1)).astype(int)
    for h in range(len(slice_h)-1):
        for w in range(len(slice_w)-1):
            #create reigons for image
            reigon = img[slice_h[h]:slice_h[h+1], slice_w[w]: slice_w[w+1]]
            # perform feature extraction on reigon
            if func.__name__ == gabor_features.__name__:
                histogram[h][w] = func(reigon, g_kernals)
            else:
                histogram[h][w] = func(reigon)

    return histogram.flatten()


# convert img database to feature vector for a given feature descriptor
def feats_2_dataframe(db_imgs, func, reg_check = False):
    db_feat = []
    slice_n = 2
    # colour features extractor
    if func.__name__ == colour_features.__name__:
        for img in db_imgs:
            db_feat.append(func(img))

    # gabor features extractor
    elif func.__name__ == gabor_features.__name__:
        g_kernals = build_gabor_kernals(4, (0.1, 0.5, 0.8), (1, 3))
        for img in db_imgs:
            # regioncheck 
            if reg_check == True:
                db_feat.append(reigon_based(img, len(g_kernals*2), func, g_kernals))
            else: 
                db_feat.append(func(img, g_kernals))
    # haralick features extractor        
    elif func.__name__ == haralick_features.__name__:
        for img in db_imgs:
            # regioncheck 
            if reg_check == True:
                db_feat.append(reigon_based(img, 13, func))
            else:
                db_feat.append(func(img))
    # HOG features extractor   
    elif func.__name__ == HOG_features.__name__:
        for img in db_imgs:
            # regioncheck
            if reg_check == True:
                db_feat.append(reigon_based(img, 10, func))
            else:
                db_feat.append(func(img))
    else:
        print("Invalid Function")
        return
    # return features as dataframe
    db_feat_df = DataFrame(db_feat)
    return db_feat_df


# calculate distance metric for query img and img database for a given feature extraction method
def calc_feat_distances(query_img, db_df, func, reg_check = False):
    # convert features to list
    feat_vectors = db_df.values.tolist()
    distances = {}
    for a in range(len(feat_vectors)):
        if func.__name__ == gabor_features.__name__:
            g_kernals = build_gabor_kernals(4, (0.1, 0.5, 0.8), (1, 3))
            if reg_check == True:
                query_features = reigon_based(query_img, len(g_kernals*2), func, g_kernals)
            else:
                query_features = func(query_img, g_kernals)
            dist = euclidean(query_features, feat_vectors[a])
            distances[a] = dist

        elif func.__name__ == haralick_features.__name__:
            if reg_check == True:
                query_features = reigon_based(query_img, 13, func)
            else:
                query_features = func(query_img)
            dist = euclidean(query_features, feat_vectors[a])
            distances[a] = dist

        elif func.__name__ == HOG_features.__name__:
            if reg_check == True:
                query_features = reigon_based(query_img, 10, func)
            else:
                query_features = func(query_img)
            dist = euclidean(query_features, feat_vectors[a])
            distances[a] = dist
        else:
            query_features = func(query_img)
            dist = euclidean(query_features, feat_vectors[a])
            distances[a] = dist
    # scale distances from 0 - 20 
    return gf.normalise(distances, 20)