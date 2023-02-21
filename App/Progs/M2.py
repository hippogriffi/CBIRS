
import math
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from skimage.feature import hog
import numpy as np
import cv2
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
import operator
import os
import pickle

from sklearn.cluster import KMeans
from itertools import chain
import mahotas as mt
import itertools
import imageio.v3 as iio


# ==================== HOG ==================== #

def HOG_features(img, bin_n, norm):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fd = hog(img, orientations=8, pixels_per_cell=(
        2, 2), cells_per_block=(1, 1))
    b = np.linspace(0, np.max(fd), bin_n+1)
    histogram, _ = np.histogram(fd, bins=b)

    if norm:
        histogram = (histogram - histogram.mean()) / (histogram.std())

    return histogram


def HOG_hist(img, type_h, bin_n, slice_n, norm):

    if type_h == 'global':
        histogram = HOG_features(img, bin_n, norm)

    elif type_h == 'reigon':
        histogram = np.zeros((slice_n, slice_n, bin_n))
        slice_h = np.around(np.linspace(
            0, img.shape[0], slice_n + 1)).astype(int)
        slice_w = np.around(np.linspace(
            0, img.shape[1], slice_n + 1)).astype(int)

        for h in range(len(slice_h)-1):
            for w in range(len(slice_w)-1):
                # create reigons for image
                reigon = img[slice_h[h]:slice_h[h+1], slice_w[w]: slice_w[w+1]]
                histogram[h][w] = HOG_features(reigon, bin_n, norm)

    return histogram.flatten()


def HOG_db_feats(img_db, type_h, bin_n, slice_n, norm):
    db_features = []
    for img in img_db:
        db_features.append(HOG_hist(img, type_h, bin_n, slice_n, norm))
    db_df = DataFrame(db_features)
    return db_df


# ==================== COLOUR  ==================== #

def colour_features(img):
    feats = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    channels = cv2.split(img)
    channel_n = ('h', 's', 'v')
    for (channel, channel_n) in zip(channels, channel_n):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        feats.extend(hist)
    return feats


def colour_hist(img, type_h, bin_n, slice_n, norm):
    bins = np.linspace(0, 256, bin_n+1, endpoint=True)
    if type_h == 'global':
        histogram = colour_features(img)

    elif type_h == 'reigon':
        histogram = np.zeros((slice_n, slice_n, 256 * img.shape[2]))
        slice_h = np.around(np.linspace(
            0, img.shape[0], slice_n + 1)).astype(int)
        slice_w = np.around(np.linspace(
            0, img.shape[1], slice_n + 1)).astype(int)

        for h in range(len(slice_h)-1):
            for w in range(len(slice_w)-1):
                # create reigons for image
                reigon = img[slice_h[h]:slice_h[h+1], slice_w[w]: slice_w[w+1]]
                histogram[h][w] = colour_features(reigon)

        if norm:
            histogram = (histogram - histogram.mean()) / (histogram.std())

    return np.array(histogram)


def colour_db_feats(img_db, type_h, bin_n, slice_n, norm):
    db_features = []
    for img in img_db:
        db_features.append(colour_hist(img, type_h, bin_n, slice_n, norm))
    db_hist_df = DataFrame(db_features)
    return db_hist_df


# ==================== GABOR ==================== #

# create kernals at different orientations

def build_gabor_kernals(theta, freq, sigma):
    kern_l = []
    for t in range(theta):
        t = t / float(theta) * np.pi
        for f in freq:
            for s in sigma:
                kern = gabor_kernel(f, theta=t, sigma_x=s, sigma_y=s)
                kern_l.append(kern)
    return kern_l


def convolve_filters2(img, kern):
    img = (img - img.mean()) / img.std()
    conv = np.sqrt(ndi.convolve(img, np.real(kern), mode='wrap')**2 +
                   ndi.convolve(img, np.imag(kern), mode='wrap')**2)

    features = np.zeros(2, dtype=np.double)
    features[0] = conv.mean()
    features[1] = conv.var()
    return features


def gabor_features2(img, g_kernals, norm):
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = []

    for kern in g_kernals:
        results.append(convolve_filters2(g_img, kern))

    histogram = np.array(results)
    if norm:
        histogram = (histogram - histogram.mean()) / (histogram.std())

    return histogram.T.flatten()


def gabor(img, g_kernals, type_h, slice_n, norm):
    if type_h == 'global':
        histogram = gabor_features2(img, g_kernals, norm)

    elif type_h == 'reigon':
        histogram = np.zeros((slice_n, slice_n, len(g_kernals)*2))
        slice_h = np.around(np.linspace(
            0, img.shape[0], slice_n + 1)).astype(int)
        slice_w = np.around(np.linspace(
            0, img.shape[1], slice_n + 1)).astype(int)

        for h in range(len(slice_h)-1):
            for w in range(len(slice_w)-1):
                # create reigons for image
                reigon = img[slice_h[h]:slice_h[h+1], slice_w[w]: slice_w[w+1]]
                histogram[h][w] = gabor_features2(reigon, g_kernals, norm)

    return histogram.flatten()


def gabor_db_feats(img_db, g_kernals, type_h, slice_n, norm):
    db_features = []
    for img in img_db:
        db_features.append(gabor(img, g_kernals, type_h, slice_n, norm))
    db_df = DataFrame(db_features)
    return db_df


# ==================== HARALICK ==================== #

def haralick_features(img, norm):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    histogram = mt.features.haralick(img).mean(axis=0)
    if norm:
        histogram = np.array(histogram) / np.sum(histogram)
    return histogram


def haralick(img, type_h, slice_n, norm):
    if type_h == 'global':
        histogram = haralick_features(img, norm)

    elif type_h == 'reigon':
        histogram = np.zeros((slice_n, slice_n, 13))
        slice_h = np.around(np.linspace(
            0, img.shape[0], slice_n + 1)).astype(int)
        slice_w = np.around(np.linspace(
            0, img.shape[1], slice_n + 1)).astype(int)

        for h in range(len(slice_h)-1):
            for w in range(len(slice_w)-1):
                # create reigons for image
                img_r = img[slice_h[h]:slice_h[h+1], slice_w[w]: slice_w[w+1]]
                histogram[h][w] = haralick_features(img_r, norm)

        if norm:
            histogram = (histogram - histogram.mean()) / (histogram.std())

    return histogram.flatten()


def haralick_db_feats(img_db, type_h, slice_n, norm):
    db_features = []
    for img in img_db:
        db_features.append(haralick(img, type_h, slice_n, norm))
    db_df = DataFrame(db_features)
    return db_df

# ==================== SVM ==================== #


def load_svm():
    print('Loading Model')
    filepath = r'C:\Users\Joe\Desktop\UNI\Yr3\Dissertation\System\saved_models\BOW_svm.pk1'
    clf = pickle.load(open(filepath, 'rb'))
    return clf


def predict(model, query_feats):
    query_f = query_feats.reshape(1, -1)
    predict = model.predict(query_f)
    print("Predicted: {}".format(predict))
    return predict

# ==================== Feature Extraction ==================== #

# feature extraction for full db


def process_M2(img_db, slice_n=2, type_h='reigon', norm=False):
    scaler = MinMaxScaler()

    # colour features
    colour_f = colour_db_feats(img_db, 'global', 12, 3, norm)
    colour_f = scaler.fit_transform(colour_f)

    # texture features
    gabor_kernals = build_gabor_kernals(4, (0.1, 0.5, 0.8), (1, 3))
    gab_f = gabor_db_feats(img_db, gabor_kernals, type_h, slice_n, norm)
    gab_f = scaler.fit_transform(gab_f)

    hara_f = haralick_db_feats(img_db, type_h, slice_n, norm)
    hara_f = scaler.fit_transform(hara_f)

    # shape features
    hog_b = 10  # number of bins for each img
    HOG_f = HOG_db_feats(img_db, type_h, hog_b, slice_n, norm)
    HOG_f = scaler.fit_transform(HOG_f)

    # feature fusion
    feat_vector = np.concatenate((colour_f, gab_f, hara_f, HOG_f), axis=1)

    return feat_vector


# feature extraction for query image
def process_query_M2(query_img, slice_n=2, type_h='reigon', norm=False):
    scaler = MinMaxScaler()

    colour_f = colour_hist(query_img, 'global', 12, 3, norm).reshape(-1, 1)
    colour_f = scaler.fit_transform(colour_f)

    # texture features
    gabor_kernals = build_gabor_kernals(4, (0.1, 0.5, 0.8), (1, 3))
    gab_f = gabor(query_img, gabor_kernals, type_h,
                  slice_n, norm).reshape(-1, 1)
    gab_f = scaler.fit_transform(gab_f)

    hara_f = haralick(query_img, type_h, slice_n, norm).reshape(-1, 1)
    hara_f = scaler.fit_transform(hara_f)

    # shape features
    hog_b = 10  # number of bins for each img
    HOG_f = HOG_hist(query_img, type_h, hog_b, slice_n, norm).reshape(-1, 1)
    HOG_f = scaler.fit_transform(HOG_f)

    # feature fusion
    query_feats = np.concatenate((colour_f, gab_f, hara_f, HOG_f), axis=0)

    # svm prediction
    clf = load_svm()
    result = predict(clf, query_feats)
    return result, query_feats
