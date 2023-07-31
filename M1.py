import numpy as np
import cv2
from pandas import DataFrame
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans

import mahotas as mt
import operator

import global_functions as gf

# Look at model 2 comments or report for more detail 

def histogram_features(img):
    features = []
    pp_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    channels = cv2.split(pp_img)
    channel_names = ('h', 's', 'v')
    for (channel, channel_name) in zip(channels, channel_names):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        features.extend(hist.flatten())
        return features


def hist_features_database(data_imgs):
    db_features = []
    for img in data_imgs:
        db_features.append(histogram_features(img))
    db_hist_df = DataFrame(db_features)
    return db_hist_df


def calc_hist_distance(query_img, db_df):
    feature_vectors = db_df.values.tolist()
    distances = {}
    for a in range(len(feature_vectors)):
        query_features = histogram_features(query_img)
        dist = euclidean(query_features, feature_vectors[a])
        distances[a] = dist
    return gf.normalise(distances, 20)


def build_filters():
    filters = []
    kernal_size = 9
    for theta in np.arange(0, np.pi, np.pi / 8):
        for deg in np.arange(0, 6*np.pi/4, np.pi / 4):
            kernal = cv2.getGaborKernel(
                (kernal_size, kernal_size), 1.0, theta, deg, 0.5, 0, ktype=cv2.CV_32F)
            kernal /= 1.5 * kernal.sum()
            filters.append(kernal)
    return filters


def convolve_filters(img, filters):
    conv = np.zeros_like(img)
    for kernal in filters:
        filter_img = cv2.filter2D(img, cv2.CV_8UC3, kernal)
        np.maximum(conv, filter_img, conv)
    return conv


def gabor_features(img):
    features = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filters = np.asarray(build_filters())

    for a in range(20):
        energy = 0
        conv = convolve_filters(img, filters[a])
        for b in range(100):
            for c in range(100):
                energy += int(conv[b][c]) * int(conv[b][c])
        features.append(energy)
    for a in range(20):
        mean = 0
        conv = convolve_filters(img, filters[a])
        for b in range(100):
            for c in range(100):
                mean += abs(conv[a][b])
        features.append(mean)
    features = np.array(features)
    return features


def gabor_features_database(data_imgs):
    db_feat = []
    for img in data_imgs:
        db_feat.append(gabor_features(img))
    db_gabor_df = DataFrame(db_feat)
    return db_gabor_df


def calc_gabor_distance(query_img, db_df):
    distances = {}
    query_feat = gabor_features(query_img)
    feature_vector = db_df.values.tolist()

    for a in range(len(feature_vector)):
        distances[a] = euclidean(query_feat, feature_vector[a])
    distances = gf.normalise(distances, 20)
    return distances

#  Haralick Features


def haralick_features(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texture = mt.features.haralick(img)
    features = texture.mean(axis=0)
    return features


def haralick_features_database(img_db):
    features = []
    for img in img_db:
        feat = haralick_features(img)
        features.append(feat)
    haralick_df = DataFrame(features)
    return haralick_df


def calc_haralick_distance(query_img, db_df):
    distances = {}
    query_feat = haralick_features(query_img)
    haralick_fv = db_df.values.tolist()
    for a in range(len(haralick_fv)):
        img_feats = haralick_fv[a]
        dist = euclidean(query_feat, img_feats)
        distances[a] = dist
    distances = gf.normalise(distances, 20)
    return distances

# Dominant Colour Features
def dom_colour_features(img, colour_num):
    dom_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ratio = dom_img.shape[0]/dom_img.shape[1]
    height = int(dom_img.shape[1] * ratio)
    dimentions = (50, height)
    dom_img = cv2.resize(dom_img, dimentions)

    channels = dom_img.reshape((dom_img.shape[0] * dom_img.shape[1], 3))

    k = KMeans(n_clusters=colour_num, random_state=31, n_init=10)
    k.fit(channels)
    clusters = k.cluster_centers_.astype(int)
    return clusters.flatten()


def dom_colour_features_database(img_db, colour_num):
    features = []
    for img in img_db:
        feat = dom_colour_features(img, colour_num)
        features.append(feat)
    dominant_df = DataFrame(features)
    return dominant_df


def calc_dominant_distance(query_img, db_df, colour_num):
    distances = {}
    query_feat = dom_colour_features(query_img, colour_num)
    dominant_fv = db_df.values.tolist()
    for a in range(len(dominant_fv)):
        img_feats = dominant_fv[a]
        dist = euclidean(query_feat, img_feats)
        distances[a] = dist
    distances = gf.normalise(distances, 20)
    return distances


# distance metric calculation
def calc_distances_total(hist_dist, gabor_dist, hara_dist, dom_dist, db_length, feat_weights):
    total_dist = []

    # weight distances
    for a in hist_dist:
        hist_dist[a] *= feat_weights[0]
        gabor_dist[a] *= feat_weights[1]
        hara_dist[a] *= feat_weights[2]
        dom_dist[a] *= feat_weights[3]
        # combine distances
        total_dist.append(
            hist_dist[a] + gabor_dist[a] + hara_dist[a] + dom_dist[a])
    # sort distances from least to most and maintain img key 
    dist_final = dict(sorted(dict(zip(np.arange(0, db_length), (np.array(
        total_dist)))).items(), key=operator.itemgetter(1)))
    return dist_final


def model_compute(query_img, img_data, feat_check, feat_weights):
    # initialise empty dict for distances
    hist_dist = {b: 0 for b in range(len(img_data))}
    gabor_dist = {b: 0 for b in range(len(img_data))}
    hara_dist = {b: 0 for b in range(len(img_data))}
    dom_dist = {b: 0 for b in range(len(img_data))}

    if (feat_check[0]):
        hist_dist = calc_hist_distance(
            query_img, hist_features_database(img_data))
    if (feat_check[1]):
        gabor_dist = calc_gabor_distance(
            query_img, gabor_features_database(img_data))
    if (feat_check[2]):
        hara_dist = calc_haralick_distance(
            query_img, haralick_features_database(img_data))
    if (feat_check[3]):
        colour_num = 1
        dom_dist = calc_dominant_distance(
            query_img, dom_colour_features_database(img_data, colour_num), colour_num)
    # calculate distances from query image
    final_dist = calc_distances_total(
        hist_dist, gabor_dist, hara_dist, dom_dist, len(img_data), feat_weights)

    return final_dist
