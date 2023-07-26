import os
import os.path
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import operator

import M1 as M1
import M2 as M2
from M3 import M3

# create img keys based on file names
def path_keys(fnames):
    return dict(zip(np.arange(0, len(fnames)), fnames))

# retrive images based on lowest distance 
def retrival_imgs(fnames, sorted_dist):
    sorted_img_names = dict(sorted(
        path_keys(fnames).items(), key=lambda x: sorted_dist.get(x[0])))
    return dict(zip(sorted_img_names.values(), sorted_dist.values()))

# normalise distances 
def normalise(distances, scale):
    scaler = MinMaxScaler((0, scale))
    keys = distances.keys()
    distances = np.array(list(distances.values()))
    distances = scaler.fit_transform(distances.reshape(-1, 1))
    distances = dict(zip(keys, distances))
    return distances

# compute retrieval using model 1
def M1_compute(fnames, query_img, img_data, feat_check, feat_weights):
    dist_dict = M1.model_compute(
        query_img, img_data, feat_check, feat_weights)
    file_metric = retrival_imgs(fnames, dist_dict)
    print("== Database features extracted ==")
    return file_metric

# compute retrieval using model 2
def M2_compute(fnames, query_img, img_data, feat_check, reg_check):
    distances, combined_distances = [], [] 
    # create list of all feature extraction methods
    feature_functions = [
        M2.colour_features,
        M2.gabor_features,
        M2.haralick_features,
        M2.HOG_features]

    for feat in range(4):
        # check if extraction method is selected
        if feat_check[feat] == True:
                print("Extracting using {}".format(feature_functions[feat].__name__))
                extractor = feature_functions[feat]
                # extract features for img database 
                db_df = M2.feats_2_dataframe(img_data, extractor, reg_check[feat])
                distances.append(M2.calc_feat_distances(query_img, db_df, extractor, reg_check[feat]))
    
    for b in range(len(distances[0])):
        temp = 0
        for a in range(len(distances)):    
                temp += distances[a][b]
        combined_distances.append(temp)
    # sort distances from least to most and maintain img key 
    combined_distances = dict(sorted(dict(zip(np.arange(0, len(img_data)), (np.array(
        combined_distances)))).items(), key=operator.itemgetter(1)))
    retrival_sorted = retrival_imgs(fnames, combined_distances)
    print("== Database features extracted ==")
    return retrival_sorted


# compute retrieval using model 3
def M3_compute(fnames, query_img, img_data):
    # create model instances 
    model = M3(img_data)
    db_distances = model.calc_feat_distances(query_img, img_data)
    # sort distances from least to most and maintain img key 
    db_distances = dict(sorted(zip(np.arange(0, len(img_data)), db_distances.values()),
                                key=operator.itemgetter(1)))
    retrival_sorted = retrival_imgs(fnames, db_distances)
    return retrival_sorted


