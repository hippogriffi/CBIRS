# imports
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from numpy import linalg
import cv2
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
import operator
from keras import models
from keras import layers
import joblib
import os
import global_functions as gf


class M3(object):

    def __init__(self, img_data):
        # model param setup
        input_shape = (img_data[0].shape[0],
                       img_data[0].shape[1], img_data[0].shape[2])
        weights = 'imagenet'
        pooling = 'max'
        self.model = models.Sequential()
        conv_layer = VGG16(weights=weights, input_shape=input_shape,
                           pooling=pooling, include_top=False)
        conv_layer.trainable = False
        self.model.add(conv_layer)
        self.model.add(layers.Flatten())

    def vgg_feat_extract(self, img):
        img = np.expand_dims(img, axis=0)
        p_img = preprocess_input(img)
        feats = self.model.predict(p_img, verbose= 0 )
        return (feats[0] / np.linalg.norm(feats[0]))

    def vgg_db_feats(self, img_db):
        db_f_vector = []
        for img in img_db:
            db_f_vector.append(self.vgg_feat_extract(img))

        print("== Database features extracted ==")
        return db_f_vector

    def SVM_train(self, db_feats, db_classes, save=False):

        X_train, X_test, y_train, y_test = train_test_split(db_feats,
                                                            db_classes,
                                                            test_size=0.30)
        self.clf = LinearSVC(C=0.1, loss='squared_hinge', max_iter=5000)
        self.clf.fit(X_train, y_train)
        predicted = self.clf.predict(X_test)
        print("SVM Accuracy Score = {}".format(
            accuracy_score(y_test, predicted)))
        if save:
            print("Saving Model")

            dirname = os.path.dirname(__file__)
            filename = os.path.join(
                dirname, r'C:\Users\Joe\Desktop\UNI\Yr3\Dissertation\System\saved_models/BOW_svm.pk1')
            joblib.dump((self.clf), filename, compress=0)

        return self.clf

    def load_SVM(self):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(
            dirname, r'C:\Users\Joe\Desktop\UNI\Yr3\Dissertation\System\saved_models/BOW_svm.pk1')
        self.clf = joblib.load(filename)

    def predict_img(self, img):
        query_f = self.vgg_feat_extract(img)
        query_f = query_f.reshape(1, -1)
        predict = self.clf.predict(query_f)
        print("Predicted: {}".format(predict))
        return predict
    
    def calc_feat_distances(self, query_img, img_db):
        distances = {}
        query_feats = self.vgg_feat_extract(query_img)
        db_feats = self.vgg_db_feats(img_db)
        for a in range(len(db_feats)):
            distances[a] = euclidean(query_feats, db_feats[a])

        return gf.normalise(distances, 20)
