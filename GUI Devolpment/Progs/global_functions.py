import os
import os.path
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler


file_path = 'C:/Users/Joe/Desktop/UNI/Yr3/Dissertation/Datasets/101_ObjectCategories'
folder_names = []
folder_names = [f for f in sorted(os.listdir(file_path))]


def load_caltech(path, cat_num, img_num):
    img_data = []
    folder_names = [f for f in sorted(os.listdir(path))]
    selected_categories = np.random.sample(range(101), cat_num)

    for a, cat in enumerate(selected_categories):
        folder_path = file_path + '/' + folder_names[cat]
        image_names = [a for a in sorted(
            os.listdir(folder_path))][:img_num]

    for b, img_name in enumerate(image_names):
        img_path = folder_path + '/' + img_name
        img = cv2.imread(img_path)
        img = cv2.resize(img, (100, 100))

        if img is not None:
            img_data.append(img)

    return img_data


def normalise(distances, scale):
    scaler = MinMaxScaler((0, scale))
    keys = distances.keys()
    distances = np.array(list(distances.values()))
    distances = scaler.fit_transform(distances.reshape(-1, 1))
    distances = dict(zip(keys, distances))
    return distances
