# Intro 
* This repository contains a CBIR (Content Based Image Retival) system.
* Completed using 3 models
* Models tested using Caltech101 dataset
# Model 1 
This model utilises 4 different feature extraction methods:
* HSV Histograms
* Dominant Colours
* Gabor Features
* Haralick Features
For each feature vector produced the eucldian distance is calculated for a distances metric.

# Model 2 
This model also utalises 4 feature extraction methods and are improved upon model 1:
* HSV Histograms
* Gabor Features
* Haralick Features
* HoG Features
A similar distance metric method is used as that in model 1.

# Model 3
This model uses the VGG-16 model for feature extraction, Distance metric is calculated using same method as model 1 + 2. 
