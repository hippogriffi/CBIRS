# Introduction
* This repository contains a CBIR (Content Based Image Retival) system.
* Completed using 3 models
* Models tested using Caltech101 dataset
* Interface built using [PySimpleGUI](https://github.com/PySimpleGUI)
* An extension to a project I made when doing my dissertation

-------------------------------
### Interface
The GUI surports single and multi folder upload, can be used to select what model and the query image to carry out the search. it can also be used to adjust model parameters such as selecting feature extraction methods or reigion based extraction. The text field is just there if I come around to implementing the BoW approach into the GUI. 

![GUI_1](https://github.com/hippogriffi/CBIRS/blob/main/imgs/GUI_1.png)
![GUI_4](https://github.com/hippogriffi/CBIRS/blob/main/imgs/GUI_4.png)
-------------------------------
### Model 1 
This model utilises 4 different feature extraction methods:
* HSV Histograms
* Dominant Colours
* Gabor Features
* Haralick Features

For each feature vector produced, the eucldian distance is calculated for a distances metric. The following image shows the top 10 retrival results for this model using the query image (top left).

![M1_r1](https://github.com/hippogriffi/CBIRS/blob/main/imgs/M1_run1.png)

-------------------------------
### Model 2 
This model also utalises 4 feature extraction methods and are improved upon model 1:
* HSV Histograms
* Gabor Features
* Haralick Features
* HoG Features
  
A similar distance metric method is used as that in model 1. **This model was adapted from a BoW approach using a linear SVM to learn image classes, it was not intended to be used with a distance metric so performance isnt optimal**. A example run using this model can be seen below:  

![M2_r1](https://github.com/hippogriffi/CBIRS/blob/main/imgs/M2_run1.png)


# Model 3
This model uses the [VGG-16](https://arxiv.org/abs/1409.1556) model for feature extraction, Distance metric is calculated using same method as model 1 + 2. Example runs can be seen below:

![M3_r1](https://github.com/hippogriffi/CBIRS/blob/main/imgs/M3_run1.png)
![M3_r2](https://github.com/hippogriffi/CBIRS/blob/main/imgs/M3_run2.png)
