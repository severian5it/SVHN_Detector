# SVHN  digit Predictor

The digit predictor requires Yolo V2 Region of Interest Detector, as downloadable [here](https://github.com/penny4860/Yolo-digit-detector),
that is based on a ResNet 50. The model for The predictor is CNN Convnet, able to achieve 0.95 accuracy on cropped images.

## Usage for python code

#### 0. Requirement

* python 3.5
* anaconda 4.4.0
* tensorflow 1.2.1
* keras 2.1.1
* opencv 3.3.0
* imgaug
* Etc.

I recommend to create an anaconda env that is independent of your project. You can create anaconda env for this project 
by following these simple steps. This process has been verified on MAC OS X10.

```
$ conda create -n SVHNDetector python=3.5 anaconda=4.4.0
$ source activate SVHNDetector"
(yolo) $ pip install -r requirements.txt
```

### 1. Test execution 

In this project, 2 distinct model will be used: 

* a detector for the region of interest, which pretrained weight are 
called `detector.h5`
* a predictor model on single digit called `predictor.h5`.

Please notice that all the image to be predicted must be in the folder `tests\dataset\svhn\imgs`; for each call the code 
will load both models.



### 2. Downloading and training 

This project provides a way to execute following action from the scratch:
* Download Training and Test data
* Unzip them
* Preprocess them in numpy format containing single images cropped
* start the training test.

Please notice that is a really time consumning project, spanning for 5-6 hours.


### 3. Jupyter Notebook example

Results are testable with visual plotting, running a jupyter notebok

``
jupyter notebook detection_example.ipynb
``
