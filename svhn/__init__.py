import os
import yolo
from svhn.test import test
from svhn.traintest import traintest
import logging

logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PKG_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(PKG_ROOT)

DETECTOR_WEIGHT = os.path.join(yolo.PROJECT_ROOT, "detector.h5")
PREDICTOR_WEIGHT = 'predictor.h5'
DEFAULT_IMAGE_FOLDER = os.path.join(yolo.PROJECT_ROOT, "tests", "dataset", "svhn", "imgs")
THRESHOLD = 0.3
DEFAULT_DATA_FOLDER = os.path.join(yolo.PROJECT_ROOT, "data")
DEFAULT_PREPROCESSED_DATA_FOLDER = os.path.join(yolo.PROJECT_ROOT, "data", "preprocessed")
URL_TRAIN = "http://ufldl.stanford.edu/housenumbers/train.tar.gz"
URL_TEST = "http://ufldl.stanford.edu/housenumbers/test.tar.gz"
TEST = "test"
TRAIN = "train"
MAT_NAME = "digitStruct.mat"
