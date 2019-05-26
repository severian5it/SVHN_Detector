# -*- coding: utf-8 -*-
# This module is responsible for communicating with the outside of the yolo package.
# Outside the package, someone can use yolo detector accessing with this module.

import os
import numpy as np

from yolo.backend.decoder import YoloDecoder
from yolo.backend.loss import YoloLoss
from yolo.backend.network import create_yolo_network
from yolo.backend.batch_gen import create_batch_generator

from yolo.backend.utils.fit import train
from yolo.backend.utils.annotation import get_train_annotations, get_unique_labels
from yolo.backend.utils.box import to_minmax


def get_object_labels(ann_directory):
    files = os.listdir(ann_directory)
    files = [os.path.join(ann_directory, fname) for fname in files]
    return get_unique_labels(files)


def create_yolo(architecture,
                labels,
                input_size = 416,
                anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
                coord_scale=1.0,
                class_scale=1.0,
                object_scale=5.0,
                no_object_scale=1.0):

    n_classes = len(labels)
    n_boxes = int(len(anchors)/2)
    yolo_network = create_yolo_network(architecture, input_size, n_classes, n_boxes)
    yolo_loss = YoloLoss(yolo_network.get_grid_size(),
                         n_classes,
                         anchors,
                         coord_scale,
                         class_scale,
                         object_scale,
                         no_object_scale)

    yolo_decoder = YoloDecoder(anchors)
    yolo = YOLO(yolo_network, yolo_loss, yolo_decoder, labels, input_size)
    return yolo


class YOLO(object):
    def __init__(self,
                 yolo_network,
                 yolo_loss,
                 yolo_decoder,
                 labels,
                 input_size = 416):
        """
        # Args
            feature_extractor : BaseFeatureExtractor instance
        """
        self._yolo_network = yolo_network
        self._yolo_loss = yolo_loss
        self._yolo_decoder = yolo_decoder

        self._labels = labels
        self._input_size = input_size

    def load_weights(self, weight_path, by_name=False):
        if os.path.exists(weight_path):
            print("Loading pre-trained weights in", weight_path)
            self._yolo_network.load_weights(weight_path, by_name=by_name)
        else:
            print("Fail to load pre-trained weights. Make sure weight file path.")

    def predict(self, image, threshold=0.3):
        """
        # Args
            image : 3d-array (BGR ordered)

        # Returns
            boxes : array, shape of (N, 4)
            probs : array, shape of (N, nb_classes)
        """
        def _to_original_scale(boxes):
            height, width = image.shape[:2]
            minmax_boxes = to_minmax(boxes)
            minmax_boxes[:,0] *= width
            minmax_boxes[:,2] *= width
            minmax_boxes[:,1] *= height
            minmax_boxes[:,3] *= height
            return minmax_boxes.astype(np.int)

        netout = self._yolo_network.forward(image)
        boxes, probs = self._yolo_decoder.run(netout, threshold)

        if len(boxes) > 0:
            boxes = _to_original_scale(boxes)
            return boxes, probs
        else:
            return [], []

