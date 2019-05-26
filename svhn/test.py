from yolo.frontend import create_yolo
import os
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array, array_to_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Activation
import svhn


def getKey(item):
    return item[0]


def test(img_to_predict):
    # 1. create yolo instance
    yolo_detector = create_yolo("ResNet50", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], 416)

    # 2. load pretrained weighted file
    yolo_detector.load_weights(svhn.DETECTOR_WEIGHT)

    # 3. load images
    img_files = [os.path.join(svhn.DEFAULT_IMAGE_FOLDER, img_to_predict)]

    imgs = []
    for fname in img_files:
        try:
            img = cv2.imread(fname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        except cv2.error:
            print("NO IMAGE FOUND! please place image to be predicted here:{}".format(svhn.DEFAULT_IMAGE_FOLDER))

    #4 create Image Detector model

    input_shape = (32, 32, 3)
    num_classes = 10

    model = Sequential()
    model.add(Conv2D(32, kernel_size=3,
                     input_shape=input_shape,
                     padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.load_weights('predictor.h5')

    # 4. Predict digit region

    for img in imgs:
        boxes, probs = yolo_detector.predict(img, svhn.THRESHOLD)

        numbers_predicted = []
        for box in sorted(boxes,key=getKey):
            (x1, y1, x2, y2) = box
            image = array_to_img(img)
            stc = image.crop((x1, y1, x2, y2))
            stc = stc.resize((32, 32))
            stc = img_to_array(stc)
            stc = stc.astype('float32')
            stc /= 255
            pred = model.predict(stc.reshape(1, 32, 32, 3))
            numbers_predicted.append(np.argmax(pred) + 1)
        return 'number predicted is: '+''.join(str(x) for x in numbers_predicted)


