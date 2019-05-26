import PIL.Image
import PIL.ImageDraw
from keras.preprocessing.image import img_to_array, array_to_img
import h5py
import svhn
import numpy as np
import os
import keras
from keras.models import Sequential
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import operator
import urllib.request
import tarfile


class MatReader(object):
    def __init__(self, filepath):
        self.f = h5py.File(filepath)

    def get_name(self, index):
        ref = self.f.get("/digitStruct/name").value[index][0]
        filename = "".join(map(chr, map(operator.itemgetter(0), self.f[ref].value)))
        return filename

    def get_images_number(self):
        images_number = self.f.get('/digitStruct/name').shape[0]
        return images_number

    def get_digit_number(self, index):
        bbox = self.get_bbox(index)
        digits = len(bbox['label'])
        return digits

    def get_bbox(self, index):
        ref = self.f.get("/digitStruct/bbox")
        res = self.f[ref.value[index][0]]

        d = {}

        for k, v in dict(res).items():
            if v.shape[0] > 1:
                values = np.array([self.f[item[0]].value[0][0] for item in v], dtype=np.int)
            else:
                values = np.array([v.value[0][0], ], dtype=np.int)
            d[k] = values
        d["label"] = [0 if label == 10 else label for label in d["label"]]
        return d

    def get_image(self, env, index):
        filename = self.get_name(index)
        bbox = self.get_bbox(index)
        fullpath = os.path.join(svhn.DEFAULT_DATA_FOLDER, env, filename)
        image = PIL.Image.open(fullpath)
        draw = PIL.ImageDraw.Draw(image)

        for i in range(len(bbox['label'])):
            x0 = bbox['left'][i]
            y0 = bbox['top'][i]
            x1 = x0 + bbox['width'][i]
            y1 = y0 + bbox['height'][i]

            draw.rectangle([x0, y0, x1, y1])

        return image

    def get_clipped_bbox(self, index, number_index, border):
        bbox = self.get_bbox(index)
        left = bbox['left'][number_index]
        top = bbox['top'][number_index]
        width = bbox['width'][number_index]
        height = bbox['height'][number_index]

        x1 = left - border
        y1 = top - border
        x2 = left + width + border
        y2 = top + height + border
        label = bbox["label"][number_index]
        return (x1, y1, x2, y2), label

    def clip(self, env, index):
        filename = self.get_name(index)
        fullpath = os.path.join(svhn.DEFAULT_DATA_FOLDER, env, filename)
        image = PIL.Image.open(fullpath)

        digits = self.get_digit_number(index)
        images, labels = [[], []]

        for i in range(digits):
            (x1, y1, x2, y2), label = self.get_clipped_bbox(index, i, 0)
            labels.append(label)
            images.append(image.crop((x1, y1, x2, y2)).resize((32, 32)))
        return images, labels

    def save_preprocessed_file(self, env):
        image_number = self.get_images_number()
        images_to_save, labels_to_save = [[], []]
        for index in range(image_number):
            images, labels = self.clip(env, index)
            images_array = [img_to_array(x) for x in images]
            print('processed img:{}'.format(index))
            images_to_save += images_array
            labels_to_save += labels

        np.save(os.path.join(svhn.DEFAULT_PREPROCESSED_DATA_FOLDER, env, 'images.npy'), images_to_save)
        np.save(os.path.join(svhn.DEFAULT_PREPROCESSED_DATA_FOLDER, env, 'label.npy'), labels_to_save)


def preprocess_data(location, env, mat):
    mat_file = os.path.join(location, env, mat)
    reader = MatReader(mat_file)
    print('preprocessing of {} started'.format(mat_file))
    reader.save_preprocessed_file(env)
    print('preprocessing of {} completed'.format(mat_file))


def download_unzip_data(url):
    output = url.rsplit('/', 1)[-1]
    file_name = os.path.join(svhn.DEFAULT_DATA_FOLDER, output)

    print('download of {} initiated'.format(output))
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    print('download of {} completed'.format(output))

    print('extracting {}'.format(output))
    tar = tarfile.open(file_name, "r")
    tar.extractall(svhn.DEFAULT_DATA_FOLDER)
    tar.close()
    print('extract of {} completed'.format(output))


def data_split():
    r"""return processed train and test data,
    the actual process takes place here includes:
    1. reshape to nofrows, 32, 32, 3
    2. shuffle
    3. dummify labels
    """
    x_train = np.load(os.path.join(svhn.DEFAULT_PREPROCESSED_DATA_FOLDER, svhn.TRAIN, "images.npy"))
    y_train = np.load(os.path.join(svhn.DEFAULT_PREPROCESSED_DATA_FOLDER, svhn.TRAIN, "label.npy"))
    x_test = np.load(os.path.join(svhn.DEFAULT_PREPROCESSED_DATA_FOLDER, svhn.TEST, "images.npy"))
    y_test = np.load(os.path.join(svhn.DEFAULT_PREPROCESSED_DATA_FOLDER, svhn.TEST, "label.npy"))

    print(x_train.shape)
    print(x_test.shape)

    img_rows, img_cols = x_train.shape[1], x_train.shape[2]
    num_classes = 10  # starts with 1 not 0

    y_test1 = y_test.reshape((y_test.shape[0],))
    y_test1 = [y - 1 for y in y_test1]

    y_train1 = y_train.reshape((y_train.shape[0],))
    y_train1 = [y - 1 for y in y_train1]

    input_shape = (img_rows, img_cols, 3)

    X_train = x_train.astype('float32')
    X_test = x_test.astype('float32')

    X_train /= 255
    X_test /= 255
    print('x_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train2 = keras.utils.to_categorical(y_train1, num_classes)
    y_test2 = keras.utils.to_categorical(y_test1, num_classes)

    y_train2 = y_train2.astype('int32')
    y_test2 = y_test2.astype('int32')

    print(
        "after process: X train shape: {}, X test shape: {}, y train shape: {}, y test shape: {}".format(x_train.shape,
                                                                                                         x_test.shape,
                                                                                                         y_train2.shape,
                                                                                                         y_test2.shape))
    return input_shape, X_train, X_test, y_train2, y_test2


def train(input_shape, x_train, y_train, x_test, y_test):
    num_classes = 10
    batch_size = 128
    epochs = 30

    callbacks = [EarlyStopping(monitor='val_loss', patience=5),
                 ModelCheckpoint(filepath='best_model.h5',
                                 monitor='val_loss', save_best_only=True)]

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
    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer="adam",
                  metrics=['accuracy', f1])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=callbacks)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test f1 score:', score[2])
    return print(score[2])


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
