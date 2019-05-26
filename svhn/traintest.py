import svhn
from digit_predictor import frontend


def traintest(download_file=False, preprocess_file=False):

    if download_file:
        #download of train and unzip
        frontend.download_unzip_data(svhn.URL_TRAIN)
        #download of test and unzip
        frontend.download_unzip_data(svhn.URL_TEST)

    if preprocess_file:
        #preprocess of Test to numpy files
        frontend.preprocess_data(svhn.DEFAULT_DATA_FOLDER, svhn.TEST, svhn.MAT_NAME)
        #preprocess of Train to numpy files
        frontend.preprocess_data(svhn.DEFAULT_DATA_FOLDER, svhn.TRAIN, svhn.MAT_NAME)

    #data split
    input_shape, x_train, x_test, y_train, y_test = frontend.data_split()
    #train procedure
    frontend.train(input_shape, x_train, y_train, x_test, y_test)

