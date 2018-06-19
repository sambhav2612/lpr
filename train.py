import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu

letters = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H' 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]


def read_data(training_data):
    image_data = []
    target_data = []

    for letter in letters:
        for each in range(10):
            image_path = os.path.join(
                training_data, letter, letter + '_' + str(each) + '.jpg')
            # read each image of each character detected
            image_details = imread(image_path, as_gray=True)
            binary_image = image_details < threshold_otsu(image_details)
            # reshape to adapt to machine learning (one-D arrays)
            flat_bin_image = binary_image.reshape(-1)
            image_data.append(flat_bin_image)
            target_data.append(letter)

    return (np.array(image_data), np.array(target_data))


# mesaures the accuracy of a model using cross validation
def validation(model, folds, data, label):
    # folds: type of cross validation
    # ex: folds = 4, 4-fold cross validation i.e., it will divide the dataset into 4 and use 1/4 of it for testing and the remaining 3/4 for the training

    accuracy_result = cross_val_score(model, data, label, cv=folds)
    print("Cross Validation Result for ", str(folds), " -fold")
    print(accuracy_result*100)


current_dir = os.path.dirname(os.path.realpath(__file__))
training_dataset_dir = os.path.join(current_dir, 'train/')
image_data, target_data = read_data(training_dataset_dir)

svc_model = SVC(kernel='linear', probability=True)
validation(svc_model, 4, image_data, target_data)

svc_model.fit(image_data, target_data)
save_directory = os.path.join(current_dir, 'models/svc/')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

joblib.dump(svc_model, save_directory + '/svc.pkl')
