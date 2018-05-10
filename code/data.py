import numpy as np
import argparse
import os
import cv2
import matplotlib.pyplot as plt

FLAGS = None
TRAIN_DATA = 'train.csv'
TEST_DATA = 'test.csv'

def readCsvFile():
    train_data_raw_filename, _ = os.path.splitext(TRAIN_DATA)
    train_data_raw_filename = 'raw_' + train_data_raw_filename + '.npz'
    train_data_raw_path = os.path.join(FLAGS.data_dir, train_data_raw_filename)
    if not os.path.exists(train_data_raw_path):
        TRAIN_DATA_PATH = os.path.join(FLAGS.data_dir, TRAIN_DATA)
        assert os.path.exists(TRAIN_DATA_PATH)
        train_data = np.genfromtxt(TRAIN_DATA_PATH, delimiter=',', dtype=np.uint8, skip_header=1, usecols=range(1, 785))
        train_label = np.genfromtxt(TRAIN_DATA_PATH, delimiter=',', dtype=np.uint8, skip_header=1, usecols=range(1))
        np.savez(train_data_raw_path, data=train_data, label=train_label)
    else:
        train_data_file = np.load(train_data_raw_path)
        train_data = train_data_file['data']
        train_label = train_data_file['label']

    test_data_raw_filename, _ = os.path.splitext(TEST_DATA)
    test_data_raw_filename = 'raw_' + test_data_raw_filename + '.npz'
    test_data_raw_path = os.path.join(FLAGS.data_dir, test_data_raw_filename)
    if not os.path.exists(test_data_raw_path):
        TEST_DATA_PATH = os.path.join(FLAGS.data_dir, TEST_DATA)
        assert os.path.exists(TEST_DATA_PATH)
        test_data = np.genfromtxt(TEST_DATA_PATH, delimiter=',', dtype=np.uint8, skip_header=1, usecols=range(784))
        np.savez(test_data_raw_path, data=test_data)
    else:
        test_data_file = np.load(test_data_raw_path)
        test_data = test_data_file['data']

    return train_data, train_label, test_data

def splitTrainingDataset(train_data, train_label):
    splited_train_data_filename, _ = os.path.splitext(TRAIN_DATA)
    splited_train_data_filename = 'splited_' + splited_train_data_filename + '.npz'
    splited_train_data_path = os.path.join(FLAGS.data_dir, splited_train_data_filename)
    if not os.path.exists(splited_train_data_path):
        shuffled_index = np.random.permutation(train_data.shape[-2])
        train_index = shuffled_index[0:int(0.95*shuffled_index.shape[-1])]
        validate_index = shuffled_index[int(0.95*shuffled_index.shape[-1]) : ]
        splited_train_data = train_data[train_index]
        splited_train_label = train_label[train_index]
        splited_validate_data = train_data[validate_index]
        splited_validate_label = train_label[validate_index]
        np.savez(splited_train_data_path, train_data=splited_train_data, train_label=splited_train_label,
                 validate_data=splited_validate_data, validate_label=splited_validate_label)
    else:
        splited_train_data_file = np.load(splited_train_data_path)
        splited_train_data = splited_train_data_file['train_data']
        splited_train_label = splited_train_data_file['train_label']
        splited_validate_data = splited_train_data_file['validate_data']
        splited_validate_label = splited_train_data_file['validate_label']

    return splited_train_data, splited_train_label, splited_validate_data, splited_validate_label

def cropDigitImage(image):
    # find the bounding rect of the digit
    _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_bouding_area = -1
    max_bouding_index = 0
    for ind, contour in enumerate(contours):
        temp_contour_area = cv2.contourArea(contours[ind])
        if temp_contour_area > max_bouding_area:
            max_bouding_area = temp_contour_area
            max_bouding_index = ind
    bounding_rect = cv2.boundingRect(contours[max_bouding_index])
    digit_image = image[bounding_rect[1]:bounding_rect[1] + bounding_rect[3],
                  bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
    return digit_image, bounding_rect

def generateTransformedDigitImage(image):
    image = np.reshape(image, [28, -1])
    _, bounding_box = cropDigitImage(image)
    height = bounding_box[3]
    width = bounding_box[2]

    # scale and rotate
    scale_ratio = np.random.ranf() * (1.1-0.8) + 0.8
    if scale_ratio > 1:
        height_ratio = 1.0 * 28 / height
        width_ratio = 1.0 * 28 / width
        scale_ratio = min(scale_ratio, width_ratio)
        scale_ratio = min(height_ratio, scale_ratio)
    rotation_degree = np.random.randint(-10, 10)
    affine_matrix = cv2.getRotationMatrix2D((bounding_box[1]+height/2, bounding_box[0]+width/2), rotation_degree, scale_ratio)
    warp_affine_image = cv2.warpAffine(image, affine_matrix, (28, 28))

    # translate
    digit_image, bounding_box = cropDigitImage(warp_affine_image)
    new_location = np.random.randint(10, size=2)
    new_x = min(new_location[1], 28-bounding_box[3])
    new_y = min(new_location[0], 28-bounding_box[2])
    translate_image = np.zeros((28, 28), np.uint8)
    translate_image[new_x:new_x+bounding_box[3], new_y:new_y+bounding_box[2]] = digit_image
    translate_image = np.reshape(translate_image, [1, -1])
    return translate_image

def augmentTrainData(data, label, filename=TRAIN_DATA, augment_factor=5):
    augmented_data_filename, _ = os.path.splitext(filename)
    augmented_data_filename = 'augmented_' + augmented_data_filename + '.npz'
    augmented_data_path = os.path.join(FLAGS.data_dir, augmented_data_filename)
    if not os.path.exists(augmented_data_path):
        augmented_data = np.zeros((data.shape[0]*augment_factor, data.shape[1]), data.dtype)
        augmented_label = np.zeros((label.shape[0]*augment_factor), label.dtype)
        for ind in range(data.shape[0]):
            for count in range(augment_factor):
                augmented_data[ind*augment_factor+count, :] = generateTransformedDigitImage(data[ind, :])
                augmented_label[ind*augment_factor+count] = label[ind]
        np.savez(augmented_data_path, data=augmented_data, label=augmented_label)
    else:
        augmented_data_file = np.load(augmented_data_path)
        augmented_data = augmented_data_file['data']
        augmented_label = augmented_data_file['label']

    return augmented_data, augmented_label

def convertDatasetFotmat(data, label, dataset_name):
    data_filename = 'converted_' + dataset_name + '.npz'
    data_path = os.path.join(FLAGS.data_dir, data_filename)
    if not os.path.exists(data_path):
        data = data.astype(np.float32) / 255.0
        # convert label to one-hot formation
        one_hot_label = np.zeros((label.shape[0], 10), dtype=np.float32)
        for ind in range(label.shape[0]):
            one_hot_label[ind, label[ind]] = 1.0
        np.savez(data_path, data=data, label=one_hot_label)
    else:
        data_file = np.load(data_path)
        data = data_file['data']
        one_hot_label = data_file['label']

    return data, one_hot_label

def convertDatasetFormationWithoutLabel(data, dataset_name):
    data_filename = 'converted_' + dataset_name + '.npz'
    data_path = os.path.join(FLAGS.data_dir, data_filename)
    if not os.path.exists(data_path):
        data = data.astype(np.float32) / 255.0
        np.savez(data_path, data=data)
    else:
        data_file = np.load(data_path)
        data = data_file['data']

    return data

if __name__ == '__main__':
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--data_dir', type=str,
                                   default='../data',
                                   help='Directory for storing input data')
    FLAGS, _ = commandLineParser.parse_known_args()

    train_data, train_label, test_data = readCsvFile()
    splited_train_data, splited_train_label, splited_validate_data, splited_validate_label = \
        splitTrainingDataset(train_data, train_label)
    augmented_train_data, augmented_train_label = augmentTrainData(splited_train_data, splited_train_label)
    _, _ = convertDatasetFotmat(augmented_train_data, augmented_train_label, 'train')
    _, _ = convertDatasetFotmat(splited_validate_data, splited_validate_label, 'validate')
    _ = convertDatasetFormationWithoutLabel(test_data, 'test')