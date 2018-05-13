import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2

FLAGS = None

class Dataset(object):
    def __init__(self, data_dir, augment_factor=5, train_data='train.csv', test_data='test.csv'):
        self.__data_dir = data_dir
        self.__augment_factor = augment_factor
        self.__train_data = train_data
        self.__test_data = test_data
        self.__train_data_path = ''
        self.__test_data_path =''

    def readCsvFile(self):
        print('read raw data from the csv file...')
        train_data_raw_filename, _ = os.path.splitext(self.__train_data)
        train_data_raw_filename = 'raw_' + train_data_raw_filename + '.npz'
        train_data_raw_path = os.path.join(self.__data_dir, train_data_raw_filename)
        if not os.path.exists(train_data_raw_path):
            self.__train_data_path = os.path.join(self.__data_dir, self.__train_data)
            print(self.__train_data_path)
            assert os.path.exists(self.__train_data_path)
            train_data = np.genfromtxt(self.__train_data_path, delimiter=',', dtype=np.uint8, skip_header=1, usecols=range(1, 785))
            train_label = np.genfromtxt(self.__train_data_path, delimiter=',', dtype=np.uint8, skip_header=1, usecols=range(1))
            np.savez(train_data_raw_path, data=train_data, label=train_label)
        else:
            train_data_file = np.load(train_data_raw_path)
            train_data = train_data_file['data']
            train_label = train_data_file['label']

        test_data_raw_filename, _ = os.path.splitext(self.__test_data)
        test_data_raw_filename = 'raw_' + test_data_raw_filename + '.npz'
        test_data_raw_path = os.path.join(self.__data_dir, test_data_raw_filename)
        if not os.path.exists(test_data_raw_path):
            self.__test_data_path = os.path.join(self.__data_dir, self.__test_data)
            print(self.__test_data_path)
            assert os.path.exists(self.__test_data_path)
            test_data = np.genfromtxt(self.__test_data_path, delimiter=',', dtype=np.uint8, skip_header=1, usecols=range(784))
            np.savez(test_data_raw_path, data=test_data)
        else:
            test_data_file = np.load(test_data_raw_path)
            test_data = test_data_file['data']

        return train_data, train_label, test_data

    def splitTrainingDataset(self, train_data, train_label):
        print('split validation dataset from the training dataset...')
        splited_train_data_filename, _ = os.path.splitext(self.__train_data)
        splited_train_data_filename = 'splited_' + splited_train_data_filename + '.npz'
        splited_train_data_path = os.path.join(self.__data_dir, splited_train_data_filename)
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

    def cropDigitImage(self, image):
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

    def generateTransformedDigitImage(self, image):
        image = np.reshape(image, [28, -1])
        _, bounding_box = self.cropDigitImage(image)
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
        digit_image, bounding_box = self.cropDigitImage(warp_affine_image)
        new_x = np.random.randint(28-bounding_box[3])
        new_y = np.random.randint(28-bounding_box[2])
        translate_image = np.zeros((28, 28), np.uint8)
        translate_image[new_x:new_x+bounding_box[3], new_y:new_y+bounding_box[2]] = digit_image
        translate_image = np.reshape(translate_image, [1, -1])
        return translate_image

    def augmentTrainData(self, data, label):
        print('augment training dataset...')
        augmented_data_filename, _ = os.path.splitext(self.__train_data)
        augmented_data_filename = 'augmented_' + augmented_data_filename + '.npz'
        augmented_data_path = os.path.join(self.__data_dir, augmented_data_filename)
        if not os.path.exists(augmented_data_path):
            augmented_data = np.zeros((data.shape[0]*self.__augment_factor, data.shape[1]), data.dtype)
            augmented_label = np.zeros((label.shape[0]*self.__augment_factor), label.dtype)
            for ind in range(data.shape[0]):
                for count in range(self.__augment_factor):
                    augmented_data[ind*self.__augment_factor+count, :] = self.generateTransformedDigitImage(data[ind, :])
                    augmented_label[ind*self.__augment_factor+count] = label[ind]
            np.savez(augmented_data_path, data=augmented_data, label=augmented_label)
        else:
            augmented_data_file = np.load(augmented_data_path)
            augmented_data = augmented_data_file['data']
            augmented_label = augmented_data_file['label']

        return augmented_data, augmented_label

    def convertDatasetFotmat(self, data, label, dataset_name):
        data_filename = 'converted_' + dataset_name + '.npz'
        data_path = os.path.join(self.__data_dir, data_filename)
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

    def convertDatasetFormationWithoutLabel(self, data, dataset_name):
        data_filename = 'converted_' + dataset_name + '.npz'
        data_path = os.path.join(self.__data_dir, data_filename)
        if not os.path.exists(data_path):
            data = data.astype(np.float32) / 255.0
            np.savez(data_path, data=data)
        else:
            data_file = np.load(data_path)
            data = data_file['data']

        return data

    def viewImages(self, index=None):
        data_file_path = os.path.join(self.__data_dir, 'augmented_train.npz')
        data_file = np.load(data_file_path)
        data = data_file['data']
        if index is None:
            index = np.random.randint(data.shape[0] / self.__augment_factor)

        label = data_file['label']
        print('index=%d, label=%d'%(index, label[index*self.__augment_factor]))

        images = data[index * self.__augment_factor:(index + 1) * self.__augment_factor, :]

        plt.figure(figsize=[10, 5])
        plt.suptitle('%d-th image in the dataset' % (index,))
        plt.subplot(2, 3, 1)
        plt.title('%d' % (1,))
        plt.imshow(images[0, :].reshape(28, -1))
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.title('%d' % (2,))
        plt.imshow(images[1, :].reshape(28, -1))
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.title('%d' % (3,))
        plt.imshow(images[2, :].reshape(28, -1))
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.title('%d' % (3,))
        plt.imshow(images[3, :].reshape(28, -1))
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.title('%d' % (4,))
        plt.imshow(images[4, :].reshape(28, -1))
        plt.axis('off')

        plt.show()

if __name__ == '__main__':
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--data_dir', type=str,
                                   default='../data',
                                   help='Directory for storing input data')
    commandLineParser.add_argument('--train_data',
                                   type=str,
                                   default='train.csv',
                                   help='Filename for training dataset')
    commandLineParser.add_argument('--test_data',
                                   type=str,
                                   default='test.csv',
                                   help='Filename for testing datasets')
    FLAGS, _ = commandLineParser.parse_known_args()

    dataset = Dataset(FLAGS.data_dir, 5, FLAGS.train_data, FLAGS.test_data)

    train_data, train_label, test_data = dataset.readCsvFile()
    splited_train_data, splited_train_label, splited_validate_data, splited_validate_label = \
        dataset.splitTrainingDataset(train_data, train_label)
    augmented_train_data, augmented_train_label = dataset.augmentTrainData(splited_train_data, splited_train_label)
    _, _ = dataset.convertDatasetFotmat(augmented_train_data, augmented_train_label, 'train')
    _, _ = dataset.convertDatasetFotmat(splited_validate_data, splited_validate_label, 'validate')
    _ = dataset.convertDatasetFormationWithoutLabel(test_data, 'test')

    dataset.viewImages()