import numpy as np
import os
import cv2

class MNIST_Dataset(object):

    def __init__(self, dataset_dir, train_file, test_file):
        train_file_path = os.path.join(dataset_dir, train_file)
        test_file_path = os.path.join(dataset_dir, test_file)
        if os.path.isfile(train_file_path):
            self.__train_file = train_file_path
        if os.path.isfile(test_file_path):
            self.__test_file = test_file_path

        self.__train_data = None
        self.__train_label = None
        self.__test_data = None
        self.__split_index = None
        self.__train_data_npyfile = os.path.join(dataset_dir, 'train_data.npy')
        self.__train_label_npyfile = os.path.join(dataset_dir, 'train_label.npy')
        self.__test_data_npyfile = os.path.join(dataset_dir, 'test_data.npy')
        self.__split_index_npyfile = os.path.join(dataset_dir, 'split_index.npy')

    def parseTrainData(self):
        if os.path.isfile(self.__train_data_npyfile) and os.path.isfile(self.__train_label_npyfile):
            self.__train_data = np.load(self.__train_data_npyfile)
            self.__train_label = np.load(self.__train_label_npyfile)
        else:
            self.__train_data = np.genfromtxt(self.__train_file, delimiter=',', dtype=np.uint8, skip_header=1, usecols=range(1,785))
            self.__train_label = np.genfromtxt(self.__train_file, delimiter=',', dtype=np.uint8, skip_header=1, usecols=range(1))
            np.save(self.__train_data_npyfile, self.__train_data)
            np.save(self.__train_label_npyfile, self.__train_label)
        print("Parse %d training data items and %d training data labels." % (self.__train_data.shape[0], self.__train_label.shape[0]))

    def parseTestData(self):
        if os.path.isfile(self.__test_data_npyfile):
            self.__test_data = np.load(self.__test_data_npyfile)
        else:
            self.__test_data = np.genfromtxt(self.__test_file, delimiter=',', dtype=np.uint8, skip_header=1, usecols=range(784))
            np.save(self.__test_data_npyfile, self.__test_data)
        print("Parse %d testing data items." % (self.__test_data.shape[0]))

    def __getAugmentedData(self, data, label):
        augmentedData = np.zeros((data.shape[0]*4, data.shape[1]), dtype=np.uint8)
        augmentedLabel = np.zeros(label.shape[0]*4, dtype=np.uint8)
        for row in range(data.shape[0]):
            img = data[row, :]
            img_label = label[row]
            img = np.reshape(img, [28, 28])
            points = cv2.findNonZero(img)
            rects = cv2.boundingRect(points)

            img_left_top = np.zeros([28, 28], dtype=np.uint8)
            img_left_top[0:rects[3], 0:rects[2]] = img[rects[1]:rects[1]+rects[3], rects[0]:rects[0]+rects[2]]
            img_left_top = np.reshape(img_left_top, [1, -1])
            augmentedData[row*4, :] = img_left_top
            augmentedLabel[row*4] = img_label

            img_left_bottom = np.zeros([28, 28], dtype=np.uint8)
            img_left_bottom[28-rects[3]:28, 0:rects[2]] = img[rects[1]:rects[1]+rects[3], rects[0]:rects[0]+rects[2]]
            img_left_bottom = np.reshape(img_left_bottom, [1, -1])
            augmentedData[row*4+1, :] = img_left_bottom
            augmentedLabel[row*4+1] = img_label

            img_right_top = np.zeros([28, 28], dtype=np.uint8)
            img_right_top[0:rects[3], 28-rects[2]:28] = img[rects[1]:rects[1]+rects[3], rects[0]:rects[0]+rects[2]]
            img_right_top = np.reshape(img_right_top, [1, -1])
            augmentedData[row*4+2, :] = img_right_top
            augmentedLabel[row*4+2] = img_label

            img_right_bottom = np.zeros([28, 28], dtype=np.uint8)
            img_right_bottom[28-rects[3]:28, 28-rects[2]:28] = img[rects[1]:rects[1]+rects[3], rects[0]:rects[0]+rects[2]]
            img_right_bottom = np.reshape(img_right_bottom, [1, -1])
            augmentedData[row*4+3, :] = img_right_bottom
            augmentedLabel[row*4+3] = img_label
        return augmentedData, augmentedLabel

    def getTraingData(self, size=38000):
        if os.path.isfile(self.__split_index_npyfile):
            self.__split_index = np.load(self.__split_index_npyfile)
        else:
            self.__split_index = np.random.permutation(42000)
            np.save(self.__split_index_npyfile, self.__split_index)

        augmentedData, augmentedLabel = self.__getAugmentedData(self.__train_data[self.__split_index[0:size], :],
                                                              self.__train_label[self.__split_index[0:size]])
        augmentedData = np.append(augmentedData, self.__train_data[self.__split_index[0:size], :], axis=0)
        augmentedLabel = np.append(augmentedLabel, self.__train_label[self.__split_index[0:size]], axis=0)

        return augmentedData, augmentedLabel

    def getValidationData(self, size=38000):
        return self.__train_data[self.__split_index[size:], :], self.__train_label[self.__split_index[size:]]

    def getTestingData(self):
        return self.__test_data

    def convertLabel2OneHot(self, label):
        oneHot = np.zeros((label.shape[0], 10), dtype=np.float32)
        for row in range(label.shape[0]):
            oneHot[row, label[row]] = 1.0
        return oneHot


if __name__ == '__main__':
    dataset = MNIST_Dataset('../data', 'train.csv', 'test.csv')
    dataset.parseTrainData()
    dataset.parseTestData()

    train_data, train_label = dataset.getTraingData()
    train_data = np.float32(train_data) / 255.0
    train_label = dataset.convertLabel2OneHot(train_label)

    validation_data, validation_label = dataset.getValidationData()
    validation_data = np.float32(validation_data) / 255.0
    validation_label = dataset.convertLabel2OneHot(validation_label)

    test_data = dataset.getTestingData()
    test_data = np.float32(test_data) / 255.0

    np.save(os.path.join('../data', 'mnist_train_data.npy'), train_data)
    np.save(os.path.join('../data', 'mnist_train_label.npy'), train_label)
    np.save(os.path.join('../data', 'mnist_validation_data.npy'), validation_data)
    np.save(os.path.join('../data', 'mnist_validation_label.npy'), validation_label)
    np.save(os.path.join('../data', 'mnist_test_data.npy'), test_data)