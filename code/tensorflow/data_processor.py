#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from opt import opt
from log import logging
from pathlib import Path


class MNISTDataProcessor(object):
    def __init__(self, train_path, test_path):
        if (not isinstance(train_path, Path)) or (not train_path.exists()) or \
                (not isinstance(test_path, Path)) or (not test_path.exists()):
            logging.error('Type error!')
            return

        self.__train_path = train_path
        self.__test_path = test_path

        npz_file_path = self.__train_path.parent / 'mnist_original_dataset.npz'
        self.__train_dataset = dict()
        self.__test_dataset = dict()
        if not npz_file_path.exists():
            train_dataset = np.genfromtxt(str(self.__train_path), dtype=np.uint8, delimiter=',', skip_header=1)

            self.__train_dataset['data'] = train_dataset[:, 1:]
            self.__train_dataset['label'] = train_dataset[:, 0]

            self.__test_dataset['data'] = np.genfromtxt(str(self.__test_path), dtype=np.uint8, delimiter=',', skip_header=1)
            np.savez(str(npz_file_path),
                     train_data=self.__train_dataset['data'],
                     train_label=self.__train_dataset['label'],
                     test_data=self.__test_dataset['data'])
        else:
            dataset = np.load(str(npz_file_path))
            self.__train_dataset['data'] = dataset['train_data']
            self.__train_dataset['label'] = dataset['train_label']
            self.__test_dataset['data'] = dataset['test_data']

        logging.debug('Train dataset shape: %r', self.__train_dataset['data'].shape)
        logging.debug('Test dataset shape: %r', self.__test_dataset['data'].shape)

    def augment_data(self, factor=10):
        npz_file_path = self.__train_path.parent / 'mnist_augmented_dataset.npz'
        if not npz_file_path.exists():
            augmented_data = np.zeros((self.__train_dataset['data'].shape[0] * factor, self.__train_dataset['data'].shape[1]), dtype=np.uint8)
            augmented_label = np.zeros((self.__train_dataset['label'].shape[0] * factor, ), dtype=np.uint8)
            for org_idx, (data_item, label_item) in enumerate(zip(self.__train_dataset['data'], self.__train_dataset['label'])):
                for aug_idx in range(factor):
                    aug_img = np.copy(data_item)
                    aug_img = self.__get_transformed_data(aug_img)
                    augmented_data[org_idx*factor+aug_idx, :] = aug_img
                    augmented_label[org_idx*factor+aug_idx] = label_item
            np.savez(str(npz_file_path), data=augmented_data, label=augmented_label)
            self.__train_dataset['data'] = augmented_data
            self.__train_dataset['label'] = augmented_label
        else:
            dataset = np.load(str(npz_file_path))
            self.__train_dataset['data'] = dataset['data']
            self.__train_dataset['label'] = dataset['label']

    def __get_transformed_data(self, image):
        image = np.reshape(image, [28, 28])
        bounding_box = cv2.boundingRect(image)
        height = bounding_box[3]
        width = bounding_box[2]

        # scale and rotate
        scale_ratio = np.random.ranf() * (1.1 - 0.8) + 0.8
        if scale_ratio > 1:
            height_ratio = 1.0 * 28 / height
            width_ratio = 1.0 * 28 / width
            scale_ratio = min(scale_ratio, width_ratio)
            scale_ratio = min(height_ratio, scale_ratio)
        rotation_degree = np.random.randint(-10, 10)
        affine_matrix = cv2.getRotationMatrix2D((bounding_box[1] + height / 2, bounding_box[0] + width / 2),
                                                rotation_degree, scale_ratio)
        warp_affine_image = cv2.warpAffine(image, affine_matrix, (28, 28))

        # translate
        digit_image, bounding_box = self.__crop_digit_image(warp_affine_image)
        new_x = np.random.randint(28 - bounding_box[3])
        new_y = np.random.randint(28 - bounding_box[2])
        translate_image = np.zeros((28, 28), np.uint8)
        translate_image[new_x:new_x + bounding_box[3], new_y:new_y + bounding_box[2]] = digit_image
        translate_image = np.reshape(translate_image, [1, -1])
        return translate_image

    def __crop_digit_image(self, image):
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


if __name__ == '__main__':
    TRAIN_PATH = Path(opt.train)
    TEST_PATH = Path(opt.test)
    logging.debug('Load train dataset from %s', TRAIN_PATH)
    logging.debug('Load test dataset from %s', TEST_PATH)

    data_processor = MNISTDataProcessor(TRAIN_PATH, TEST_PATH)
    data_processor.augment_data()
