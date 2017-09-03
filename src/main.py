import argparse
import sys
import os
import tensorflow as tf

FLAGS = None
TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'

def loadMNIST(data_dir):
    trainPath=os.path.join(data_dir,TRAIN_FILENAME)
    testPath=os.path.join(data_dir,TEST_FILENAME)
    if (not os.path.exists(trainPath)) or (not os.path.exists(testPath)):
        print "%s or %s not exists in the directory %s"%(TRAIN_FILENAME, TEST_FILENAME, data_dir)
        return None

    trainDataset=tf.contrib.data.Dataset.from_tensor_slices(trainPath)
    trainDataset=trainDataset.flat_map(
        lambda filename : (
            tf.contrib.data.TextLineDataset(filename).skip(1)
        )
    )

    testDataset=tf.contrib.data.Dataset.from_tensor_slices(testPath)
    testDataset=testDataset.flat_map(
        lambda filename : (
            tf.contrib.data.TextLineDataset(filename).skip(1)
        )
    )

    return trainDataset, testDataset

def main(_):
    MNIST=loadMNIST(FLAGS.data_dir)

if __name__ == '__main__':
    commandLineParser=argparse.ArgumentParser()
    commandLineParser.add_argument('--data_dir', type=str,
                                   default='./',
                                   help='Directory for storing input data')
    FLAGS, unparsed=commandLineParser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)