import argparse
import sys
import os
import csv
import numpy as np
import tensorflow as tf

FLAGS=None
TRAIN_FILENAME='train.csv'
TEST_FILENAME='test.csv'
MODEL_FILENAME='model'

def loadDataItems(data_dir, train_filename):
    train_path = os.path.join(data_dir, train_filename)
    reader = csv.reader(open(train_path))
    train_data = np.zeros([42000, 784], dtype=np.float)
    train_label = np.zeros([42000, 10], dtype=np.float)
    for count, line in enumerate(reader):
        if count == 0:
            continue
        else:
            line = map(float, line)
            train_data[count-1, :] = line[1:]
            train_label[count-1, int(line[0])] = 1.0

    print 'Parse train data done!'
    shuffled_index = np.random.permutation(42000)
    train_data = train_data / 255.0
    return train_data[shuffled_index[:41000]], train_label[shuffled_index[:41000]], train_data[shuffled_index[41000:]], train_label[shuffled_index[41000:]]

def main(_):
    trainData, trainLabel, validationData, validationLabel = loadDataItems(FLAGS.data_dir, TRAIN_FILENAME)

    x = tf.placeholder('float', shape=[None, 784], name='raw_data')
    y_ = tf.placeholder('float', shape=[None, 10], name='label')
    inputs = tf.reshape(x, shape=[-1, 28, 28, 1], name='input_data')

    with tf.variable_scope('layer1'):
        layer1_weights = tf.get_variable('weights', shape=[5, 5, 1, 32],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer1_biases = tf.get_variable('biases', shape=[32],
                                        initializer=tf.constant_initializer(0.1))
        layer1 = tf.nn.conv2d(inputs, layer1_weights, strides=[1, 1, 1, 1], padding='SAME') + layer1_biases
        layer1 = tf.nn.relu(layer1)
        layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer2'):
        layer2_weights = tf.get_variable('weights', shape=[5, 5, 32, 64],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer2_biases = tf.get_variable('biases', shape=[64],
                                        initializer=tf.constant_initializer(0.1))

        layer2 = tf.nn.conv2d(layer1, layer2_weights, strides=[1, 1, 1, 1], padding='SAME') + layer2_biases
        layer2 = tf.nn.relu(layer2)
        layer2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer3'):
        layer3_weights = tf.get_variable('weights', shape=[7 * 7 * 64, 1024],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer3_biases = tf.get_variable('biases', shape=[1024],
                                        initializer=tf.constant_initializer(0.1))
        layer3 = tf.reshape(layer2, shape=[-1, 7 * 7 * 64])
        layer3 = tf.matmul(layer3, layer3_weights) + layer3_biases
        layer3 = tf.nn.relu(layer3)

    with tf.variable_scope('layer4'):
        layer4_weights = tf.get_variable('weights', shape=[1024, 10],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer4_biases = tf.get_variable('biases', shape=[10],
                                        initializer=tf.constant_initializer(0.1))
        layer4 = tf.matmul(layer3, layer4_weights) + layer4_biases

    outputs = tf.nn.softmax(layer4)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(outputs))
    train_op = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(outputs, axis=1), tf.argmax(y_, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for count in range(5000):
            sampled_index = np.random.permutation(trainData.shape[0])[:100]
            sess.run(train_op, feed_dict={x:trainData[sampled_index], y_:trainLabel[sampled_index]})
            if count%50 == 0:
                print "step=%d, accuracy=%f"%(count, sess.run(accuracy, feed_dict={x:validationData, y_:validationLabel}))

        saver = tf.train.Saver()
        saver.save(sess, os.path.join(FLAGS.model_dir, MODEL_FILENAME))

if __name__ == '__main__':
    commandLineParser=argparse.ArgumentParser()
    commandLineParser.add_argument('--data_dir', type=str,
                                   default='../data',
                                   help='Directory for storing input data')
    commandLineParser.add_argument('--model_dir', type=str,
                                   default='../model',
                                   help='Directory for storing model file')
    FLAGS, unparsed=commandLineParser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)