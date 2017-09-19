import argparse
import sys
import os
import csv
import numpy as np
import tensorflow as tf

FLAGS=None
TRAIN_FILENAME='train_split.csv'
VALIDATION_FILENAME='validation_split.csv'
TEST_FILENAME='test.csv'
RESULT_FILENAME='submission.csv'
MODEL_FILENAME='model'

def parse_test_dataset(file_path, item_count):
    with open(file_path) as csv_file_handler:
        csv_file_reader = csv.reader(csv_file_handler)
        next(csv_file_reader)
        data = np.zeros([item_count, 784])
        for count, line in enumerate(csv_file_reader):
            data[count, :] = map(float, line)
        data = data / 255
    return data

def parse_train_dataset(file_path, item_count):
    with open(file_path) as csv_file_handler:
        csv_file_reader = csv.reader(csv_file_handler)
        next(csv_file_reader)
        data = np.zeros([item_count, 784], dtype=np.float)
        label = np.zeros([item_count, 10], dtype=np.float)
        for count, line in enumerate(csv_file_reader):
            data[count, :] = map(float, line[1:])
            label[count, int(line[0])] = 1.0
        data = data / 255
    return data, label

def write_result(result, data_dir, result_filename):
    result_path = os.path.join(data_dir, result_filename)
    csvfile = open(result_path, 'w')
    writer = csv.writer(csvfile)
    writer.writerow(['ImageId', 'Label'])
    data = []
    for count in range(result.shape[0]):
        line = (str(count+1), str(result[count]))
        data.append(line)
    writer.writerows(data)
    csvfile.close()

def main(_):
    train_data, train_label = parse_train_dataset(os.path.join(FLAGS.data_dir, TRAIN_FILENAME), 41501)
    validation_data, validation_label = parse_train_dataset(os.path.join(FLAGS.data_dir, VALIDATION_FILENAME), 499)
    test_data = parse_test_dataset(os.path.join(FLAGS.data_dir, TEST_FILENAME), 28000)

    # construct network
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

    predict_op = tf.argmax(outputs, axis=1)

    # train and test
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for count in range(14001):
            batch_index = np.random.permutation(28000)[:100]
            train_data_batch = train_data[batch_index, :]
            train_label_batch = train_label[batch_index, :]
            sess.run(train_op, feed_dict={x:train_data_batch, y_:train_label_batch})
            if count%50 == 0:
                print "step=%d, accuracy=%f"%(count, sess.run(accuracy, feed_dict={x:validation_data, y_:validation_label}))


        saver = tf.train.Saver()
        saver.save(sess, os.path.join(FLAGS.model_dir, MODEL_FILENAME))

        prediction = sess.run(predict_op, feed_dict={x:test_data})
        write_result(prediction, FLAGS.data_dir, RESULT_FILENAME)


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