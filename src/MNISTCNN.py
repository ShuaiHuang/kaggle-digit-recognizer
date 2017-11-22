import argparse
import numpy as np
import tensorflow as tf
import os
import sys
import csv

FLAGS=None
TRAIN_FILENAME='train.csv'
TEST_FILENAME='test.csv'
RESULT_FILENAME='submission.csv'

MODEL_FILENAME='model'

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
    train_data = np.load(os.path.join(FLAGS.data_dir, 'mnist_train_data.npy'))
    train_label = np.load(os.path.join(FLAGS.data_dir, 'mnist_train_label.npy'))
    validation_data = np.load(os.path.join(FLAGS.data_dir, 'mnist_validation_data.npy'))
    validation_label = np.load(os.path.join(FLAGS.data_dir, 'mnist_validation_label.npy'))
    test_data = np.load(os.path.join(FLAGS.data_dir, 'mnist_test_data.npy'))

    # construct the network
    x = tf.placeholder('float', shape=[None, 784], name='raw_data')
    y_ = tf.placeholder('float', shape=[None, 10], name='label')
    inputs = tf.reshape(x, shape=[-1, 28, 28, 1], name='input_data')

    with tf.variable_scope('layer1'):
        layer1_weights = tf.get_variable('weights', shape=[3, 3, 1, 32],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer1_biases = tf.get_variable('biases', shape=[32],
                                        initializer=tf.constant_initializer(0.1))
        layer1 = tf.nn.conv2d(inputs, layer1_weights, strides=[1, 1, 1, 1], padding='SAME') + layer1_biases
        layer1 = tf.nn.relu(layer1)

    with tf.variable_scope('layer2'):
        layer2_weights = tf.get_variable('weights', shape=[3, 3, 32, 32],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer2_biases = tf.get_variable('biases', shape=[32],
                                        initializer=tf.constant_initializer(0.1))
        layer2 = tf.nn.conv2d(layer1, layer2_weights, strides=[1, 1, 1, 1], padding='SAME') + layer2_biases
        layer2 = tf.nn.relu(layer2)
        layer2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        layer2 = tf.nn.dropout(layer2, 0.7)

    with tf.variable_scope('layer3'):
        layer3_weights = tf.get_variable('weights', shape=[3, 3, 32, 64],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer3_biases = tf.get_variable('biases', shape=[64],
                                        initializer=tf.constant_initializer(0.1))

        layer3 = tf.nn.conv2d(layer2, layer3_weights, strides=[1, 1, 1, 1], padding='SAME') + layer3_biases
        layer3 = tf.nn.relu(layer3)

    with tf.variable_scope('layer4'):
        layer4_weights = tf.get_variable('weights', shape=[3, 3, 64, 64],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer4_biases = tf.get_variable('biases', shape=[64],
                                        initializer=tf.constant_initializer(0.1))

        layer4 = tf.nn.conv2d(layer3, layer4_weights, strides=[1, 1, 1, 1], padding='SAME') + layer4_biases
        layer4 = tf.nn.relu(layer4)
        layer4 = tf.nn.max_pool(layer4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer5'):
        layer5_weights = tf.get_variable('weights', shape=[7 * 7 * 64, 1024],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer5_biases = tf.get_variable('biases', shape=[1024],
                                        initializer=tf.constant_initializer(0.1))
        layer5 = tf.reshape(layer4, shape=[-1, 7 * 7 * 64])
        layer5 = tf.matmul(layer5, layer5_weights) + layer5_biases
        layer5 = tf.nn.dropout(layer5, 0.7)
        layer5 = tf.nn.relu(layer5)

    with tf.variable_scope('layer6'):
        layer6_weights = tf.get_variable('weights', shape=[1024, 10],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer6_biases = tf.get_variable('biases', shape=[10],
                                        initializer=tf.constant_initializer(0.1))
        layer6 = tf.matmul(layer5, layer6_weights) + layer6_biases

    outputs = tf.nn.softmax(layer6)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(outputs))
    train_op = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(outputs, axis=1), tf.argmax(y_, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    predict_op = tf.argmax(outputs, axis=1)

    saver = tf.train.Saver()
    max_accuracy = 0.0
    # train and test
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        iterations = 40000
        for count in range(iterations + 1):
            batch_index = np.random.permutation(190000)[:50]
            train_data_batch = train_data[batch_index, :]
            train_label_batch = train_label[batch_index, :]
            sess.run(train_op, feed_dict={x:train_data_batch, y_:train_label_batch})
            temp_accuracy = sess.run(accuracy, feed_dict={x:validation_data, y_:validation_label})
            if count%50 == 0:
                print "step=%d, accuracy=%f"%(count, temp_accuracy)
            if temp_accuracy > 0.99 and temp_accuracy > max_accuracy:
                max_accuracy = temp_accuracy
                saver.save(sess, os.path.join(FLAGS.model_dir, MODEL_FILENAME))
        print "max_accuracy=%f"%(max_accuracy,)

        saver.restore(sess, os.path.join(FLAGS.model_dir, MODEL_FILENAME))
        result = np.zeros([test_data.shape[0] / 5], dtype=np.int8)
        for count in range(test_data.shape[0] / 5):
            test_data_batch = test_data[count*5:count*5+5, :]
            prediction = sess.run(predict_op, feed_dict={x:test_data_batch})
            numCount = np.bincount(prediction)
            result[count] = np.argmax(numCount)
        write_result(result, FLAGS.data_dir, RESULT_FILENAME)


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