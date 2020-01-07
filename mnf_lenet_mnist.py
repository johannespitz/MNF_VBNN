import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
from tensorflow import keras
from mnist import MNIST
import time, os
from wrappers import MNFLeNet

from mutual_information import plot_mi

def get_data():
    # (xtrain, ytrain), (xtest, ytest) = keras.datasets.mnist.load_data()
    # xtrain, xtest = np.reshape(xtrain, (xtrain.shape[0], 28, 28, 1)), np.reshape(xtest, (xtest.shape[0], 28, 28, 1))
    (xtrain, ytrain), (xtest, ytest) = keras.datasets.cifar10.load_data()

    def normalize(batch):
        return batch / 255. - 0.5

    xtrain = normalize(xtrain)
    xtest  = normalize(xtest)

    threshold = int(0.9 * xtrain.shape[0])
    xvalid = xtrain[threshold:]
    yvalid = ytrain[threshold:]
    xtrain = xtrain[:threshold]
    ytrain = ytrain[:threshold]

    ytrain, yvalid, ytest = keras.utils.to_categorical(ytrain, 10), keras.utils.to_categorical(yvalid, 10), keras.utils.to_categorical(ytest, 10)

    return (xtrain, ytrain), (xvalid, yvalid), (xtest, ytest)



def generate_hold_out_masks(train_labels, test_labels, force_labels=None):
    num_labels = train_labels.shape[1]

    if force_labels is None:
        keep_labels = np.random.choice(num_labels, num_labels // 10)
    else:
        keep_labels = force_labels
    # keep_labels = [13]
    print('Holding out the following labels: ', str(keep_labels))

    train_labels, test_labels = np.argmax(train_labels, 1), np.argmax(test_labels, 1)

    train_keep_mask = (np.zeros(train_labels.size) > 0)
    for label in keep_labels:
        train_keep_mask += (train_labels == label)
    test_keep_mask = (np.zeros(test_labels.size) > 0)
    for label in keep_labels:
        test_keep_mask += (test_labels == label)
    train_use_mask = np.invert(train_keep_mask)
    test_use_mask = np.invert(test_keep_mask)

    return (train_keep_mask, train_use_mask), (test_keep_mask, test_use_mask)



def train(force_labels=None):
    (xtrain, ytrain), (xvalid, yvalid), (xtest, ytest) = get_data()

    (train_keep_mask, train_use_mask), (test_keep_mask, test_use_mask) = generate_hold_out_masks(ytrain, ytest, force_labels=force_labels)

    N, height, width, n_channels = xtrain[train_use_mask].shape
    batchsize = 1000
    iter_per_epoch = N // batchsize

    sess = tf.InteractiveSession()

    input_shape = [None, height, width, n_channels]
    x = tf.placeholder(tf.float32, input_shape, name='x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

    model = MNFLeNet(N, input_shape=input_shape, flows_q=FLAGS.fq, flows_r=FLAGS.fr, use_z=not FLAGS.no_z,
                     learn_p=FLAGS.learn_p, thres_var=FLAGS.thres_var, flow_dim_h=FLAGS.flow_h)

    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    y = model.predict(x)
    yd = model.predict(x, sample=False)
    pyx = tf.nn.softmax(y)

    with tf.name_scope('KL_prior'):
        regs = model.get_reg()
        tf.summary.scalar('KL prior', regs)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
        tf.summary.scalar('Loglike', cross_entropy)

    global_step = tf.Variable(0, trainable=False)
    if FLAGS.anneal:
        number_zero, original_zero = FLAGS.epzero, FLAGS.epochs / 2
        with tf.name_scope('annealing_beta'):
            max_zero_step = number_zero * iter_per_epoch
            original_anneal = original_zero * iter_per_epoch
            beta_t_val = tf.cast((tf.cast(global_step, tf.float32) - max_zero_step) / original_anneal, tf.float32)
            beta_t = tf.maximum(beta_t_val, 0.)
            annealing = tf.minimum(1., tf.cond(global_step < max_zero_step, lambda: tf.zeros((1,))[0], lambda: beta_t))
            tf.summary.scalar('annealing beta', annealing)
    else:
        annealing = 1.

    with tf.name_scope('lower_bound'):
        lowerbound = cross_entropy + annealing * regs
        tf.summary.scalar('Lower bound', lowerbound)

    train_step = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(lowerbound, global_step=global_step)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(yd, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)

    tf.add_to_collection('logits', y)
    tf.add_to_collection('logits_map', yd)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y_)
    saver = tf.train.Saver(tf.global_variables())

    tf.global_variables_initializer().run()

    idx = np.arange(N)
    steps = 0
    model_dir = './models/mnf_lenet_mnist_fq{}_fr{}_usez{}_thres{}/model/'.format(FLAGS.fq, FLAGS.fr, not FLAGS.no_z,
                                                                                  FLAGS.thres_var)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print('Will save model as: {}'.format(model_dir + 'model'))
    # Train
    for epoch in range(FLAGS.epochs):


        if epoch % 1 == 0:
            class FakeModel:
                def __init__(self, sess, target):
                    self.sess = sess
                    self.target = target
                def predict(self, samples, batch_size=None):
                    return sess.run(self.target, feed_dict={x: samples})
            fake_model = FakeModel(sess, pyx)
            tests_dict = dict(
                random=np.random.uniform(size=list((2000,) + xtrain.shape[1:])),
                trained_labels_train=xtrain[train_use_mask][:2000],
                trained_labels_test=xtest[test_use_mask][:2000],
                new_label=xtrain[train_keep_mask][:2000],
            )
            plt = plot_mi(tests_dict, fake_model)
            if force_labels is None:
                name = f'plots/cifar10/MiEnt_{epoch}.svg'
            else:
                name = f'plots/cifar10/MiEnt_keep{force_labels[0]}_{epoch}.svg'
            plt.savefig(name)
            plt.close()

            # if epoch > 3:
            #     break



        widgets = ["epoch {}/{}|".format(epoch + 1, FLAGS.epochs), Percentage(), Bar(), ETA()]
        pbar = ProgressBar(iter_per_epoch, widgets=widgets)
        pbar.start()
        np.random.shuffle(idx)
        t0 = time.time()
        for j in range(iter_per_epoch):
            steps += 1
            pbar.update(j)
            batch = np.random.choice(N, batchsize)
            if j == (iter_per_epoch - 1):
                summary, _ = sess.run([merged, train_step], feed_dict={x: xtrain[train_use_mask][batch], y_: ytrain[train_use_mask][batch]})
                train_writer.add_summary(summary,  steps)
                train_writer.flush()
            else:
                sess.run(train_step, feed_dict={x: xtrain[train_use_mask][batch], y_: ytrain[train_use_mask][batch]})
            # if j == (iter_per_epoch - 1):
            #     summary, _ = sess.run([merged, train_step], feed_dict={x: xtrain[train_use_mask][j * batchsize:(j + 1) * batchsize], y_: ytrain[train_use_mask][j * batchsize:(j + 1) * batchsize]})
            #     train_writer.add_summary(summary,  steps)
            #     train_writer.flush()
            # else:
            #     sess.run(train_step, feed_dict={x: xtrain[train_use_mask][j * batchsize:(j + 1) * batchsize], y_: ytrain[train_use_mask][j * batchsize:(j + 1) * batchsize]})

        # the accuracy here is calculated by a crude MAP so as to have fast evaluation
        # it is much better if we properly integrate over the parameters by averaging across multiple samples
        tacc = sess.run(accuracy, feed_dict={x: xvalid, y_: yvalid})
        string = 'Epoch {}/{}, valid_acc: {:0.3f}'.format(epoch + 1, FLAGS.epochs, tacc)

        if (epoch + 1) % 10 == 0:
            string += ', model_save: True'
            saver.save(sess, model_dir + 'model')

        string += ', dt: {:0.3f}'.format(time.time() - t0)
        print(string)

    saver.save(sess, model_dir + 'model')
    train_writer.close()

    preds = np.zeros_like(ytest)
    widgets = ["Sampling |", Percentage(), Bar(), ETA()]
    pbar = ProgressBar(FLAGS.L, widgets=widgets)
    pbar.start()
    for i in range(FLAGS.L):
        pbar.update(i)
        for j in range(xtest.shape[0] // batchsize):
            pyxi = sess.run(pyx, feed_dict={x: xtest[j * batchsize:(j + 1) * batchsize]})
            preds[j * batchsize:(j + 1) * batchsize] += pyxi / FLAGS.L
    print()
    sample_accuracy = np.mean(np.equal(np.argmax(preds, 1), np.argmax(ytest, 1)))
    print('Sample test accuracy: {}'.format(sample_accuracy))

    sess.close()


def main():
    # if tf.io.gfile.exists(FLAGS.summaries_dir):
    #     tf.io.gfile.rmtree(FLAGS.summaries_dir)
    # tf.io.gfile.makedirs(FLAGS.summaries_dir)
    # train()
    for i in range(10):
        if tf.io.gfile.exists(FLAGS.summaries_dir):
            tf.io.gfile.rmtree(FLAGS.summaries_dir)
        tf.io.gfile.makedirs(FLAGS.summaries_dir)
        tf.keras.backend.clear_session()
        train(force_labels=[i])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries_dir', type=str, default='logs/mnf_lenet',
                        help='Summaries directory')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-epzero', type=int, default=1)
    parser.add_argument('-fq', default=2, type=int)
    parser.add_argument('-fr', default=2, type=int)
    parser.add_argument('-no_z', action='store_true')
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-thres_var', type=float, default=0.5)
    parser.add_argument('-flow_h', type=int, default=50)
    parser.add_argument('-L', type=int, default=100)
    parser.add_argument('-anneal', action='store_true')
    parser.add_argument('-learn_p', action='store_true')
    FLAGS = parser.parse_args()
    main()