import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import model_cnn
import preprocessing_classification as pre_c
import itertools as it
from scipy import signal
import scipy
from makedata_classification import sliding
import glob
from sklearn.cluster import MeanShift, estimate_bandwidth
import time
import sys


sess = tf.InteractiveSession()

text_length = 1000
num_authors = 5
input_cnn = tf.placeholder(tf.float32, [None, len(pre_c.alphabet), text_length, 1], name="input_x")
with tf.variable_scope("cnn"):
    cnn_logits, cnn_variables, _ = model_cnn.inference(
            input_x=input_cnn, keep_prob=1.0, num_authors=num_authors)
known_vars = []
known_vars = tf.global_variables()
[print(v) for v in known_vars]
saver = tf.train.Saver(var_list=known_vars)
saver.restore(sess, "../resources/1511967049/saves/cnn.ckpt-00009384")
print("cnn_classifier restored")

w_1 = sess.run('cnn/conv-maxpool-1/W:0')
b_1 = sess.run('cnn/conv-maxpool-1/b:0')
w_2 = sess.run('cnn/conv-maxpool-2/W:0')
b_2 = sess.run('cnn/conv-maxpool-2/b:0')
w_3 = sess.run('cnn/conv-3/W:0')
b_3 = sess.run('cnn/conv-3/b:0')
w_4 = sess.run('cnn/conv-4/W:0')
b_4 = sess.run('cnn/conv-4/b:0')
w_5 = sess.run('cnn/conv-5/W:0')
b_5 = sess.run('cnn/conv-5/b:0')
w_6 = sess.run('cnn/conv-maxpool-6/W:0')
b_6 = sess.run('cnn/conv-maxpool-6/b:0')
w_fc = sess.run('cnn/fc/W-fc:0')
b_fc = sess.run('cnn/fc/b-fc:0')
weights = [w_1, w_2, w_3, w_4, w_5, w_6, w_fc]
biases = [b_1, b_2, b_3, b_4, b_5, b_6, b_fc]


def prepare_cnn_input(textlist):
    encoded_list = []
    for sample in textlist:
        # shorten text if it is too long
        if len(sample) > text_length:  # tf.flags.FLAGS.text_length:
            text_end_extracted = sample.lower()[0:text_length]
        else:
            text_end_extracted = sample.lower()
        # pad text with spaces if it is too short
        num_padding = text_length - len(text_end_extracted)
        padded = text_end_extracted + " " * num_padding
        text_int8_repr = np.array([pre_c.alphabet.find(char) for char in padded], dtype=np.int8)
        x_batch_one_hot = np.zeros(shape=[len(pre_c.alphabet), len(text_int8_repr), 1])
        for char_pos_in_seq, char_seq_char_ind in enumerate(text_int8_repr):
            if char_seq_char_ind != -1:
                x_batch_one_hot[char_seq_char_ind][char_pos_in_seq][0] = 1
        encoded_list.append(x_batch_one_hot)
    return np.array(encoded_list)


def calculate_activations(textlist):
    logits, activations = sess.run([cnn_logits, cnn_variables], feed_dict={input_cnn: prepare_cnn_input(textlist)[:-1]})
    logit_array = np.reshape(np.argmax(logits, axis=1), [len(logits), 1])
    return logit_array, activations


def data_generation(file):
    with open(file) as f:
        eof = False
        ctr = 0
        while eof == False:
            try:
                read = [line.split("|SEPERATOR|") for line in f.read(5000*1013).split("\n")[:-1]][0::10]
                texts = [r[1] for r in read]
                authors = np.expand_dims(np.array([int(r[0])-1 for r in read][:-1]), axis=1)
                logits, act = calculate_activations(texts)
                logits_save = np.append(logits, authors, axis=1)
                np.savez_compressed("../resources/activations-five-authors/"+file+"-activations-"+str(ctr), logits=logits_save, act_1=act[0], act_2=act[1], act_3=act[2], act_4=act[3], act_5=act[4], act_6=act[5], act_7=act[6])
                ctr += 1
            except:
                eof = True
    print(ctr)


data_generation(file="TrainSet-five_authors.txt")