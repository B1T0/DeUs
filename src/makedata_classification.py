import os
import tensorflow as tf


def sliding(string, wnSize, step):
    list = []
    start = 0
    while start + wnSize < len(string):
        list.append(string[start:start+wnSize])
        start = start + step
    return list


def make(filename):
    path = "../resources/texts"
    author_ctr = 1
    allAuthors = []
    for infile in os.listdir(path):
        allAuthors.append(open(path + "/" + infile, 'r', encoding="utf-8"))
        author_ctr = author_ctr + 1
        if author_ctr > tf.flags.FLAGS.num_authors and not tf.flags.FLAGS.use_all_available_texts:
            break
    output = open(filename, 'w', encoding="utf-8")
    for a in range(len(allAuthors)):
        wholeAuthor = ' '.join(allAuthors[a].readlines()).replace("\n", " ")
        texts = sliding(wholeAuthor, tf.flags.FLAGS.text_length, 1)
        for t in texts:
            output.write(str(a+1) + "|SEPERATOR|" + t + "\n")
