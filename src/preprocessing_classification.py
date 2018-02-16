import numpy as np
import tensorflow as tf

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"

def get_batched_one_hot(char_seqs_indices, labels, start_index, end_index, nn="cnn"):
    x_batch = char_seqs_indices[start_index:end_index]
    y_batch = labels[start_index:end_index]
    if nn == "rnn":
        x_batch_one_hot = np.zeros(shape=[len(x_batch), len(x_batch[0]), len(alphabet)])
        for example_i, char_seq_indices in enumerate(x_batch):
            for char_pos_in_seq, char_seq_char_ind in enumerate(char_seq_indices):
                if char_seq_char_ind != -1:
                    x_batch_one_hot[example_i][char_pos_in_seq][char_seq_char_ind] = 1
    elif nn == "cnn":
        x_batch_one_hot = np.zeros(shape=[len(x_batch), len(alphabet), len(x_batch[0]), 1])
        for example_i, char_seq_indices in enumerate(x_batch):
            for char_pos_in_seq, char_seq_char_ind in enumerate(char_seq_indices):
                if char_seq_char_ind != -1:
                    x_batch_one_hot[example_i][char_seq_char_ind][char_pos_in_seq][0] = 1
    return [x_batch_one_hot, y_batch]


def load_data(filename):
    examples = []
    labels = []
    examples_train = []
    examples_eval = []
    labels_train = []
    labels_eval = []
    with open(filename, encoding="utf-8") as f:
        i = 0
        author, authordouble = 1, 1
        for line in f:
            author, text = line.split("|SEPERATOR|")
            author = int(author)
            if author != authordouble:
                print(str(author-1) + " ends at " + str(i))
                # split = int(i/25)
                # examples_train.extend(examples[:-split])
                # examples_eval.extend(examples[-split:])
                # labels_train.extend(labels[:-split])
                # labels_eval.extend(labels[-split:])
                examples_train.extend(examples[0:20000])
                examples_eval.extend(examples[20001:26000])
                labels_train.extend(labels[0:20000])
                labels_eval.extend(labels[20001:26000])
                examples = []
                labels = []
            authordouble = author
            # shorten text if it is too long
            if len(text) > tf.flags.FLAGS.text_length:
                text_end_extracted = text.lower()[-tf.flags.FLAGS.text_length:]
            else:
                text_end_extracted = text.lower()
            # pad text with spaces if it is too short
            num_padding = tf.flags.FLAGS.text_length - len(text_end_extracted)
            padded = text_end_extracted + " " * num_padding
            text_int8_repr = np.array([alphabet.find(char) for char in padded], dtype=np.int8)
            author_one_hot = []
            for author_i in range(tf.flags.FLAGS.num_authors):
                if author_i == author-1:
                    author_one_hot.append(1)
                else:
                    author_one_hot.append(0)
            labels.append(author_one_hot)
            examples.append(text_int8_repr)
            i += 1
        print(str(author - 1) + " ends at " + str(i))
        split = int(i / 25)
        # examples_train.extend(examples[:-split])
        # examples_eval.extend(examples[-split:])
        # labels_train.extend(labels[:-split])
        # labels_eval.extend(labels[-split:])
        examples_train.extend(examples[0:20000])
        examples_eval.extend(examples[20001:26000])
        labels_train.extend(labels[0:20000])
        labels_eval.extend(labels[20001:26000])
        print("Non-neutral instances processed: " + str(i))
    x_train = np.array(examples_train, dtype=np.int8)
    x_eval = np.array(examples_eval, dtype=np.int8)
    y_train = np.array(labels_train, dtype=np.int8)
    y_eval = np.array(labels_eval, dtype=np.int8)
    print("x_char_seq_ind=" + str(x_train.shape))
    print("y shape=" + str(y_train.shape))
    return x_train, x_eval, y_train, y_eval


def batch_iterator(x, y, batch_size, num_epochs, nn, shuffle=True):
    data_size = len(x)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch + 1))
        # Shuffle the data at each epoch
        if shuffle or epoch > 0:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
        else:
            x_shuffled = x
            y_shuffled = y
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            x_batch, y_batch = get_batched_one_hot(x_shuffled, y_shuffled, start_index, end_index, nn)
            batch = list(zip(x_batch, y_batch))
            yield batch


def prepare_cnn_api_input(txt, text_length):
    # shorten text if it is too long
    if len(txt) > text_length:  # tf.flags.FLAGS.text_length:
        text_end_extracted = txt.lower()[0:text_length]
    else:
        text_end_extracted = txt.lower()
    # pad text with spaces if it is too short
    num_padding = text_length - len(text_end_extracted)
    padded = text_end_extracted + " " * num_padding
    text_int8_repr = np.array([alphabet.find(char) for char in padded], dtype=np.int8)
    x_batch_one_hot = np.zeros(shape=[1, len(alphabet), len(text_int8_repr), 1])
    for char_pos_in_seq, char_seq_char_ind in enumerate(text_int8_repr):
        if char_seq_char_ind != -1:
            x_batch_one_hot[0][char_seq_char_ind][char_pos_in_seq][0] = 1
    return x_batch_one_hot


def prepare_rnn_api_input(txt, text_length):
    # shorten text if it is too long
    if len(txt) > text_length:  # tf.flags.FLAGS.text_length:
        text_end_extracted = txt.lower()[0:text_length]
    else:
        text_end_extracted = txt.lower()
    # pad text with spaces if it is too short
    num_padding = text_length - len(text_end_extracted)
    padded = text_end_extracted + " " * num_padding
    text_int8_repr = np.array([alphabet.find(char) for char in padded], dtype=np.int8)
    x_batch_one_hot = np.zeros(shape=[1, text_length, len(alphabet)])
    for char_pos_in_seq, char_seq_char_ind in enumerate(text_int8_repr):
        if char_seq_char_ind != -1:
            x_batch_one_hot[0][char_pos_in_seq][char_seq_char_ind] = 1
    return x_batch_one_hot