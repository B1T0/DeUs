import tensorflow as tf
import preprocessing_classification


# Function for MANY summaries
def variable_summaries(var, name):
    pass
    # with tf.name_scope('summaries'):
    #     mean = tf.reduce_mean(var)
    #     tf.summary.scalar('mean/' + name, mean)
    #     with tf.name_scope('stddev'):
    #         stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    #     tf.summary.scalar('sttdev/' + name, stddev)
    #     tf.summary.scalar('max/' + name, tf.reduce_max(var))
    #     tf.summary.scalar('min/' + name, tf.reduce_min(var))
    #     tf.summary.histogram(name, var)


def conv_layer(filter_shape, tf_name_scope, layer_input):
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
    variable_summaries(W, tf_name_scope+"/weights")
    b = tf.Variable(tf.constant(0.1, shape=[256]), name="b")
    variable_summaries(b, tf_name_scope+"/biases")
    conv = tf.nn.conv2d(layer_input, W, strides=[1, 1, 1, 1], padding="VALID", name=tf_name_scope+"conv")
    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
    return W, b, h
    
def inference(input_x, keep_prob, num_authors):
    tf.summary.image("input", input_x, max_outputs=10)
    # Layer 1 ===============================
    with tf.name_scope("conv-1"):
        filter_shape = [len(preprocessing_classification.alphabet), 7, 1, 256]
        W_1, b_1, h_1 = conv_layer(filter_shape, "conv-1", input_x)

    # Layer 2 ===============================
    with tf.name_scope("conv-2"):
        filter_shape = [1, 7, 256, 256]
        W_2, b_2, h_2 = conv_layer(filter_shape, "conv-2", h_1)
        
    # Layer 3 ===============================
    with tf.name_scope("conv-3"):
        filter_shape = [1, 3, 256, 256]
        W_3, b_3, h_3 = conv_layer(filter_shape, "conv-3", h_2)
        
    # Layer 4 ===============================
    with tf.name_scope("conv-4"):
        filter_shape = [1, 3, 256, 256]
        W_4, b_4, h_4 = conv_layer(filter_shape, "conv-4", h_3)
        
    # Layer 5 ===============================
    with tf.name_scope("conv-5"):
        filter_shape = [1, 3, 256, 256]
        W_5, b_5, h_5 = conv_layer(filter_shape, "conv-5", h_4)

    # Layer 6 ===============================
    with tf.name_scope("conv-6"):
        filter_shape = [1, 3, 256, 256]
        W_6, b_6, h_6 = conv_layer(filter_shape, "conv-6", h_5)
        
    # Flattening and dropout ================
    h_pool_flat = tf.reshape(h_6, [-1, int(h_6._shape.dims[2])*int(h_6._shape.dims[3])])
    with tf.name_scope("dropout"):
        drop = tf.nn.dropout(h_pool_flat, keep_prob)

    # Fully connected layer =================
    with tf.name_scope("fc"):
        W_fc = tf.Variable(
            tf.truncated_normal([h_6._shape[2].value * h_6._shape[3].value, num_authors],
                                stddev=0.05), name="W-fc")
        variable_summaries(W_fc, "fc/weights")
        b_fc = tf.Variable(tf.constant(0.1, shape=[num_authors]), name="b-fc")
        variable_summaries(b_fc, "fc/biases")

        logits = tf.nn.softmax(tf.matmul(drop, W_fc) + b_fc, name="output")

    return logits,\
           [h_1, h_2, h_3, h_4, h_5, h_6, h_pool_flat],\
           [W_1, b_1, W_2, b_2, W_3, b_3, W_4, b_4, W_5, b_5, W_6, b_6, W_fc, b_fc]


def loss(logits, labels):
    # Cross-entropy loss
    with tf.name_scope("loss"):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(losses)
    return loss


def training(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(1e-2)
    train_op = optimizer.minimize(loss=loss)
    return train_op


def evaluation(logits, labels):
    predictions = prediction(logits)
    with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(predictions, tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    return accuracy


def prediction(logits):
    return tf.argmax(logits, 1, name="predictions")
