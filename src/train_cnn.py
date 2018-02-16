import datetime
import os
import time
import numpy as np
import tensorflow as tf
import makedata_classification as md
import model_cnn
import preprocessing_classification

# Parameters ==============================================
# Model Hyperparameters
tf.flags.DEFINE_float("keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("num_authors", os.getenv('NUM_AUTHORS', 5), "Number of authors (default: 5)")
tf.flags.DEFINE_boolean("use_all_available_texts", os.getenv('USE_ALL_AVAILABLE_TEXTS', False), "Should all available texts be used (default: False)")
tf.flags.DEFINE_integer("text_length", os.getenv('TEXT_LENGTH', 1000), "Length of the texts read by the model (default: 1000)")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", os.getenv('NUM_EPOCHS', 25), "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1, "Evaluate model on dev set after this many steps (default: 100)") # used only for debugging
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_float("learning_rate", 1e-2, "Learning Rate (default: 1e-5)")
tf.flags.DEFINE_integer("n_eval_samples", 10000, "Number of evaluation samples (default: 500)")
# Misc Parameters
tf.flags.DEFINE_boolean("train", True, "Train the model (default: False)")
tf.flags.DEFINE_string("model_name", "cnn", "Filename for saved model (default: cnn)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation ========================================
# Create data
#print("Creating data...")
#md.make("TrainSet-five_authors.txt")
# Load data
print("Loading data...")
x_train, x_eval, y_train, y_eval = preprocessing_classification.load_data("TrainSet-five_authors.txt")
# Shuffle training and evaluation data
shuffle_indices_train = np.random.permutation(np.arange(len(y_train)))
shuffle_indices_eval = np.random.permutation(np.arange(len(y_eval)))
x_train = x_train[shuffle_indices_train]
x_eval = x_eval[shuffle_indices_eval]
y_train = y_train[shuffle_indices_train]
y_eval = y_eval[shuffle_indices_eval]
# Output information
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_eval)))
tf.flags.DEFINE_integer("num_batches_per_epoch", int(len(x_train)/FLAGS.batch_size)+1, "Number of batches per epoch")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


# Training ================================================
def run_training():
    with tf.Graph().as_default():

        with tf.variable_scope("cnn"):
            input_x_pl = tf.placeholder(tf.float32, [None, len(preprocessing_classification.alphabet), FLAGS.text_length, 1], name="input_x")
            keep_prob = tf.placeholder(tf.float32)
            logits, h_vals, variables = model_cnn.inference(input_x_pl, keep_prob, FLAGS.num_authors)

        input_y_pl = tf.placeholder(tf.float32, [None, tf.flags.FLAGS.num_authors], name="input_y")
        loss = model_cnn.loss(logits, input_y_pl)
        train_op = model_cnn.training(loss, tf.flags.FLAGS.learning_rate)
        eval_correct = model_cnn.evaluation(logits, input_y_pl)
        prediction = model_cnn.prediction(logits)
        # Configurate and start TensorFlow-Session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(config=config)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "cnn", timestamp))
        save_dir = os.path.abspath(os.path.join(out_dir, "saves"))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("Writing to {}\n".format(out_dir))

        # Saver
        saver = tf.train.Saver(max_to_keep=100, pad_step_number=True)

        # Train Summaries
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        tf.summary.scalar("accuracy", eval_correct)
        tf.summary.scalar("loss", loss)
        tf.summary.histogram("predictions", prediction)
        summary_op = tf.summary.merge_all()
        summary_dir_train = os.path.join(out_dir, "summaries", "training")
        summary_dir_test = os.path.join(out_dir, "summaries", "testing")
        summary_writer_train = tf.summary.FileWriter(summary_dir_train, sess.graph)
        summary_writer_test = tf.summary.FileWriter(summary_dir_test, sess.graph)

        sess.run(tf.global_variables_initializer())

        with sess.as_default():
            # Generate batches in one-hot-encoding format
            batches = preprocessing_classification.batch_iterator(x_train, y_train, FLAGS.batch_size, FLAGS.num_epochs, nn="cnn")
            # Training loop. For each batch...
            step = 0
            plot_ctr = 1
            for batch in batches:
                step += 1
                # Training Step ===========================
                x_batch, y_batch = zip(*batch)
                feed_dict = {
                    input_x_pl: x_batch,
                    input_y_pl: y_batch,
                    keep_prob: FLAGS.keep_prob
                }
                _, loss_value, accuracy_value, predictions, ls, summaries = sess.run([train_op, loss, eval_correct, prediction, logits, summary_op], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss_value, accuracy_value))
                summary_writer_train.add_summary(summaries, step)
                # Evaluation Step =========================
                if step % FLAGS.num_batches_per_epoch == 0:  # FLAGS.evaluate_every == 0: (used for debugging)
                    print("Evaluation:")
                    eval_size = len(x_eval)
                    max_batch_size = 32
                    num_batches = int(eval_size / max_batch_size)
                    # Prepare variables for evaluation
                    acc = []
                    losses = []
                    print("Number of batches in dev set is " + str(num_batches))
                    for i in range(num_batches):
                        x_batch_dev, y_batch_dev = preprocessing_classification.get_batched_one_hot(
                            x_eval, y_eval, i * max_batch_size, (i + 1) * max_batch_size, nn="cnn")
                        feed_dict = {
                            input_x_pl: x_batch_dev,
                            input_y_pl: y_batch_dev,
                            keep_prob: 1.0
                        }
                        # Evaluation operation
                        accuracy_value, summaries, loss_value = sess.run(
                            [eval_correct, summary_op, loss], feed_dict)
                        acc.append(accuracy_value)
                        losses.append(loss_value)
                        time_str = datetime.datetime.now().isoformat()
                        print("batch " + str(i + 1) + " in dev >>" +
                              " {}: loss {:g}, acc {:g}".format(time_str, loss_value, accuracy_value))
                        summary_writer_test.add_summary(summaries, step)
                    print("Mean accuracy=" + str(sum(acc) / len(acc)))
                    print("Mean loss=" + str(sum(losses) / len(losses)))
                    # Export Model ========================
                    print("saving brain...")
                    saved_path = saver.save(sess=sess, save_path=os.path.join(save_dir, "cnn.ckpt"), global_step=step)
                    print("brain saved to: " + str(saved_path))

if FLAGS.train:
    run_training()
