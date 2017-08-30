

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
import util
import model
import shutil
import datetime
import os


tf.logging.set_verbosity(tf.logging.FATAL)


def main():
    tf.flags.DEFINE_string("output_dir", "/Users/iiskin/Documents/workspace/PreProduction/sentiment_beam/output", "Directory to export the model run results")
    tf.flags.DEFINE_string("input_data_dir", "/Users/iiskin/Documents/workspace/PreProduction/sentiment_beam/data", "Path to directory containing training and testing data")
    tf.flags.DEFINE_string("version", "default", "Version of your model")
    tf.flags.DEFINE_boolean("transform_data", False, "Preprocess raw data")

    tf.flags.DEFINE_integer("vocab_size", 20000, "Vocabulary size")
    tf.flags.DEFINE_integer("train_batch_size", 1000, "Batch size for training")
    tf.flags.DEFINE_integer("train_num_epochs", 10000, "Number of epochs for training")
    tf.flags.DEFINE_integer("num_train_instances", 2000, "Number of training instances")
    tf.flags.DEFINE_integer("num_test_instances", 2000, "Number of test instances")
    tf.flags.DEFINE_string("delimiters", ".,!?() ", "Delimiters to be used in splitting text")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()

    raw_file_dir = os.path.join(FLAGS.input_data_dir, 'raw')
    raw_metadata_dir = os.path.join(raw_file_dir, 'metadata')
    train_neg_file_pattern = os.path.join(raw_file_dir, 'train/negative/*')
    train_pos_file_pattern = os.path.join(raw_file_dir, 'train/positive/*')
    test_neg_file_pattern = os.path.join(raw_file_dir, 'test/negative/*')
    test_pos_file_pattern = os.path.join(raw_file_dir, 'test/positive/*')

    transformed_file_dir = os.path.join(FLAGS.input_data_dir, 'transformed')
    transformed_metadata_dir = os.path.join(transformed_file_dir, 'metadata')
    transformed_train_file_pattern = os.path.join(transformed_file_dir, 'train/*')
    transformed_test_file_pattern = os.path.join(transformed_file_dir, 'test/*')

    temp_dir = os.path.join(FLAGS.output_dir, "tmp")
    #model_run_dir = os.path.join(FLAGS.output_dir, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    model_run_dir = os.path.join(FLAGS.output_dir, FLAGS.version)

    if not FLAGS.transform_data:
        if not os.path.exists(transformed_file_dir):
            raise Exception("It doesn't look like the raw data has been transformed yet. Use transform_data flag to transform the raw data.")
    else:
        shutil.rmtree(transformed_file_dir, ignore_errors=True)
        util.preprocess_data(train_neg_file_pattern=train_neg_file_pattern,
                        train_pos_file_pattern=train_pos_file_pattern,
                        test_neg_file_pattern=test_neg_file_pattern,
                        test_pos_file_pattern=test_pos_file_pattern,
                        transformed_train_file_pattern=transformed_train_file_pattern,
                        transformed_test_file_pattern=transformed_test_file_pattern,
                        transformed_metadata_dir=transformed_metadata_dir,
                        raw_metadata_dir=raw_metadata_dir,
                        transform_func_dir=model_run_dir,
                        temp_dir=temp_dir,
                        vocab_size=FLAGS.vocab_size,
                        delimiters=FLAGS.delimiters)

    print("\nRun \"tensorboard --logdir {}\" to see the results on Tensorboard\n\n".format(FLAGS.output_dir))
    learn_runner.run(experiment_fn=model.generate_experiment_fn(transformed_train_file_pattern=transformed_train_file_pattern,
                                                          transformed_test_file_pattern=transformed_test_file_pattern,
                                                          transformed_metadata_dir=transformed_metadata_dir,
                                                          raw_metadata_dir=raw_metadata_dir,
                                                          vocab_size=FLAGS.vocab_size,
                                                          train_batch_size=FLAGS.train_batch_size,
                                                          train_num_epochs=FLAGS.train_num_epochs,
                                                          num_train_instances=FLAGS.num_train_instances,
                                                          num_test_instances=FLAGS.num_test_instances), output_dir=model_run_dir)

if __name__ == '__main__':
    main()
