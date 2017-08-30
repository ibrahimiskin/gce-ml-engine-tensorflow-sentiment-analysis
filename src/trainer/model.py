import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.layers import feature_column
from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow.contrib.learn.python.learn.utils import (saved_model_export_utils)
import constants as const


def generate_experiment_fn(transformed_train_file_pattern, transformed_test_file_pattern, transformed_metadata_dir, raw_metadata_dir, vocab_size, train_batch_size, train_num_epochs, num_train_instances, num_test_instances):
    def train_and_evaluate(output_dir):
        review_column = feature_column.sparse_column_with_integerized_feature(const.REVIEW_COLUMN, bucket_size=vocab_size + 1, combiner='sum')
        weighted_reviews = feature_column.weighted_sparse_column(review_column, const.REVIEW_WEIGHT)

        estimator = learn.LinearClassifier(feature_columns=[weighted_reviews],
                                         n_classes=2,
                                         model_dir=output_dir,
                                         config=tf.contrib.learn.RunConfig(save_checkpoints_secs=30))

        transformed_metadata = metadata_io.read_metadata(transformed_metadata_dir)
        raw_metadata = metadata_io.read_metadata(raw_metadata_dir)

        train_input_fn = input_fn_maker.build_training_input_fn(
            transformed_metadata,
            transformed_train_file_pattern,
            training_batch_size=train_batch_size,
            label_keys=[const.LABEL_COLUMN])

        eval_input_fn = input_fn_maker.build_training_input_fn(
            transformed_metadata,
            transformed_test_file_pattern,
            training_batch_size=1,
            label_keys=[const.LABEL_COLUMN])

        serving_input_fn = input_fn_maker.build_default_transforming_serving_input_fn(
            raw_metadata=raw_metadata,
            transform_savedmodel_dir=output_dir + '/transform_fn',
            raw_label_keys=[],
            raw_feature_keys=[const.REVIEW_COLUMN])

        export_strategy = saved_model_export_utils.make_export_strategy(
            serving_input_fn,
            exports_to_keep=5,
            default_output_alternative_key=None)

        return tf.contrib.learn.Experiment(
            estimator=estimator,
            train_steps=train_num_epochs * num_train_instances / train_batch_size,
            eval_steps=num_test_instances,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            export_strategies=export_strategy,
            min_eval_frequency=500)
    return train_and_evaluate
