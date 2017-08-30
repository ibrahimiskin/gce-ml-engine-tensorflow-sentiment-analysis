import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema


REVIEW_COLUMN = 'review'
REVIEW_WEIGHT = 'review_weight'
LABEL_COLUMN = 'label'

INPUT_SCHEMA = dataset_schema.from_feature_spec({
    REVIEW_COLUMN: tf.FixedLenFeature(shape=[], dtype=tf.string),
    LABEL_COLUMN: tf.FixedLenFeature(shape=[], dtype=tf.int64)
})

RAW_METADATA = dataset_metadata.DatasetMetadata(schema=INPUT_SCHEMA)


