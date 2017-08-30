import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.beam import tft_beam_io
from tensorflow_transform.beam.tft_beam_io import beam_metadata_io
import apache_beam as beam
from apache_beam.io import textio
from apache_beam.io import tfrecordio
import constants as const
import tempfile
import datetime





@beam.ptransform_fn
def Shuffle(pcoll):
    """Shuffles a PCollection.  Collection should not contain duplicates."""
    return (pcoll
        | 'PairWithHash' >> beam.Map(lambda x: (hash(x), x))
        | 'GroupByHash' >> beam.GroupByKey()
        | 'DropHash' >> beam.FlatMap(lambda (k, vs): vs))


@beam.ptransform_fn
def ReadAndShuffleData(pcoll, filepatterns):
    """Read a train or test dataset from disk and shuffle it."""
    # NOTE: we pass filepatterns as a tuple instead of two args, as the current
    # version of beam assumes that if the first arg to a ptransfrom_fn is a
    # string, then that string is the label.
    neg_filepattern, pos_filepattern = filepatterns

    # Read from each file pattern and create a tuple of the review text and the
    # correct label.
    negative_examples = (
        pcoll
        | 'ReadNegativeExamples' >> textio.ReadFromText(neg_filepattern)
        | 'PairWithZero' >> beam.Map(lambda review: (review, 0))
    )
    positive_examples = (
        pcoll
        | 'ReadPositiveExamples' >> textio.ReadFromText(pos_filepattern)
        | 'PairWithOne' >> beam.Map(lambda review: (review, 1))
    )
    all_examples = ([negative_examples, positive_examples]
        | 'Merge' >> beam.Flatten())

    # Shuffle the data.  Note that the data does in fact contain duplicate reviews
    # for reasons that are unclear.  This means that NUM_TRAIN_INSTANCES and
    # NUM_TRAIN_INSTANCES are slightly wrong for the preprocessed data.
    # pylint: disable=no-value-for-parameter
    shuffled_examples = (
        all_examples
        | 'RemoveDuplicates' >> beam.RemoveDuplicates()
        | 'Shuffle' >> Shuffle())

    # Put the data in the format that can be accepted directly by tf.Transform.
    return shuffled_examples | 'MakeInstances' >> beam.Map(lambda p: {const.REVIEW_COLUMN: p[0], const.LABEL_COLUMN: p[1]})


def generate_preprocessing_fn(vocab_size, delimiters):
    def preprocessing_fn(inputs):
        """Preprocess input columns into transformed columns."""
        review = inputs[const.REVIEW_COLUMN]
        review_tokens = tft.map(lambda x: tf.string_split(x, delimiters), review)
        review_indices = tft.string_to_int(review_tokens, top_k=vocab_size)
        # Add one for the oov bucket created by string_to_int.
        review_weight = tft.tfidf_weights(review_indices, vocab_size + 1)

        output = {
            const.REVIEW_COLUMN: review_indices,
            const.REVIEW_WEIGHT: review_weight,
            const.LABEL_COLUMN: inputs[const.LABEL_COLUMN]
        }
        return output
    return preprocessing_fn


def preprocess_data(train_neg_file_pattern,
                    train_pos_file_pattern,
                    test_neg_file_pattern,
                    test_pos_file_pattern,
                    transformed_train_file_pattern,
                    transformed_test_file_pattern,
                    transformed_metadata_dir,
                    raw_metadata_dir,
                    transform_func_dir,
                    temp_dir,
                    vocab_size,
                    delimiters):
    """Transform the data and write out as a TFRecord of Example protos.
    Read in the data from the positive and negative examples on disk, and
    transform it using a preprocessing pipeline that removes punctuation,
    tokenizes and maps tokens to int64 values indices.

    Args:
    train_neg_filepattern: Filepattern for training data negative examples
    train_pos_filepattern: Filepattern for training data positive examples
    test_neg_filepattern: Filepattern for test data negative examples
    test_pos_filepattern: Filepattern for test data positive examples
    transformed_train_filebase: Base filename for transformed training data shards
    transformed_test_filebase: Base filename for transformed test data shards
    transformed_metadata_dir: Directory where metadata for transformed data should be written


    raw_metadata = dataset_metadata.DatasetMetadata(dataset_schema.Schema({
        REVIEW_COLUMN: dataset_schema.ColumnSchema(tf.string, [], dataset_schema.FixedColumnRepresentation()),
        LABEL_COLUMN: dataset_schema.ColumnSchema(tf.int64, [], dataset_schema.FixedColumnRepresentation()),
    }))
    """
    pipeline_name = 'DataflowRunner'
    options = {
        'job_name': ('cloud-ml-hazmat-preprocess-{}'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))),
        'temp_location': temp_dir,
        'project': "stone-outpost-636",
        'max_num_workers': 8
    }
    pipeline_options = beam.pipeline.PipelineOptions(flags=[], **options)
    #with beam.Pipeline(pipeline_name, options=pipeline_options) as pipeline:
    #    with beam_impl.Context(temp_dir=temp_dir):
    with beam.Pipeline() as pipeline:
        with beam_impl.Context(temp_dir=tempfile.mkdtemp()):

            train_data = pipeline | 'ReadTrain' >> ReadAndShuffleData((train_neg_file_pattern, train_pos_file_pattern))
            test_data = pipeline | 'ReadTest' >> ReadAndShuffleData((test_neg_file_pattern, test_pos_file_pattern))
            preprocessing_fn = generate_preprocessing_fn(vocab_size, delimiters)

            (transformed_train_data, transformed_metadata), transform_fn = ((train_data, const.RAW_METADATA)
              | 'AnalyzeAndTransform' >> beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

            _ = (transform_fn | 'WriteTransformFn' >> tft_beam_io.WriteTransformFn(transform_func_dir))

            transformed_test_data, _ = (((test_data, const.RAW_METADATA), transform_fn)
              | 'Transform' >> beam_impl.TransformDataset())

            _ = (transformed_train_data
              | 'WriteTrainData' >> tfrecordio.WriteToTFRecord(transformed_train_file_pattern,
                  coder=example_proto_coder.ExampleProtoCoder(transformed_metadata.schema)))

            _ = (transformed_test_data
              | 'WriteTestData' >> tfrecordio.WriteToTFRecord(transformed_test_file_pattern,
                  coder=example_proto_coder.ExampleProtoCoder(transformed_metadata.schema)))

            _ = (transformed_metadata
              | 'WriteTransformedMetadata' >> beam_metadata_io.WriteMetadata(transformed_metadata_dir, pipeline=pipeline))

            _ = (const.RAW_METADATA
              | 'WriteRawMetadata' >> beam_metadata_io.WriteMetadata(raw_metadata_dir, pipeline=pipeline))


