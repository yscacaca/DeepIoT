import os

import tensorflow as tf
import numpy as np


# todo: change the path to your own data folder path
DATA_FOLDER_PATH = 'sepHARData_a'
TF_RECORD_PATH = 'sepHARData_a'


SEPCTURAL_SAMPLES = 10
FEATURE_DIM = SEPCTURAL_SAMPLES*6*2
WIDE = 20
OUT_DIM = 6#len(idDict)
BATCH_SIZE = 64


def csv_to_example(fname):
    text = np.loadtxt(fname, delimiter=',')
    features = text[:WIDE*FEATURE_DIM]
    label = text[WIDE*FEATURE_DIM:]

    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=label)),
        'example': tf.train.Feature(float_list=tf.train.FloatList(value=features))
    }))

    return example


def read_and_decode(tfrec_path):
    filename_queue = tf.train.string_input_producer([tfrec_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([OUT_DIM], tf.float32),
                                           'example': tf.FixedLenFeature([WIDE*FEATURE_DIM], tf.float32),
                                       })
    return features['example'], features['label']


def input_pipeline_har(tfrec_path, batch_size, shuffle_sample=True, num_epochs=None):
    example, label = read_and_decode(tfrec_path)
    example = tf.reshape(example, [WIDE, FEATURE_DIM])
    example = tf.expand_dims(example, 0)
    example = tf.reshape(example, shape=(WIDE, FEATURE_DIM))
    min_after_dequeue = 1000  # int(0.4*len(csvFileList)) #1000
    capacity = min_after_dequeue + 3 * batch_size
    if shuffle_sample:
        example_batch, label_batch = tf.train.shuffle_batch(
            [example, label], batch_size=batch_size, num_threads=16, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
    else:
        example_batch, label_batch = tf.train.batch(
            [example, label], batch_size=batch_size, num_threads=16)

    return example_batch, label_batch


def main(_):
    writer = tf.python_io.TFRecordWriter(os.path.join(TF_RECORD_PATH, 'train.tfrecord'))
    train_path = os.path.join(DATA_FOLDER_PATH, 'train')
    train_files = os.listdir(train_path)
    print 'train_path', train_path, len(train_files)
    for f in train_files:
        f_pre, f_suf = f.split('.')
        if f_suf == 'csv':
            f_path = os.path.join(train_path, f)
            example = csv_to_example(f_path)
            writer.write(example.SerializeToString())
    writer.close()

    writer = tf.python_io.TFRecordWriter(os.path.join(TF_RECORD_PATH, 'eval.tfrecord'))
    train_path = os.path.join(DATA_FOLDER_PATH, 'eval')
    train_files = os.listdir(train_path)
    print 'train_path', train_path, len(train_files)
    for f in train_files:
        f_pre, f_suf = f.split('.')
        if f_suf == 'csv':
            f_path = os.path.join(train_path, f)
            example = csv_to_example(f_path)
            writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    tf.app.run()
