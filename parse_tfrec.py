import tensorflow as tf
from utils.configs import *

# train, rgb, 1-layer
def _parse_function(example_proto):
    features = tf.parse_single_example(
        example_proto,
        features={
            'train': tf.FixedLenFeature(shape=(config.data.patch_size*config.data.patch_size*3,), dtype=tf.float32),
            'label': tf.FixedLenFeature(shape=(config.data.patch_size*config.data.patch_size*3,), dtype=tf.float32)
        }
    )
    img = features['train']
    label = features['label']

    label = tf.reshape(label, [config.data.patch_size, config.data.patch_size, 3])
    img = tf.reshape(img, [config.data.patch_size, config.data.patch_size, 3])

    return img, label


# train, rgb, multi-layer
def _parse_function_multilayer(example_proto):
    """
    returns an array of training images in img.
    """
    feature_labels={
        'label': tf.FixedLenFeature(shape=(config.data.patch_size*config.data.patch_size*3,), dtype=tf.float32)
    }
    # Add training layers equal to the number of layers in the pyramid.
    for l in range(0, config.data.py_lev+1):
        feature_labels['train{0}'.format(l)] = tf.FixedLenFeature(shape=(config.data.patch_size*config.data.patch_size*3,), dtype=tf.float32)

    features = tf.parse_single_example(
        example_proto,
        features=feature_labels
    )

    img = []
    for l in range(0, config.data.py_lev+1):
        img.append(features['train{0}'.format(l)])
        img[-1] = tf.reshape(img[-1], [config.data.patch_size, config.data.patch_size, 3])

    label = features['label']
    label = tf.reshape(label, [config.data.patch_size, config.data.patch_size, 3])

    return img, label


def _parse_function_new_high_gray(example_proto):
    features = tf.parse_single_example(
        example_proto,
        features={
            'train': tf.FixedLenFeature(shape=(config.data.patch_size*config.data.patch_size,), dtype=tf.float32),
            'label': tf.FixedLenFeature(shape=(config.data.patch_size*config.data.patch_size,), dtype=tf.float32),
        }
    )
    img = features['train']
    label = features['label']

    img = tf.reshape(img, [config.data.patch_size, config.data.patch_size, 1])
    label = tf.reshape(label, [config.data.patch_size, config.data.patch_size, 1])

    return img, label


def _parse_function_new_bot_gray(example_proto):
    features = tf.parse_single_example(
        example_proto,
        features={
            'train': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
            'label': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
            'h': tf.FixedLenFeature([], dtype=tf.int64),
            'w': tf.FixedLenFeature([], dtype=tf.int64),

        }
    )
    img = features['train']
    label = features['label']

    h = tf.cast(features['h'], tf.int32)
    w = tf.cast(features['w'], tf.int32)

    img = tf.reshape(img, [h, w, 1])
    label = tf.reshape(label, [h, w, 1])

    return img, label


def _parse_function_new_ft_gray(example_proto):
    features = tf.parse_single_example(
        example_proto,
        features={
            'train1': tf.FixedLenFeature(shape=(256 * 256,), dtype=tf.float32),
            'label': tf.FixedLenFeature(shape=(256 * 256,), dtype=tf.float32),
            'train2': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
            'h2': tf.FixedLenFeature([], dtype=tf.int64),
            'w2': tf.FixedLenFeature([], dtype=tf.int64),

        }
    )
    img_h = features['train1']
    img_b = features['train2']
    label = features['label']

    h2 = tf.cast(features['h2'], tf.int32)
    w2 = tf.cast(features['w2'], tf.int32)

    img_h = tf.reshape(img_h, [256, 256, 1])
    img_b = tf.reshape(img_b, [h2, w2, 1])
    label = tf.reshape(label,  [256, 256, 1])

    return img_h, img_b, label


def _parse_function_new_high_rgb(example_proto):
    features = tf.parse_single_example(
        example_proto,
        features={
            'train': tf.FixedLenFeature(shape=(config.data.patch_size*config.data.patch_size*3,), dtype=tf.float32),
            'label': tf.FixedLenFeature(shape=(config.data.patch_size*config.data.patch_size*3,), dtype=tf.float32),
        }
    )
    img = features['train']
    label = features['label']

    img = tf.reshape(img, [config.data.patch_size, config.data.patch_size, 3])
    label = tf.reshape(label, [config.data.patch_size, config.data.patch_size, 3])

    return img, label


def _parse_function_new_bot_rgb(example_proto):
    features = tf.parse_single_example(
        example_proto,
        features={
            'train': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
            'label': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
            'h': tf.FixedLenFeature([], dtype=tf.int64),
            'w': tf.FixedLenFeature([], dtype=tf.int64),
            'c': tf.FixedLenFeature([], dtype=tf.int64),

        }
    )
    img = features['train']
    label = features['label']

    h = tf.cast(features['h'], tf.int32)
    w = tf.cast(features['w'], tf.int32)
    c = tf.cast(features['c'], tf.int32)

    img = tf.reshape(img, [h, w, c])
    label = tf.reshape(label, [h, w, c])

    return img, label


def _parse_function_new_ft_rgb(example_proto):
    features = tf.parse_single_example(
        example_proto,
        features={
            'train1': tf.FixedLenFeature(shape=(256 * 256 * 3,), dtype=tf.float32),
            'label': tf.FixedLenFeature(shape=(256 * 256 * 3,), dtype=tf.float32),
            'train2': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
            'h2': tf.FixedLenFeature([], dtype=tf.int64),
            'w2': tf.FixedLenFeature([], dtype=tf.int64),
            'c2': tf.FixedLenFeature([], dtype=tf.int64),

        }
    )
    img_h = features['train1']
    img_b = features['train2']
    label = features['label']

    h2 = tf.cast(features['h2'], tf.int32)
    w2 = tf.cast(features['w2'], tf.int32)
    c2 = tf.cast(features['c2'], tf.int32)

    img_h = tf.reshape(img_h, [256, 256, 3])
    img_b = tf.reshape(img_b, [h2, w2, c2])
    label = tf.reshape(label,  [256, 256, 3])

    return img_h, img_b, label


# train, gray, 1-layer
def _parse_function_gray(example_proto):
    features = tf.parse_single_example(
        example_proto,
        features={
            'train': tf.FixedLenFeature(shape=(config.data.patch_size*config.data.patch_size,), dtype=tf.float32),
            'label': tf.FixedLenFeature(shape=(config.data.patch_size*config.data.patch_size,), dtype=tf.float32),
            'index': tf.FixedLenFeature([], dtype=tf.int64)
        }
    )
    img = features['train']
    label = features['label']
    index = tf.cast(features['index'], tf.int32)

    label = tf.reshape(label, [config.data.patch_size, config.data.patch_size])
    img = tf.reshape(img, [config.data.patch_size, config.data.patch_size])

    return img, label, index


# train, gray, bottom-layer only
def _parse_function_gray_bottom(example_proto):
    features = tf.parse_single_example(
        example_proto,
        features={
            'train': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
            'label': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
            'h': tf.FixedLenFeature([], dtype=tf.int64),
            'w': tf.FixedLenFeature([], dtype=tf.int64),
            'index': tf.FixedLenFeature([], dtype=tf.int64)
        }
    )
    img = features['train']
    label = features['label']
    index = tf.cast(features['index'], tf.int32)
    h = tf.cast(features['h'], tf.int32)
    w = tf.cast(features['w'], tf.int32)

    img = tf.reshape(img, [h, w])
    label = tf.reshape(label, [h, w])

    return img, label, index


# train, gray, dual, ft layer
def _parse_function_gray_dual_heter_ft(example_proto):
    features = tf.parse_single_example(
        example_proto,
        features={
            'train': tf.FixedLenFeature(shape=(config.data.patch_size*config.data.patch_size,), dtype=tf.float32),
            'label': tf.FixedLenFeature(shape=(config.data.patch_size*config.data.patch_size,), dtype=tf.float32),
            'bottom': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
            'h': tf.FixedLenFeature([], dtype=tf.int64),
            'w': tf.FixedLenFeature([], dtype=tf.int64),
        }
    )
    img = features['train']
    label = features['label']
    bottom = features['bottom']
    h = tf.cast(features['h'], tf.int32)
    w = tf.cast(features['w'], tf.int32)

    img = tf.reshape(img, [config.data.patch_size, config.data.patch_size])
    label = tf.reshape(label, [config.data.patch_size, config.data.patch_size])
    bottom = tf.reshape(bottom, [h, w])

    return img, label, bottom


# train, gray, dual, ft layer, no global branch
def _parse_function_gray_dual_heter_ft_no_global(example_proto):
    features = tf.parse_single_example(
        example_proto,
        features={
            'train': tf.FixedLenFeature(shape=(config.data.patch_size*config.data.patch_size,), dtype=tf.float32),
            'label': tf.FixedLenFeature(shape=(config.data.patch_size*config.data.patch_size,), dtype=tf.float32),
        }
    )
    img = features['train']
    label = features['label']

    img = tf.reshape(img, [config.data.patch_size, config.data.patch_size])
    label = tf.reshape(label, [config.data.patch_size, config.data.patch_size])

    return img, label


# train, gray, multi-layer
def _parse_function_gray_multilayer(example_proto):
    feature_labels = {
        'label': tf.FixedLenFeature(shape=(config.data.patch_size * config.data.patch_size,), dtype=tf.float32)
    }
    # Add training layers equal to the number of layers in the pyramid.
    for l in range(0, config.data.py_lev + 1):
        feature_labels['train{0}'.format(l)] = tf.FixedLenFeature(
            shape=(config.data.patch_size * config.data.patch_size,), dtype=tf.float32)

    feature_labels['index'] = tf.FixedLenFeature([], dtype=tf.int64)

    features = tf.parse_single_example(
        example_proto,
        features=feature_labels
    )

    img = []
    for l in range(0, config.data.py_lev + 1):
        img.append(features['train{0}'.format(l)])
        img[-1] = tf.reshape(img[-1], [config.data.patch_size, config.data.patch_size])

    label = features['label']
    label = tf.reshape(label, [config.data.patch_size, config.data.patch_size])

    index = tf.cast(features['index'], tf.int32)

    return img, label, index


# train, gray, dual-layer, with label
def _parse_function_gray_duallayer_gen_train_ft(example_proto):
    feature_labels = {
        'label': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
        'h1': tf.FixedLenFeature([], dtype=tf.int64),
        'w1': tf.FixedLenFeature([], dtype=tf.int64),
        'h2': tf.FixedLenFeature([], dtype=tf.int64),
        'w2': tf.FixedLenFeature([], dtype=tf.int64)
    }
    # Add training layers equal to the number of layers in the pyramid.
    for l in range(0, 2):
        feature_labels['train{0}'.format(l)] = tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)

    features = tf.parse_single_example(
        example_proto,
        features=feature_labels
    )

    h1 = tf.cast(features['h1'], tf.int32)
    w1 = tf.cast(features['w1'], tf.int32)
    h2 = tf.cast(features['h2'], tf.int32)
    w2 = tf.cast(features['w2'], tf.int32)

    label = features['label']
    label = tf.reshape(label, [h1, w1])

    train0 = features['train{0}'.format(0)]
    train0 = tf.reshape(train0, [h1, w1])

    train1 = features['train{0}'.format(1)]
    train1 = tf.reshape(train1, [h2, w2])

    return train0, train1, label


# eval, rgb, multi-layer
def _parse_eval_function(example_proto):
    feature_labels={
        'name': tf.FixedLenFeature([], dtype=tf.string),
        'h': tf.FixedLenFeature([], dtype=tf.int64),
        'w': tf.FixedLenFeature([], dtype=tf.int64)
    }

    for l in range(0, config.data.py_lev+1):
        feature_labels['eval{0}'.format(l)] = tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)

    features = tf.parse_single_example(
        example_proto,
        features=feature_labels
    )

    name = features['name']
    h = tf.cast(features['h'], tf.int32)
    w = tf.cast(features['w'], tf.int32)

    img = []
    for l in range(0, config.data.py_lev+1):
        img.append(features['eval{0}'.format(l)])
        img[-1] = tf.reshape(img[-1], [h, w, 3])

    return img, name


# eval, gray, multi-layer
def _parse_eval_function_gray(example_proto):
    feature_labels = {
        'name': tf.FixedLenFeature([], dtype=tf.string),
        'h': tf.FixedLenFeature([], dtype=tf.int64),
        'w': tf.FixedLenFeature([], dtype=tf.int64)
    }

    for l in range(0, config.data.py_lev + 1):
        feature_labels['eval{0}'.format(l)] = tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)

    features = tf.parse_single_example(
        example_proto,
        features=feature_labels
    )

    name = features['name']
    h = tf.cast(features['h'], tf.int32)
    w = tf.cast(features['w'], tf.int32)

    img = []
    for l in range(0, config.data.py_lev + 1):
        img.append(features['eval{0}'.format(l)])
        img[-1] = tf.reshape(img[-1], [h, w])

    return img, name


def _parse_eval_function_new_gray(example_proto):
    """
    parse dual layer for evaluation.  h1, w1 are the size of high layer. h2, w2 are the size of bottom layer.
    :param example_proto:
    :return:
    """
    feature_labels = {
        'name': tf.FixedLenFeature([], dtype=tf.string),
        'h1': tf.FixedLenFeature([], dtype=tf.int64),
        'w1': tf.FixedLenFeature([], dtype=tf.int64),
        'h2': tf.FixedLenFeature([], dtype=tf.int64),
        'w2': tf.FixedLenFeature([], dtype=tf.int64)
    }

    for l in range(0, config.data.py_lev + 1):
        feature_labels['eval{0}'.format(l)] = tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)

    features = tf.parse_single_example(
        example_proto,
        features=feature_labels
    )

    name = features['name']
    h1 = tf.cast(features['h1'], tf.int32)
    w1 = tf.cast(features['w1'], tf.int32)
    h2 = tf.cast(features['h2'], tf.int32)
    w2 = tf.cast(features['w2'], tf.int32)

    eval0 = features['eval{0}'.format(0)]
    eval0 = tf.reshape(eval0, [h1, w1, 1])

    eval1 = features['eval{0}'.format(1)]
    eval1 = tf.reshape(eval1, [h2, w2, 1])

    return eval0, eval1, h1, w1, name


# eval, gray, dual
def _parse_eval_function_new_rgb(example_proto):
    """
    parse dual layer for evaluation.  h1, w1 are the size of high layer. h2, w2 are the size of bottom layer.
    :param example_proto:
    :return:
    """
    feature_labels = {
        'name': tf.FixedLenFeature([], dtype=tf.string),
        'h1': tf.FixedLenFeature([], dtype=tf.int64),
        'w1': tf.FixedLenFeature([], dtype=tf.int64),
        'h2': tf.FixedLenFeature([], dtype=tf.int64),
        'w2': tf.FixedLenFeature([], dtype=tf.int64)
    }

    for l in range(0, config.data.py_lev + 1):
        feature_labels['eval{0}'.format(l)] = tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)

    features = tf.parse_single_example(
        example_proto,
        features=feature_labels
    )

    name = features['name']
    h1 = tf.cast(features['h1'], tf.int32)
    w1 = tf.cast(features['w1'], tf.int32)
    h2 = tf.cast(features['h2'], tf.int32)
    w2 = tf.cast(features['w2'], tf.int32)

    eval0 = features['eval{0}'.format(0)]
    eval0 = tf.reshape(eval0, [h1, w1, 3])

    eval1 = features['eval{0}'.format(1)]
    eval1 = tf.reshape(eval1, [h2, w2, 3])

    return eval0, eval1, h1, w1, name


# eval, gray, dual, heter, ft
def _parse_eval_function_gray_dual_heter_ft(example_proto):
    """
    parse dual layer for evaluation.  h1, w1 are the size of high layer. h2, w2 are the size of bottom layer.
    :param example_proto:
    :return:
    """
    feature_labels = {
        'eval': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
        'name': tf.FixedLenFeature([], dtype=tf.string),
        'h': tf.FixedLenFeature([], dtype=tf.int64),
        'w': tf.FixedLenFeature([], dtype=tf.int64)
    }

    features = tf.parse_single_example(
        example_proto,
        features=feature_labels
    )

    name = features['name']
    h = tf.cast(features['h'], tf.int32)
    w = tf.cast(features['w'], tf.int32)

    eval = features['eval'.format(0)]
    eval= tf.reshape(eval, [h, w])

    return eval, name


def data_iterator1(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_function)
    data = data.batch(1).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def data_iterator12(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_function)
    data = data.shuffle(buffer_size=200).batch(config.train.batch_size).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def data_iterator_multilayer(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_function_multilayer)
    data = data.shuffle(buffer_size=200).batch(config.train.batch_size).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def data_iterator_new_gray_high(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_function_new_high_gray)
    data = data.shuffle(buffer_size=200, reshuffle_each_iteration=True).batch(config.train.batch_size_high).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def data_iterator_new_gray_bot(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_function_new_bot_gray)
    data = data.shuffle(buffer_size=2000, reshuffle_each_iteration=True).batch(config.train.batch_size_bot).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def data_iterator_new_gray_ft(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_function_new_ft_gray)
    data = data.shuffle(buffer_size=500, reshuffle_each_iteration=True).batch(config.train.batch_size_ft).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def eval_iterator_new_gray_ft(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_eval_function_new_gray)
    data = data.repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def data_iterator_new_rgb_high(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_function_new_high_rgb)
    data = data.shuffle(buffer_size=200, reshuffle_each_iteration=True).batch(config.train.batch_size_high).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def data_iterator_new_rgb_bot(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_function_new_bot_rgb)
    data = data.shuffle(buffer_size=2000, reshuffle_each_iteration=True).batch(config.train.batch_size_bot).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def data_iterator_new_rgb_ft(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_function_new_ft_rgb)
    data = data.shuffle(buffer_size=500, reshuffle_each_iteration=True).batch(config.train.batch_size_ft).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def eval_iterator_new_rgb_ft(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_eval_function_new_rgb)
    data = data.repeat()
    iterater = data.make_one_shot_iterator()
    return iterater






def data_iterator_gray(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_function_gray)
    data = data.shuffle(buffer_size=200).batch(config.train.batch_size).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def data_iterator_gray_bottom(tfrecord, lev_scale):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_function_gray_bottom)
    # lev_scale == 1 means size of data is (750, 1500), batch_size == 2 is max for Titan X GPU.
    if lev_scale == '1':
        data = data.shuffle(buffer_size=180).batch(config.train.batch_size1).repeat()
    elif lev_scale == '2':
        data = data.shuffle(buffer_size=180).batch(config.train.batch_size1).repeat()
    elif lev_scale == '3':
        data = data.shuffle(buffer_size=180).batch(config.train.batch_size4).repeat()
    elif lev_scale == '4':
        data = data.shuffle(buffer_size=180).batch(config.train.batch_size8).repeat()
    else:
        data = data.shuffle(buffer_size=180).batch(config.train.batch_size).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def data_iterator_gray_multilayer(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_function_gray_multilayer)
    data = data.shuffle(buffer_size=200).batch(config.train.batch_size_ft).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def data_iterator_gray_duallayer_gen_train_ft(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_function_gray_duallayer_gen_train_ft)
    data = data.repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def data_iterator_gray_dual_heter_ft(tfrecord):
    """
    batch size is 12
    :param tfrecord:
    :return:
    """
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_function_gray_dual_heter_ft)
    data = data.shuffle(buffer_size=1000).batch(config.train.batch_size).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def data_iterator_gray_dual_heter_ft_lev_scale_1(tfrecord):
    """
    same as func'data_iterator_gray_dual_heter_ft', but with smaller batch size and shuffle buffer size.
    batch size is 4
    :param tfrecord:
    :return:
    """
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_function_gray_dual_heter_ft)
    data = data.shuffle(buffer_size=100).batch(config.train.batch_size4).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def data_iterator_gray_dual_heter_ft_no_global(tfrecord):
    """
    batch size is 12
    :param tfrecord:
    :return:
    """
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_function_gray_dual_heter_ft_no_global)
    data = data.shuffle(buffer_size=1000).batch(config.train.batch_size).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def eval_iterator(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_eval_function)
    data = data.repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def eval_iterator_gray(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_eval_function_gray)
    data = data.repeat()
    iterater = data.make_one_shot_iterator()
    return iterater



def eval_iterator_gray_dual_heter_ft(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_eval_function_gray_dual_heter_ft)
    data = data.repeat()
    iterater = data.make_one_shot_iterator()
    return iterater
