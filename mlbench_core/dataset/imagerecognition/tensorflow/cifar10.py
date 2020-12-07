r"""Test the tensorflow load and preprocess cifar-10 correctly.

Credit https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py
"""
import logging
import os
import sys
import tarfile

import tensorflow as tf
from six.moves import urllib, xrange


class DatasetCifar(object):
    """
    This clas is adapted from the following script
    https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py

    Args:
        dataset (str): Name of the dataset e.g. `cifar-10`, `cifar-100`.
        dataset_root (str): Root directory to the dataset.
        batch_size (int): Size of batch.
        world_size (int): Size of the world size.
        rank (int): Rank of the process.
        seed (int): Seed of random number.
        tf_dtype (tensorflow.python.framework.dtypes.DType, optional): Defaults to tf.float32.
            Datatypes of the tensor.
    """

    def __init__(
        self,
        dataset,
        dataset_root,
        batch_size,
        world_size,
        rank,
        seed,
        tf_dtype=tf.float32,
    ):
        # define image size and some commonly used parameters.
        self.data_url = "http://www.cs.toronto.edu/~kriz/{}-binary.tar.gz".format(
            dataset
        )

        self.dataset = dataset
        self.dataset_dir = dataset_root
        self.seed = seed

        self.batch_size = batch_size * world_size
        self.num_examples_per_epoch_for_train = 50000
        self.num_examples_per_epoch_for_eval = 10000
        self.num_batches_per_epoch_for_train = (
            self.num_examples_per_epoch_for_train // self.batch_size
        )
        self.num_batches_per_epoch_for_eval = (
            self.num_examples_per_epoch_for_eval // self.batch_size
        )

        self.image_size = 32
        self.image_channel = 3

        if dataset == "cifar-10":
            self.label_bytes = 1
            self.label_offset = 0
            self.num_classes = 10
        else:
            self.label_bytes = 1
            self.label_offset = 1
            self.num_classes = 100

        # Every record consists of a label followed by the image,
        # with a fixed number of bytes for each.
        self.image_bytes = self.image_size * self.image_size * self.image_channel
        self.record_bytes = self.label_bytes + self.label_offset + self.image_bytes

        # download the dataset.
        self.maybe_download_and_extract()

        # Define dataset for both training and validation
        self.train_dataset = self.input_fn(
            is_train=True, repeat_count=-1, num_shards=world_size, shard_index=rank
        )
        self.validation_dataset = self.input_fn(
            is_train=False, repeat_count=-1, num_shards=world_size, shard_index=rank
        )

        # Define reinitializable iterators for dataset.
        # Initialize operations like train_init_op and validation_init_op when switch mode.
        iterator = tf.data.Iterator.from_structure(
            self.train_dataset.output_types, self.train_dataset.output_shapes
        )
        self.train_init_op = iterator.make_initializer(self.train_dataset)
        self.validation_init_op = iterator.make_initializer(self.validation_dataset)

        # do not need to feed_dict for `inputs`, `labels`, `training`
        (self.inputs, self.labels), self.training = iterator.get_next()

    def maybe_download_and_extract(self):
        """Download and extract the tarball from Alex's website."""
        dest_directory = os.path.join(self.dataset_dir, self.dataset)

        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = self.data_url.split("/")[-1]
        filepath = os.path.join(dest_directory, filename)

        if not os.path.exists(filepath):

            def _progress(count, block_size, total_size):
                sys.stdout.write(
                    "\r>> Downloading %s %.1f%%"
                    % (filename, float(count * block_size) / float(total_size) * 100.0)
                )
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(self.data_url, filepath, _progress)
            logging.debug("download file to the path:" + filepath)
        else:
            logging.debug("retrieve file to the path:" + filepath)

        if self.dataset == "cifar-10":
            self.data_dir = os.path.join(dest_directory, self.dataset + "-batches-bin")
        else:
            self.data_dir = os.path.join(dest_directory, self.dataset + "-binary")

        if not os.path.exists(self.data_dir):
            logging.debug("does not exist extracted file: {}".format(self.data_dir))
            tarfile.open(filepath, "r:gz").extractall(dest_directory)

    def record_dataset(self, filenames):
        """Returns an input pipeline Dataset from `filenames`."""
        return tf.data.FixedLengthRecordDataset(filenames, self.record_bytes)

    def get_filenames(self, is_training=True):
        if is_training:
            filenames = (
                [
                    os.path.join(self.data_dir, "data_batch_%d.bin" % i)
                    for i in xrange(1, 6)
                ]
                if self.dataset == "cifar-10"
                else [os.path.join(self.data_dir, "train.bin")]
            )
        else:
            filenames = (
                [os.path.join(self.data_dir, "test_batch.bin")]
                if self.dataset == "cifar-10"
                else [os.path.join(self.data_dir, "test.bin")]
            )
        return filenames

    def parse_record(self, raw_record):
        """Parse CIFAR-10/100 image and label from a raw record."""
        # Convert bytes to a vector of uint8 that is record_bytes long.
        # record_vector = tf.decode_raw(raw_record, tf.uint8)
        record = tf.reshape(tf.decode_raw(raw_record, tf.uint8), [self.record_bytes])

        # The first byte represents the label,
        # which we convert from uint8 to int32 and then to one-hot.
        label = tf.squeeze(
            tf.cast(tf.slice(record, [self.label_offset], [self.label_bytes]), tf.int32)
        )
        label = tf.one_hot(label, self.num_classes)

        # The remaining bytes after the label represent the image,
        # which we reshape from
        # [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(
            tf.slice(
                record, [self.label_offset + self.label_bytes], [self.image_bytes]
            ),
            [self.image_channel, self.image_size, self.image_size],
        )

        # Convert from [depth, height, width] to [height, width, depth],
        # and cast as float32.
        image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
        return image, label

    def preprocess_image(self, image, is_training):
        """Preprocess a single image of layout [height, width, depth]."""
        if is_training:
            # Randomly flip the image horizontally.
            image = tf.image.random_flip_left_right(image)

            # decide to augment the data and only for training data.
            # Resize the image to add four extra pixels on each side.
            pad = 4
            image = tf.image.resize_image_with_crop_or_pad(
                image, self.image_size + pad * 2, self.image_size + pad * 2
            )

            # Randomly crop a [image_size, image_size] section of the image.
            image = tf.random_crop(
                image, [self.image_size, self.image_size, self.image_channel]
            )

            # consider to add more stronger augmentation.
            # image = tf.image.random_brightness(image, max_delta=63)
            # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        else:
            image = tf.image.resize_image_with_crop_or_pad(
                image, self.image_size, self.image_size
            )

        if self.dataset == "cifar-10":
            stats = {
                "mean": tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32),
                "std": tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32),
            }
            image = (image / 256 - stats["mean"]) / stats["std"]
        else:
            # Subtract off the mean and divide by the variance of the pixels.
            # image = tf.image.per_image_standardization(image)
            raise NotImplementedError
        return image

    def input_fn(self, is_train, repeat_count=-1, num_shards=1, shard_index=0):
        """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

        In synchronized training, faster nodes may use more batches than the number
        of batches availble. Thus repeat dataset for engouh times to avoid throwing
        error.

        In the distributed settings, datasets are split into `num_shards`
        non-overlapping parts and each process takes one shard by its index.

        Args:
            is_train (bool): A boolean denoting whether the input is for training.
            repeat_count (int): Defaults to -1. Count of dataset repeated times
                with -1 for infinite.
            num_shards (int): Defaults to 1. Number of Shards the dataset is splitted.
            shard_index (int): Defaults to 0. Index of shard to use.

        Returns
            tf.data.Dataset object of `((inputs, labels), is_train)`.
        """

        dataset = self.record_dataset(self.get_filenames(is_train))

        dataset = dataset.shard(num_shards, shard_index)

        if is_train:
            # When choosing shuffle buffer sizes, larger sizes result in better
            # randomness, while smaller sizes have better performance.
            # Because CIFAR-10 is a relatively small dataset,
            # we choose to shuffle the full epoch.
            dataset = dataset.shuffle(
                buffer_size=self.num_examples_per_epoch_for_train,
                seed=self.seed,
                reshuffle_each_iteration=True,
            )

        dataset = dataset.map(self.parse_record, num_parallel_calls=8)
        dataset = dataset.map(
            lambda image, label: (self.preprocess_image(image, is_train), label),
            num_parallel_calls=8,
        )
        # TODO: change prefetch size? to `tf.contrib.data.AUTOTUNE`
        # https://github.com/tensorflow/models/blob/master/official/resnet/resnet_run_loop.py#L103
        dataset = dataset.prefetch(2 * self.batch_size)

        # We call repeat after shuffling, rather than before,
        # to prevent separate epochs from blending together.
        dataset = dataset.repeat(repeat_count)

        # Batch results by up to batch_size,
        # and then fetch the tuple from the iterator.
        dataset = dataset.batch(self.batch_size)

        # A boolean indicating the mode / dataset type : True for training and False for validation.
        mode = tf.data.Dataset.from_tensor_slices([is_train]).repeat()
        return tf.data.Dataset.zip((dataset, mode))
