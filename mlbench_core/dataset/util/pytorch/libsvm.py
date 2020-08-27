"""
Utility functions that allow the download and transformation of LIBSVM datasets
into LMDB format.

https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

"""
import os

import click
from sklearn.datasets import load_svmlight_file, make_classification
from tensorpack.dataflow import LMDBSerializer, PrefetchDataZMQ

from mlbench_core.dataset.util.tools import maybe_download_and_extract_bz2

_DATASET_MAP = {
    "australian_train": {
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools"
        "/datasets/binary/australian",
        "sparse": False,
    },
    "duke_train": {
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
        "/binary/duke.tr.bz2",
        "sparse": True,
    },
    "duke_test": {
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
        "/binary/duke.val.bz2",
        "sparse": True,
    },
    "epsilon_train": {
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools"
        "/datasets/binary/epsilon_normalized.bz2",
        "sparse": False,
    },
    "epsilon_test": {
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
        "/binary/epsilon_normalized.t.bz2",
        "sparse": False,
    },
    "rcv1_train": {
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
        "/binary/rcv1_train.binary.bz2",
        "sparse": True,
    },
    "rcv1_test": {
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
        "/binary/rcv1_test.binary.bz2",
        "sparse": True,
    },
    "webspam_train": {
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
        "/webspam_wc_normalized_trigram.svm.bz2",
        "sparse": True,
    },
}


def _get_dense_tensor(tensor):
    if "sparse" in str(type(tensor)):
        return tensor.toarray()
    elif "numpy" in str(type(tensor)):
        return tensor


def _correct_binary_labels(labels, is_01_classes=True):
    classes = set(labels)

    if -1 in classes and is_01_classes:
        labels[labels == -1] = 0
    return labels


def _get_features_and_labels(data, is_sparse):
    features, labels = data

    features = _get_dense_tensor(features) if not is_sparse else features
    labels = _get_dense_tensor(labels)
    labels = _correct_binary_labels(labels)
    return features, labels


def _load_libsvm_data(root_path, name, data_type):
    data = _DATASET_MAP["{}_{}".format(name, data_type)]
    data_url = data["url"]
    is_sparse = data["sparse"]

    file_name = data_url.split("/")[-1]
    # Downloads and extracts the data (deletes downloaded file after)
    file_path = maybe_download_and_extract_bz2(root_path, file_name, data_url)

    print("Loading SVM file {}".format(file_path))
    data = load_svmlight_file(file_path)
    features, labels = _get_features_and_labels(data, is_sparse)
    return features, labels, is_sparse


class LIBSVMDataset(object):
    def __init__(self, features, labels, is_sparse):

        self.is_sparse = is_sparse

        self.features, self.labels = features, labels

    def __len__(self):
        return self.features.shape[0]

    def __iter__(self):
        idxs = list(range(self.__len__()))
        for k in idxs:
            features = self.features[k]
            label = self.labels[k]
            if self.is_sparse:
                features = features.tocoo()
                yield [(features.data, features.row, features.col), label]
            else:
                yield [features, label]

    def get_data(self):
        return self.__iter__()

    def size(self):
        return self.__len__()

    def reset_state(self):
        """
        Reset state of the dataflow.
        It **has to** be called once and only once before producing datapoints.
        Note:
            1. If the dataflow is forked, each process will call this method
               before producing datapoints.
            2. The caller thread of this method must remain alive to keep
            this dataflow alive.
        For example, RNG **has to** be reset if used in the DataFlow,
        otherwise it won't work well with prefetching, because different
        processes will have the same RNG state.
        """
        pass


def sequential_epsilon_or_rcv1(root_path, name, data_type):
    features, labels, is_sparse = _load_libsvm_data(root_path, name, data_type)
    data = LIBSVMDataset(features, labels, is_sparse)
    lmdb_file_path = os.path.join(root_path, "{}_{}.lmdb".format(name, data_type))

    ds1 = PrefetchDataZMQ(data, nr_proc=1)
    LMDBSerializer.save(ds1, lmdb_file_path)

    print("Dumped dataflow to {} for {}".format(lmdb_file_path, name))


def sequential_synthetic_dataset(root_path, dataset_name, data_type):
    """Generate a synthetic dataset for regression."""
    if data_type == "dense":
        X, y = make_classification(
            n_samples=10000,
            n_features=100,
            n_informative=90,
            n_classes=2,
            random_state=42,
        )
    else:
        raise NotImplementedError(
            "{} synthetic dataset is " "not supported.".format(data_type)
        )

    data = LIBSVMDataset(X, y, False)
    lmdb_file_path = os.path.join(
        root_path, "{}_{}.lmdb".format(dataset_name, data_type)
    )

    ds1 = PrefetchDataZMQ(data, nr_proc=1)
    LMDBSerializer.save(ds1, lmdb_file_path)

    print("Dumped dataflow to {} for {}".format(lmdb_file_path, dataset_name))


@click.command()
@click.argument("data")
@click.argument("data_type")
@click.argument("data_dir")
def generate_lmdb_from_libsvm(data, data_type, data_dir):
    """Utility script that downloads data from LIBSVM and transforms them
    into `.lmdb` files.

    Args:
        data (str): Name of dataset
        data_type (str): One of `train` `test`
        data_dir (str): Directory where to download, extract and transform
    """
    if "epsilon" in data or "rcv1" in data or "webspam" in data:
        sequential_epsilon_or_rcv1(data_dir, data, data_type)
    elif "australian" in data or "duke" in data:
        # These two are small datasets for testing purpose.
        sequential_epsilon_or_rcv1(data_dir, data, data_type)
    elif "synthetic" in data:
        sequential_synthetic_dataset(data_dir, data, data_type)
    else:
        raise NotImplementedError("Dataset {} not supported.".format(data))


if __name__ == "__main__":
    generate_lmdb_from_libsvm()
