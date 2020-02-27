"""
Utility functions that allow the download and transformation of LIBSVM datasets
into LMDB format.

https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

"""
import os
from sklearn.datasets import load_svmlight_file
from tensorpack.dataflow import dataset, PrefetchDataZMQ, LMDBSerializer
import sys
from urllib.request import urlretrieve
import logging
import bz2
from sklearn.datasets import make_classification

_logger = logging.getLogger('mlbench')

_DATASET_MAP = {
    'australian_train': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools'
                        '/datasets/binary/australian',
    'duke_train': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets'
                  '/binary/duke.bz2',
    'epsilon_train': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools'
                     '/datasets/binary/epsilon_normalized.bz2',
    'epsilon_test': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets'
                    '/binary/epsilon_normalized.t.bz2',
    'rcv1_train': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets'
                  '/binary/rcv1_train.binary.bz2',
    'rcv1_test': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets'
                 '/binary/rcv1_test.binary.bz2',
    'url': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary'
           '/url_combined.bz2',
    'webspam_train':
        'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary'
        '/webspam_wc_normalized_trigram.svm.bz2',
}


def _get_dense_tensor(tensor):
    if 'sparse' in str(type(tensor)):
        return tensor.toarray()
    elif 'numpy' in str(type(tensor)):
        return tensor


def _correct_binary_labels(labels, is_01_classes=True):
    classes = set(labels)

    if -1 in classes and is_01_classes:
        labels[labels == -1] = 0
    return labels


class LIBSVMDataset(object):
    def __init__(self, root_path, name, data_type, is_sparse):
        self.is_sparse = is_sparse

        # get file url and file path.
        data_url = _DATASET_MAP['{}_{}'.format(name, data_type)]
        file_name = data_url.split('/')[-1]

        file_path = maybe_download_and_extract_bz2(root_path, file_name,
                                                   data_url)
        data = load_svmlight_file(file_path)
        self.features, self.labels = self._get_features_and_labels(data)

    def _get_features_and_labels(self, data):
        features, labels = data

        features = _get_dense_tensor(features) if not self.is_sparse else \
            features
        labels = _get_dense_tensor(labels)
        labels = _correct_binary_labels(labels)
        return features, labels

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


class SyntheticLIBSVMDataset(object):
    def __init__(self, features, labels):
        self.features, self.labels = features, labels

    def __len__(self):
        return self.features.shape[0]

    def __iter__(self):
        idxs = list(range(self.__len__()))
        for k in idxs:
            features = self.features[k]
            label = [self.labels[k]]
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


def maybe_download_and_extract_bz2(root, file_name, data_url):
    """ Downloads file from given URL and extracts if bz2

    Args:
        root (str): The root directory
        file_name (str): File name to download to
        data_url (str): Url of data
    """
    if not os.path.exists(root):
        os.makedirs(root)

    file_path = os.path.join(root, file_name)

    # Download file if not present
    if len([x for x in os.listdir(root) if x == file_name]) == 0:
        progress_download(data_url, file_path)

    # Extract downloaded file if compressed
    if file_name.endswith(".bz2"):
        file_basename = os.path.splitext(file_name)[0]
        extracted_fpath = os.path.join(root, file_basename)
        extract_bz2_file(file_path, extracted_fpath)

        os.remove(file_path)
        file_path = extracted_fpath
    return file_path


def progress_download(url, dest):
    """ Downloads a file from `url` to `dest` and shows progress

    Args:
        url (src): Url to retrieve file from
        dest (src): Destination file
    """
    print(url)
    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            os.path.basename(dest),
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    urlretrieve(url, dest, _progress)
    _logger.info("Downloaded {} to {}".format(url, dest))


def extract_bz2_file(source, dest):
    """ Extracts a bz2 archive

    Args:
        source (str): Source file (must have .bz2 extension)
        dest (str): Destination file

    """
    assert source.endswith(".bz2"), "Extracting non bz2 archive"
    with open(source, 'rb') as s, open(dest, 'wb') as d:
        d.write(bz2.decompress(s.read()))


def sequential_epsilon_or_rcv1(root_path, name, data_type, is_sparse):
    data = LIBSVMDataset(root_path, name, data_type, is_sparse)
    lmdb_file_path = os.path.join(root_path, '{}_{}.lmdb'.format(name,
                                                                 data_type))

    ds1 = PrefetchDataZMQ(data, nr_proc=1)
    LMDBSerializer.save(ds1, lmdb_file_path)
    _logger.info('Dumped dataflow to {} for {}'.format(lmdb_file_path, name))


def sequential_synthetic_dataset(root_path, dataset_name, data_type):
    """Generate a synthetic dataset for regression."""
    if data_type == 'dense':
        X, y = make_classification(n_samples=10000,
                                   n_features=100,
                                   n_informative=90,
                                   n_classes=2,
                                   random_state=42)
    else:
        raise NotImplementedError("{} synthetic dataset is "
                                  "not supported.".format(data_type))

    data = SyntheticLIBSVMDataset(X, y)
    lmdb_file_path = os.path.join(root_path, '{}_{}.lmdb'.format(dataset_name,
                                                                 data_type))

    _logger.info('Dumped dataflow to {} for {}'.format(lmdb_file_path,
                                                       dataset_name))
    ds1 = PrefetchDataZMQ(data, nr_proc=1)
    LMDBSerializer.save(ds1, lmdb_file_path)


def generate_lmdb_from_libsvm(data, data_type, data_dir, sparse):
    if 'epsilon' in data or 'rcv1' in data or 'webspam' in data:
        sequential_epsilon_or_rcv1(data_dir, data, data_type, sparse)
    elif 'australian' in data or 'duke' in data:
        # These two are small datasets for testing purpose.
        sequential_epsilon_or_rcv1(data_dir, data, data_type, sparse)
    elif 'synthetic' in data:
        sequential_synthetic_dataset(data_dir, data, data_type)
    else:
        raise NotImplementedError(
            "Dataset {} not supported.".format(data))
