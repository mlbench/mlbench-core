import argparse


class DatasetLoaderParser(argparse.ArgumentParser):
    def __init__(self, add_help=True, pytorch=True, tensorflow=True, dataset_version=True, dataset_root=True,
                 shuffle_before_partition=True, batch_size=True, num_parallel_workers=True):
        super(DatasetLoaderParser, self).__init__(add_help=add_help)

        self.add_argument('--dataset', type=str, metavar='<DN>', required=True,
                          help="[default: %(default)s] name of the dataset.")

        backend_group = self.add_mutually_exclusive_group()

        if pytorch:
            backend_group.add_argument('--pytorch', action='store_true', default=False,
                                       help="[default: %(default)s] Use PyTorch DataLoader.")

        if tensorflow:
            backend_group.add_argument('--tensorflow', action='store_true', default=False,
                                       help="[default: %(default)s] Use tensorflow DataLoader.")

        if dataset_version:
            self.add_argument('--dataset_version', type=str, metavar='<DV>',
                              help="[default: %(default)s] default preprocessing methods.")

        if dataset_root:
            self.add_argument('--dataset_root', type=str, metavar='<DR>',
                              help="[default: %(default)s] root folder of dataset.")

        if shuffle_before_partition:
            self.add_argument('--shuffle_before_partition', type=str, metavar='<SBP>',
                              help="[default: %(default)s] reshuffle the partitioned index before partition.")

        if batch_size:
            self.add_argument('--batch_size', type=int, metavar='<BS>',
                              help="[default: %(default)s] batch size.")

        if num_parallel_workers:
            self.add_argument('--num_parallel_workers', type=int, metavar='<NPW>',
                              help="[default: %(default)s] number of workers to load dataset.")
