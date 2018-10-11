from .parser import *
from .dataset import create_dataset, _DATASET_NAMES


def load(train, name='', line_args=None, parser=None):
    """Create a data loader.

    If `line_args` and `parser` are not None, then use line arguments to parse.

    :param name: str, defaults to ''
    :type name: str, optional
    :param line_args: line argument containing configs of datasets, defaults to None
    :type line_args: list of strs, optional
    :param parser: parser for line arguments, defaults to DatasetLoaderParser()
    :type parser: ArgumentParser, optional
    """
    if line_args is None:
        line_args = ['--dataset', name]

    if not (isinstance(line_args, list) and all(isinstance(el, str) for el in line_args)):
        raise ValueError("line_args should be a list of strings. Got {}".format(line_args))

    # Parser
    if parser is None:
        parser = DatasetLoaderParser()

    if not isinstance(parser, argparse.ArgumentParser):
        raise ValueError("parser should be argparse.ArgumentParser type. Got {}".format(type(parser)))

    options, unused_args = parser.parse_known_args(line_args)
    return create_dataset(train, options)
