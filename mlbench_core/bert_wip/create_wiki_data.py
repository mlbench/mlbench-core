from wiki_downloader import WikiDownloader
from wiki_extractor import WikiExtractorWrapper
from wiki_formatter import WikiFormatter
import os
import argparse

"""
    Script for testing WikiCorpus helpers
"""

# Should be specified by the user (argparse)
DOWNLOADED = 'data/downloaded/'
EXTRACTED = 'data/extracted/'
FORMATTED = 'data/formatted/'

if __name__ == '__main__':
    downloader = WikiDownloader('en', DOWNLOADED)
    downloader.download(unzip=True)

    extractor = WikiExtractorWrapper(EXTRACTED)
    extractor(os.path.join(DOWNLOADED, 'wikicorpus_en/wikicorpus_en.xml'))

    formatter = WikiFormatter(FORMATTED, recursive=True)
    formatter(os.path.join(EXTRACTED, 'wikicorpus_en'))
