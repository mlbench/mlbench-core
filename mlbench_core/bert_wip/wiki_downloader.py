import os
import urllib.request
import bz2
import sys
from pathlib import Path


class WikiDownloader:


    def __init__(self, language, save_path):
        save_path = os.path.join(save_path, 'wikicorpus_' + language)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.download_urls = {
            'en' : 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-p1p41242.bz2',
            'zh' : 'https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles1.xml-p1p187712.bz2'
        }

        self.output_files = {
            'en' : 'wikicorpus_en.xml.bz2',
            'zh' : 'wikicorpus_zh.xml.bz2'
        }

        if language not in self.download_urls:
            raise NotImplementedError("WikiDownloader not implemented for this language yet.")

        filename = self.output_files[language]
        self.zip_path = os.path.join(save_path, filename)
        self.url = self.download_urls[language]


    def download(self, unzip=False):
        file_path = Path(self.zip_path).with_suffix('')
        if os.path.isfile(file_path):
            return
        response = urllib.request.urlopen(self.url)
        if unzip:
            data = bz2.decompress(response.read())
            with open(file_path, 'wb') as decompressed:
                decompressed.write(data)
        else:
            with open(self.zip_path, 'wb') as handle:
                handle.write(response.read())


    def decompress(self):
        data = bz2.BZ2File(self.zip_path).read()
        file_path = Path(self.zip_path).with_suffix('')
        with open(file_path, 'wb') as decompressed:
            decompressed.write(data)
