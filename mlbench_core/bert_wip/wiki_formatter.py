import glob
import os
import logging
import re
import nltk
from pathlib import Path


class WikiFormatter:


    def __init__(self, save_path, recursive=False):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
        self.recursive = recursive


    def __call__(self, wiki_path):
        filename = Path(os.path.basename(wiki_path)).with_suffix('.txt')
        output_path = os.path.join(self.save_path, filename)
        if os.path.exists(output_path):
            return

        with open(output_path, 'w', newline='\n') as output_file:
            regex = re.compile(r'<doc id=.*?>([\s\S]*?)<\/doc>')
            mid_folders = os.path.join(wiki_path, '*')

            for dirname in glob.glob(mid_folders, recursive=False):
                extracted_files = os.path.join(dirname, 'wiki_*')

                for filename in glob.glob(extracted_files, recursive=self.recursive):
                    with open(filename, 'r', newline='\n') as file:
                        documents = re.finditer(regex, file.read())

                        for doc in documents:
                            article = re.sub(r'\n', ' ', doc.group(1).strip())
                            sentences = nltk.tokenize.sent_tokenize(article)
                            output_file.write('\n'.join(sentences))
                            output_file.write('\n\n')
