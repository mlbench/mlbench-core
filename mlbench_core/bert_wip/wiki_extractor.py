import subprocess
import os
from pathlib import Path


class WikiExtractorWrapper:


    def __init__(self, save_path, *args):
        self.save_path = save_path
        self.args = args # for later


    def __call__(self, dump_path):
        output_folder = Path(os.path.basename(dump_path)).with_suffix('')
        output_path = os.path.join(self.save_path, output_folder)
        if os.path.exists(output_path):
            return
        subprocess.run(f'wikiextractor {dump_path} -o {output_path}', shell=True, check=True)
