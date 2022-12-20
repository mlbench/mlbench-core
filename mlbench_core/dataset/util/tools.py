import bz2
import os
import sys
import tarfile
import zipfile
from urllib.request import urlretrieve


def progress_download(url, dest):
    """Downloads a file from `url` to `dest` and shows progress

    Args:
        url (src): Url to retrieve file from
        dest (src): Destination file
    """

    def _progress(count, block_size, total_size):
        percentage = float(count * block_size) / float(total_size) * 100.0
        if percentage % 25 == 0:
            sys.stdout.write(
                "\r>> Downloading %s %.1f%%" % (os.path.basename(dest), percentage)
            )
            sys.stdout.flush()

    urlretrieve(url, dest, _progress)
    print("\nDownloaded {} to {}\n".format(url, dest))


def extract_bz2_file(source, dest, delete=True):
    """Extracts a bz2 archive

    Args:
        source (str): Source file (must have .bz2 extension)
        dest (str): Destination file
        delete (bool): Delete compressed file after decompression

    """
    assert source.endswith(".bz2"), "Extracting non bz2 archive"

    if os.path.isfile(dest):
        print("File {} already extracted to {}".format(source, dest))
        return
    with open(dest, "wb") as d, open(source, "rb") as s:
        decompressor = bz2.BZ2Decompressor()
        for data in iter(lambda: s.read(1000 * 1024), b""):
            d.write(decompressor.decompress(data))

    if delete:
        os.remove(source)


def compress_to_bz2_file(source, delete=True):
    """Extracts a bz2 archive

    Args:
        source (str): Source file to compress
        delete (bool): Delete un-compressed file
    """

    dest = source + ".bz2"
    with open(source, "rb") as s, open(dest, "wb") as d:
        compressor = bz2.BZ2Compressor()
        for data in iter(lambda: s.read(1000 * 1024), b""):
            d.write(compressor.compress(data))

    if delete:
        os.remove(source)


def maybe_download_and_extract_bz2(root, file_name, data_url):
    """Downloads file from given URL and extracts if bz2

    Args:
        root (str): The root directory
        file_name (str): File name to download to
        data_url (str): Url of data
    """
    if not os.path.exists(root):
        os.makedirs(root)

    file_path = os.path.join(root, file_name)
    file_basename = os.path.splitext(file_name)[0]
    extracted_fpath = os.path.join(root, file_basename)

    if os.path.isfile(extracted_fpath):
        return extracted_fpath

    # Download file if not present
    if len([x for x in os.listdir(root) if x == file_name]) == 0:
        progress_download(data_url, file_path)

    # Extract downloaded file if compressed
    if file_name.endswith(".bz2"):
        # Extract file
        extract_bz2_file(file_path, extracted_fpath, delete=True)
        file_path = extracted_fpath
    return file_path


def maybe_download_and_extract_tar_gz(root, file_name, data_url):
    """Downloads file from given URL and extracts if compressed as tar.gz

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

    if file_name.endswith(".tar.gz"):
        with tarfile.open(file_path, "r:gz") as tar:
            dirs = [member for member in tar.getmembers()]
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=root, members=dirs)


def maybe_download_and_extract_zip(root, file_name, data_url):
    """Downloads file from given URL and extracts if compressed as zip

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

    if file_name.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zip:
            zip.extractall(root)
