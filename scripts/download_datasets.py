# Python
import zipfile
import os
from pathlib import Path
import math

# Third Party
import requests
from tqdm import tqdm


def download_sintel():
    # https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
    url = "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip"

    root = Path.cwd() / 'datasets' / 'sintel'

    if not root.exists():
        root.mkdir()

    file_name = str(root / 'MPI-Sintel-complete.zip')
    unzip_file_name = str(root)
    download(url, file_name, unzip_file_name)


def download(url, file_name, unzip_file_name):
    # https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
    resp = requests.get(url, stream=True)
    total_size = int(resp.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0

    print('############### DOWNLOADING SINTEL DATA ###############')

    with open(file_name, 'wb') as f:
        for data in tqdm(resp.iter_content(block_size), total=math.ceil(total_size // block_size), unit='KB', unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)

        if total_size != 0 and wrote != total_size:
            print("ERROR, something went wrong")

    print('############### UNZIPPING SINTEL DATA ###############')

    unzip_file(file_name, unzip_file_name)

    print('############### COMPLETE SINTEL DATA ###############')


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def unzip_file(file_name, unzip_path):
    zip_ref = zipfile.ZipFile(file_name, 'r')
    zip_ref.extractall(unzip_path)
    zip_ref.close()
    # os.remove(file_name)


if __name__ == '__main__':
    download_sintel()
