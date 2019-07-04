# Download code taken from Code taken from https://stackoverflow.com questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
# Python
import zipfile
import os
import math

# Third Party
import requests
from tqdm import tqdm


def download_file_from_google_drive(id, destination):
    print('############### DOWNLOADING FLOWNET2 MODEL ###############')

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(
        URL,
        params={'id': id},
        stream=True
    )
    token = get_confirm_token(response)
    headers = {'Range':'bytes=0-'}

    if token:
        params = {
            'id': id,
            'confirm': token
        }
        response = session.get(URL, params=params, headers=headers, stream=True)
    save_response_content(response, destination)

    print('############### COMPLETE DOWNLOADING FLOWNET2 MODEL ###############')


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    # https://stackoverflow.com/questions/52044489/how-to-get-content-length-for-google-drive-download
    total_size = int(response.headers.get('Content-Range', '0').partition('/')[-1])

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE), total=math.ceil(total_size // CHUNK_SIZE), unit='KB', unit_scale=True):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def unzip_file(file_name, unzip_path):
    zip_ref = zipfile.ZipFile(file_name, 'r')
    zip_ref.extractall(unzip_path)
    zip_ref.close()
    os.remove(file_name)
