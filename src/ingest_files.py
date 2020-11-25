import zipfile
import os
import gzip
import pandas as pd


def unzip_main_file(path_to_zip_file, directory_to_extract_to):
    """
    Extracts contents into a directory
    :param path_to_zip_file: Path to the zip file downloaded
    :param directory_to_extract_to: Directory to extract all the contents
    :return: None
    """
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


def open_gzip_file(path_to_gzip_file):
    """Takes a path to the gzip file and ungzips it"""
    with gzip.open(path_to_gzip_file, 'rb') as file:
        ungzipped = file.read().decode('utf-8').split('\t')

    print(ungzipped[:20])
    return ungzipped


def frame_the_data(ungzipped_file):
    """Takes an ungzipped file and converts to a pandas dataframe"""
    samples = []

    for idx in range(0, len(ungzipped_file)):

        if idx != 0 and idx % 4 == 0:
            samples.extend(ungzipped_file[idx - 4:idx])

    data = pd.DataFrame(samples, columns=['source', 'alignment_type', 'alignment_quality', 'english', 'hindi'])

    print(data.head(10))
    return data


if __name__ == '__main__':
    ## zip file to download from https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0001-BD17-1
    path_to_zip_file='HindEnCorp 0.5.zip'
    directory_to_extract_to = os.getcwd()
    unzip_main_file(path_to_zip_file,directory_to_extract_to)

    path_to_gzip_file = 'hindencorp05.plaintext.gz'
    read_file = open_gzip_file(path_to_gzip_file)

    data = frame_the_data(read_file)

    data.to_csv('data.csv',index=True)







