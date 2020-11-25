
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import string
from string import digits
import matplotlib.pyplot as plt
%matplotlib inline
import re

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', -1)

def clean_data(file_path='./truncated_data/Hindi_English_Truncated_Corpus.csv',sample_size=80000):
    """
    Removes duplicates and null values and samples data
    :param file_path:Saved data path
    :param sample_size: Sample size to use
    :return:
    """

    ##extract contents of truncated_data/archives.zip and use the csv file there..its a sample
    data = pd.read_csv(file_path, encoding='utf-8')

    #remove null pairs
    data = data[~pd.isnull(data['english_sentence'])]

    data.drop_duplicates(inplace=True)
    return data.sample(sample_size,random_state=42) if sample_size else data

def normalize_text(data):
    """
    Takes in data and lowercases it, removes punctuations,digits,
    removes special tokens and characters, quotes
    Finally it adds in start and end tokens to the target sequences
    :param data: Cleaned data
    :return: Text Normalized data
    """
    # Lowercase all characters
    data['english_sentence'] = data['english_sentence'].apply(lambda x: x.lower())
    data['hindi_sentence'] = data['hindi_sentence'].apply(lambda x: x.lower())

    # Remove quotes
    data['english_sentence'] = data['english_sentence'].apply(lambda x: re.sub("'", '', x))
    data['hindi_sentence'] = data['hindi_sentence'].apply(lambda x: re.sub("'", '', x))

    exclude = set(string.punctuation)  # Set of all special characters
    # Remove all the special characters
    data['english_sentence'] = data['english_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    data['hindi_sentence'] = data['hindi_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

    # Remove all numbers from text
    remove_digits = str.maketrans('', '', digits)
    data['english_sentence'] = data['english_sentence'].apply(lambda x: x.translate(remove_digits))
    data['hindi_sentence'] = data['hindi_sentence'].apply(lambda x: x.translate(remove_digits))

    data['hindi_sentence'] = data['hindi_sentence'].apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))

    # Remove extra spaces
    data['english_sentence'] = data['english_sentence'].apply(lambda x: x.strip())
    data['hindi_sentence'] = data['hindi_sentence'].apply(lambda x: x.strip())
    data['english_sentence'] = data['english_sentence'].apply(lambda x: re.sub(" +", " ", x))
    data['hindi_sentence'] = data['hindi_sentence'].apply(lambda x: re.sub(" +", " ", x))

    # Add start and end tokens to target sequences
    data['hindi_sentence'] = data['hindi_sentence'].apply(lambda x: 'START_ ' + x + ' _END')

    return data

def preprocess():
    """Wrapper that cleans and normalizes text, saves file"""
    data = clean_data(file_path='./truncated_data/Hindi_English_Truncated_Corpus.csv', sample_size=80000)

    data = normalize_text(data)

    data.to_csv('./truncated_data/cleaned_data.csv')

if __name__=='__main__':
    preprocess()