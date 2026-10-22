# Standard libraries
from numpy._core.defchararray import lower
import re
import numpy as np
import pandas as pd

# Special libraries
import contractions
from autocorrect import Speller
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import emoji
import gc
from collections import Counter

# Setting up spell checks and word tokenization
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
spell = Speller(lang = 'en')


# Function to run cleaning steps on Trustpilot outputs

def clean_text(dataset, spell, stop_words):

    # Some format Date column as datetime and filter to use only data
    # since the beginning of the year 2025
    dataset['Date'] = pd.to_datetime(dataset['Date'], format = '%Y-%m-%d')
    dataset = dataset[dataset['Date'] >= '2025-01-01 00:00:00']
    dataset = dataset.reset_index(drop = True)

    # Combine sentences into one text (and also add the review title at the beginning)
    reviews = dataset['Review_Sentences']
    l = len(reviews)
    review_adj = [''] * l
    for i in range(0, l):
        try:
            text0 = dataset['Review_Title'][i]
            len_list = len(reviews[i])
            for j in range(0, len_list):
                text0 = text0 + ' ' + reviews[i][j]
            review_adj[i] = text0
        except TypeError:
            review_adj[i] = dataset['Review_Title'][i]

    dataset['Review_Sentences'] = review_adj

    # Replace emojis by text replacement
    dataset['Review_Sentences'] = [emoji.demojize(review) for review in dataset['Review_Sentences']]
    # Replace everything except letters, numbers, whitespace, !, ?, and ' with a whitespace
    dataset['Review_Sentences'] = dataset['Review_Sentences'].str.replace(r'[^\w\s\'?!]', ' ', regex = True)
    # Replace all numbers and multiple whitespaces in a row by a single whitespace
    dataset['Review_Sentences'] = dataset['Review_Sentences'].str.replace(r'[\s0-9]+', ' ', regex = True)
    # Remove leading and trailing whitespace
    dataset['Review_Sentences'] = dataset['Review_Sentences'].str.strip()
    # Make everything lower case
    dataset['Review_Sentences'] = dataset['Review_Sentences'].str.lower()

    # Create a subobject of the review texts
    reviews = dataset['Review_Sentences']
    # Split English contractions into two words, for example "isn't" into "is not"
    reviews = [contractions.fix(review) for review in reviews]
    # Now replace all remaining ' by an empty string
    reviews = [re.sub(r'\'+', '', review) for review in reviews]
    # Break review texts into individual words
    reviews = [word_tokenize(review) for review in reviews]
    # Correct spelling
    reviews = [[spell(word) for word in review] for review in reviews]
    # Remove stop words like "and"
    reviews = [[word for word in word_list if word not in stop_words] for word_list in reviews]
    # Stemming
    snowball = SnowballStemmer(language = 'english')
    reviews = [[snowball.stem(w) for w in review] for review in reviews]

    l = len(reviews)
    review_adj = [''] * l
    for i in range(0, l):
        len_list = len(reviews[i])
        text0 = ''
        for j in range(0, len_list):
            text0 = text0 + ' ' + reviews[i][j]
        review_adj[i] = text0

    dataset['Review_Sentences'] = review_adj
    dataset['Review_Sentences'] = dataset['Review_Sentences'].str.strip()
    dataset['Rating_Class'] = 'Bad'
    dataset.loc[dataset['Review_Score'] >= 4, 'Rating_Class'] = 'Good'

    dataset = dataset.drop(['User_ID', 'Date', 'User_Name', 'User_Country', 'Review_Score', 'Review_Title'], axis = 1)

    return dataset


# Original data download from GitHub
data_full = pd.read_feather("https://github.com/dschulz13/dataset-amazon_trustpilot_reviews/raw/refs/heads/main/amazon_com_reviews_trustpilot_20261017_anon.feather")

# Run cleaning steps on the data
data_full = clean_text(data_full, spell, stop_words)

# Review texts are now fully cleaned and prepared for further analysis
data_full.to_feather(f'amazon_reviews_cleaned.feather')

# Delete data from memory
del data_full
gc.collect()
