# Standard libraries
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

# Setting up spell checks and word tokenization
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
spell = Speller(lang = 'en')

# Function to reduce memory usage of data frame
def reduce_df_size(df, verbose = True):
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32',
'float64']
    if verbose:
        old_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    df['User_Country'] = df['User_Country'].astype('category')
    df['Review_Date'] = pd.to_datetime(df['Review_Date'], format = '%Y-%m-%d %H:%M:%S', utc = True)
    df['Experience_Date'] = pd.to_datetime(df['Experience_Date'], format = '%Y-%m-%d', utc = True)

    if verbose:
        mem_new = (df.memory_usage().sum() / 1024**2).item()
        change_mem = abs(mem_new - old_mem)
        rel_mem = 100 * change_mem / old_mem
        print(f'Old memory usage: {old_mem:.2f}MB; new memory usage: {mem_new:.2f}MB; reduction of {change_mem:.2f}MB ({rel_mem:.2f}%)')
    return df

# Function to run cleaning steps on Trustpilot outputs

def clean_text(dataset, spell, stop_words):

    # Combine sentences into one text (and also add the review title at the beginning)
    reviews = dataset['Review_Sentences']
    l = len(reviews)
    review_adj = [''] * l
    for i in range(0, l):
        try:
            review_adj[i] = dataset['Review_Title'][i] + ' ' + ' '.join(dataset['Review_Sentences'][i])
        except TypeError:
            review_adj[i] = dataset['Review_Title'][i]

    dataset['Review_Sentences'] = review_adj
    del review_adj
    gc.collect()

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
        review_adj[i] = ' '.join(reviews[i])

    dataset['Review_Sentences'] = review_adj
    del(reviews, review_adj)
    gc.collect()

    dataset['Review_Sentences'] = dataset['Review_Sentences'].str.strip()
    dataset['Rating_Class'] = 'Bad'
    dataset.loc[dataset['Review_Score'] >= 4, 'Rating_Class'] = 'Good'
    dataset['Rating_Class'] = dataset['Rating_Class'].astype('category')

    dataset = dataset.drop(['User_ID', 'Review_Date', 'Experience_Date', 'User_Name', 'User_Country', 'Review_Score', 'Review_Title'], axis = 1)
    gc.collect()

    return dataset


# Data URL 
url = 'https://github.com/dschulz13/dataset-amazon_trustpilot_reviews/raw/refs/heads/main/amazon_com_reviews_trustpilot_20261026_anon.feather'

# Original data download from GitHub
data_filt = pd.read_feather(url)

# Apply reformatting of columns
data_filt['Review_Score'] = data_filt['Review_Score'].astype(int)
data_filt = reduce_df_size(data_filt)

# Keep observations since 2025-01-01 and reset index
data_filt = (
    data_filt[data_filt['Review_Date'] >= '2025-01-01 00:00:00']
        .reset_index(drop = True)
)

# Run cleaning steps on the data
data_filt = clean_text(data_filt, spell, stop_words)

# Review texts are now fully cleaned and prepared for further analysis
data_filt.to_feather(f'amazon_reviews_cleaned.feather')

# Delete data from memory
del data_filt
gc.collect()
