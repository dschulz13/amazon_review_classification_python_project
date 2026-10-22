# Identifying Negative Customer Experiences Through Text Classification Using Python: A Case Study of Amazon.com Reviews on Trustpilot
Dominik Schulz
October 22, 2026

<div data-align="justify">

# Executive summary

# Objective

# Data cleaning

The available review data in
[dschulz13/dataset-amazon_trustpilot_reviews](https://github.com/dschulz13/dataset-amazon_trustpilot_reviews)
is unformatted and raw and hence requires preliminary cleaning. Most
notably, the variable `Review_Sentences` contains a list for each review
entry, where each of the list’s elements is a string representing an
entire sentence of the review text. Consequently, the review sentences
are split into individual strings in the given data. Furthermore, the
review titles are available in a separate variable `Review_Title`.

Initially, we only keep only reviews starting from January 01, 2025,
that are thus providing some (debatable) relevance to the present day.
The goal of the cleaning step is then two-fold: firstly, we wish to
categorize the star rating into two groups. Those groups will be
labelled `Good` and `Bad`. A good review is a review with a four- or
five-start rating. A bad review covers the remaining star ratings. Of
course other class constructions are also possible and could be checked
in further analyses. For the prupose of this project, we, however,
assume that a rating of three stars or less indicates a notable
dissatisfaction with the companies’ services. Secondly, we clean the
text data. The following diagram highlights the individual cleaning
steps. We proceed by combining the review title and the review sentences
of a given observation into a single string. As the main steps, we
replace emojis by specific textual representations, remove numbers and
some punctutation, but not all such as `?` or `!`, remove leading and
trailing whitespace, turn all letters into lower case, separate word
contractions, do spelling corrections, remove stop words, and ultimately
implement word stemming. This form of cleaning can be done on the entire
data at the same time, as each observation is treated individually and
the described procedure thus does not produce data leakage.

``` python
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
from collections import Counter

# Setting up spell checks and word tokenization
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
spell = Speller(lang = 'en')
```

The following code chunk provides the main cleaning function for the
reviews.

``` python
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
```

After the illustrated setup, the actual data treatment can be summarized
within the following few lines of code. The result is a cleaned dataset
`amazon_reviews_cleaned.feather` that can be used for the actual
analysis.

``` python
# Original data download from GitHub
data_full = pd.read_feather("https://github.com/dschulz13/dataset-amazon_trustpilot_reviews/raw/refs/heads/main/amazon_com_reviews_trustpilot_20261017_anon.feather")

# Run cleaning steps on the data
data_full = clean_text(data_full, spell, stop_words)

# Review texts are now fully cleaned and prepared for further analysis
data_full.to_feather(f'amazon_reviews_cleaned.feather')

# Delete data from memory
del data_full
gc.collect()
```

# Analysis

</div>
