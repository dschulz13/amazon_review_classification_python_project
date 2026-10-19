# Standard libraries
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Special libraries
import contractions
from autocorrect import Speller
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

spell = Speller(lang = 'en')

# Data download from GitHub
data_full = pd.read_feather("https://github.com/dschulz13/dataset-amazon_trustpilot_reviews/raw/refs/heads/main/amazon_com_reviews_trustpilot_20261017_anon.feather")

# Some format Date column as datetime and filter to use only data
# since the beginning of the year 2025
data_full['Date'] = pd.to_datetime(data_full['Date'], format = '%Y-%m-%d')
data_full = data_full[data_full['Date'] >= '2025-01-01 00:00:00']
data_full = data_full.reset_index(drop = True)

# Combine sentences into one text
reviews = data_full['Review_Sentences']
l = len(reviews)
review_adj = [''] * l
for i in range(0, l):
    try:
        text0 = ''
        len_list = len(reviews[i])
        for j in range(0, len_list):
            text0 = text0 + ' ' + reviews[i][j]
        review_adj[i] = text0
    except TypeError:
        review_adj[i] = ''

data_full['Review_Sentences'] = review_adj

# Replace everything except letters, numbers, whitespace, and ' with a whitespace
data_full['Review_Sentences'] = data_full['Review_Sentences'].str.replace(r'[^\w\s\']', ' ', regex = True)
# Replace all numbers and multiple whitespaces in a row by a single whitespace
data_full['Review_Sentences'] = data_full['Review_Sentences'].str.replace(r'[\s0-9]+', ' ', regex = True)
# Remove leading and trailing whitespace
data_full['Review_Sentences'] = data_full['Review_Sentences'].str.strip()
# Make everything lower case
data_full['Review_Sentences'] = data_full['Review_Sentences'].str.lower()

# Create a subobject of the review texts
reviews = data_full['Review_Sentences']
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

data_full['Review_Sentences'] = review_adj
data_full['Review_Sentences'] = data_full['Review_Sentences'].str.strip()
data_full['Rating_Class'] = 'Bad'
data_full.loc[data_full['Review_Score'] >= 4, 'Rating_Class'] = 'Good'

data_full = data_full.drop(['User_ID', 'Date', 'User_Name', 'User_Country', 'Review_Score', 'Review_Title'], axis = 1)

# Review texts are now fully cleaned and prepared for further analysis
data_full.to_feather(f'amazon_reviews_cleaned.feather')

# Barplot of distribution of 2025 2026 review scores
fig, ax = plt.subplots(1, 1, figsize = (8, 4))
score_counts = data_full.Review_Score.value_counts().sort_index()
score_counts.plot(kind = 'bar', color = 'skyblue', ax = ax, edgecolor = 'black', alpha = 0.8)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'));
ax.grid(axis = 'y');
ax.set_axisbelow(True);
ax.set_xlabel('Review score');
ax.set_ylabel('Number of reviews');
plt.xticks(rotation = 0)

for i in score_counts.index:
    ax.text(int(i) - 1, score_counts.loc[i].item() + 250, s = f"{score_counts.loc[i].item():,}",
        ha = 'center', va = 'center',
        fontsize = 10);

ax.set_title('Distribution of 2025 and 2026 review scores');
ax.set_ylim(0, 8200)
plt.show()



