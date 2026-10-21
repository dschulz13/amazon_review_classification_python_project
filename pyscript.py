# Standard libraries
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

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

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

spell = Speller(lang = 'en')

# Data download from GitHub
data_full = pd.read_feather("https://github.com/dschulz13/dataset-amazon_trustpilot_reviews/raw/refs/heads/main/amazon_com_reviews_trustpilot_20261017_anon.feather")

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

data_full = clean_text(data_full, spell, stop_words)

# Review texts are now fully cleaned and prepared for further analysis
data_full.to_feather(f'amazon_reviews_cleaned.feather')

del data_full
gc.collect()


######################################################

# Get words that appear at least five times

data_cleaned = pd.read_feather('amazon_reviews_cleaned.feather')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



dtm = vectorizer.fit_transform(data_cleaned.Review_Sentences.to_numpy())

X_train, X_test, y_train, y_test = train_test_split(
    data_cleaned.Review_Sentences.tolist(),
    data_cleaned.Rating_Class.tolist(),
    test_size = 0.1,
    random_state = 1,
    stratify = data_cleaned.Rating_Class.tolist()
)

### Model 1: Standard logistic regression (as a baseline)
base_model = Pipeline([
    ('dtm', CountVectorizer(
        tokenizer = str.split,
        preprocessor = None,
        token_pattern = None,
        lowercase = False,
        min_df = 5,
        max_df = 100000000
    )),
    ('model', LogisticRegression(
        solver = "saga",
        C = np.inf,
        random_state = 1,
        max_iter = 5000  
    ))
])

base_model.fit(X_train, y_train)
y_pred_base = base_model.predict(X_test)
base_report = classification_report(y_test, y_pred_base)
print(base_report)

from sklearn.ensemble import RandomForestClassifier

# Model 2: A basic random forecast classifier

model2 = Pipeline([
    ('dtm', CountVectorizer(
        tokenizer = str.split,
        preprocessor = None,
        token_pattern = None,
        lowercase = False,
        min_df = 5,
        max_df = 100000000
    )),
    ('model', RandomForestClassifier(
        criterion = 'entropy',
        random_state = 1
    ))
])

model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)
mod2_report = classification_report(y_test, pred2)
print(mod2_report)

forest_importances = pd.Series(model2.feature_importances_, index = frequent_words).sort_values(
    ascending = False
)
forest_importances.head(50).plot(kind = 'bar')



####### Need to adjust the following

# L1-regularized logistic regression
model = LogisticRegression(
    solver = "saga",
    l1_ratio = 1.0,
    C = 1.0,
    max_iter = 5000,
    random_state = 1
)

# Fit the model
model.fit(X_train, y_train)

# Predict categories in the test set
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

model2 = Pipeline([
    ('standard', TfidfTransformer(use_idf = True, norm = "l2", sublinear_tf = True)),
    ('regression', LogisticRegression(solver = "saga", l1_ratio = 1.0, C = 1.0, max_iter = 5000, random_state = 1))
])

model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)

print(classification_report(y_test, y_pred2))

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



