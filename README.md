# Identifying Negative Customer Experiences Through Text Classification Using Python: A Case Study of Amazon.com Reviews on Trustpilot
Dominik Schulz
October 25, 2026

- [1 Executive summary](#1-executive-summary)
- [2 Data](#2-data)
- [3 Objective](#3-objective)
- [4 Data cleaning](#4-data-cleaning)
- [5 Analysis](#5-analysis)
  - [5.1 Train-test split](#51-train-test-split)
  - [5.2 Baseline prediction](#52-baseline-prediction)
  - [5.3 Logistic regression](#53-logistic-regression)

# 1 Executive summary

# 2 Data

The anonymized dataset
`amazon_com_reviews_trustpilot_20261017_anon.feather` in the GitHub
repository
[dschulz13/dataset-amazon_trustpilot_reviews](https://github.com/dschulz13/dataset-amazon_trustpilot_reviews.git)
is considered. It contains all **32,666 English Trustpilot reviews** for
**Amazon.com** published between August 27, 2007, and October 17, 2026.
The data was originally scraped directly from the [Trustpilot webpage of
Amazon.com](https://de.trustpilot.com/review/www.amazon.com?languages=en)
using **Python’s Selenium** extension.

For the sake of this project, only the available **data since January
01, 2025**, is used. This approximately gives a **cross-sectional
overview of recent customer feedback**. Filtered this way, **9,522
reviews** remain for the analysis.

<details>

<summary>

Show Python code
</summary>

With the following code chunk, the dataset is read into Python and
filtered, so that only reviews published on January 01, 2025, or later
are kept.

``` python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

# URL to the data on GitHub
url = 'https://github.com/dschulz13/dataset-amazon_trustpilot_reviews/raw/refs/heads/main/amazon_com_reviews_trustpilot_20261017_anon.feather'

# Read in the data
data_filt = pd.read_feather(url)
# Make Date column a datetime column
data_filt['Date'] = pd.to_datetime(data_filt['Date'])
# Keep observations since 2025-01-01 and reset index
data_filt = (
    data_filt[data_filt['Date'] >= '2025-01-01 00:00:00']
        .reset_index(drop = True)
)
```

The following code chunk then provides general information about the
filtered dataset and its variables.

``` python
# Show general information
print(data_filt.info())
print(data_filt[['Date', 'Review_Score']].describe())
print(data_filt['Review_Sentences'][6])
```

A barplot of the star ratings in the filtered dataset can be produced as
follows.

``` python
star_counts = data_filt['Review_Score'].value_counts().sort_index(ascending = True)

def star_formatter(x, pos):
    n_star = int(round(x)) + 1
    return '★' * n_star 

fig, ax = plt.subplots(1, 1, figsize = (6, 3))
star_counts.plot(kind = 'bar', ax = ax, color = 'lightblue',
    edgecolor = 'black', alpha = 1, width = 0.9)
ax.set_xlabel('Star rating');
ax.set_ylabel('Number of ratings');
ax.set_title('Star rating distribution');
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'));
ax.xaxis.set_major_formatter(ticker.FuncFormatter(star_formatter));
plt.xticks(rotation = 0, ha = 'center');
ax.grid(axis = 'y');
ax.set_axisbelow(True);
ax.set_ylim(0, 8300);
for i in range(0, 5):
    ax.text(x = i, y = star_counts.to_numpy()[i].item() + 300,
        s = f'{star_counts.to_numpy()[i].item():,}',
        ha = 'center', va = 'center')
```

</details>

    <class 'pandas.DataFrame'>
    RangeIndex: 9522 entries, 0 to 9521
    Data columns (total 7 columns):
     #   Column            Non-Null Count  Dtype         
    ---  ------            --------------  -----         
     0   User_ID           9522 non-null   int32         
     1   Date              9522 non-null   datetime64[us]
     2   User_Name         9522 non-null   str           
     3   User_Country      9522 non-null   str           
     4   Review_Score      9522 non-null   int32         
     5   Review_Title      9522 non-null   str           
     6   Review_Sentences  9373 non-null   object        
    dtypes: datetime64[us](1), int32(2), object(1), str(3)
    memory usage: 951.8+ KB
    None


                                 Date  Review_Score
    count                        9522   9522.000000
    mean   2025-12-18 20:42:38.790170      1.552195
    min           2025-01-01 00:00:00      1.000000
    25%           2025-07-19 00:00:00      1.000000
    50%           2026-01-03 00:00:00      1.000000
    75%           2026-06-01 00:00:00      1.000000
    max           2026-10-17 00:00:00      5.000000
    std                           NaN      1.210385


    ['You have to spend time checking whether it’s a genuine seller'
     'I now only shop at Amazon if i need a cheap, temporary item quickly.']

From the data output, we can see that the dataset has the seven
variables `User_ID`, `Date`, `User_Name`, `User_Country`,
`Review_Score`, `Review_Title`, and `Review_Sentences`. Each row in the
dataset is a unique review entry. `User_ID` is an `int` variable that
uniquely identifies each author (and can be used to see, which person
wrote multiple reviews). `Date` is a `datetime` variable, while
`Review_Score` is also an `int` column. `User_Name`, `User_Country`, and
`Review_Title` are `str` columns containing the author’s user name, the
author’s registered country, and the the document’s review title,
respectively. `Review_Sentences` is an `object` column, where in each
row it contains a list, whose entries are the different review sentences
as strings.

Only `Review_Sentences` contains missing values sometimes, as review
author’s must specify a review title, but they are not obligated to
write a review text. Therefore, in 149 instances no review text was
written.

The star ratings are apparently strongly skewed / imbalanced. The mean
rating is approx. 1.55, the median and even the 75th percentile of the
observed rating distribution are simply 1.

<div id="fig-macro-plot-run">

<img
src="README_files/figure-commonmark/fig-macro-plot-run-output-1.png"
data-fig-align="center" />

Figure 2.1: The distribution of the star ratings in the filtered
dataset.

</div>

<a href="#fig-macro-plot-run" class="quarto-xref">Figure 2.1</a>
supports the previous statement on the strong imbalance in the data:
7,480 reviews are one-star reviews, where customers were as dissatisfied
as possible. The remaining 2,042 reviews are spread across the two- to
five-star reviews. Given that one-, two-, and five-star reviews appear
most frequently (with a clear dominance of the one-star reviews), we
conclude that strong emotional reactions to Amazon’s services, either
strongly positive or strongly negative, lead to a correspondingly higher
chance of a review entry on Trustpilot. Since customers with
intermediate-level (and also to some degree positive) satisfaction do
not choose to write a review as often, the observed distribution can
under no circumstances be understood as the real distribution of
customer satisfaction.

# 3 Objective

A crucial task for any company is dealing with customer feedback.
Commonly, companies offer direct channels to contact customer service
and receive support, if they are dissatisfied with the companies service
or product. Unfortunately, companies receive vast amounts of messages
from customers on a daily basis, some of which do not contain urgent
issues (or do not state any problems at all) the companies need to
address immediately. However, each message requires manual inspection
first to determine the implied urgency; a process which **costs a
substantial amount of human labor**.

An algorithm inspecting and manually classifying incoming messages would
filter out messages that do not state any severely negative customer
experiences and thus **immensely recude the required human input** for
this task.

Of course, training classification algorithms needs **pre-labelled
training data**, which is not the case for incoming messages through
common customer service channels. In case of Amazon.com, we train
classification models on the **reviews from Trustpilot**, because they
already include ratings, thus allowing for **simple labelling** of
positive and negative customer experiences. Nonetheless,
**transferability** to other channels needs to **inspected elsewhere**.

In the provided context, a natural basline approach is to simply predict
every review as negative, thus enforcing the manual inspection of every
message afterwards. Such an approach has both advantages and
disadvantages: on one hand, no crucial negative review is missed, while
on the other hand this approach raises many false alarms by labelling
every non-negative message as negative. Treating four- and five-star
reviews as `Good` and all other reviews as `Bad`, the share of bad
reviews is approx. 89%. This native approach consequently leads to an
**out-of-the-box overall classification accuracy of 89%** and a **recall
of 100%** on the **`Bad` class** (for stratified test samples), but has
**poor precision and recall** on the **`Good` class**.

The main objective of this project is hence the development of a
classification algorithm to classify the Amazon.com reviews into good
and bad reviews. Overall classification **accuracy** should be
**improved**, while **ensuring a high precision and recall** on the
**`Bad` class**, so that (a) the amount of **false alarms** is kept as
**low** as possible, and (b) actual **bad reviews** are **identified
reliably**.

# 4 Data cleaning

<a href="#fig-cleaning-steps" class="quarto-xref">Figure 4.1</a>
demonstrates the cleaning pipeline for the pre-filtered review data.

<div id="fig-cleaning-steps">

![](Data%20cleaning%20steps.png)

Figure 4.1: The data cleaning steps.

</div>

The goal of the cleaning step is two-fold: firstly, we wish to
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

<details>

<summary>

Show Python code
</summary>

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
# Run cleaning steps on the data
data_filt = clean_text(data_filt, spell, stop_words)

# Review texts are now fully cleaned and prepared for further analysis
data_filt.to_feather(f'amazon_reviews_cleaned.feather')

# Delete data from memory
del data_filt
gc.collect()
```

</details>

# 5 Analysis

Throughout the analysis, we derive numerical features from 1-grams of
the cleaned review texts, i.e. variables taken from individual words.
This keeps the feature engineering simple enough, avoids high
dimensionality in the feature space and reduces computation times. More
in-depth approaches not considered here may also include 2-grams or
3-grams, and thus variables derived from chains of multiple words.

## 5.1 Train-test split

The data is split into a training set and a test set before any further
preprocessing of the cleaned texts. To ensure that the test set contains
at least 1,000 observations, we consider a **test share** of **11%**,
resulting in a **test sample size** of **1,048**. For simplicity, we
only consider such a fixed split; more elaborate but also time- and
memory-consuming approaches use nested cross-validation ideas with
multiple training-test splits. For the purpose of this project, the size
of the random test sample is, however, large enough to allow for a
generalization of the findings. The split here is stratified, ensuring
that both the training and the test sample reflect the 11% and 89%
shares of `Good` and `Bad` reviews in the total dataset.

<details>

<summary>

Show Python code
</summary>

``` python
from sklearn.model_selection import train_test_split
import pandas as pd

data_all = pd.read_feather(f'amazon_reviews_cleaned.feather')

X_train, X_test, y_train, y_test = train_test_split(
    data_all.Review_Sentences.tolist(),
    data_all.Rating_Class.tolist(),
    test_size = 0.11,
    random_state = 1,
    stratify = data_all.Rating_Class.tolist()
)
```

</details>

## 5.2 Baseline prediction

As indicated earlier, the baseline approach of simply always predicting
the `Bad` class results in a high overall classification accuracy of 89%
(all `Bad` reviews are classified correctly, whereas all `Good` reviews
are classified incorrectly). Hence, for the `Bad` class, we achieve with
minimum effort high scores (precision: 0.89; recall: 1.00), while the
method produces the worst possible scores for the `Good` class
(precision: 0.00; recall: 0.00). Note that for the precision to be 0.00,
we defined a division by zero to be equal to zero in the reporting
function.

<details>

<summary>

Show Python code
</summary>

``` python
from sklearn.metrics import classification_report
import numpy as np

pred_baseline = np.array(['Bad'] * len(y_test))
```

``` python
print(classification_report(y_test, pred_baseline,
        zero_division = 0.0))
```

</details>

                  precision    recall  f1-score   support

             Bad       0.89      1.00      0.94       933
            Good       0.00      0.00      0.00       115

        accuracy                           0.89      1048
       macro avg       0.45      0.50      0.47      1048
    weighted avg       0.79      0.89      0.84      1048

## 5.3 Logistic regression

``` python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TunedThresholdClassifierCV, StratifiedKFold
from imblearn.pipeline import Pipeline

### General settings for CountVectorizer
tokenizer = str.split    # create words by splitting cleaned texts by whitespace
preprocessor = None      # no additional preprocessing
token_pattern = None     # no further token identification
lowercase = False        # no further lower case transformation need
min_df = 10              # consider only words appearing in at least ten documents
max_df = 1000000000      # (practically) no limit for number of documents with a specific word

### Base settings for logistic regression
solver = 'saga'
C_val = np.inf
random_state = 1
max_iter = 5000
class_weight = 'balanced'

model1 = Pipeline([
    # Create word frequency variables for each review from the training set
    ('dtm', CountVectorizer(
        tokenizer = tokenizer,    
        preprocessor = preprocessor,      
        token_pattern = token_pattern,     
        lowercase = lowercase,        
        min_df = min_df,               
        max_df = max_df        
    )),
    # Model definition
    ('model', LogisticRegression(
        solver = solver,     # SAGA solver (good for large datasets with many variables)
        C = C_val,          # unlimited inverse penalty implies standard logistic regression
        random_state = random_state,
        max_iter = max_iter
    ))
])

model1.fit(X_train, y_train)
pred_model1 = model1.predict(X_test)
report_model1 = classification_report(y_test, pred_model1)
print(report_model1)
```

                  precision    recall  f1-score   support

             Bad       0.97      0.97      0.97       933
            Good       0.75      0.77      0.76       115

        accuracy                           0.95      1048
       macro avg       0.86      0.87      0.87      1048
    weighted avg       0.95      0.95      0.95      1048
