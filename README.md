# Identifying Negative Customer Experiences Through Text Classification Using Python: A Case Study of Amazon.com Reviews on Trustpilot
Dominik Schulz
October 27, 2026

- [1 Executive summary](#1-executive-summary)
- [2 Data](#2-data)
- [3 Objective](#3-objective)
- [4 Data cleaning](#4-data-cleaning)
- [5 Analysis](#5-analysis)
  - [5.1 Train-test split](#51-train-test-split)
  - [5.2 Word cloud](#52-word-cloud)
  - [5.3 Baseline prediction](#53-baseline-prediction)
  - [5.4 Logistic regression](#54-logistic-regression)

# 1 Executive summary

# 2 Data

The anonymized dataset
`amazon_com_reviews_trustpilot_20261026_anon.feather` in the GitHub
repository
[dschulz13/dataset-amazon_trustpilot_reviews](https://github.com/dschulz13/dataset-amazon_trustpilot_reviews.git)
is considered. It contains all **32,632 English Trustpilot reviews** for
**Amazon.com** published between August 27, 2007, and October 26, 2026.
The data was originally scraped directly from the [Trustpilot webpage of
Amazon.com](https://de.trustpilot.com/review/www.amazon.com?languages=en)
using **Python’s Selenium** extension.

For the sake of this project, only the available **reviews published
since January 01, 2025** are used. This approximately gives a
**cross-sectional overview of recent customer feedback**. Filtered this
way, **9,688 reviews** remain for the analysis.

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
url = 'https://github.com/dschulz13/dataset-amazon_trustpilot_reviews/raw/refs/heads/main/amazon_com_reviews_trustpilot_20261026_anon.feather'

# Read in the data
data_filt = pd.read_feather(url)
# Make Date column a datetime column
data_filt['Review_Date'] = pd.to_datetime(data_filt['Review_Date'], utc = True, format = '%Y-%m-%d %H:%M:%S')
# Keep observations since 2025-01-01 and reset index
data_filt = (
    data_filt[data_filt['Review_Date'] >= '2025-01-01 00:00:00']
        .reset_index(drop = True)
)
```

The following code chunk then provides general information about the
filtered dataset and its variables.

``` python
# Show general information
print(data_filt.info())
print(data_filt[['Review_Date', 'Review_Score']].describe())
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
    RangeIndex: 9688 entries, 0 to 9687
    Data columns (total 8 columns):
     #   Column            Non-Null Count  Dtype              
    ---  ------            --------------  -----              
     0   User_ID           9688 non-null   int32              
     1   Review_Date       9688 non-null   datetime64[us, UTC]
     2   Experience_Date   9688 non-null   str                
     3   User_Name         9688 non-null   str                
     4   User_Country      9688 non-null   str                
     5   Review_Score      9688 non-null   int32              
     6   Review_Title      9688 non-null   str                
     7   Review_Sentences  9551 non-null   object             
    dtypes: datetime64[us, UTC](1), int32(2), object(1), str(4)
    memory usage: 1.1+ MB
    None


           Review_Score
    count   9688.000000
    mean       1.529727
    std        1.190940
    min        1.000000
    25%        1.000000
    50%        1.000000
    75%        1.000000
    max        5.000000


    ['I have to say that they the make mistakes but have always made up for them with me'
     "I've gotten things for free because they were late"
     'The offer 0% interest for 6 months and everything comes so fast'
     'Whenever I order from another company, I realize just how much more efficient and accommodating Amazon is'
     "And I hope this doesn't infuriate anyone on here who's had real problems with them, I'm just being honest"
     "I almost always come to Trust Pilot to complain but I came here today because I was curious of what Amazon's average review score was and must say that I'm shocked because I can't imagine better customer service."]

From the data output, we can see that the dataset has the eight
variables `User_ID`, `Review_Date`, `Experience_Date`, `User_Name`,
`User_Country`, `Review_Score`, `Review_Title`, and `Review_Sentences`.
Each row in the dataset is a unique review entry. `User_ID` is an `int`
variable that uniquely identifies each author (and can be used to see,
which person wrote multiple reviews). `Review_Date` is a `datetime`
variable, while `Experience_Date` is a `str` variable. `Review_Score` is
also an `int` column. `User_Name`, `User_Country`, and `Review_Title`
are `str` columns containing the author’s user name, the author’s
registered country, and the the document’s review title, respectively.
`Review_Sentences` is an `object` column, where in each row it contains
a list, whose entries are the different review sentences as strings.

Only `Review_Sentences` contains missing values sometimes, as review
author’s must specify a review title, but they are not obligated to
write a review text. Therefore, in 137 instances no review text was
written.

The star ratings are apparently strongly skewed / imbalanced. The mean
rating is approx. 1.53, the median and even the 75th percentile of the
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
7,688 reviews are one-star reviews, where customers were as dissatisfied
as possible. The remaining 2,000 reviews are spread across the two- to
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

<img src="Data%20cleaning%20steps.png"
style="width:60.0%;height:60.0%" />

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

    dataset = dataset.drop(['User_ID', 'Review_Date', 'Experience_Date', 'User_Name', 'User_Country', 'Review_Score', 'Review_Title'], axis = 1)

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
resulting in a **test sample size** of **1,066**. For simplicity, we
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

## 5.2 Word cloud

<details>

<summary>

Show Python code
</summary>

``` python
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

all_text = ' '.join(X_train)

x, y = np.ogrid[:1000, :1000]
mask = (x - 500) ** 2 + (y - 500) ** 2 > 400 ** 2
mask = 255 * mask.astype(int)

wcloud = (WordCloud(
    background_color = "white",
    collocations = False,
    random_state = 2,
    colormap = "coolwarm",
    mask = mask)
        .generate(all_text)
)
plt.imshow(wcloud, interpolation = "bilinear")
plt.axis("off")
plt.show()
```

</details>

<div id="fig-wordcloud">

<img src="README_files/figure-commonmark/fig-wordcloud-output-1.png"
data-fig-align="center" />

Figure 5.1: A word cloud of the 200 most frequent training data word
stems.

</div>

<a href="#fig-wordcloud" class="quarto-xref">Figure 5.1</a> represents a
word cloud of the 200 most frequent word stems in the training data. It
provides an overview of important terms to look out for in the data.
Obviously, `amazon` appears most often, followed by stems such as
`custom` (customer/s), `deliveri` (delivery/deliveries) and `order`.
Frequent terms also include `refund`, `return`, `time`, `day`, `servic`,
hinting the importance of short-term deliveries and of return / refund
policies.

## 5.3 Baseline prediction

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

             Bad       0.89      1.00      0.94       954
            Good       0.00      0.00      0.00       112

        accuracy                           0.89      1066
       macro avg       0.45      0.50      0.47      1066
    weighted avg       0.80      0.89      0.85      1066

## 5.4 Logistic regression

### 5.4.1 Estimation framework

<div id="fig-cv-steps">

![](CV%20framework.png)

Figure 5.2: The validation and testing framework.

</div>

### 5.4.2 No standardization of word frequencies

As a first, more elaborate approach, a fine-tuned regularized logistic
regression is implemented. In the outer cross-validation, we tune the
penalty term (`C`), the class weights (`class_weight`), and the form of
regularization (`l1_ratio`), while the inner cross-validation handles
the probability threshold selection. Ultimately, `C = 0.1`,
`class_weight = 'balanced'`, and `l1_ratio = 0.0` are selected,
corresponding to a strongly regularized ridge logistic regression, where
incorrect classification of the `Good` class costs 8.5 times more than
an incorrect classification of the `Bad` class. With these
hyperparameters, the model is then refitted to the entire training
sample, while re-selecting the probability threshold with
cross-validation to be approx. 0.4490.

<details>

<summary>

Show Python code
</summary>

``` python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TunedThresholdClassifierCV, StratifiedKFold
from sklearn.metrics import make_scorer, balanced_accuracy_score
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
random_state = 1
max_iter = 5000

bas_adj = make_scorer(balanced_accuracy_score) 

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
        solver = solver,
        random_state = random_state,
        max_iter = max_iter
    ))
])

model1_tuned_inner = TunedThresholdClassifierCV(
    estimator = model1,
    scoring = bas_adj,
    cv = StratifiedKFold(3, random_state = 1, shuffle = True),
    thresholds = 50,
    refit = True,
    n_jobs = 1
)

model1_params = {
    'estimator__model__C': [0.01, 0.1, 1, 10, 100],
    'estimator__model__class_weight': [None, 'balanced', {'Bad': 2.0, 'Good': 1.0}, {'Bad': 1.0, 'Good': 2.0}, {'Bad': 1.0, 'Good': 4.0}],
    'estimator__model__l1_ratio': [0.0, 1.0]
}

model1_tuned = GridSearchCV(
    model1_tuned_inner,
    model1_params,
    scoring = bas_adj,
    cv = StratifiedKFold(5, random_state = 1, shuffle = True),
    n_jobs = -1
)

model1_tuned.fit(X_train, y_train)
pred_model1 = model1_tuned.predict(X_test)
report_model1 = classification_report(y_test, pred_model1)
print(report_model1)
```

</details>

The test sample evaluation is summarized in the following Python report.
In comparison to the naive approach, the overall accuracy could be
increased from 0.89 to 0.92. The average (unweighted) recall increased
from 0.5 to 0.89. Most indicators increase, with the exception of the
`Bad` class recall, which decreased from 1.00 to 0.93: the improvement
in the other measures comes at the cost of not being able to identify
all the `Bad` labels.

                  precision    recall  f1-score   support

             Bad       0.98      0.93      0.95       954
            Good       0.57      0.85      0.68       112

        accuracy                           0.92      1066
       macro avg       0.78      0.89      0.82      1066
    weighted avg       0.94      0.92      0.92      1066

### 5.4.3 Standardization of word frequencies

<details>

<summary>

Show Python code
</summary>

``` python
from sklearn.feature_extraction.text import TfidfTransformer

model2 = Pipeline([
    ('dtm', CountVectorizer(
        tokenizer = tokenizer,    
        preprocessor = preprocessor,      
        token_pattern = token_pattern,     
        lowercase = lowercase,        
        min_df = min_df,               
        max_df = max_df        
    )),
    ('standard', TfidfTransformer(
        use_idf = True,
        norm = "l2", 
        sublinear_tf = True
    )),
    ('model', LogisticRegression(
        solver = solver,
        random_state = random_state,
        max_iter = max_iter
    ))
])

model2_tuned_inner = TunedThresholdClassifierCV(
    estimator = model2,
    scoring = bas_adj,
    cv = StratifiedKFold(3, random_state = 1, shuffle = True),
    thresholds = 50,
    refit = True,
    n_jobs = 1
)

model2_params = {
    'estimator__model__C': [0.01, 0.1, 1, 10, 100],
    'estimator__model__class_weight': [None, 'balanced', {'Bad': 2.0, 'Good': 1.0}, {'Bad': 1.0, 'Good': 2.0}, {'Bad': 1.0, 'Good': 4.0}],
    'estimator__model__l1_ratio': [0.0, 1.0]
}

model2_tuned = GridSearchCV(
    model2_tuned_inner,
    model2_params,
    scoring = bas_adj,
    cv = StratifiedKFold(5, random_state = 1, shuffle = True),
    n_jobs = -1
)

model2_tuned.fit(X_train, y_train)
pred_model2 = model2_tuned.predict(X_test)
report_model2 = classification_report(y_test, pred_model2)
print(report_model2)
```

</details>

                  precision    recall  f1-score   support

             Bad       0.98      0.94      0.96       954
            Good       0.64      0.84      0.72       112

        accuracy                           0.93      1066
       macro avg       0.81      0.89      0.84      1066
    weighted avg       0.94      0.93      0.94      1066
