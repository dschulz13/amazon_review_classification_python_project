### Standard libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import make_scorer, f1_score, average_precision_score
from sklearn.model_selection import GridSearchCV, TunedThresholdClassifierCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE



### Load in cleaned data
data_cleaned = pd.read_feather('amazon_reviews_cleaned.feather')

### Split data into training and test samples
X_train, X_test, y_train, y_test = train_test_split(
    data_cleaned.Review_Sentences.tolist(),
    data_cleaned.Rating_Class.tolist(),
    test_size = 0.1,
    random_state = 1,
    stratify = data_cleaned.Rating_Class.tolist()
)

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

### Base model: Always predicting the majority class 'Bad'
pred_base = np.array(['Bad'] * len(y_test))
base_report = classification_report(y_test, pred_base, zero_division = 0.0)
print(base_report)
#               precision    recall  f1-score   support
#
#          Bad       0.89      1.00      0.94       848
#         Good       0.00      0.00      0.00       105
#
#     accuracy                           0.89       953
#    macro avg       0.44      0.50      0.47       953
# weighted avg       0.79      0.89      0.84       953

### Model 1: Standard logistic regression (as a baseline)
### Features:
###     - no standardization of word counts
###     - no adjustment for class imbalance

base_model = Pipeline([
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

base_model.fit(X_train, y_train)
y_pred_base = base_model.predict(X_test)
base_report = classification_report(y_test, y_pred_base)
print(base_report)

### Model 2: Standard logistic regression (as a baseline)
### Features:
###     - no standardization of word counts
###     - adjustment for class imbalance through class weights in the model

base_model2 = Pipeline([
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
        max_iter = max_iter,
        class_weight = class_weight
    ))
])

base_model2.fit(X_train, y_train)
y_pred_base2 = base_model2.predict(X_test)
base_report2 = classification_report(y_test, y_pred_base2)
print(base_report2)

###

### Model 3: Standard logistic regression (as a baseline)
### Features:
###     - no standardization of word counts
###     - adjustment for class imbalance through oversampling

base_model3 = Pipeline([
    # Create word frequency variables for each review from the training set
    ('dtm', CountVectorizer(
        tokenizer = tokenizer,    
        preprocessor = preprocessor,      
        token_pattern = token_pattern,     
        lowercase = lowercase,        
        min_df = min_df,               
        max_df = max_df        
    )),
    ("oversampler", RandomOverSampler(
        random_state = 1
    )),
    # Model definition
    ('model', LogisticRegression(
        solver = solver,     # SAGA solver (good for large datasets with many variables)
        C = C_val,          # unlimited inverse penalty implies standard logistic regression
        random_state = random_state,
        max_iter = max_iter
    ))
])

f1_neg = make_scorer(f1_score, pos_label = 'Good')
avprec_neg = make_scorer(average_precision_score, pos_label = 'Good')

param_grid = {
    "oversampler__sampling_strategy": np.arange(0.4, 1.1, 0.1).tolist()
}

cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1)

search = GridSearchCV(base_model3, param_grid, cv = cv, scoring = "average_precision", n_jobs = -1)
search.fit(X_train, y_train)

threshold_tuned_model = TunedThresholdClassifierCV(     # select best probability threshold
    estimator = search.best_estimator_,
    scoring = f1_neg,
    cv = 5,                 # Inner loop folds for threshold evaluation
    thresholds = 100,       # Number of threshold variations to test (0.0 to 1.0),
    n_jobs = -1
)
threshold_tuned_model.fit(X_train, y_train)

logit_ovs_model = threshold_tuned_model
y_pred_base3 = logit_ovs_model.predict(X_test)
base_report3 = classification_report(y_test, y_pred_base3)
print(base_report3)

###

### Model 4: Standard logistic regression (as a baseline)
### Features:
###     - no standardization of word counts
###     - adjustment for class imbalance through SMOTE oversampling

base_model4 = Pipeline([
    # Create word frequency variables for each review from the training set
    ('dtm', CountVectorizer(
        tokenizer = tokenizer,    
        preprocessor = preprocessor,      
        token_pattern = token_pattern,     
        lowercase = lowercase,        
        min_df = min_df,               
        max_df = max_df        
    )),
    ("oversampler", SMOTE(
        random_state = 1
    )),
    # Model definition
    ('model', LogisticRegression(
        solver = solver,    # SAGA solver (good for large datasets with many variables)
        C = C_val,          # unlimited inverse penalty implies standard logistic regression
        random_state = random_state,
        max_iter = max_iter
    ))
])

param_grid = {
    "oversampler__sampling_strategy": np.arange(0.4, 1.1, 0.1).tolist(),
    "oversampler__k_neighbors": [3, 5]
}

recall_neg = make_scorer(recall_score, pos_label = 'Good')

cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1)

search = GridSearchCV(base_model4, param_grid, cv = cv, scoring = recall_neg, n_jobs = -1)
search.fit(X_train, y_train)

logit_smote_model = search.best_estimator_
y_pred_base4 = logit_smote_model.predict(X_test)
base_report4 = classification_report(y_test, y_pred_base4)
print(base_report4)


###

### Lasso logistic regression
### Features:
###     - no standardization of features
###     - no adjustment of class imbalance




###

from sklearn.ensemble import RandomForestClassifier

# Model 2: A random forecast classifier

rf_model = 

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



