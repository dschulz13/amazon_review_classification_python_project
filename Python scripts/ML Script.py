### Standard libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, make_scorer, recall_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TunedThresholdClassifierCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
import gc
import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer
from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgb

# Custom scorers

## Computes the inverse AUC for recall >= 0.95
## Identifies estimators with high 'Good' class recall
## when 'Bad' class recall >= 0.95
def high_sensitivity_auc(
    y_true,
    y_proba,
    min_recall = 0.95,
    n_points = 1000,
    pos_label = 'Bad'
):
    fpr, tpr, thresholds = roc_curve(
        y_true, y_proba, pos_label = pos_label
    )

    unique_tpr, indices = np.unique(
        tpr, return_index = True
    )
    unique_fpr = fpr[indices]

    tpr_grid = np.linspace(
        min_recall, 1.0, n_points
    )
    fpr_grid = np.interp(
        tpr_grid,
        unique_tpr,
        unique_fpr
    )

    score = 1 - np.trapezoid(
        fpr_grid, tpr_grid
    ) / (1 - min_recall)

    return score

def hs_auc_scorer(estimator, X, y):

    classes = list(estimator.classes_)
    bad_idx = classes.index('Bad')

    proba = estimator.predict_proba(X)[:, bad_idx]

    return high_sensitivity_auc(
        y,
        proba,
        min_recall = 0.95,
        pos_label = 'Bad'
    )

def good_recall(y_true, y_pred):
    bad_rec = recall_score(y_true, y_pred, pos_label = 'Bad')
    if bad_rec < 0.95:
        return -1.0
    return recall_score(y_true, y_pred, pos_label = 'Good')

good_recall_scorer = make_scorer(good_recall)

### Load in cleaned data
data_all = pd.read_feather('amazon_reviews_cleaned.feather')

### Split data into training and test samples
X_train, X_test, y_train, y_test = train_test_split(
    data_all.Review_Sentences.tolist(),
    data_all.Rating_Class.tolist(),
    test_size = 0.11,
    random_state = 1,
    stratify = data_all.Rating_Class.tolist()
)

del data_all
gc.collect()

### General settings for CountVectorizer
tokenizer = str.split    # create words by splitting cleaned texts by whitespace
preprocessor = None      # no additional preprocessing
token_pattern = None     # no further token identification
lowercase = False        # no further lower case transformation need
min_df = 5               # consider only words appearing in at least five documents
max_df = 1.0             # no limit for number of documents with a specific word

### Base settings for logistic regression
solver = 'saga'
random_state = 1
max_iter = 5000

### Model 1: Always predicting the majority class 'Bad'
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


### Model 2: Standard logistic regression
logit_model = Pipeline([
    # Create word frequency variables for each review from the training set
    ('dtm', CountVectorizer(
        tokenizer = tokenizer,    
        preprocessor = preprocessor,      
        token_pattern = token_pattern,     
        lowercase = lowercase,        
        min_df = min_df,               
        max_df = max_df,
        dtype = np.int16
    )),
    # Word frequency tranformer
    ('tfidf', TfidfTransformer()),
    # Model definition
    ('model', LogisticRegression(
        solver = solver,
        random_state = random_state,
        max_iter = max_iter
    ))
])

logit_params = {
    'model__C': [0.01, 0.1, 1, 10],
    'model__class_weight': [None, 'balanced', {'Bad': 1.0, 'Good': 2.0}, {'Bad': 1.0, 'Good': 4.0}],
    'model__l1_ratio': [0.0, 0.5, 1.0],
    'tfidf': [
        TfidfTransformer(
            norm = 'l2',
            use_idf = True,
            smooth_idf = True,
            sublinear_tf = True
        ),
        TfidfTransformer(
            norm = 'l2',
            use_idf = True,
            smooth_idf = True,
            sublinear_tf = False
        ),        
        'passthrough'
    ],
    'dtm__ngram_range': [(1, 1), (1, 2)]
}

logit_step1 = RandomizedSearchCV(
    logit_model,
    logit_params,
    scoring = hs_auc_scorer,
    cv = StratifiedKFold(3, random_state = 1, shuffle = True),
    n_jobs = -1,
    random_state = 1,
    n_iter = 30
)

logit_step1.fit(X_train, y_train)

pickle.dump(logit_step1, open('Predictor models/logit_base.bin', 'wb'))

logit_tuned = TunedThresholdClassifierCV(
    estimator = logit_step1.best_estimator_,
    scoring = good_recall_scorer,
    cv = StratifiedKFold(3, random_state = 1, shuffle = True),
    thresholds = 100,
    refit = True,
    n_jobs = -1,
    store_cv_results = True
)
logit_tuned.fit(X_train, y_train)

pickle.dump(logit_tuned, open('Predictor models/logit_tuned.bin', 'wb'))

logit_pred = logit_tuned.predict(X_test)

np.save('Test predictions/logit_pred.npy', logit_pred)

print(classification_report(y_test, logit_pred))

# Model 3: Naive Bayes

def fun_toarray(x):
    return x.toarray()

nbayes_model = Pipeline([
    ('dtm', CountVectorizer(
        tokenizer = tokenizer,    
        preprocessor = preprocessor,      
        token_pattern = token_pattern,     
        lowercase = lowercase,        
        min_df = min_df,               
        max_df = max_df,
        dtype = np.int16   
    )),
    ('tfidf', TfidfTransformer()),
    ('oversampler', RandomOverSampler(random_state = 1)),
    ('to_dense', FunctionTransformer(fun_toarray, accept_sparse = True)),
    ('model', GaussianNB())
])

nbayes_params = {
    'oversampler__sampling_strategy': [0.12, 0.2, 0.4, 0.6, 0.8, 1.0],
    'dtm__ngram_range': [(1, 1), (1, 2)],
    'tfidf': [
        TfidfTransformer(
            norm = 'l2',
            use_idf = True,
            smooth_idf = True,
            sublinear_tf = True
        ),
        TfidfTransformer(
            norm = 'l2',
            use_idf = True,
            smooth_idf = True,
            sublinear_tf = False
        ),        
        'passthrough'
    ]
}

nbayes_step1 = GridSearchCV(
    nbayes_model,
    nbayes_params,
    scoring = hs_auc_scorer,
    cv = StratifiedKFold(3, random_state = 1, shuffle = True),
    n_jobs = -1
)
nbayes_step1.fit(X_train, y_train)

pickle.dump(nbayes_step1, open('Predictor models/nbayes_base.bin', 'wb'))

nbayes_tuned = TunedThresholdClassifierCV(
    estimator = nbayes_step1.best_estimator_,
    scoring = good_recall_scorer,
    cv = StratifiedKFold(3, random_state = 1, shuffle = True),
    thresholds = 100,
    refit = True,
    n_jobs = -1
)
nbayes_tuned.fit(X_train, y_train)

pickle.dump(nbayes_tuned, open('Predictor models/nbayes_tuned.bin', 'wb'))
nbayes_pred = nbayes_tuned.predict(X_test)
np.save('Test predictions/nbayes_pred.npy', nbayes_pred)
print(classification_report(y_test, nbayes_pred))



# Model 4: Random forest

rf_model = Pipeline([
    ('dtm', CountVectorizer(
        tokenizer = tokenizer,    
        preprocessor = preprocessor,      
        token_pattern = token_pattern,     
        lowercase = lowercase,        
        min_df = min_df,               
        max_df = max_df,
        dtype = np.int16        
    )),
    ('tfidf', TfidfTransformer()),
    ('model', RandomForestClassifier(criterion = 'gini', random_state = 1))
])

rf_params = {
    'model__n_estimators': [100, 200, 400, 800],
    'model__max_depth': [None, 10, 20, 40, 60],
    'model__max_features': ['sqrt', 'log2', 0.1, 0.3, 0.5],
    'model__min_samples_split': [2, 5, 10, 20],
    'model__min_samples_leaf': [1, 2, 5, 10],
    'model__class_weight': [None, 'balanced', {'Bad': 1.0, 'Good': 2.0}],
    'tfidf': [
        TfidfTransformer(
            norm = 'l2',
            use_idf = True,
            smooth_idf = True,
            sublinear_tf = True
        ),
        TfidfTransformer(
            norm = 'l2',
            use_idf = True,
            smooth_idf = True,
            sublinear_tf = False
        ),        
        'passthrough'
    ],
    'dtm__ngram_range': [(1, 1), (1, 2)]
}

rf_step1 = RandomizedSearchCV(
    rf_model,
    rf_params,
    cv = StratifiedKFold(3, shuffle = True, random_state = 1),
    refit = True,
    scoring = hs_auc_scorer,
    n_jobs = -1,
    random_state = 1,
    n_iter = 30
)
rf_step1.fit(X_train, y_train)

pickle.dump(rf_step1, open('Predictor models/rf_base.bin', 'wb'))

rf_tuned = TunedThresholdClassifierCV(
    rf_step1.best_estimator_,
    scoring = good_recall_scorer,
    thresholds = 100,
    cv = StratifiedKFold(3, random_state = 1, shuffle = True),
    refit = True,
    n_jobs = -1
)
rf_tuned.fit(X_train, y_train)
pickle.dump(rf_tuned, open('Predictor models/rf_tuned.bin', 'wb'))

rf_pred= rf_tuned.predict(X_test)
np.save('Test predictions/rf_pred.npy', rf_pred)

print(classification_report(y_test, rf_pred))


### Model 5: Basic gradient boosting

boost_model = Pipeline([
    ('dtm', CountVectorizer(
        tokenizer = tokenizer,    
        preprocessor = preprocessor,      
        token_pattern = token_pattern,     
        lowercase = lowercase,        
        min_df = min_df,               
        max_df = max_df,
        dtype = np.int16        
    )),
    ('tfidf', TfidfTransformer()),
    ('sampler', RandomOverSampler(random_state = 1)),
    ('model', GradientBoostingClassifier(random_state = 1))
])

boost_params = {
    'sampler__sampling_strategy': [0.2, 0.4, 0.6, 0.8, 1.0],
    'model__n_estimators': [100, 200, 400, 800],
    'model__max_depth': [None, 10, 20, 40, 60],
    'model__max_features': ['sqrt', 'log2', 0.1, 0.3, 0.5],
    'model__min_samples_split': [2, 5, 10, 20],
    'model__min_samples_leaf': [1, 2, 5, 10],
    'model__learning_rate': [0.01, 0.1, 1],
    'model__loss': ['log_loss', 'exponential'],
    'tfidf': [
        'passthrough',
        TfidfTransformer(
            sublinear_tf = True
        ),
        TfidfTransformer(
            sublinear_tf = False
        )
    ],
    'dtm__ngram_range': [(1, 1), (1, 2)]
}

boost_step1 = RandomizedSearchCV(
    boost_model,
    boost_params,
    cv = StratifiedKFold(3, shuffle = True, random_state = 1),
    refit = True,
    scoring = hs_auc_scorer,
    n_jobs = -1,
    random_state = 1,
    n_iter = 30
)
boost_step1.fit(X_train, y_train)

pickle.dump(boost_step1, open('Predictor models/boost_base.bin', 'wb'))

boost_tuned = TunedThresholdClassifierCV(
    boost_step1.best_estimator_,
    scoring = good_recall_scorer,
    thresholds = 100,
    cv = StratifiedKFold(3, random_state = 1, shuffle = True),
    refit = True,
    n_jobs = -1
)
boost_tuned.fit(X_train, y_train)
pickle.dump(boost_tuned, open('Predictor models/boost_tuned.bin', 'wb'))

boost_pred = boost_tuned.predict(X_test)
np.save('Test predictions/boost_pred.npy', boost_pred)

print(classification_report(y_test, boost_pred))


### Model 6: LightGBM

pipeline = Pipeline([
    ('dtm', CountVectorizer(
        tokenizer = tokenizer,    
        preprocessor = preprocessor,      
        token_pattern = token_pattern,     
        lowercase = lowercase,        
        min_df = min_df,               
        max_df = max_df,
        dtype = np.float32
    )),
    ('tfidf', TfidfTransformer()),
    ('model', lgb.LGBMClassifier(random_state = 1))
])

lgbm_params = {
    'dtm__ngram_range': [(1, 1), (1, 2)],
    'model__learning_rate': [0.001, 0.01, 0.1],
    'model__boosting_type': ['gbdt'],
    'model__n_estimators': [300, 500, 800, 1200],
    'model__num_leaves': [15, 31, 63, 127],
    'model__max_depth': [-1, 5, 10, 15],
    'model__min_child_samples': [10, 20, 40, 80],
    'model__colsample_bytree': [0.7, 0.85, 1.0],
    'model__reg_alpha': [0, 0.1, 1, 5],
    'model__reg_lambda': [0, 0.1, 1, 5],
    'model__is_unbalance': [True, False],
    'tfidf': [
        'passthrough',
        TfidfTransformer(sublinear_tf = True),
        TfidfTransformer(sublinear_tf = False),
    ]
}

lgbm_base = RandomizedSearchCV(
    pipeline,
    lgbm_params,
    random_state = 1,
    n_iter = 30,
    scoring = hs_auc_scorer,
    cv = StratifiedKFold(3, shuffle = True, random_state = 1),
    refit = True,
    n_jobs = -1   
)

lgbm_base.fit(X_train, y_train)

pickle.dump(lgbm_base, open('Predictor models/lgbm_base.bin', 'wb'))

lgbm_tuned = TunedThresholdClassifierCV(
    lgbm_base.best_estimator_,
    scoring = good_recall_scorer,
    thresholds = 100,
    cv = StratifiedKFold(3, random_state = 1, shuffle = True),
    refit = True,
    n_jobs = -1,
    store_cv_results = True
)    

lgbm_tuned.fit(X_train, y_train)

pickle.dump(lgbm_tuned, open('Predictor models/lgbm_tuned.bin', 'wb'))

lgbm_pred = lgbm_tuned.predict(X_test)
np.save('Test predictions/lgbm_pred.npy', lgbm_pred)
print(classification_report(y_test, lgbm_pred))
