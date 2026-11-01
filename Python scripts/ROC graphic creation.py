
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import GaussianNB
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import gc

data_all = pd.read_feather(f'amazon_reviews_cleaned.feather')

X_train, X_test, y_train, y_test = train_test_split(
    data_all.Review_Sentences.tolist(),
    data_all.Rating_Class.tolist(),
    test_size = 0.11,
    random_state = 1,
    stratify = data_all.Rating_Class.tolist()
)

del data_all
gc.collect()

def fun_toarray(x):
    return x.toarray()

basic_mod = Pipeline([
    ('dtm', CountVectorizer(
        tokenizer = str.split,    
        preprocessor = None,      
        token_pattern = None,     
        lowercase = False,        
        min_df = 5,               
        max_df = 1.0,
        dtype = np.int16
    )),
    ('to_dense', FunctionTransformer(fun_toarray, accept_sparse = True)),    
    ('model', GaussianNB())
])
basic_mod.fit(X_train, y_train)

classes = list(basic_mod.classes_)
idx = classes.index('Bad')

proba = basic_mod.predict_proba(X_train)[:, idx]
min_recall = 0.95

fpr, tpr, _ = roc_curve(y_train, proba, pos_label = 'Bad')

unique_tpr, indeces = np.unique(tpr, return_index = True)
unique_fpr = fpr[indeces]

tpr_grid = np.linspace(
    0.0, 1.0, 1000000
)

fpr_grid = np.interp(
    tpr_grid,
    unique_tpr,
    unique_fpr
)

tpr_grid_s = np.linspace(
    0.95, 1.0, 1000
)

fpr_grid_s = np.interp(
    tpr_grid_s,
    unique_tpr,
    unique_fpr
)

area = np.trapezoid(fpr_grid_s, tpr_grid_s).item()
mean_val = area / 0.05

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize = (8, 4))

ax[0].plot(fpr_grid, tpr_grid, color = 'black');
ax[0].grid(axis = 'y');
ax[0].set_axisbelow(True);
ax[0].set_xlabel('False positive rate (positive label: Bad)');
ax[0].set_ylabel('True positive rate (positive label: Bad)');
ax[0].set_title('ROC curve');

ax[1].plot(tpr_grid, fpr_grid, color = 'black');
#ax.set_ylim(-0.001, 0.02);
#ax.set_xlim(0.9 - 0.005, 1.0 + 0.005);
ax[1].grid(axis = 'y');
ax[1].set_axisbelow(True);
ax[1].set_xlabel('True positive rate (positive label: Bad)');
ax[1].set_ylabel('False positive rate (positive label: Bad)');
ax[1].set_title('Inverse ROC curve');
ax[1].fill_between(tpr_grid[tpr_grid >= 0.95], 0, fpr_grid[tpr_grid >= 0.95],
    color = "lightblue", alpha = 0.8, label = 'Area under the inverse ROC\ncurve for TPR >= 0.95');
ax[1].axhline(y = mean_val, color = 'blue', linestyle = '--', linewidth = 2,
    label = 'Average FPR over blue area')
ax[1].legend();
ax[1].text(0.5, mean_val + 0.02, s = f'{mean_val:.2f}', color = 'blue');

fig.savefig('Pictures/ROC-Example.png')
