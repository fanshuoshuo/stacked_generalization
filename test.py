from sklearn.datasets import load_digits
from stacked_generalizer import StackedGeneralizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

VERBOSE = True
N_FOLDS = 5

# load data and shuffle observations
data = load_digits()

X = data.data
y = data.target

shuffle_idx = np.random.permutation(y.size)

X = X[shuffle_idx]
y = y[shuffle_idx]

# hold out 20 percent of data for testing accuracy
train_prct = 0.8
n_train = int(round(X.shape[0]*train_prct))

# define base models
base_models = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
               RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
               ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini')]

# define blending model
blending_model = LogisticRegression()

# initialize multi-stage model
sg = StackedGeneralizer(base_models, blending_model, 
	                    n_folds=N_FOLDS, verbose=VERBOSE)

# fit model
sg.fit(X[:n_train],y[:n_train])

# test accuracy
pred = sg.predict(X[n_train:])
pred_classes = [np.argmax(p) for p in pred]

_ = sg.evaluate(y[n_train:], pred_classes)
