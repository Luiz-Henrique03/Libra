import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_validate
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

for item in range(len(x_train)):
    if len(x_train[item]) > 42:
        del x_train[item][42:]
for item in y_train:
    if int(item) < 0 or int(item) > 22:
        print(item)

n_estimators = [int(x) for x in np.linspace(start=10, stop=300, num=10)]
criterion = ['gini', 'entropy']
min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=3)]
max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
max_features = ['auto', 'sqrt', 'log2']

rf_grid = {
    'n_estimators': n_estimators,
    'criterion': criterion,
    'min_samples_split': min_samples_split,
    'max_depth': max_depth,
    'max_features': max_features
}

rf = RandomForestClassifier()
rf_hyper = RandomizedSearchCV(estimator = rf, param_distributions = rf_grid, n_iter = 10, cv = 3, verbose = 1, n_jobs = -1, error_score="raise")

rf_hyper.fit(x_train, y_train)

rf = RandomForestClassifier(**rf_hyper.best_params_)
rf = rf.fit(x_train, y_train)

y_predict = rf.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': rf}, f)
f.close()
