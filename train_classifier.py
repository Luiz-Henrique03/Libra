import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score

#Carrega os dados a partir do arquivo pickle
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

#Divide os dados em conjunto de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

#Remove elementos excedentes em x_train
x_train = [x[:42] for x in x_train if len(x) > 42]

#Remove elementos inválidos em y_train
y_train = [y for y in y_train if 0 <= int(y) <= 22]

#Define os parâmetros para a busca aleatória
n_estimators = [int(x) for x in np.linspace(start=10, stop=300, num=10)]
criterion = ['gini', 'entropy']
min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=3)]
max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
max_features = ['auto', 'sqrt', 'log2']

#Cria o grid de parâmetros
rf_grid = {
'n_estimators': n_estimators,
'criterion': criterion,
'min_samples_split': min_samples_split,
'max_depth': max_depth,
'max_features': max_features
}

#Realiza a busca aleatória para encontrar os melhores parâmetros
rf = RandomForestClassifier()
rf_hyper = RandomizedSearchCV(estimator=rf, param_distributions=rf_grid, n_iter=10, cv=3, verbose=1, n_jobs=-1, error_score="raise")
rf_hyper.fit(x_train, y_train)

#Cria o modelo final com os melhores parâmetros encontrados
rf = RandomForestClassifier(**rf_hyper.best_params_)
rf.fit(x_train, y_train)

#Realiza as previsões no conjunto de teste
y_predict = rf.predict(x_test)

#Calcula a pontuação de acurácia
score = accuracy_score(y_predict, y_test)

#Imprime o resultado da acurácia
print('{}% das amostras foram classificadas corretamente!'.format(score * 100))

#Salva o modelo em um arquivo pickle
with open('model.p', 'wb') as f:
    pickle.dump({'model': rf}, f)