import pandas as pd
from utils import scale_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

def main():
    data = pd.read_csv('flowers.csv')
    data.drop(['Area_Code', 'Region_Code'], axis=1, inplace=True)

    features = list(data.columns.drop('Class'))
    target = 'Class'

    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], shuffle=True)
    print(f'Обучающая выборка: {X_train.shape[0]} наблюдений')
    print(f'Тестовая выборка: {X_test.shape[0]} наблюдений\n')

    X_train, X_test = scale_data(X_train, X_test)

    classifier = KNeighborsClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print('Подсчёт метрик на тестовой выборке')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred, average='weighted', zero_division=0))
    print('Матрица ошибок:\n', confusion_matrix(y_test, y_pred))

    print('\nПоиск оптимальных параметров с помощью GridSearch')
    pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])
    params = {'knn__n_neighbors': range(1, 10), 'knn__weights': ['uniform', 'distance']}
    knn_grid = GridSearchCV(pipe, params, cv=5, n_jobs=-1, verbose=True)
    knn_grid.fit(X_train, y_train)
    print('Оптимальные параметры:', knn_grid.best_params_)
    print('Лучшая accuracy по сетке параметров:', knn_grid.best_score_)

    print('\nПодсчёт метрик на тестовой выборке с оптимальной моделью')
    print('Accuracy:', accuracy_score(y_test, knn_grid.predict(X_test)))
    print('Precision:', precision_score(y_test, knn_grid.predict(X_test), average='weighted', zero_division=0))

if __name__=='__main__':
    main()