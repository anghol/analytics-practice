import pandas as pd
from utils import scale_data, count_metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def main():
    data = pd.read_csv('flowers.csv')
    data.drop(['Area_Code', 'Region_Code'], axis=1, inplace=True)

    features = list(data.columns.drop('Class'))
    target = 'Class'

    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], stratify=data[target])
    print(f'Обучающая выборка: {X_train.shape[0]} наблюдений')
    print(f'Тестовая выборка: {X_test.shape[0]} наблюдений\n')

    X_train, X_test = scale_data(X_train, X_test)

    classifier = QuadraticDiscriminantAnalysis()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print('Подсчёт метрик на тестовой выборке')
    metrics_default = count_metrics(y_test, y_pred)

    print('\nПоиск оптимальных параметров с помощью GridSearch')
    params = {
        'reg_param': (0.00001, 0.0001, 0.001,0.01, 0.1), 
        'store_covariance': (True, False),
        'tol': (0.0001, 0.001,0.01, 0.1)
        }
    grid = GridSearchCV(classifier, params, cv=5, n_jobs=-1, verbose=True)
    grid.fit(X_train, y_train)
    print('Оптимальные параметры:', grid.best_params_)
    print('Лучшая accuracy по сетке параметров:', grid.best_score_)

    print('\nПодсчёт метрик на тестовой выборке с оптимальной моделью')
    metrics_optim = count_metrics(y_test, grid.predict(X_test))

if __name__=='__main__':
    main()