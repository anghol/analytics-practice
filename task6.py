import pandas as pd
from utils import scale_data, count_metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

def main():
    data = pd.read_csv('flowers.csv')
    data.drop(['Area_Code', 'Region_Code'], axis=1, inplace=True)

    features = list(data.columns.drop('Class'))
    target = 'Class'

    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], shuffle=True)
    print(f'Обучающая выборка: {X_train.shape[0]} наблюдений')
    print(f'Тестовая выборка: {X_test.shape[0]} наблюдений\n')

    X_train, X_test = scale_data(X_train, X_test)

    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print('Подсчёт метрик на тестовой выборке')
    metrics = count_metrics(y_test, y_pred)

    print('\nПоиск оптимальных параметров с помощью GridSearch')
    params = {'criterion': ['gini', 'entropy', 'log_loss'], 'max_depth': range(1, 11)}
    tree_grid = GridSearchCV(classifier, params, cv=5, n_jobs=-1, verbose=True)
    tree_grid.fit(X_train, y_train)
    print('Оптимальные параметры:', tree_grid.best_params_)
    print('Лучшая accuracy по сетке параметров:', tree_grid.best_score_)

    print('\nПодсчёт метрик на тестовой выборке с оптимальной моделью')
    metrics = count_metrics(y_test, tree_grid.predict(X_test))

if __name__=='__main__':
    main()