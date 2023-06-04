import pandas as pd
from utils import scale_data, count_metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def main():
    data = pd.read_csv('flowers.csv')
    data.drop(['Area_Code', 'Region_Code'], axis=1, inplace=True)

    features = list(data.columns.drop('Class'))
    target = 'Class'

    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], stratify=data[target])
    print(f'Обучающая выборка: {X_train.shape[0]} наблюдений')
    print(f'Тестовая выборка: {X_test.shape[0]} наблюдений\n')

    X_train, X_test = scale_data(X_train, X_test)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print('Подсчёт метрик на тестовой выборке')
    metrics = count_metrics(y_test, y_pred)

if __name__=="__main__":
    main()