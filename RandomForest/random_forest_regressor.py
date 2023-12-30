import itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from data_utils import generate_combinations, NewThread
import pandas as pd

id = 0


def search_parameter(params):
    global id
    id += 1
    n_estimator, max_depth, min_samples_split, min_samples_leaf = params
    data = pd.read_csv('../data/data.csv')
    data['rating'] = pd.qcut(data['rating'], q=5, labels=False, duplicates='drop')

    X = data[data.columns[3:23]]
    y = data[data.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22, shuffle=True)
    model = RandomForestRegressor(n_estimators=n_estimator, max_depth=max_depth,
                                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                  random_state=22)
    model.fit(X_train, y_train)
    test_mae = mean_absolute_error(model.predict(X_test), y_test)
    print(f'{id}:({params}), test_mae: {test_mae}')
    return test_mae


def main():
    n_estimators = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    max_depths = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_samples_splits = [2, 3, 4, 5, 10, 15, 20, 25, 30]
    min_samples_leafs = [1, 2, 3, 4, 5, 10, 15, 20, 25]
    combinations = generate_combinations(n_estimators, max_depths, min_samples_splits, min_samples_leafs)

    best_mae = float('inf')
    best_params = None

    for i in range(0, len(combinations), 8):
        threads = [NewThread(target=search_parameter, args=(params,)) for params in combinations[i:i + 8]]
        for thread in threads:
            thread.start()
        for thread in threads:
            test_mae = thread.join()
            if test_mae < best_mae:
                best_mae = test_mae

    print(f'best accuracy: {best_mae}')


if __name__ == '__main__':
    main()
