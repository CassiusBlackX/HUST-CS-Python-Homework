import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44, shuffle=True)
    model = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth,
                                   min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                   random_state=44)
    model.fit(X_train, y_train)
    test_acc = accuracy_score(model.predict(X_test), y_test)
    print(f'{id}:({params}), test_acc: {test_acc}')
    return test_acc


def main():
    n_estimators = [128, 64, 100, 110, 120, 130, 140, 10, 512, 1000]
    max_depths = [20, 32, 64, 10, None, 40]
    min_samples_splits = [2, 4, 16, 20, 40]
    min_samples_leafs = [1, 2, 4, 8, 16, 20, 32]
    combinations = generate_combinations(n_estimators, max_depths, min_samples_splits, min_samples_leafs)

    best_acc = 0

    for i in range(0, len(combinations), 8):
        threads = [NewThread(target=search_parameter, args=(params,)) for params in combinations[i:i+8]]
        for thread in threads:
            thread.start()
        for thread in threads:
            test_acc = thread.join()
            if test_acc > best_acc:
                best_acc = test_acc

    print(f'best accuracy: {best_acc}')


if __name__ == '__main__':
    main()



