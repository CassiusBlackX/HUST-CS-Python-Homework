import itertools
from threading import Thread
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def generate_combinations(list1, list2, list3, list4):
    return list(itertools.product(list1, list2, list3, list4))


def load_data(batch_size):
    movie = pd.read_csv("../data/data.csv")
    movie['rating'] = pd.qcut(movie['rating'], q=5, labels=False, duplicates='drop')
    X = movie[movie.columns[3:23]]
    y = movie[movie.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2, shuffle=True)
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
    X_train = torch.from_numpy(X_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.int64))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.int64))

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return X_train, train_loader, test_loader

class NewThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}):
        Thread.__init__(self, group, target, name, args, kwargs)
    def run(self):
        if self._target != None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

