import pickle
import numpy as np
import os
import glob
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class DataLoader(object):

    def __init__(self, data_directory: str, time_interval: str, train: float, valid: float, device: str, horizon: int, window: int, normalize: int):

        data_files = glob.glob(os.path.join(data_directory, f"*{time_interval}*.csv"))

        if not data_files:
            print(f"There are no data files in provided directory {data_directory} and time interval = {time_interval}")
            exit()

        self.currencies = list(map(lambda x: os.path.split(x)[-1].replace(f"{time_interval}.csv", ""), data_files))
        currencies_data = dict(map(lambda x: (x[0], pd.read_csv(x[1], index_col = 0)), zip(self.currencies, data_files)))

        for currency, data in currencies_data.items():
            data.columns = [f"{currency.lower()}_{x}" for x in data.columns]
            self._build_features(currency, data)


        self.raw_data = pd.concat(currencies_data.values(), axis = 1)
        self.raw_data.index = pd.to_datetime(self.raw_data.index)


        self.data = np.zeros(self.raw_data.shape)
        self.n, self.m = self.data.shape

        self.P = window
        self.h = horizon
        self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

    def _build_features(self, currency: str, data: pd.DataFrame):
        # Build all the features (e.g. indicators) needed
        pass

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.data = self.rawdat

        if (normalize == 1):
            self.data = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.data[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.data[start:end, :])
            Y[i, :] = torch.from_numpy(self.data[idx_set[i], :])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size
