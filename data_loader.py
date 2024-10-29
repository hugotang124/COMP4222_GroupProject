import pickle
from typing import List
import numpy as np
import os
import glob
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import linalg
from statsmodels.tsa.stattools import adfuller
import torch
from torch.autograd import Variable
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class DataLoader(object):

    def __init__(self, data_directory: str, time_interval: str, train: float, valid: float, device: str, horizon: int, window: int, normalize: int, stationary_check: bool):

        data_files = glob.glob(os.path.join(data_directory, f"*{time_interval}*.csv"))

        if not data_files:
            print(f"There are no data files in provided directory {data_directory} and time interval = {time_interval}")
            exit()

        self.currencies = list(map(lambda x: os.path.split(x)[-1].replace(f"{time_interval}.csv", ""), data_files))
        currencies_data = dict(map(lambda x: (x[0], pd.read_csv(x[1], index_col = 0)), zip(self.currencies, data_files)))

        for currency, data in currencies_data.items():
            self._build_currency_features(currency, data)
            data.columns = [f"{currency.lower()}_{x}" for x in data.columns]

        self.raw_data = pd.concat(currencies_data.values(), axis = 1)
        self.raw_data.index = pd.to_datetime(self.raw_data.index)
        self._build_data_features()

        self.raw_data.dropna(inplace = True)

        # Stationarity checking
        if stationary_check:
            self._check_stationarity()
            self._check_cointegration()
            self._noise_removal()


        self.data = np.zeros(self.raw_data.shape)
        self.n, self.m = self.data.shape

        self.P = window
        self.h = horizon
        self.normalize = normalize
        self.scale = np.ones([3, self.m])
        self._split(train, valid)
        self._normalized(normalize)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale[-1, :].expand(self.test[1].size(0), self.m)

        self.scale = self.scale.to(device)

        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device
        exit()

    def _build_currency_features(self, currency: str, data: pd.DataFrame):
        '''
        Build same features for each currency
        Args:
            currency (str): Currency
            data (pd.DataFrame): Data for the currency
        '''
        pass

    def _build_data_features(self):
        '''
        Build features based on the entirety of data
        '''
        pass

    def _check_stationarity(self):
        '''
        Check stationarity of each column by the ADF test (Alternate hypothesis being the time series is stationary)
        '''

        self.stationary_vars = []
        p_value_threshold = 0.05

        for col in self.raw_data.columns:
            adftest = adfuller(self.raw_data[col], autolag = "AIC", regression = "CT")
            if adftest[1] < p_value_threshold:
                self.stationary_vars.append(col)


    def _check_cointegration(self):
        pass

    def _noise_removal(self):
        pass

    def _normalized(self, normalize: int):
        '''
        Normalize data with one of the multiple methods
        Args:
            normalize (int): Normalization mode
        '''

        for i, (X, Y) in enumerate([self.train, self.valid, self.test]):
            if (normalize == 0):
                data = data

            elif (normalize == 1):
                # Standardisation

                means = X.mean(dim = 1, keepdim = True)
                stds = X.std(dim = 1, keepdim = True)
                X = (X - means) / stds

            elif (normalize == 2):
                # Normalization based on the maximum of each column
                # Need adjustment in the evaluation and training function regarding error calculation

                self.scale[i, :] = X.max(dim = 1).values
                X = X / X.max(dim = 1, keepdim = True)

    def _split(self, train: float, valid: float):
        '''
        Split dataset into train, validation and test set in chronological order (test = 1 - train - valid), with batches created specifically
        Args:
            train (float): Proportion of train data
            valid (float): Proportion of validation data
        '''
        train_set = range(self.P + self.h - 1, int(train * self.n))
        valid_set = range(int(train * self.n), int((train + valid) * self.n))
        test_set = range(int((train + valid) * self.n), self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set: range, horizon: int) -> List[torch.Tensor]:
        '''
        Create batches from provided dataset ranges (train, valid, test) with batch size = horizon
        Args:
            idx_set (range): Range of indexes for data slicing
            horizon (int): Batch size
        '''
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.raw_data.iloc[start:end, :].to_numpy())
            Y[i, :] = torch.from_numpy(self.raw_data.iloc[idx_set[i], :].to_numpy())
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle = True):
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
