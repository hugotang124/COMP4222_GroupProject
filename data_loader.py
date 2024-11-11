import pickle
from typing import List, Optional
import numpy as np
import os
import glob
import pandas as pd
from datetime import datetime
import scipy.sparse as sp
from scipy.sparse import linalg
from statsmodels.tsa.stattools import adfuller
import torch
from torch.autograd import Variable
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from ta import add_all_ta_features
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volume import VolumeWeightedAveragePrice


def normal_std(x: torch.FloatTensor) -> float:
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class StandardScaler():
    def __init__(self, mean: torch.FloatTensor, std: torch.FloatTensor, device: str):
        """
        Standardisation scaler for data

        Arg types:
            **mean** (PyTorch Float Tensor): Mean matrix of data
            **std** (PyTorch Float Tensor): Standard deviation matrix of data
            device (str): Device used for model training
        """
        self.mean = mean.to(device)
        self.std = std.to(device)

    def transform(self, data: torch.FloatTensor) -> torch.FloatTensor:
        '''
        Transformation with standardisation

        Arg types:
            **data*8 (PyTorch Float Tensor): Data.

        Return types:
            **X** (PyTorch Float Tensor): Standardised Tensor.
        '''
        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.FloatTensor, id: torch.Tensor = None) -> torch.FloatTensor:
        '''
        Inverse transformation of standardisation regarding output

        Arg types:
            data (PyTorch Float Tensor): Data
            id (PyTorch Tensor):  Input indices, a permutation of the num_nodes, default None.

        Return types:
            **X** (PyTorch Float Tensor): Standardised Tensor
        '''

        if id is None:
            return (data * self.std) + self.mean

        return (data * self.std[:, id]) + self.mean[:, id]

class MaxScaler():
    def __init__(self, max: torch.FloatTensor, device: str):
        """
        Max value standardisation scaler for data

        Arg types:
            **max** (PyTorch Float Tensor): Max matrix of data
            device (str): Device used for model training
        """

        self.max = max.unsqueeze(1).to(device)

    def transform(self, data: torch.FloatTensor) -> torch.FloatTensor:
        '''
        Transformation with max value standardisation

        Arg types:
            **data*8 (PyTorch Float Tensor): Data.

        Return types:
            **X** (PyTorch Float Tensor): Standardised Tensor.
        '''

        return data / self.max

    def inverse_transform(self, data: torch.FloatTensor, id: Optional[torch.Tensor] = None) -> torch.FloatTensor:
        '''
        Inverse transformation of standardisation regarding output

        Arg types:
            data (PyTorch Float Tensor): Data
            id (PyTorch Tensor):  Input indices, a permutation of the num_nodes, default None.

        Return types:
            **X** (PyTorch Float Tensor): Standardised Tensor
        '''

        if id is None:
            return data * self.max

        return data * self.max[:, id]


class DataLoader(object):

    start_time = datetime(2021, 10, 1)

    def __init__(self, data_directory: str, time_interval: str, train: float, valid: float, device: str, horizon: int, window: int, normalize: int, stationary_check: bool, noise_removal: bool):

        data_files = glob.glob(os.path.join(data_directory, f"*{time_interval}*.csv"))

        if not data_files:
            print(f"There are no data files in provided directory {data_directory} and time interval = {time_interval}")
            exit()

        self.currencies = list(map(lambda x: os.path.split(x)[-1].replace(f"{time_interval}.csv", ""), data_files))
        currencies_data = dict(map(lambda x: (x[0], pd.read_csv(x[1], index_col = 0, parse_dates = True)), zip(self.currencies, data_files)))

        for currency, data in currencies_data.items():
            self._build_currency_features(currency, data)
            data.columns = [f"{currency.lower()}_{x}" for x in data.columns]

        self.raw_data = pd.concat(currencies_data.values(), axis = 1)
        self.raw_data = self.raw_data.loc[self.start_time:]
        self.raw_data.dropna(axis = 1, inplace = True)
        self.currencies = list(set(map(lambda x: x.split("_")[0], self.raw_data.columns)))
        self._build_data_features()

        # Save memory
        currencies_data.clear()

        # Stationarity checking
        if stationary_check:
            self._check_stationarity()
            self._check_cointegration()

        self.data = np.zeros(self.raw_data.shape)
        if noise_removal:
            for i in range(self.raw_data.shape[1]):
                self.data[i, :] = self._noise_removal(self.raw_data.iloc[i, :].to_numpy())
        else:
            self.data = self.raw_data.to_numpy()

        self.n, self.m = self.data.shape
        print(f"There are {len(self.currencies)} currencies involved in the data.")
        print(f"Dataframe Shape: {self.data.shape}")

        self.P = window
        self.h = horizon
        self.normalize = normalize
        self._split(train, valid)
        self.device = device


        self.rse = normal_std(self.test[1])
        self.rae = torch.mean(torch.abs(self.test[1] - torch.mean(self.test[1])))


    def _build_currency_features(self, currency: str, data: pd.DataFrame):
        '''
        Build same features for each currency
        Args:
            currency (str): Currency
            data (pd.DataFrame): Data for the currency
        '''


        windows = [5, 10, 20]

        data['volume_quote_ratio'] = data['volume'] / data['quote_volume']
        data['buy_sell_volume_ratio'] = data['buy_base_vol'] / data['volume']
        data['buy_sell_quote_ratio'] = data['buy_quote_vol'] / data['quote_volume']


        # Stochastic Oscillator
        stoch = StochasticOscillator(high = data['high'], low = data['low'], close = data['close'])
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()

        # VWAP
        vwap = VolumeWeightedAveragePrice(high = data['high'], low = data['low'],
                                         close = data['close'], volume = data['volume'])
        data['vwap'] = vwap.volume_weighted_average_price()

        for window in windows:
            # Parkinson Volatility
            data[f'parkinson_vol_{window}'] = np.sqrt(
                (1.0 / (4.0 * np.log(2.0))) *
                (np.log(data['high'] / data['low']) ** 2).rolling(window).mean()
            )

            # Garman-Klass Volatility
            data[f'garman_klass_vol_{window}'] = np.sqrt(
                (0.5 * np.log(data['high'] / data['low']) ** 2) -
                (2.0 * np.log(2.0) - 1.0) * (np.log(data['close'] / data['open']) ** 2)
            ).rolling(window).mean()

        data['avg_trade_size'] = data['volume'] / data['trades']
        data['avg_trade_quote_size'] = data['quote_volume'] / data['trades']


        data.drop(columns = ["open", "high", "low"], inplace = True)


    def _build_data_features(self):
        '''
        Build features based on the entirety of data
        '''
        # Time-based features
        time_df = pd.DataFrame(
            {
                "year": self.raw_data.index.year,
                "month": self.raw_data.index.month,
                "day": self.raw_data.index.day,
                "hour": self.raw_data.index.hour,
                "day_of_week": self.raw_data.index.dayofweek,
                "week_of_year": self.raw_data.index.isocalendar().week
            }
        )
        time_df = time_df.astype(int)

        self.raw_data = pd.concat([self.raw_data, time_df], axis = 1)


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

    def _noise_removal(self, signal: np.ndarray, window: int = 100):
        '''
        Remove noise by using moving average filter
        Args:
            signal (pd.Series): Pandas Series to be smoothened with moving average filter
            window (int): Number of rows accounted for calculating the moving average
        '''

        return np.convolve(signal, np.ones(window) / window, mode = 'same')

    def _normalized(self, X, Y, normalize: int = 1):
        '''
        Normalize data with one of the multiple methods
        Args:
            normalize (int): Normalization mode
        '''
        if (normalize == 0):
            data = data
            X_scaler = StandardScaler(mean = 0, std = 1, device = self.device)
            Y_scaler = StandardScaler(mean = 0, std = 1, device = self.device)

        elif (normalize == 1):
            # Standardisation

            X_scaler = StandardScaler(mean = X.mean(dim = 0, keepdim = True),  std = X.std(dim = 0, keepdim = True), device = self.device)
            X = X_scaler.transform(X)
            del X_scaler

            Y_scaler = StandardScaler(mean = Y.mean(dim = 0, keepdim = True),  std = Y.std(dim = 0, keepdim = True), device = self.device)

        elif (normalize == 2):
            # Normalization based on the maximum of each column

            X_scaler = MaxScaler(max = X.max(dim = 0).values, device = self.device)
            X = X_scaler.transform(X)
            del X_scaler

            Y_scaler = MaxScaler(max = Y.max(dim = 0).values, device = self.device)

        return X, Y, Y_scaler

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
            X[i, :, :] = torch.from_numpy(self.data[start:end, :])
            Y[i, :] = torch.from_numpy(self.raw_data.iloc[idx_set[i], :].to_numpy())

        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle = True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.arange(length)  # More efficient than LongTensor

        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]

            # Ensure inputs and targets are tensors
            X = inputs[excerpt].to(self.device)
            Y = targets[excerpt].to(self.device)
            X, Y, Y_scaler = self._normalized(X, Y, self.normalize)
            X = X.to(self.device)
            Y = Y.to(self.device)

            yield X, Y, Y_scaler # No need for Variable

            start_idx += batch_size
