import numpy as np
import pandas as pd


class Data:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        try:
            self.pd = pd.read_csv(csv_path)
        except:
            raise ValueError('csv路径错误')

    def get_feature(self):
        self.ft_str = list()
        for dt1 in self.pd.columns:
            self.ft_str.append(str(dt1))

    def get_col(self, str):
        return self.pd.loc[str]

    def get_x(self, int_x):
        d = self.pd.columns[int_x:int_x + 1]
        return np.array(self.pd[d])

    def get_y(self, int_y):
        d = self.pd.columns[int_y:int_y + 1]
        return np.array(self.pd[d])

    def get_pd_x(self, int_x):
        d = self.pd.columns[int_x:int_x + 1]
        return self.pd[d]

    def get_pd_y(self, int_y):
        d = self.pd.columns[int_y:int_y + 1]
        return self.pd[d]

    def display_data(self):
        self.pd.head()

    def data_check(self):
        self.pd.info()

    def data_describe(self):
        self.pd.describe()
