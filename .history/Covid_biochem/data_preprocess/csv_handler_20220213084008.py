import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import numpy as np


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

class CSVHandler:

    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        csv_handle.preprocess_csv(useless_cols_list=["to_patient_id"])

    def preprocess_csv(self, useless_cols_list):
        self.drop_cols(useless_cols_list)
        self.cat_cols, self.num_cols = self.identify_d_type()
        self.do_imputation()

    def do_imputation(self):
        self.apply_imputation_num()
        self.apply_imputation_cat()

    def apply_imputation_cat(self):
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = "NA")
        self.df[self.cat_cols] = pd.DataFrame(
            imp.fit_transform(self.df[self.cat_cols]))

    def apply_imputation_num(self):
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.df[self.num_cols] = pd.DataFrame(
            imp.fit_transform(self.df[self.num_cols]))

    def identify_d_type(self):
        num_cols = list(self.df.select_dtypes("number").columns)
        cat_cols = list(self.df.select_dtypes(exclude=["number"]).columns)
        return cat_cols, num_cols

    def drop_cols(self, useless_cols_list):
        for col in useless_cols_list:
            self.df = self.df.drop(col, 1)
