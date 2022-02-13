import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

from data_preprocess.multi_column_encoder import MultiColumnLabelEncoder


class CSVHandler:

    def __init__(self, csv_file, useless_cols_list, target_col):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        self.cat_cols, self.num_cols = None, None
        self.preprocess_csv(useless_cols_list=useless_cols_list)
        self.remove_target_col(target_col)

    def remove_target_col(self, target_col):
        if(target_col in self.cat_cols):
            self.cat_cols.remove(target_col)
        elif(target_col in self.num_cols):

    def preprocess_csv(self, useless_cols_list):
        self.drop_cols(useless_cols_list)
        self.cat_cols, self.num_cols = self.identify_d_type()
        self.handle_bools_cols()
        self.do_imputation()
        self.encode_cat_cols()

    def encode_cat_cols(self):
        multi_encoder = MultiColumnLabelEncoder(self.cat_cols)
        self.df = multi_encoder.fit_transform(self.df)

    def handle_bools_cols(self):
        mask = self.df.applymap(type) != bool
        d = {True: 'TRUE', False: 'FALSE'}
        self.df = self.df.where(mask, self.df.replace(d))

    def do_imputation(self):
        self.apply_imputation_num()
        self.apply_imputation_cat()

    def apply_imputation_cat(self):
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value="NA")
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
