import pandas as pd
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
from data_preprocess.multi_column_encoder import MultiColumnLabelEncoder
from sklearn.impute import KNNImputer


class CSVHandler:

    def __init__(self,
                 csv_file,
                 useless_cols_list,
                 target_col,
                 input_cols=None,
                 n_neighbors=5):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        self.df = shuffle(self.df)
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
        self.input_cols = input_cols
        self.correct_col_names()
        self.cat_cols, self.num_cols = None, None
        self.preprocess_csv(useless_cols_list=useless_cols_list)
        self.update_cat_num_list(target_col)
        self.quantile_transformer = None

    def correct_col_names(self):
        self.df = self.df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    def update_cat_num_list(self, target_col):
        if (target_col in self.cat_cols):
            self.cat_cols.remove(target_col)
        elif (target_col in self.num_cols):
            self.num_cols.remove(target_col)
        else:
            raise Exception("target column not in csv!")

    def preprocess_csv(self, useless_cols_list):
        self.drop_cols(useless_cols_list)
        self.cat_cols, self.num_cols = self.identify_d_type()
        self.remove_input_col()
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
        self.df[self.cat_cols] = pd.DataFrame(imp.fit_transform(self.df[self.cat_cols]))

    def apply_imputation_num(self):
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-10)
        self.df[self.num_cols] = pd.DataFrame(imp.fit_transform(self.df[self.num_cols]))

    def remove_input_col(self):
        if (self.input_cols is not None):
            if (self.input_cols in self.cat_cols):
                self.cat_cols.remove(self.input_cols)
            elif (self.input_cols in self.num_cols):
                self.num_cols.remove(self.input_cols)

    def identify_d_type(self):
        num_cols = list(self.df.select_dtypes("number").columns)
        cat_cols = list(self.df.select_dtypes(exclude=["number"]).columns)
        return cat_cols, num_cols

    def drop_cols(self, useless_cols_list):
        for col in useless_cols_list:
            self.df = self.df.drop(col, 1)

    def initilize_quantile_transformer(self, train_set):
        self.quantile_transformer = QuantileTransformer(output_distribution="normal")
        data = pd.DataFrame(self.quantile_transformer.fit_transform(train_set))
        return data
        # self.quantile_transformer = MinMaxScaler()
        # return pd.DataFrame(self.quantile_transformer.fit_transform(train_set))

    def apply_quantile_transformer(self, test_set):
        if (self.quantile_transformer is None):
            raise Exception("Quantile transformer not initilized!")
        return pd.DataFrame(self.quantile_transformer.transform(test_set))
