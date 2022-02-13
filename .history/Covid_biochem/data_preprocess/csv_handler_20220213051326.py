import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

class CSVHandler:

    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)

    def preprocess_csv(self, useless_cols_list):
        self.drop_cols(useless_cols_list)
        self.cat_cols, self.num_cols = self.identify_d_type()
        self.do_imputation()

    def do_imputation(self):
        imp = SimpleImputer(missing_values=[None, np.nan], strategy='mean')
        
    def apply_imputation_num(self):
        self.df[self. self.num_cols]

    def identify_d_type(self):
        num_cols = list(self.df.select_dtypes("number").columns)
        cat_cols = list(self.df.select_dtypes(exclude=["number"]).columns)
        return cat_cols, num_cols

    def drop_cols(self, useless_cols_list):
        for col in useless_cols_list:
            self.df = self.df.drop(col, 1)
