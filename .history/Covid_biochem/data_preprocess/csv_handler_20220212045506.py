import pandas as pd


class CSVHandler:
    
    def __init__(self, csv_file):
        self.csv_file=csv_file
        self.df = pd.read_csv(csv_file)
        
        
    def preprocess_csv(self,useless_cols_list):
        self.drop_cols(useless_cols_list)
        
    def drop_cols(self, useless_cols_list):
        for col in useless_cols_list:
            self.df = self.df.drop(col, 1)
        
    
        