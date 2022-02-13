import pandas as pd
import os


class CSVHandler:
    
    def __init__(self, csv_file):
        self.csv_file=csv_file
        
        
    def preprocess_csv(self,useless_cols):
        self.drop_cols(useless_cols)
        
    
        