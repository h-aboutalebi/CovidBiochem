
class CSVHandler:
    
    def __init__(self, csv_file):
        self.csv_file=csv_file
        
    def preprocess_csv(self):
        self.drop_cols()
        