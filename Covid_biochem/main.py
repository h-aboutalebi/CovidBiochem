import pandas as pd
import os
import argparse
import logging
import datetime

from sklearn.model_selection import train_test_split
from models.model_selection import Model_select
from utility.utils import print_metrics, seed_everything
from data_preprocess.csv_handler import CSVHandler
from utility.utils import File_Manager

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description='Covid-Net BioChem')
dirname = os.path.dirname(__file__)

# *********************************** General Setting *********************************************
parser.add_argument('-o', '--output_path', default=os.path.expanduser('~') + '/results_covid_biochem',
                    help='output path for files produced by the agent')
parser.add_argument('--csv_path', default=os.path.join(dirname, "pytorch_tabular-main/data/clinical_data.csv"),
                    help='path of csv file for BioChem')
parser.add_argument('--cuda_n', type=str, default="0", help='random seed (default: 4)')
parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')

# *********************************** Model Setting **********************************************
parser.add_argument('-m', '--model_name', type=str, default="lightgbm",
                    help='Available Model: lightgbm, tabtransformer')

# *********************************** Dataset Setting ********************************************
parser.add_argument('--test_size', type=float, default=0.2, help='test size for experiment')
parser.add_argument('-t', '--target_col', type=str, default="therapeuticexnoxBoolean",
                    help='Target column to be used for prediction on Biochem.'
                    'If your col name  has special character other than "_", remove them in the name')
parser.add_argument('-u', '--useless_cols', nargs='+', default=["therapeuticexnoxBoolean"],
                    help='Useless columns to be removed for prediction on Biochem.')


args = parser.parse_args()
seed_everything(args.seed)

# *********************************** Logging Config *********************************************
current_time = (str(datetime.datetime.now()).replace(" ", "#")).replace(":", "-")
output_path = args.output_path
file_path_results = output_path + "/" + current_time
File_Manager.set_file_path(file_path_results)
if not os.path.exists(output_path):
    os.makedirs(output_path)
os.mkdir(file_path_results)
logging.basicConfig(level=logging.DEBUG, filename=file_path_results + "/log.txt")
logging.getLogger().addHandler(logging.StreamHandler())

csv_file = args.csv_path
csv_handle = CSVHandler(csv_file, useless_cols_list=args.useless_cols, target_col=target_col)
train_set, test_set = train_test_split(csv_handle.df, test_size=args.test_size, random_state=args.seed)

model = Model_select(
    model_name=args.model_name,
    categorical_feature=csv_handle.cat_cols,
    target_col=args.target_col,
    seed=args.seed,
)
model.create_model()
model.train_model(train_set)
test_pred = model.test_model(test_set)

print_metrics(test_set[args.target_col], test_pred, "Holdout")
