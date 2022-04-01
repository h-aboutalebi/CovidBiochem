import pandas as pd
import os
import argparse
import logging
import datetime

from Covid_biochem.graph.his import create_his
from utility.file_manager import File_Manager
from utility.utils import seed_everything

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True
parser = argparse.ArgumentParser(description='Covid-Net BioChem')
dirname = os.path.dirname(__file__)

# *********************************** General Setting *********************************************
parser.add_argument('-o', '--output_path', default=os.path.expanduser('~') + '/results_covid_biochem',
                    help='output path for files produced by the agent')
parser.add_argument('--csv_path', default=os.path.join(dirname, "pytorch_tabular_main/data/clinical_data.csv"),
                    help='path of csv file for BioChem')
parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')

# *********************************** Dataset Setting ********************************************
parser.add_argument('-t', '--target_col', type=str, default="last.status",
                    help='Target column to be used for prediction on Biochem.'
                    'If your col name  has special character other than "_", remove them in the name')


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

header = "===================== Experiment configuration ========================"
logger.info(header)
args_keys = sorted(vars(args).keys())
max_k = len(max(args_keys, key=lambda x: len(x)))
for k in args_keys:
    s = k + '.' * (max_k - len(k)) + ': %s' % repr(getattr(args, k))
    logger.info(s + ' ' * max((len(header) - len(s), 0)))
logger.info("=" * len(header))

# *********************************** Environment Building ********************************************

csv_file = args.csv_path
df = pd.read_csv(csv_file)
create_his(df,args.target_col)
