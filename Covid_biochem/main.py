import torch
import os
import argparse
import logging
import datetime

from sklearn.model_selection import train_test_split
from models_parent.model_select import Model_select
from utility.file_manager import File_Manager
from utility.utils import seed_everything
from data_preprocess.csv_handler import CSVHandler

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description='Covid-Net BioChem')
dirname = os.path.dirname(__file__)

# *********************************** General Setting *********************************************
parser.add_argument('-o',
                    '--output_path',
                    default=os.path.expanduser('~') + '/results_covid_biochem',
                    help='output path for files produced by the agent')
parser.add_argument('--csv_path',
                    default=os.path.join(dirname,
                                         "pytorch_tabular_main/data/clinical_data.csv"),
                    help='path of csv file for BioChem')
parser.add_argument('--cuda_n', type=str, default="3", help='random seed (default: 4)')
parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')

# *********************************** Model Setting **********************************************
parser.add_argument('-m',
                    '--model_name',
                    type=str,
                    default="lightgbm",
                    help='Available Model: lightgbm, tabtransformer, XGBoost')
parser.add_argument('--gradient_clip_val',
                    type=float,
                    default=0.0,
                    help='Gradient clipping value')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--early_stopping_patience',
                    type=int,
                    default=30,
                    help='Number of epochs to wait before early stopping')
parser.add_argument('-b', "--batch_size", type=int, default=128)
parser.add_argument('--checkpoints_save_top_k',
                    type=int,
                    default=0,
                    help='Number of best models to save')
parser.add_argument('--lr', type=float, default=0.0004, help="Initial learning rate")
parser.add_argument(
    '--auto_lr_find',
    action="store_true",
    help=
    'Runs a learning rate finder algorithm (see this paper) when calling trainer.tune(), to find optimal initial learning rate.'
)

# *********************************** Optimizer Setting **********************************************
parser.add_argument(
    '--lr_scheduler',
    type=str,
    default=None,
    help=
    'The name of the LearningRateScheduler to use, if any, from [torch.optim.lr_scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate). If None, will not use any scheduler. Defaults to `None`'
)

# *********************************** Dataset Setting ********************************************
parser.add_argument('--test_size',
                    type=float,
                    default=0.2,
                    help='test size for experiment')
parser.add_argument('--val_size',
                    type=float,
                    default=0.1,
                    help='validation size for experiment')
parser.add_argument('--n_neighbors', type=int, default=5, help='KNN imputaor')
parser.add_argument(
    '-t',
    '--target_col',
    type=str,
    default="laststatus",
    help='Target column to be used for prediction on Biochem.'
    'If your col name  has special character other than "_", remove them in the name')
parser.add_argument(
    '-u',
    '--useless_cols',
    nargs='+',
    default=[
        #categorical features:
        "to_patient_id",
        "covid19_statuses",
        "blood_pHbetween735and745",
        "blood_pHbelow735",
        "blood_pHabove745",
        "A1C8to99",
        "A1Cover10",
        "A1C66to79",
        "A1Cunder65",
        "A1Cover65",
        "visit_start_datetime",
        #numerical features:
        "invasive_vent_days",
        "45484_HemoglobinA1cHemoglobintotalinBlood",
        "332544_pHofArterialbloodadjustedtopatientsactualtemperature"
    ],
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

header = "===================== Experiment configuration ========================"
logger.info(header)
args_keys = sorted(vars(args).keys())
max_k = len(max(args_keys, key=lambda x: len(x)))
for k in args_keys:
    s = k + '.' * (max_k - len(k)) + ': %s' % repr(getattr(args, k))
    logger.info(s + ' ' * max((len(header) - len(s), 0)))
logger.info("=" * len(header))

# *********************************** Environment Building ********************************************
device = torch.device("cuda:" + args.cuda_n if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBILE_DEVICES"] = args.cuda_n
logger.info("device is set for: {}".format(device))

csv_file = args.csv_path
csv_handle = CSVHandler(csv_file,
                        useless_cols_list=args.useless_cols,
                        target_col=args.target_col,
                        n_neighbors=args.n_neighbors)
train_set, test_set = train_test_split(csv_handle.df,
                                       test_size=args.test_size,
                                       random_state=args.seed,
                                       stratify=csv_handle.df[args.target_col])
train_set, val_set = train_test_split(csv_handle.df,
                                      test_size=args.val_size,
                                      random_state=args.seed,
                                      stratify=csv_handle.df[args.target_col])
num_classes = csv_handle.df[args.target_col].nunique()
model = Model_select(model_name=args.model_name,
                     num_col_names=csv_handle.num_cols,
                     categorical_feature=csv_handle.cat_cols,
                     target_col=args.target_col,
                     num_classes=num_classes,
                     lr_scheduler=args.lr_scheduler,
                     init_lr=args.lr,
                     seed=args.seed)

model.create_model()
model.train_model(train_set,
                  gradient_clip_val=args.gradient_clip_val,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  early_stopping_patience=args.early_stopping_patience,
                  checkpoints_save_top_k=args.checkpoints_save_top_k,
                  auto_lr_find=args.auto_lr_find,
                  cuda_n=args.cuda_n,
                  seed=args.seed,
                  val_set=val_set)
test_pred = model.test_model(test_set)
