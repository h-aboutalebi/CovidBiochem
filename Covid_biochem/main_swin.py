import torch
import os
import argparse
import logging
import datetime

from sklearn.model_selection import train_test_split
from data_preprocess.image_dataloader import DataLoaderCXR
from models_parent.model_select import Model_select
from torch.utils.data import DataLoader
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
parser.add_argument('--image_path',
                    default=os.path.join(
                        dirname, "/storage/disk2/covid_biochem/stonybrook_cleaned"),
                    help='path of csv file for BioChem')
parser.add_argument('--cuda_n', type=str, default="7", help='random seed (default: 4)')
parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')

# *********************************** Model Setting **********************************************
parser.add_argument('-m',
                    '--model_name',
                    type=str,
                    default="swintransformer",
                    help='Available Model: swintransformer')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('-b', "--batch_size", type=int, default=32)

# *********************************** Optimizer Setting **********************************************
parser.add_argument('--lr', type=float, default=0.008, help="Initial learning rate")
parser.add_argument('--milestones',
                    type=int,
                    nargs='+',
                    default=[50, 100, 150],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma',
                    type=float,
                    default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay',
                    default=0.0002,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-2)')

# *********************************** Dataset Setting ********************************************
parser.add_argument('--test_size',
                    type=float,
                    default=0.2,
                    help='test size for experiment')
parser.add_argument(
    '-t',
    '--target_col',
    type=str,
    default="therapeuticexnoxBoolean",
    help='Target column to be used for prediction on Biochem.'
    'If your col name  has special character other than "_", remove them in the name')
parser.add_argument('-u',
                    '--useless_cols',
                    nargs='+',
                    default=[],
                    help='Useless columns to be removed for prediction on Biochem.')
parser.add_argument("--img_size", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument('--input_col', type=str, default="to_patient_id")

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

#Dataset creation:
csv_file = args.csv_path
csv_handle = CSVHandler(csv_file,
                        useless_cols_list=args.useless_cols,
                        target_col=args.target_col,
                        input_cols=args.input_col)
train_set, test_set = train_test_split(csv_handle.df,
                                       test_size=args.test_size,
                                       random_state=args.seed)
num_classes = csv_handle.df[args.target_col].nunique()
train_dataset = DataLoaderCXR(image_folder=args.image_path,
                              df=train_set,
                              target=args.target_col,
                              input_col=args.input_col,
                              img_size=args.img_size)
test_dataset = DataLoaderCXR(image_folder=args.image_path,
                             df=test_set,
                             target=args.target_col,
                             input_col=args.input_col,
                             img_size=args.img_size)
train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.num_workers)
test_loader = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         shuffle=True,
                         num_workers=args.num_workers)

#Model creation:
model = Model_select(model_name=args.model_name,
                     num_col_names=csv_handle.num_cols,
                     categorical_feature=csv_handle.cat_cols,
                     target_col=args.target_col,
                     num_classes=num_classes,
                     lr_scheduler=None,
                     init_lr=args.lr,
                     seed=args.seed)
model.create_model(device=device)

#Training model:
model.train_model(train_set=train_loader,
                  testset=test_loader,
                  epochs=args.epochs,
                  lr=args.lr,
                  milestones=args.milestones,
                  gamma=args.gamma,
                  momentum=args.momentum,
                  weight_decay=args.weight_decay)
test_pred = model.test_model(test_loader)
