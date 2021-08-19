from data_loader.balanced_sampler import BalancedBatchSampler
from data_loader.build_dataset import BuildDataset
from tensorboardX import SummaryWriter
from data_loader.data_extractor import Data_extractor
from torch.utils.data import DataLoader
from models.tcn import TCN
from torch.optim.lr_scheduler import MultiStepLR
from trainer.tcn_train import TCNTrainer
from utils.file_manager import File_Manager
from data_loader.dataset import TCNDataset

import numpy as np
import datetime
import argparse
import os
import random
import torch
import logging
import torch.optim as optim

from utils.tensor_writer import Tensor_Writer

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='TCN for Privacy Adversarial Attack')

# *********************************** General Setting ********************************************
parser.add_argument('-o', '--output_path', default=os.path.expanduser('~') + '/results_privacy',
                    help='output path for files produced by the agent')
parser.add_argument('-d', '--data_dir', default='/home/hossein.aboutalebi/data/PrivAttack-Data/100/1',
                    help='output path for files produced by the agent')
parser.add_argument('--cuda_n', type=str, default="0", help='random seed (default: 4)')
parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')

# *********************************** Dataset Loading Setting ********************************************
parser.add_argument('--action_shape', type=int, default=3,
                    help='trajectory length (default: 10)')
parser.add_argument('--max_num_trj', type=int, default=25000,
                    help='trajectory length (default: 10)')
parser.add_argument('--trj_len', type=int, default=100,
                    help='trajectory length (default: 10)')
parser.add_argument('--n_output', type=int, default=2,
                    help='number of hidden units per layer (default: 600)')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 16)')
parser.add_argument('--num_workers', type=int, default=4,
                    help='number of workers for torchvision Dataloader')
# parser.add_argument("--seeds_shadow", nargs="+", default=[100, 700])
# parser.add_argument("--seeds_target", nargs="+", default=[5, 45])

parser.add_argument("--seeds_shadow", nargs="+", default=[700, 5])
parser.add_argument("--seeds_target", nargs="+", default=[ 75, 90])

# *********************************** Model Setting ********************************************
parser.add_argument('--dropout', type=float, default=0.45,
                    help='dropout applied to layers (default: 0.45)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--nhid', type=int, default=600,
                    help='number of hidden units per layer (default: 600)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')

# *********************************** Training Setting ********************************************
parser.add_argument('--clip', type=float, default=0.35,
                    help='gradient clip, -1 means no clip (default: 0.35)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--lr', type=float, default=0.0003,
                    help='initial learning rate (default: 4)')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='initial learning rate (default: 4)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer type (default: Adam)')
args = parser.parse_args()

# sets the seed for making it comparable with other implementations
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# *********************************** Logging Config ********************************************
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
args_keys = list(vars(args).keys())
args_keys.sort()
max_k = len(max(args_keys, key=lambda x: len(x)))
for k in args_keys:
    s = k + '.' * (max_k - len(k)) + ': %s' % repr(getattr(args, k))
    logger.info(s + ' ' * max((len(header) - len(s), 0)))
logger.info("=" * len(header))


# for tensorboard
file_path_tensorboard = output_path + "/tensorboard/" + current_time
if not os.path.exists(output_path + "/tensorboard"):
    os.makedirs(output_path + "/tensorboard")
try:
    writer = SummaryWriter(logdir=file_path_tensorboard)
except:
    writer = SummaryWriter(file_path_tensorboard)
Tensor_Writer.set_writer(writer)

# *********************************** Environment Building ********************************************
device = torch.device("cuda:" + args.cuda_n if torch.cuda.is_available() else "cpu")
logger.info("device is set for: {}".format(device))

if __name__ == '__main__':
    num_chans = [args.nhid] * (args.levels - 1) + [args.n_output]
    batch_size = args.batch_size
    epochs = args.epochs
    k_size = args.ksize
    dropout = args.dropout
    buildDataset=BuildDataset(seeds_shadow=args.seeds_shadow,seeds_target=args.seeds_target,data_dir=args.data_dir,action_shape=args.action_shape,trj_len=args.trj_len,batch_size=args.batch_size,
                              max_num_trj=args.max_num_trj)
    train_loader = buildDataset.load_trainset(max_num_trj=args.max_num_trj)
    test_loader = buildDataset.load_testset(max_num_trj=2000)

    # loading the model
    model = TCN(args.trj_len,args.action_shape*2, args.n_output, num_chans, dropout=dropout, kernel_size=k_size,trj_len=args.trj_len)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[33, 100], gamma=0.1)
    trainer = TCNTrainer(train_loader, test_loader,model, optimizer, device)
    trainer.run(epochs, scheduler)
