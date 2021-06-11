from data_loader.balanced_sampler import BalancedBatchSampler
from data_loader.data_extractor import Data_extractor
from torch.utils.data import DataLoader
from models.tcn import TCN
from trainer.tcn_train import TCNTrainer
from utils.file_manager import File_Manager
from utils.files import get_trj_end_npy
from data_loader.dataset import TCNDataset

import numpy as np
import datetime
import argparse
import os
import random
import torch
import logging
import torch.optim as optim

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='TCN for Privacy Adversarial Attack')

# *********************************** General Setting ********************************************
parser.add_argument('-o', '--output_path', default=os.path.expanduser('~') + '/results_privacy',
                    help='output path for files produced by the agent')
parser.add_argument('--cuda_n', type=str, default="0", help='random seed (default: 4)')
parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')

# *********************************** Dataset Loading Setting ********************************************
parser.add_argument('--action_shape', type=int, default=3,
                    help='trajectory length (default: 10)')
parser.add_argument('--trj_len', type=int, default=10,
                    help='trajectory length (default: 10)')
parser.add_argument('--n_output', type=int, default=2,
                    help='number of hidden units per layer (default: 600)')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 16)')
parser.add_argument('--num_workers', type=int, default=4,
                    help='number of workers for torchvision Dataloader')

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
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer type (default: SGD)')
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

# *********************************** Environment Building ********************************************
device = torch.device("cuda:" + args.cuda_n if True else "cpu")
logger.info("device is set for: {}".format(device))

if __name__ == '__main__':
    num_chans = [args.nhid] * (args.levels - 1) + [args.n_output]
    batch_size = args.batch_size
    epochs = args.epochs
    k_size = args.ksize
    dropout = args.dropout
    de = Data_extractor(trj_len=args.trj_len, action_shape=args.action_shape, max_num_trj=100000)
    shadow_trj_ddpg_fp, shadow_end_ddpg_fp = get_trj_end_npy(
        "/home/hossein.aboutalebi/data/PrivAttack-Data/shadow/seed_5/DDPG_Robust_Hopper-v2_20_5")
    shadow_trj_ddpg_p = de.extract(shadow_trj_ddpg_fp, shadow_end_ddpg_fp)
    shadow_trj_bcq_f, shadow_end_bcq_f = get_trj_end_npy(
        "/home/hossein.aboutalebi/data/PrivAttack-Data/shadow/seed_5/BCQ_target_Robust_Hopper-v2_20_5_1000000_compatible")
    shadow_trj_bcq = de.extract(shadow_trj_bcq_f, shadow_end_bcq_f)
    train_dataset_positive = de.create_dataset_TCN_ch(shadow_trj_ddpg_p, shadow_trj_bcq)
    shadow_trj_ddpg_fn, shadow_end_ddpg_fn = get_trj_end_npy(
        "/home/hossein.aboutalebi/data/PrivAttack-Data/shadow/seed_100/DDPG_Robust_Hopper-v2_20_100")
    shadow_trj_ddpg_n = de.extract(shadow_trj_ddpg_fn, shadow_end_ddpg_fn)
    train_dataset_negative = de.create_dataset_TCN_ch(shadow_trj_ddpg_n, shadow_trj_bcq)
    train_dataset = TCNDataset(positive_samples=train_dataset_positive, negative_samples=train_dataset_negative)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.tensor([1, 1]), 2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4,
                              sampler=BalancedBatchSampler(train_dataset))

    # loading the model
    model = TCN(args.action_shape*2, args.n_output, num_chans, dropout=dropout, kernel_size=k_size,trj_len=args.trj_len)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = TCNTrainer(train_loader, model, optimizer, device)
    trainer.run(epochs)
