import torch
from torch.utils.data import DataLoader
import os

from data_loader.balanced_sampler import BalancedBatchSampler
from data_loader.data_extractor import Data_extractor
from data_loader.dataset import TCNDataset
from utils.files import get_trj_end_npy


class BuildDataset:

    def __init__(self,data_dir,action_shape,trj_len,batch_size,max_num_trj=100000):
        self.data_dir=data_dir
        self.action_shape=action_shape
        self.trj_len=trj_len
        self.batch_size=batch_size
        self.max_num_trj=max_num_trj

    def load_trainset(self,max_num_trj):
        de = Data_extractor(trj_len=self.trj_len, action_shape=self.action_shape, max_num_trj=max_num_trj)
        shadow_trj_ddpg_fp, shadow_end_ddpg_fp = get_trj_end_npy(
            os.path.join(self.data_dir,"shadow/seed_5/DDPG_Robust_Hopper-v2_20_5"))
        shadow_trj_ddpg_p = de.extract(shadow_trj_ddpg_fp, shadow_end_ddpg_fp)
        shadow_trj_bcq_f, shadow_end_bcq_f = get_trj_end_npy(
            os.path.join(self.data_dir,"shadow/seed_5/BCQ_target_Robust_Hopper-v2_20_5_1000000_compatible"))
        shadow_trj_bcq = de.extract(shadow_trj_bcq_f, shadow_end_bcq_f)
        train_dataset_positive = de.create_dataset_TCN_ch(shadow_trj_ddpg_p, shadow_trj_bcq)
        shadow_trj_ddpg_fn, shadow_end_ddpg_fn = get_trj_end_npy(
            os.path.join(self.data_dir,"shadow/seed_100/DDPG_Robust_Hopper-v2_20_100"))
        shadow_trj_ddpg_n = de.extract(shadow_trj_ddpg_fn, shadow_end_ddpg_fn)
        train_dataset_negative = de.create_dataset_TCN_ch(shadow_trj_ddpg_n, shadow_trj_bcq)
        train_dataset = TCNDataset(positive_samples=train_dataset_positive, negative_samples=train_dataset_negative)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=4,
                                  sampler=BalancedBatchSampler(train_dataset))
        return train_loader

    def load_testset(self,max_num_trj=10000):
        de = Data_extractor(trj_len=self.trj_len, action_shape=self.action_shape, max_num_trj=max_num_trj)
        shadow_trj_ddpg_fp, shadow_end_ddpg_fp = get_trj_end_npy(
            os.path.join(self.data_dir,"target/seed_75/DDPG_Robust_Hopper-v2_200_75"))
        shadow_trj_ddpg_p = de.extract(shadow_trj_ddpg_fp, shadow_end_ddpg_fp)
        shadow_trj_bcq_f, shadow_end_bcq_f = get_trj_end_npy(
            os.path.join(self.data_dir,"target/seed_75/BCQ_target_Robust_Hopper-v2_200_75_1000000_compatible"))
        shadow_trj_bcq = de.extract(shadow_trj_bcq_f, shadow_end_bcq_f)
        test_dataset_positive = de.create_dataset_TCN_ch(shadow_trj_ddpg_p, shadow_trj_bcq)
        shadow_trj_ddpg_fn, shadow_end_ddpg_fn = get_trj_end_npy(
            os.path.join(self.data_dir,"target/seed_700/DDPG_Robust_Hopper-v2_200_700"))
        shadow_trj_ddpg_n = de.extract(shadow_trj_ddpg_fn, shadow_end_ddpg_fn)
        test_dataset_negative = de.create_dataset_TCN_ch(shadow_trj_ddpg_n, shadow_trj_bcq)
        test_dataset = TCNDataset(positive_samples=test_dataset_positive, negative_samples=test_dataset_negative)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=4)
        return test_loader

