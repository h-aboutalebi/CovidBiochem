from data_loader.data_extractor import Data_extractor
from utils.files import get_trj_end_npy
from data_loader.dataset import TCNDataset
import numpy as np

if __name__ == '__main__':
    de = Data_extractor(trj_len=10, action_shape=3, max_num_trj=100000)
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
