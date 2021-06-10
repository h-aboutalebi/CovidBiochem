from data_loader.data_extractor import Data_extractor

if __name__ == '__main__':
    de = Data_extractor(trj_len=10, action_shape=3, max_num_trj=100000)
    shadow_trj_ddpg_f = "/home/hossein.aboutalebi/data/PrivAttack-Data/shadow/seed_5/DDPG_Robust_Hopper-v2_20_5_action.npy"
    shadow_end_ddpg_f = "/home/hossein.aboutalebi/data/PrivAttack-Data/shadow/seed_5/DDPG_Robust_Hopper-v2_20_5_trajectory_end_index.npy"
    shadow_trj_ddpg = de.extract(shadow_trj_ddpg_f, shadow_end_ddpg_f)
    shadow_trj_bcq_f = "/home/hossein.aboutalebi/data/PrivAttack-Data/shadow/seed_5/BCQ_target_Robust_Hopper-v2_20_5_1000000_compatible_action.npy"
    shadow_end_bcq_f = "/home/hossein.aboutalebi/data/PrivAttack-Data/shadow/seed_5/BCQ_target_Robust_Hopper-v2_20_5_1000000_compatible_trajectory_end_index.npy"
    shadow_trj_bcq = de.extract(shadow_trj_ddpg_f, shadow_end_ddpg_f)
    train_dataset = de.create_dataset_TCN_ch(shadow_trj_ddpg, shadow_trj_bcq)
    print("yes")
