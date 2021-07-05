import re
import os

def get_trj_end_npy_target(dir):
    regex_action = re.compile('(^BCQ_target.*action.npy$)|(^target_Robust.*action.npy$)')
    regex_end = re.compile('(^BCQ_target.*trajectory_end_index.npy$)|(^target_Robust.*trajectory_end_index.npy$)')
    action_file = None
    end_file = None
    for root, dirs, files in os.walk(dir):
        for file in files:
            if regex_action.match(file):
                action_file = file
            if regex_end.match(file):
                end_file = file
    return os.path.join(dir, action_file), os.path.join(dir, end_file)

def get_trj_end_npy_buffer(dir):
    regex_action = re.compile('(^DDPG_Robust.*action.npy$)|(^Robust.*action.npy$)')
    regex_end = re.compile('(^DDPG_Robust.*trajectory_end_index.npy$)|(^Robust.*trajectory_end_index.npy$)')
    action_file=None
    end_file=None
    for root, dirs, files in os.walk(dir):
        for file in files:
            if regex_action.match(file):
                action_file=file
            if regex_end.match(file):
                end_file = file
    return os.path.join(dir,action_file),os.path.join(dir,end_file)