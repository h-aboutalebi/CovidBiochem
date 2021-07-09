import numpy as np
import random

class Data_extractor:

    def __init__(self,trj_len=10,action_shape=6,max_num_trj=100000):
        self.action_shape=action_shape
        self.trj_len=trj_len
        self.max_num_trj=max_num_trj

    def update_trj_len(self,trj_len):
        self.trj_len=trj_len

    def extract(self,file_trj,file_end):
        trj=np.load(file_trj)
        end=np.load(file_end)
        final_list= self.create_action_trj(trj,end)
        return final_list

    def create_action_trj(self,trj,end):
        final_list=[]
        index = 0
        for i in range(len(end)):
            if(i+1>self.max_num_trj):
                break
            trajectory=[]
            for j in range(index,end[i]+1):
                if(len(trajectory)>=self.trj_len):
                    continue
                trajectory.append(trj[j])
            if(len(trajectory)<self.trj_len):
                self.pad_trajectory(trajectory)
            index=end[i]+1
            final_list.append(np.array(trajectory).transpose())
        random.shuffle(final_list)
        return np.array(final_list)

    def pad_trajectory(self,trajectory):
        dif=self.trj_len-len(trajectory)
        last_element=trajectory[-1]
        for i in range(dif):
            trajectory.append(np.ones(self.action_shape))

    def create_dataset_TCN_ch(self, trj1, trj2):
        return np.concatenate((trj1,trj2),axis=len(trj1.shape)-2)




