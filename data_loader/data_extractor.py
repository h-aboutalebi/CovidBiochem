import numpy as np

class Data_extractor:

    def __init__(self,file_trj,file_end):
        pass

    @staticmethod
    def extract(file_trj,file_end):
        trj=np.load(file_trj)
        end=np.load(file_end)
        final_list= Data_extractor.create_action_trj(trj,end)

    @staticmethod
    def create_action_trj(trj,end):
        pass



