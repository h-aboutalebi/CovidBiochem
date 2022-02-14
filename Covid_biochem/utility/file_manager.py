import pickle
import torch
import os


class File_Manager():
    file_path = None

    @staticmethod
    def set_file_path(file_path):
        File_Manager.file_path = file_path

    @staticmethod
    def write(name, obj):
        file_name = File_Manager.get_file_name(name)
        open(file_name, 'w').close()
        with open(file_name, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def save_torch_model(name, model):
        file_name = File_Manager.get_file_name(name)
        open(file_name, 'w').close()
        torch.save(model.state_dict(), file_name)

    @staticmethod
    def remove_file(name):
        file_name = File_Manager.get_file_name(name)
        os.remove(file_name)

    @staticmethod
    def get_file_name(name):
        if (name[0] != "/"):
            name = "/" + name
        file_name = File_Manager.file_path + name
        return file_name