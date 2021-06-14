import torch
import time
import logging
import pickle

from utils.file_manager import File_Manager
from utils.tensor_writer import Tensor_Writer

logger = logging.getLogger(__name__)



# This class logs and reports loss and accuracy
class Metric():

    def __init__(self):
        self.total_normal = 0
        self.n_correct_normal = 0
        self.current_time = time.time()
        self.loss_normal = 0
        self.results = {"epoch": [], "time": [], "loss_normal": [], "accuracy_normal": []}

    def update(self, outputs, labels, loss):
        _, predicted = torch.max(outputs.data, 1)
        self.n_correct_normal += (predicted == labels).sum().item()
        self.loss_normal += loss
        self.total_normal += labels.size(0)


    def get_accuracy_normal(self):
        return self.n_correct_normal / self.total_normal


    def get_loss_normal(self):
        return (self.loss_normal / self.total_normal).item()


    def reset_params(self):
        self.total_normal = 0
        self.n_correct_normal = 0
        self.loss_normal = 0
        self.current_time = time.time()

    def log(self, epoch):
        passed_time = time.time() - self.current_time
        # self.save_results_pkl(epoch, passed_time)
        # self.log_tensor(epoch, passed_time)
        logger.info(
            '[{}] accuracy normal: {:.4f} loss normal: {:.6f} time: {:.2f}'.format(
                epoch + 1,
                self.get_accuracy_normal(),
                self.get_loss_normal(),
                passed_time))

    def log_tensor(self, epoch, passed_time):
        Tensor_Writer.add_scalar("Accuracy_train", self.get_accuracy_normal(), epoch)
        Tensor_Writer.add_scalar("Loss_train", self.get_loss_normal(), epoch)
        Tensor_Writer.add_scalar("Time_train", passed_time, epoch)

    def save_results_pkl(self, epoch, passed_time):
        self.results["epoch"].append(epoch)
        self.results["time"].append(passed_time)
        self.results["loss_normal"].append(self.get_loss_normal())
        self.results["accuracy_normal"].append(self.get_accuracy_normal())
        File_Manager.write("/results_train.pkl", self.results)