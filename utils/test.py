import torch
import logging
import time

from utils.file_manager import File_Manager
from utils.tensor_writer import Tensor_Writer

logger = logging.getLogger(__name__)


class Test():
    initial_time = 0

    def __init__(self):
        self.results_test = {"accuracy": [], "epoch": []}
        self.results_valid = {"accuracy": [], "epoch": []}

    def get_accuracy_test(self, test_loader, model, device, epoch):
        model.eval()
        current_time = time.time()
        correct, total = self.cal_correct(test_loader, model, device)
        passed_time = time.time() - current_time
        Tensor_Writer.add_scalar("Time_test", passed_time, epoch)
        Tensor_Writer.add_scalar("Accuracy_test", correct / total, epoch)
        self.results_test["accuracy"].append(correct / total)
        self.results_test["epoch"].append(epoch)
        File_Manager.write("/results_test.pkl", self.results_test)
        logger.info(
            'Test Acuraccy at epoch [{}]: accuracy: {:.4f} time: {:.2f}'.format(epoch + 1, correct / total,
                                                                                passed_time))
        logger.info("Total elapsed time so far: {} min".format((time.time() - Test.initial_time) / 60))
        return correct / total

    def cal_correct(self, loader, model, device):
        model.eval()
        correct = 0
        total = 0
        for data in loader:
            images, labels = data
            outputs = model.call(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
        return correct, total
