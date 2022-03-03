import torch.optim as optim
import torch.nn as nn
import logging
import torch
from tqdm import tqdm
from time import sleep
from models_parent.swin_transformer import swin_t
from torch.optim.lr_scheduler import MultiStepLR

from utility.utils import pytorch_accuracy

logger = logging.getLogger(__name__)


class Swin:

    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        model = swin_t(num_classes=self.num_classes)
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def train(self, trainloader, testloader, epochs, lr, milestones, gamma, momentum,
              weight_decay):
        optimizer = optim.SGD(self.model.parameters(),
                              lr=lr,
                              momentum=momentum,
                              weight_decay=weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        logger.info("Training has started for SwinTranssformer ...")
        for epoch in range(1, epochs + 1):
            # import ipdb;ipdb.set_trace()
            total_loss = 0
            with tqdm(trainloader, unit="batch") as tepoch:
                total_loss = 0
                total_correct = 0
                total = 0
                for i, data in enumerate(tepoch, 1):
                    tepoch.set_description(f"Epoch {epoch}")
                    self.model.train()
                    optimizer.zero_grad()
                    inputs, labels = data
                    batch_size = inputs.shape[0]
                    outputs = self.model(inputs.to(self.device))
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == labels.to(self.device)).sum().item()
                    loss = self.criterion(outputs, labels.to(self.device))
                    loss.backward()
                    optimizer.step()
                    loss.detach_()
                    #progress bar:
                    total_loss += loss.item() / batch_size
                    total_correct += correct
                    total += batch_size
                    accuracy = correct / batch_size
                    tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
                    sleep(0.1)
            scheduler.step()
            logger.info(
                "epoch {}. Average Train Loss: {}, Average Train Accuracy {}".format(
                    epoch, round(total_loss, 3), round(total_correct / total, 3)))
            if (epoch % 10 == 0):
                self.predict(testloader)
                # File_Manager.save_torch_model("weights.pth", self.model)

    def predict(self, testloader):
        with torch.no_grad():
            correct = 0
            total = 0
            for i, data in enumerate(testloader, 1):
                self.model.eval()
                inputs, labels = data
                outputs = self.model(inputs.to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels.to(self.device)).sum().item()
                total += len(labels)
            accuracy = correct / total
        logger.info("Accuracy on test set: {}".format(accuracy))
