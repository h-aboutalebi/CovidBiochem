import torch.optim as optim
import torch.nn as nn
import logging
from models_parent.swin_transformer import swin_t
from torch.optim.lr_scheduler import MultiStepLR

from utility.utils import pytorch_accuracy

logger = logging.getLogger(__name__)


class Swin:

    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        model = swin_t(num_classes=self.num_classes, channels=1)
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
            for i, data in enumerate(trainloader, 1):
                self.model.train()
                optimizer.zero_grad()
                inputs, labels = data
                outputs = self.model(inputs.to(self.device))
                loss = self.criterion(outputs, labels.to(self.device))
                total_loss += loss.item()
                loss.backward()
                loss.detach_()
            logger.info("epoch {}. loss: {}".format(epoch, total_loss))
            scheduler.step()
            if (epoch % 10 == 0):
                self.predict(testloader)
                # File_Manager.save_torch_model("weights.pth", self.model)

    def predict(self, testloader):
        labels_list = []
        outputs_list = []
        for i, data in enumerate(testloader, 1):
            self.model.test()
            inputs, labels = data
            outputs = self.model(inputs.to(self.device))
            labels_list.append(labels)
            outputs_list.append(outputs)
        accuracy = pytorch_accuracy(labels_list, outputs_list)
        logger.info("Accuracy on test set: {}".format(accuracy))
