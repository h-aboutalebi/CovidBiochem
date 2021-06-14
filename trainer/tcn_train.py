import torch.nn as nn
import logging

from utils.metric import Metric
from utils.test import Test

logger = logging.getLogger(__name__)


class TCNTrainer:

    def __init__(self,trainloader,test_loader,model,optimizer,device):
        self.trainloader=trainloader
        self.testloader=test_loader
        self.test = Test()
        self.model=model
        self.device=device
        self.optimizer=optimizer
        self.criterion = nn.CrossEntropyLoss()

    def change_model(self,model):
        self.model=model

    def run(self,epochs):
        metric = Metric()
        for epoch in range(epochs):
            logger.info("epoch number: {}".format(epoch))
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs=inputs.to(self.device)
                self.model.train()
                self.optimizer.zero_grad()
                outputs =self.model(inputs)
                loss = self.criterion(outputs, labels.to(self.device))
                print(loss.item())
                metric.update(outputs, labels.to(self.device), loss)
                loss.backward()
                self.optimizer.step()
                loss.detach_()
            metric.log(epoch)
            metric.reset_params()
            self.test.get_accuracy_test(self.testloader, self.model, self.device, epoch)


