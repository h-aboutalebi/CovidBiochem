import torch.nn as nn
import logging
logger = logging.getLogger(__name__)


class TCNTrainer:

    def __init__(self,trainloader,model,optimizer,device):
        self.trainloader=trainloader
        self.model=model
        self.device=device
        self.optimizer=optimizer
        self.criterion = nn.CrossEntropyLoss()

    def change_model(self,model):
        self.model=model

    def run(self,epochs):
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            logger.info("epoch number: {}".format(epoch))
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                self.model.train()
                self.optimizer.zero_grad()
                outputs =self.model(inputs)
                loss = self.criterion(outputs, labels.to(self.device))
                loss.backward()
                self.optimizer.step()


