import torch.nn as nn
import logging
logger = logging.getLogger(__name__)


class TCNTrainer:

    def __init__(self,trainloader,test_loader,model,optimizer,device):
        self.trainloader=trainloader
        self.test_loader=test_loader
        self.model=model
        self.device=device
        self.optimizer=optimizer
        self.criterion = nn.CrossEntropyLoss()

    def change_model(self,model):
        self.model=model

    def run(self,epochs):
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
                loss.backward()
                self.optimizer.step()


