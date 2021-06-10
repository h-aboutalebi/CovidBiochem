import torch.optim as optim
import torch.nn as nn

class TCNTrainer:

    def __init__(self,trainloader):
        self.trainloader=trainloader
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


    def run(self,epochs):
        for epoch in range(epochs):
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data

