import torch
import pytorch_lightning as pl
from torch import nn
from sam import SAM
import torch.nn.functional as F

class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        # Reduced the number of output channels
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Reduced the input features of the fully connected layer
        self.fc1 = nn.Linear(32 * 16 * 16, 64) # reduced from 128 to 64
        self.fc2 = nn.Linear(64, 10) # output layer remains the same

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16) # match the number of output channels from conv2
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SmallNetLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        #self.model = torchvision.models.resnet18()
        self.model = SmallNet()
        #num_ftrs = self.model.fc.in_features
        #self.model.fc = nn.Linear(num_ftrs, 10)  # CIFAR10 has 10 classes
        self.loss = torch.nn.CrossEntropyLoss()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.automatic_optimization = False
        if self.config["SAM"]:
            self.optimizer = SAM(self.model.parameters(), self.config["optimizer"], **self.config["parameters"])
        else:
            self.optimizer = self.config["optimizer"](self.model.parameters(), **self.config["parameters"])
        
        
        
    def forward(self, x):
        return self.model(x)
    
    
    def training_step(self, batch, batch_idx):
        
        optimizer = self.optimizers()
        x, y = batch
        
        def closure():
            loss = self.loss(self.model(x), y)
            optimizer.zero_grad()
            self.manual_backward(loss)
            return loss
        
        loss = self.loss(self.model(x), y)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        optimizer.zero_grad()
        self.manual_backward(loss)

        if self.config["SAM"]:
            optimizer.step(closure)
        else:
            optimizer.step()
        
        return loss
        
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    
    def configure_optimizers(self):
        # takes the constructor of the optimizer as input (for example torch.optim.Adam)
        if self.config["SAM"]:
            optimizer = SAM(self.model.parameters(), self.config["optimizer"], **self.config["parameters"])
        else:
            optimizer = self.config["optimizer"](self.model.parameters(), **self.config["parameters"])
        return optimizer