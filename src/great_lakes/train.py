import os
import torch.nn as nn
import torch
import wandb
from models import AndreaNet
from torchvision.models import resnet101, resnet18
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from configs import Lake, Label, BaseConfig, TrainConfig, TestConfig, LABEL_CONVERTER
from datasets import LakesRandom
from utils import wandb_logging

wandb.init()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}.")

data_directory = "/home/dsola/repos/PGA-Net/data/patch20"
lakes = [Lake.ERIE, Lake.ONTARIO]
label = Label.BINARY
weight = torch.Tensor([1, 1]).to(device=device)
batch_size = 10
epochs = 50
train_epoch_size, test_epoch_size = 500, 250
img_paths = None  # ['/home/dsola/repos/PGA-Net/data/patch20/20140111_3_20_HH_HV_patches_erie.npy']
ice_con_paths = None  # ['/home/dsola/repos/PGA-Net/data/patch20/20140111_patch20_stride3_erie.pkl']

base_config = BaseConfig(data_directory=data_directory, lakes=lakes, label=label)

train_config = TrainConfig(*base_config, batch_size=batch_size, epochs=epochs,
                           epoch_size=train_epoch_size, weight=weight, device=device)
test_config = TestConfig(*base_config, batch_size=batch_size, epoch_size=test_epoch_size)

train_set = LakesRandom(train_config, imgs_paths=img_paths, ice_con_paths=ice_con_paths)
train_loader = DataLoader(train_set, batch_size=train_config.batch_size, shuffle=True)

test_set = LakesRandom(test_config, imgs_paths=img_paths, ice_con_paths=ice_con_paths)
test_loader = DataLoader(test_set, batch_size=test_config.batch_size, shuffle=True)

net = AndreaNet(classes=LABEL_CONVERTER[label.value]['classes'])
# net = resnet101(num_classes=classes=LABEL_CONVERTER[label]['classes'])
# net = resnet18(num_classes=classes=LABEL_CONVERTER[label]['classes'])
# net.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
softmax = nn.Softmax(dim=1)
net = net.to(device=device)
wandb.watch(net)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(train_config.epochs):
    print(f"Epoch {epoch}.")
    for i, train_batch in enumerate(train_loader):
        print(f"Training batch {i}.")
        net.train()
        inputs, labels = train_batch
        optimizer.zero_grad()
        outputs = net(inputs.to(device=device, dtype=torch.float32))
        outputs = softmax(outputs)
        loss = train_config.criterion(outputs, torch.argmax(labels, dim=1).to(device=device))
        metric = test_config.metric(outputs, labels.to(device=device, dtype=torch.float32))
        wandb.log({"Train Loss": loss})
        wandb.log({f"Train {test_config.metric.name}": metric})
        loss.backward()
        optimizer.step()

        if i % 10 == 9:
            print("Testing.")
            net.eval()
            test_loss, test_metric = [], []
            x_vals, y_vals = [], []
            for test_batch in test_loader:
                inputs, labels = test_batch
                with torch.no_grad():
                    outputs = net(inputs.to(device=device, dtype=torch.float32))
                    outputs = softmax(outputs)
                loss = train_config.criterion(outputs, torch.argmax(labels, dim=1).to(device=device))
                metric = test_config.metric(outputs, labels.to(device=device, dtype=torch.float32))
                x_vals += torch.argmax(outputs, dim=1).tolist()
                y_vals += torch.argmax(labels, dim=1).tolist()
                test_loss.append(loss.item())
                test_metric.append(metric.item())
            wandb.log({"Test Loss": np.mean(test_loss)})
            wandb_logging(x_vals, y_vals, classes=LABEL_CONVERTER[label.value]['classes'])
    try:
        os.mkdir('../checkpoints/')
    except OSError:
        pass
    torch.save(net.state_dict(), '../checkpoints/' + f'epoch{epoch + 1}.pth')
