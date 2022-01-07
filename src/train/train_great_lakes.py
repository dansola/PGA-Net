import os
import torch.nn as nn
import torch
import wandb
from src.models.basic_cnn import AndreaNet
from torchvision.models import resnet101, resnet18
import torch.optim as optim
from src.datasets.great_lakes import Lake, LakesRandom, BaseConfig, TrainConfig, TestConfig
from torch.utils.data import DataLoader
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score

wandb.init()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}.")

data_directory = "/home/dsola/repos/PGA-Net/data/patch20"
lake = Lake.erie
class_weighted = False
binary_labels = True
batch_size = 10
epochs = 10
train_epoch_size, test_epoch_size = 500, 250

base_config = BaseConfig(data_directory=data_directory, lake=lake, binary_labels=binary_labels)

train_config = TrainConfig(*base_config, batch_size=batch_size, epochs=epochs,
                           epoch_size=train_epoch_size, class_weighted=class_weighted, device=device)
test_config = TestConfig(*base_config, batch_size=batch_size, epoch_size=test_epoch_size)

train_set = LakesRandom(train_config)
train_loader = DataLoader(train_set, batch_size=train_config.batch_size, shuffle=True)

test_set = LakesRandom(test_config)
test_loader = DataLoader(test_set, batch_size=test_config.batch_size, shuffle=True)

net = AndreaNet()
# net = resnet101(num_classes=1)
# net = resnet18(num_classes=1)
# net.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
net = net.to(device=device)
wandb.watch(net)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(train_config.epochs):
    print(f"Epoch {epoch}.")
    for i, train_batch in enumerate(train_loader):
        print(f"Training batch {i}.")
        net.train()
        inputs, labels = train_batch
        # optimizer.zero_grad()
        outputs = net(inputs.to(device=device, dtype=torch.float32))
        loss = train_config.criterion(outputs.squeeze(1), labels.to(device=device, dtype=torch.float32))
        metric = test_config.metric(outputs.squeeze(1), labels.to(device=device, dtype=torch.float32))
        wandb.log({"Train Loss": loss})
        wandb.log({f"Train {test_config.metric.name}": metric})
        optimizer.zero_grad()
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
                loss = train_config.criterion(outputs.squeeze(1), labels.to(device=device, dtype=torch.float32))
                metric = test_config.metric(outputs.squeeze(1), labels.to(device=device, dtype=torch.float32))
                x_vals += ((outputs>0.5)*1).squeeze(1).cpu().detach().tolist()
                y_vals += labels.tolist()
                test_loss.append(loss.item())
                test_metric.append(metric.item())
            wandb.log({"Test Loss": np.mean(test_loss)})
            wandb.log({f"Test {test_config.metric.name}": np.mean(test_metric)})
            wandb.log({"accuracy_score": accuracy_score(x_vals, y_vals)})
            wandb.log({"precision_score": precision_score(x_vals, y_vals)})
            wandb.log({"recall_score": recall_score(x_vals, y_vals)})
            wandb.log({"f1_score": f1_score(x_vals, y_vals)})
            # wandb.log({"roc_auc_score": roc_auc_score(x_vals, y_vals)})
    try:
        os.mkdir('../checkpoints/')
    except OSError:
        pass
    torch.save(net.state_dict(), '../checkpoints/' + f'epoch{epoch + 1}.pth')
