# !pip install timm 
import os
import re
import numpy as np

import timm
import timm
import torch
from timm.optim.optim_factory import create_optimizer
from types import SimpleNamespace
from timm.data.transforms_factory import create_transform
from timm.data import ImageDataset
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

#Data
import warnings
warnings.filterwarnings('ignore', '.*interpolation.*', )

model = timm.create_model('inception_v3', pretrained=True, num_classes=1081)
# model
dataset = ImageDataset('/home/data/challenge_2022_miashs/train', 
                       transform=create_transform(224, is_training=True))
weights = torch.FloatTensor(np.load("/home/weights_crossentropy.npy"))

loader = DataLoader(dataset, batch_size=96, shuffle=True, num_workers=16)

#Cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

weights = weights.to(device)
model.to(device)


#Modèle
args = SimpleNamespace()
args.weight_decay = 1e-4
args.lr = 0.01
args.opt = 'sgd' #'lookahead_adam' to use `lookahead`
args.momentum = 0.9

optimizer = create_optimizer(args, model)
criterion = CrossEntropyLoss(weight = weights)

scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

training_stats = []

#Epoch
for epoch in range(60):
    print("Epoch:", epoch)
    if epoch % 2 == 0 and epoch > 10:
        torch.save(model.state_dict(), 'model_inception_v3_cewl_e{epoch}.torch')

    running_loss = 0
    for inputs, targets in tqdm(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
    Loss =  running_loss/len(loader)

    print('Loss:', Loss)

    
    training_stats.append(
        {
            'epoch': epoch + 1,
            'Training Loss': Loss
        })

    scheduler.step()

torch.save(model.state_dict(), 'model_inception_v3_cewl.torch')
np.save(np.array("loss_save.npy", training_stats))
