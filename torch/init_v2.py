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

import warnings
warnings.filterwarnings('ignore', '.*interpolation.*', )

model = timm.create_model('resnet50', pretrained=True, num_classes=1081)

dataset = ImageDataset(r"C:\Users\lulu5\Documents\echantillon_2", 
                       transform=create_transform(224, is_training=True))
loader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=16)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

args = SimpleNamespace()
args.weight_decay = 0
args.lr = 1e-4
args.opt = 'sgd' #'lookahead_adam' to use `lookahead`
args.momentum = 0.9

optimizer = create_optimizer(args, model)
criterion = CrossEntropyLoss()

scheduler = MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

for epoch in range(1):
    print(epoch)
    running_loss = 0
    for inputs, targets in tqdm(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
    print('Loss:', running_loss/len(loader))
    scheduler.step()

torch.save(model.state_dict(), 'model.torch')