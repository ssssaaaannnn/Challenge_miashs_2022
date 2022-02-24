import timm
import torch
import os
from timm.data import ImageDataset, create_transform
from torch.utils.data import DataLoader
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', '.*interpolation.*', )

dataset = ImageDataset('/home/data/challenge_2022_miashs/train', transform=create_transform(224))
idx_to_class = {v: k for k, v in dataset.parser.class_to_idx.items()}

test_set = ImageDataset('/home/data/challenge_2022_miashs/test', transform=create_transform(224))
loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=16)

model = timm.create_model('resnet50', num_classes=1081)
model.load_state_dict(torch.load('model.torch'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.eval()

answers = []

idx = 0
filenames = test_set.parser.filenames()
for inputs, labels in tqdm(loader):
    inputs, labels = inputs.to(device), labels.to(device)

    with torch.set_grad_enabled(False):
        prediction = model(inputs).argmax(dim=1)

        for a in prediction:
            predicted_class = a.cpu().numpy()
            answers.append(
                (os.path.basename(os.path.splitext(filenames[idx])[0]), idx_to_class[int(predicted_class)])
            )
            idx += 1

with open('predictions.csv', 'w') as f:
    f.write('Id,Category\n')
    for k, p in answers:
        f.write('{},{}\n'.format(k, p))