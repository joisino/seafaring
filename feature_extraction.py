import pickle

import os
import sys

import numpy as np

import torch
import torchvision
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from dataset import MDataset

import random


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda')

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

train_nodes, eval_nodes = get_graph_node_names(model)

model = create_feature_extractor(model, {
    'flatten': 'flatten'
})

model.to(device)
model.eval()


with open('./openimage_tag_to_image_test.pickle', 'rb') as f:
    tag_to_item_dict = pickle.load(f)
existing_images = list(map(lambda x: x.split('.')[0], os.listdir('imgs')))
existing_images = [('imgs/' + item + '.jpg', 0) for item in existing_images]
dataset = MDataset(existing_images)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

li = []
paths = []
for batch_idx, (data, target, idx) in enumerate(loader):
    data, target = data.to(device), target.to(device)
    res = model(data)
    print(batch_idx, len(loader), file=sys.stderr)
    li.append(res['flatten'].cpu().detach())
    paths += [dataset.paths[i] for i in idx]

li = torch.concat(li)

print(li.shape)

with open('feature.pt', 'wb') as f:
    torch.save(li, f)

with open('paths.pickle', 'wb') as f:
    pickle.dump(paths, f)
