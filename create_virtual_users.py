import pickle

import os

import numpy as np

import torch
import torch.nn.functional as F

import random


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

tags = [
    'Skyscraper',
    'Fox',
    'Baby',
]

n = 10

with open('./openimage_tag_to_image_test.pickle', 'rb') as f:
    tag_to_item_dict = pickle.load(f)
existing_images = set(map(lambda x: x.split('.')[0], os.listdir('imgs')))

with open('feature.pt', 'rb') as f:
    li = torch.load(f)

with open('paths.pickle', 'rb') as f:
    paths = pickle.load(f)


def item_to_id(item):
    path = 'imgs/' + item + '.jpg'
    return paths.index(path)


users = []
data = []
for tag in tags:
    items = list(set(tag_to_item_dict[tag]) & existing_images)
    items = np.random.choice(items, size=n, replace=False)
    ids = torch.LongTensor([item_to_id(item) for item in items])
    embeddings = li[ids]
    users.append(embeddings[None, ...])
    c = torch.concat([F.cosine_similarity(li, embeddings[i])[None, ...].cpu() for i in range(n)])
    print(c.shape)
    c = c.max(0)[0]
    print(c.shape)
    data.append([(float(c[i]), paths[i]) for i in range(len(paths))])

users = torch.concat(users)

print(users.shape)

with open('virtual_user_source.pt', 'wb') as f:
    torch.save(users, f)

with open('virtual_user_source.pickle', 'wb') as f:
    pickle.dump(data, f)
