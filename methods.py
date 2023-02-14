from PIL import Image
import numpy as np
import torch
from dataset import MDataset
from torchvision import transforms
from tiara import Tiara
from utils import load_glove


class Solver():
    def __init__(self):
        pass

    def set_model(self, model):
        self.model = model

    def get_dataset(self):
        return MDataset(self.cur_dataset)

    def get_class(self, item):
        return self.env.get_class(item)

    def increment_dataset(self):
        raise NotImplementedError


class RandomMethod(Solver):
    def __init__(self, init_dataset, poslabels, env, budget_per_round=1):
        super(RandomMethod, self).__init__()
        self.cur_dataset = init_dataset
        self.used = set()
        self.budget_per_round = budget_per_round
        self.poslabels = poslabels
        self.env = env

    def increment_dataset(self):
        budget_per_round = self.budget_per_round
        while budget_per_round > 0:
            item = self.env.random_item()
            if item not in self.used:
                c = self.get_class(item)
                data = (self.env.get_path(item), c)
                self.cur_dataset.append(data)
                self.used.add(item)
                budget_per_round -= 1
        return MDataset(self.cur_dataset)


class SmallExact(Solver):
    def __init__(self, init_dataset, poslabels, env, budget_per_round=1, poolsize=1000, device='cpu'):
        super(SmallExact, self).__init__()
        self.cur_dataset = init_dataset
        self.used = set()
        self.budget_per_round = budget_per_round
        self.poslabels = poslabels
        self.env = env
        self.cand = [self.env.random_item() for i in range(poolsize)]
        self.device = device

    def set_model(self, model):
        self.model = model
        self.acquisition = MaxEntAcquisition(model, self.device)

    def increment_dataset(self):
        cand = []
        for r in self.cand:
            if r in self.used:
                continue
            score = self.acquisition(self.env.get_path(r))
            cand.append((score, r))
        cand = sorted(cand)[::-1]
        for score, item in cand[:self.budget_per_round]:
            c = self.get_class(item)
            data = (self.env.get_path(item), c)
            self.used.add(item)
            self.cur_dataset.append(data)
        return MDataset(self.cur_dataset)


class MaxEntAcquisition():
    def __init__(self, model, device):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.244, 0.225]),
        ])
        self.model = model
        self.device = torch.device(device)
        self.ps = []

    def __call__(self, path):
        try:
            img = Image.open(path).convert('RGB')
        except BaseException:
            img = Image.new('RGB', (256, 256))
        data = self.transform(img)
        data = data.to(self.device)
        output = self.model(data.unsqueeze(0))[0].cpu()
        p = torch.softmax(output, dim=0)
        self.ps.append(float(p[1]))
        ent = float((- p * torch.log2(p)).sum())
        return np.exp(4 * ent)


class Seafaring(Solver):
    def __init__(self, init_dataset, env, budget_per_round=1, tiara_budget=1000, alpha=1, poslabels=None, device='cpu'):
        self.cur_dataset = init_dataset
        self.budget_per_round = budget_per_round
        self.tiara_budget = tiara_budget
        self.glove = load_glove(300, 6)
        self.downloaded = set()
        self.used = set()
        self.alpha = alpha
        self.poslabels = poslabels
        self.device = device
        self.env = env
        self.maxprob = []

    def set_model(self, model):
        self.model = model
        self.acquisition = MaxEntAcquisition(model, self.device)
        self.env.set_acquisition(self.acquisition)
        init_tags = self.env.init_tags()
        self.tiara = Tiara(self.env, self.tiara_budget, word_embedding=self.glove, alpha=self.alpha, init_tags=init_tags, verbose=True)

    def get_dataset(self):
        return MDataset(self.cur_dataset)

    def get_class(self, item):
        c = self.env.get_class(item)
        print('Selected', self.env.get_path(item), c, self.env.item_to_tag(item))
        return c

    def increment_dataset(self):
        self.acquisition.ps = []
        item_history = self.tiara.optimize()
        now_items = set([item for score, item in item_history])
        cand = []
        for r in self.downloaded - now_items:
            if r in self.used:
                continue
            score = self.acquisition(self.env.get_path(r))
            cand.append((score, r))
        self.downloaded |= now_items
        cand = cand + item_history
        cand = sorted(cand)[::-1]
        for score, item in cand[:self.budget_per_round]:
            c = self.get_class(item)
            data = (self.env.get_path(item), c)
            self.used.add(item)
            self.cur_dataset.append(data)
        mp = max(self.acquisition.ps)
        print('maxprob', mp)
        self.maxprob.append(mp)
        return MDataset(self.cur_dataset)
