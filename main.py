import pickle

import os
import json

import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

import sklearn.metrics

from dataset import MDataset
from methods import RandomMethod, SmallExact, Seafaring
from environments import get_env

import argparse
import random


def measure(method, test_dataset, n_classes, n_round=100, epoch_per_round=10, batchsize=16, device='cpu'):
    train_dataset = method.get_dataset()
    accs = []
    aucs = []
    for i_round in range(n_round + 1):
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(512, n_classes)
        method.set_model(model)
        device = torch.device(device)
        model.to(device)
        ns = train_dataset.labelcount()
        print('ns', ns)
        weight = ns.sum() / (ns + 1e-9)
        weight = torch.FloatTensor(weight)
        weight = weight.to(device)
        criterion = nn.CrossEntropyLoss(weight)
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
        for epoch in range(epoch_per_round):
            print(i_round, epoch, len(train_dataset), end=' ')
            model.train()
            losss = []
            for batch_idx, (data, target, idx) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                losss.append(float(loss))
            loss = np.mean(losss)
            print(loss, end='\r')

        model.eval()
        acc = 0
        res = []
        preds = []
        for batch_idx, (data, target, idx) in enumerate(test_dataset):
            data = data.to(device)
            output = model(data.unsqueeze(0))[0].cpu()
            pred = output.argmax()
            preds.append(int(pred))
            acc += int(pred == target)
            res.append((float(torch.softmax(output, dim=0)[1]), target))
        fpr, tpr, thresholds = sklearn.metrics.roc_curve([c for a, c in res], [a for a, c in res], pos_label=1)
        auc = sklearn.metrics.auc(fpr, tpr)
        aucs.append(auc)
        acc = acc / len(test_dataset)
        accs.append(acc)
        print()
        print(acc, auc)
        print(np.bincount(preds, minlength=n_classes))
        if i_round < n_round:
            train_dataset = method.increment_dataset()
    return [accs, aucs, model]


def build_data(n, poslabels, tag_to_image_filepath):
    with open(tag_to_image_filepath, 'rb') as f:
        tag_to_item_dict = pickle.load(f)
    existing_images = set(map(lambda x: x.split('.')[0], os.listdir('imgs')))
    tag_to_item_dict = {
        tag: list(filter(lambda x: x in existing_images, li)) for tag, li in tag_to_item_dict.items()
    }
    testlabels = np.random.randint(len(poslabels), size=n)
    n_testlabels = np.bincount(testlabels, minlength=len(poslabels))
    pos = sum([[('imgs/' + x + '.jpg', 1) for x in tag_to_item_dict[t][-n_testlabels[i]:]] for i, t in enumerate(poslabels)], [])
    tags = [tag for tag in tag_to_item_dict.keys() if len(tag_to_item_dict[tag]) >= 1]
    neg_tags = [tags[i] for i in np.random.randint(len(tags), size=n)]
    neg = [('imgs/' + tag_to_item_dict[tag][0] + '.jpg', 0) for tag in neg_tags]
    data = pos + neg
    return data


def build_user_vitual_user(n, user, test, threshold):
    with open('virtual_user_source.pickle', 'rb') as f:
        li = pickle.load(f)[user]

    pos = list(map(lambda x: (x[1], 1), filter(lambda x: x[0] > threshold, li)))
    neg = list(map(lambda x: (x[1], 0), filter(lambda x: x[0] < threshold, li)))

    print(len(pos), len(neg))

    if test == -1:
        data = pos[:n] + neg[:n]
    else:
        data = pos[test:][-n:] + neg[test:][-n:]

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', choices=['Seafaring', 'Random', 'SmallExact'], default='MaxMax')
    parser.add_argument('--env', choices=['OpenImage', 'Flickr'], default='OpenImage')
    parser.add_argument('--apikey', type=str, default=None, help='API key of Flickr. Valid only for Flickr env.')
    parser.add_argument('--tiara_budget', type=int, default=1000)
    parser.add_argument('--budget_per_round', type=int, default=1)
    parser.add_argument('--initdata', type=int, default=1, help='NumSizeber of the initial labelled data.')
    parser.add_argument('--testdata', type=int, default=100, help='Size of the test dataset.')
    parser.add_argument('--nround', type=int, default=100, help='Number of rounds of active learning.')
    parser.add_argument('--nepoch', type=int, default=100, help='Number of epochs for training the target model.')
    parser.add_argument('--alpha', type=float, default=1.0, help='The alpha parameter of Tiara.')
    parser.add_argument('--threshold', type=float, default=0.6, help='Thoreshold of Positive data. Valid only for Flickr env.')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--poolsize', type=int, default=1000, help='Size of the poolsize for SmallExact method')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--poslabels', type=str, nargs='+', default=['Cat'], help='List of positive labels. Valid only for OpenImage env.')
    parser.add_argument('--user', type=int, default=0, help='Id of the target virtual user, i.e., category. Valid only for Flickr env. See also create_virtual_users.py.')
    parser.add_argument('--initialtags', type=str, default=None, help='Path to the tag file.')
    parser.add_argument('--resdir', type=str, default='results')
    args = parser.parse_args()
    if not os.path.exists(args.resdir):
        os.makedirs(args.resdir)
    with open(f'{args.resdir}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(vars(args))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    n_classes = 2
    if args.env == 'OpenImage':
        test_data = build_data(args.testdata, args.poslabels, 'openimage_tag_to_image_test.pickle')
        test_data = MDataset(test_data)
        init_data = build_data(args.initdata, args.poslabels, 'openimage_tag_to_image.pickle')
    else:
        test_data = build_user_vitual_user(args.testdata, args.user, -1, args.threshold)
        test_data = MDataset(test_data)
        init_data = build_user_vitual_user(args.initdata, args.user, args.testdata, args.threshold)
    env = get_env(args.env, args.apikey, args.initialtags, args.poslabels, args.user, args.device, args.threshold)
    if args.method == 'Random':
        method = RandomMethod(init_data, budget_per_round=args.budget_per_round, poslabels=args.poslabels, env=env)
    elif args.method == 'SmallExact':
        method = SmallExact(init_data, budget_per_round=args.budget_per_round, poslabels=args.poslabels, poolsize=args.poolsize, device=args.device, env=env)
    else:
        method = Seafaring(init_data, budget_per_round=args.budget_per_round, tiara_budget=args.tiara_budget, alpha=args.alpha, poslabels=args.poslabels, device=args.device, env=env)
    accs, aucs, model = measure(method, test_data, n_classes, args.nround, args.nepoch, args.batchsize, args.device)
    res = [accs, aucs]
    print(res)
    if hasattr(env, 'save_cache'):
        env.save_cache()
    with open(f'{args.resdir}/res.pickle', 'wb') as f:
        pickle.dump(res, f)
    if hasattr(method, 'maxprob'):
        with open(f'{args.resdir}/maxprob.pickle', 'wb') as f:
            pickle.dump(method.maxprob, f)
    torch.save(model.state_dict(), f'{args.resdir}/model.pth')


if __name__ == '__main__':
    main()
