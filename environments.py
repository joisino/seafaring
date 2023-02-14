import pickle
import time
import os
import json
import urllib

import numpy as np
from PIL import Image
import fasteners
import requests

import torch
import torchvision
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms


class OpenImage():
    def __init__(self, poslabels, initialtags=None):
        with open('openimage_image_to_tag.pickle', 'rb') as f:
            self.item_to_tag_dict = pickle.load(f)
        with open('openimage_tag_to_image.pickle', 'rb') as f:
            self.tag_to_item_dict = pickle.load(f)
        ss = set(map(lambda x: x.split('.')[0], os.listdir('imgs')))
        self.tag_to_item_dict = {tag: list(filter(lambda x: x in ss, li)) for tag, li in self.tag_to_item_dict.items()}

        self.cand = list(ss & set(self.item_to_tag_dict.keys()))
        self.poslabels = poslabels

        if initialtags is None:
            self.initialtags = list(self.tag_to_item_dict.keys())
        else:
            with open(initialtags, 'r') as f:
                self.initialtags = [r.strip() for r in f]

    def set_acquisition(self, acquisition):
        self.acquisition = acquisition

    def init_tags(self):
        return self.initialtags.copy()

    def item_to_tag(self, item):
        if item not in self.item_to_tag_dict:
            return []
        return self.item_to_tag_dict[item]

    def tag_to_item(self, tag):
        if tag not in self.tag_to_item_dict:
            return []
        return self.tag_to_item_dict[tag]

    def random_item(self):
        i = np.random.randint(len(self.cand))
        return self.cand[i]

    def f(self, item):
        return self.acquisition(self.get_path(item))

    def get_path(self, item):
        return 'imgs/' + item + '.jpg'

    def get_image(self, item):
        img = Image.open(self.get_path(item)).convert('RGB')
        return img

    def get_class(self, item):
        tags = self.item_to_tag(item)
        poslabels = [label.lower() for label in self.poslabels]
        pos = sum([x.lower() in poslabels for x in tags])
        if pos > 0:
            return 1
        return 0


class Flicker():
    def __init__(self, api_key, initialtags, user, device, threshold):
        self.api_key = api_key

        self.item_origin = {}

        self.item_to_tag_pickle = 'flickr_objects/cache_image_to_tag.pickle'
        self.tag_to_item_pickle = 'flickr_objects/cache_tag_to_image.pickle'
        self.item_to_url_pickle = 'flickr_objects/cache_image_to_url.pickle'
        self.results_pickle = 'flickr_objects/cache_results.pickle'
        self.initial_tags = initialtags
        self.cache_lock = 'flickr_objects/cache_lock'
        self.api_log = 'flickr_objects/api_log_{}'.format(api_key)
        self.api_lock = 'flickr_objects/api_lock_{}'.format(api_key)

        with fasteners.InterProcessLock(self.cache_lock):
            self.cache_item_to_tag = {}
            if os.path.exists(self.item_to_tag_pickle):
                with open(self.item_to_tag_pickle, 'rb') as f:
                    self.cache_item_to_tag = pickle.load(f)

            self.cache_tag_to_item = {}
            if os.path.exists(self.tag_to_item_pickle):
                with open(self.tag_to_item_pickle, 'rb') as f:
                    self.cache_tag_to_item = pickle.load(f)

            self.item_to_url = {}
            if os.path.exists(self.item_to_url_pickle):
                with open(self.item_to_url_pickle, 'rb') as f:
                    self.item_to_url = pickle.load(f)

            self.cache_results = {}
            if os.path.exists(self.results_pickle):
                with open(self.results_pickle, 'rb') as f:
                    self.cache_results = pickle.load(f)

        with open(self.initial_tags) as f:
            self.init_tags_list = [r.strip() for r in f]

        if not os.path.exists('flickr_images'):
            os.makedirs('flickr_images')
        if not os.path.exists('flickr_objects'):
            os.makedirs('flickr_objects')

        self.device = device
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.model = create_feature_extractor(model, {
            'flatten': 'flatten'
        })
        self.model.to(device)
        self.model.eval()

        with open('virtual_user_source.pt', 'rb') as f:
            self.preference = torch.load(f)[user].to(device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.244, 0.225]),
        ])

        self.threshold = threshold

    def set_acquisition(self, acquisition):
        self.acquisition = acquisition

    def init_tags(self):
        return self.init_tags_list.copy()

    def wait_until_flickr_rate(self):
        with fasteners.InterProcessLock(self.api_lock):
            if os.path.exists(self.api_log):
                with open(self.api_log, 'r') as f:
                    times = f.readlines()
            else:
                times = []
            if len(times) == 3000:
                t = max(0, 3600 - (time.time() - float(times[0])))
                time.sleep(t)
                times.pop(0)
            times.append(time.time())
            assert len(times) <= 3000
            with open(self.api_log, 'w') as f:
                for r in times:
                    print(float(r), file=f)

    def item_to_tag(self, item):
        if item not in self.cache_item_to_tag:
            print(self.item_origin[item])
            print(self.tag_to_item[self.item_origin[item]])
        assert item in self.cache_item_to_tag
        return self.cache_item_to_tag[item]

    def tag_to_item(self, tag):
        if tag not in self.cache_tag_to_item:
            self.cache_tag_to_item[tag] = []
            self.wait_until_flickr_rate()
            try:
                maxUploadDate = int(time.time() - 60 * 60 * 24 * 365 * 3 * np.random.rand())
                res = requests.get('https://www.flickr.com/services/rest/', params={
                    'method': 'flickr.photos.search',
                    'api_key': self.api_key,
                    'text': tag,
                    'max_upload_date': maxUploadDate,
                    'per_page': 500,
                    'extras': 'tags,url_n',
                    'format': 'json',
                    'nojsoncallback': True
                })
                for i in json.loads(res.text)['photos']['photo']:
                    item = i['id']
                    tags = i['tags'].split()
                    if tag in tags:
                        self.cache_tag_to_item[tag].append(item)
                        self.cache_item_to_tag[item] = tags
                        self.item_to_url[item] = i['url_n']
                        self.item_origin[item] = tag
            except BaseException:
                pass
        return self.cache_tag_to_item[tag]

    def random_item(self):
        tag = np.random.choice(self.init_tags())
        items = self.tag_to_item(tag)
        item = np.random.choice(items)
        return item

    def get_path(self, item):
        filepath = 'flickr_images/{}.jpg'.format(item)
        try:
            if not os.path.exists(filepath):
                urllib.request.urlretrieve(self.item_to_url[item], filepath)
            Image.open(filepath).convert('RGB')
        except BaseException:
            filepath = 'flickr_images/notfound.jpg'
            if not os.path.exists(filepath):
                image = Image.new('RGB', (256, 256))
                image.save(filepath)
        return filepath

    def get_image(self, item):
        filename = 'flickr_images/{}.jpg'.format(item)
        try:
            if not os.path.exists(filename):
                urllib.request.urlretrieve(self.item_to_url[item], filename)
            return Image.open(filename).convert('RGB')
        except BaseException:
            return Image.new('RGB', (256, 256))

    def f(self, item):
        return self.acquisition(self.get_path(item))

    def get_class(self, item):
        img = self.get_image(item)
        img = self.transform(img)[None, ...].to(self.device)
        feature = self.model(img)['flatten']
        c = F.cosine_similarity(feature, self.preference).cpu()
        print('similarity:', c)
        c = c.max(0)[0]
        label = 1 if float(c) > self.threshold else 0
        return label

    def get_class_from_path(self, path):
        img = Image.open(path).convert('RGB')
        img = self.transform(img)[None, ...].to(self.device)
        feature = self.model(img)['flatten']
        c = F.cosine_similarity(feature, self.preference).cpu()
        print('similarity:', c)
        c = c.max(0)[0]
        label = 1 if float(c) > self.threshold else 0
        return label

    def merge_save(self, filename, dict):
        old_dict = {}
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                old_dict = pickle.load(f)
        for key, value in dict.items():
            old_dict[key] = value
        with open(filename, 'wb') as f:
            pickle.dump(old_dict, f)

    def save_cache(self):
        with fasteners.InterProcessLock(self.cache_lock):
            self.merge_save(self.item_to_tag_pickle, self.cache_item_to_tag)
            self.merge_save(self.tag_to_item_pickle, self.cache_tag_to_item)
            self.merge_save(self.item_to_url_pickle, self.item_to_url)
            self.merge_save(self.results_pickle, self.cache_results)


def get_env(env_str, api_key=None, initialtags=None, poslabels=None, user=None, device=None, threshold=0.6):
    if env_str == 'OpenImage':
        return OpenImage(poslabels, initialtags)
    elif env_str == 'Flickr':
        return Flicker(api_key, initialtags, user, device, threshold)
    raise NotImplementedError
