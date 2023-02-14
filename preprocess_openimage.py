import pickle
import numpy as np

trainfile = 'oidv6-train-annotations-human-imagelabels.csv'
testfile = 'test-annotations-human-imagelabels.csv'

print('Selecting train ids.')
with open(trainfile) as f:
    f.readline()
    images = [r.split(',')[0] for r in f]

images = sorted(list(set(images)))

np.random.seed(0)
res = np.random.choice(images, size=100000, replace=False)

print('Loading test ids.')
with open(testfile) as f:
    f.readline()
    images = [r.split(',')[0] for r in f]

images = sorted(list(set(images)))

print('Writing ids.')
with open('openimage_id.txt', 'w') as f:
    for i in res:
        print('train/{}'.format(i), file=f)
    for i in images:
        print('test/{}'.format(i), file=f)

ss = set(res) | set(images)

print('Loading tag names.')
tag_to_name = {}
with open('oidv6-class-descriptions.csv') as f:
    for r in f:
        split = r.split(',')
        tag = split[0]
        name = ','.join(split[1:])
        tag_to_name[tag] = name.strip()


def create_dictionaries(filename):
    image_to_tag = {i: [] for i in ss}
    tag_to_image = {}
    with open(filename) as f:
        f.readline()
        for r in f:
            imageid, source, tag, confidence = r.split(',')
            tag = tag_to_name[tag]
            if imageid in ss and int(confidence) == 1:
                image_to_tag[imageid].append(tag)
                if tag not in tag_to_image:
                    tag_to_image[tag] = []
                tag_to_image[tag].append(imageid)
    return image_to_tag, tag_to_image


print('Creating dictionaries.')
image_to_tag, tag_to_image = create_dictionaries(trainfile)
image_to_tag_test, tag_to_image_test = create_dictionaries(testfile)

print('Creating pickle files.')
with open('openimage_image_to_tag.pickle', 'wb') as f:
    pickle.dump(image_to_tag, f)

with open('openimage_tag_to_image.pickle', 'wb') as f:
    pickle.dump(tag_to_image, f)

with open('openimage_image_to_tag_test.pickle', 'wb') as f:
    pickle.dump(image_to_tag_test, f)

with open('openimage_tag_to_image_test.pickle', 'wb') as f:
    pickle.dump(tag_to_image_test, f)

print('Done!')
