import numpy as np


def load_glove(dim=300, token=6):
    glove = {}
    with open('glove/glove.{}B.{}d.txt'.format(token, dim), 'r') as f:
        for r in f:
            split = r.split()
            glove[''.join(split[:-dim])] = np.array(list(map(float, split[-dim:])))
    return glove
