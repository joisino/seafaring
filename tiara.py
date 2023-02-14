import numpy as np
from collections import defaultdict


class Solver():
    def __init__(self, env, init_tags, budget, seed, verbose=False):
        self.f = env.f
        self.item_to_tag = env.item_to_tag
        self.tag_to_item = env.tag_to_item
        self.budget = budget
        self.seed = seed
        self.tags = init_tags
        self.verbose = verbose

        self.tag_pos = defaultdict(int)
        self.used_images = set()

        self.bestscore = -1e9
        self.bestitem = -1
        self.second_bestscore = -1e9
        self.second_bestitem = -1
        self.history = []
        self.item_history = []

    def update_history(self, i, val, loop):
        self.used_images.add(i)
        if self.bestscore is None or val > self.bestscore:
            if self.bestscore is not None:
                self.second_bestscore = self.bestscore
                self.second_bestitem = self.second_bestitem
            self.bestscore = val
            self.bestitem = i
        elif self.second_bestscore is None or val > self.second_bestscore:
            self.second_bestscore = val
            self.second_bestitem = i
        self.history.append((loop, self.bestscore, self.bestitem))
        self.item_history.append((val, i))

    def verbose_print(self, loop):
        if self.verbose:
            print('\r {} / {}, best_score: {:.3f}, best_id: {}, second: {:.3f}, {}\033[K'.format(loop + 1, self.budget, self.bestscore, self.bestitem, self.second_bestscore, self.second_bestitem), end='', flush=True)

    def verbose_print_end(self):
        if self.verbose:
            print('', flush=True)

    def draw(self, tag):
        num_of_items = len(self.tag_to_item(tag))
        while self.tag_pos[tag] < num_of_items and self.tag_to_item(tag)[self.tag_pos[tag]] in self.used_images:
            self.tag_pos[tag] += 1
        if self.tag_pos[tag] == num_of_items:
            return None
        return self.tag_to_item(tag)[self.tag_pos[tag]]

    def random_draw(self):
        while True:
            tag = np.random.choice(self.tags)
            res = self.draw(tag)
            if res is not None:
                return res, tag


class Tiara(Solver):
    def __init__(self, env, budget, seed=0, verbose=False, budget_ini=1, word_embedding=None, lam=1.0, alpha=0.01, uncase=True, split='bar', aggregation='mean', init_tags=None):
        super(Tiara, self).__init__(env, init_tags, budget, seed, verbose)
        self.uncase = uncase
        self.split = split
        self.aggregation = aggregation

        self.budget_ini = budget_ini
        self.X = []
        self.y = []
        self.alpha = alpha
        self.word_embedding = word_embedding
        self.word_embedding_dim = next(iter(word_embedding.values())).shape[0]
        self.tag_embedding = [self.tag_to_emb(t) for t in self.tags]

        self.lam = lam
        self.A = lam * np.eye(self.word_embedding_dim)
        self.A_inv = np.eye(self.word_embedding_dim) / lam
        self.b = np.zeros(self.word_embedding_dim)

    def tag_to_words(self, tag):
        if self.uncase:
            tag = tag.lower()
        if self.split == 'bar':
            return sum([word.split('_') for word in tag.split()], [])
        elif self.split == 'all':
            return (''.join([c if c.isalpha() else ' ' for c in tag])).split()

    def tag_to_emb(self, tag):
        x = [self.word_embedding[word] for word in self.tag_to_words(tag) if word in self.word_embedding]
        if len(x) == 0:
            return np.zeros(self.word_embedding_dim)
        if self.aggregation == 'mean':
            return np.array(x).mean(0)
        if self.aggregation == 'max':
            return np.array(x).max(0)

    def update_tags(self, item):
        for tag in self.item_to_tag(item):
            if tag not in self.tags:
                self.tags.append(tag)
                self.tag_embedding.append(self.tag_to_emb(tag))

    def add_tag(self, tag, val):
        x = self.tag_to_emb(tag)
        self.X.append(x)
        self.y.append(val)

        self.A += x[:, np.newaxis] @ x[np.newaxis, :]
        self.A_inv -= (self.A_inv @ x[:, np.newaxis]) @ (x[np.newaxis, :] @ self.A_inv) / (1 + x @ self.A_inv @ x)
        self.b += val * x

    def update(self, item, tag, loop):
        val = self.f(item)
        for t in self.item_to_tag(item):
            self.add_tag(t, val)
        self.update_history(item, val, loop)
        self.update_tags(item)

    def tag_scores(self):
        embs = np.array(self.tag_embedding)
        y = embs @ (self.A_inv @ self.b) + self.alpha * np.sqrt(((embs @ self.A_inv) * embs).sum(1))
        return y

    def optimize(self):
        self.bestscore = -1e9
        self.bestitem = -1
        self.second_bestscore = -1e9
        self.second_bestitem = -1
        np.random.seed(self.seed)
        budget_ini = 0
        li = []
        for loop in range(self.budget - budget_ini):
            tag_scores = self.tag_scores()
            tags_sorted = np.argsort(tag_scores)[::-1]
            for tag_id in tags_sorted:
                item = self.draw(self.tags[tag_id])
                if item is not None:
                    break
            self.update(item, self.tags[tag_id], budget_ini + loop)
            li.append(item)
            self.verbose_print(budget_ini + loop)
        self.verbose_print_end()
        return self.item_history
