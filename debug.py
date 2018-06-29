import pickle
import numpy as np
from sklearn import metrics
import config


def __check_word_vec_matrix():
    with open(config.SE14_LAPTOP_WORD_VECS_FILE, 'rb') as f:
        vocab, We = pickle.load(f)

    word_idx = 2
    print(vocab[word_idx])
    We = We.T
    v = We[word_idx]
    v = v.reshape(1, -1)
    sims = metrics.pairwise.cosine_similarity(We, v).flatten()
    max_idxs = np.argpartition(-sims, np.arange(10))[:10]
    print(max_idxs)
    words = [vocab[idx] for idx in max_idxs]
    print(' '.join(words))


__check_word_vec_matrix()
