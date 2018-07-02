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


# __check_word_vec_matrix()
f = open('d:/data/amazon/electronics_5_text_tok.txt', encoding='utf-8')
word_cnts = dict()
for line in f:
    words = line.strip().split(' ')
    for w in words:
        cnt = word_cnts.get(w, 0)
        word_cnts[w] = cnt + 1
    # print(words)
    # break
f.close()

cnts = [0, 0]
for w, cnt in word_cnts.items():
    if cnt >= 2:
        cnts[0] += 1
    if cnt >= 3:
        cnts[1] += 1
print(cnts)
