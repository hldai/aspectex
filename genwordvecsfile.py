import numpy as np


def bin_word_vec_file_to_txt(bin_word_vec_file, dst_file):
    from gensim.models.keyedvectors import KeyedVectors
    model = KeyedVectors.load_word2vec_format(bin_word_vec_file, binary=True)
    model.save_word2vec_format(dst_file, binary=False)


def load_word_vec_file(filename, vocab=None):
    word_vecs = dict()
    f = open(filename, encoding='utf-8')
    next(f)
    for line in f:
        vals = line.strip().split(' ')
        word = vals[0]
        if vocab and word not in vocab:
            continue

        # vec = [float(v) for v in vals[1:]]
        word_vecs[word] = np.asarray([float(v) for v in vals[1:]], np.float32)
    f.close()
    return word_vecs


def trim_word_vecs_file(text_files, origin_word_vec_file, dst_word_vec_file):
    import numpy as np
    import pickle

    word_vecs_dict = load_word_vec_file(origin_word_vec_file)
    print('{} words in word vec file'.format(len(word_vecs_dict)))
    vocab = set()
    for text_file in text_files:
        f = open(text_file, encoding='utf-8')
        for line in f:
            words = line.strip().split(' ')
            for w in words:
                w = w.lower()
                if w in word_vecs_dict:
                    vocab.add(w)
        f.close()
    print(len(vocab), 'words')

    dim = next(iter(word_vecs_dict.values())).shape[0]
    print('dim: ', dim)
    vocab = list(vocab)
    word_vec_matrix = np.zeros((len(vocab) + 1, dim), np.float32)
    word_vec_matrix[0] = np.random.normal(size=dim)
    for i, word in enumerate(vocab):
        word_vec_matrix[i + 1] = word_vecs_dict[word]

    with open(dst_word_vec_file, 'wb') as fout:
        pickle.dump((vocab, word_vec_matrix), fout, protocol=pickle.HIGHEST_PROTOCOL)
