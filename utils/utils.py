import json
import numpy as np


def has_alphabet(s: str):
    for ch in s:
        if ch.isalpha():
            return True
    return False


def get_machine_name():
    import socket
    hostname = socket.gethostname()
    dot_pos = hostname.find('.')
    return hostname[:dot_pos] if dot_pos > -1 else hostname[:]


def write_terms_list(terms_list, dst_file):
    fout = open(dst_file, 'w', encoding='utf-8')
    for terms in terms_list:
        fout.write('{}\n'.format(json.dumps(terms, ensure_ascii=False)))
    fout.close()


def aspect_terms_list_from_sents(sents):
    aspect_terms_list = list()
    for sent in sents:
        aspect_terms_list.append([t['term'].lower() for t in sent.get('terms', list())])
    return aspect_terms_list


def prf1(n_true, n_sys, n_hit):
    p = n_hit / (n_sys + 1e-6)
    r = n_hit / (n_true + 1e-6)
    f1 = 2 * p * r / (p + r + 1e-6)
    return p, r, f1


def prf1_for_single_term_type(preds, token_seqs, terms_list, label_beg=1, label_in=2):
    true_cnt, sys_cnt, hit_cnt = 0, 0, 0
    for i, (p_seq, token_seq) in enumerate(zip(preds, token_seqs)):
        seq_len = len(token_seq)
        terms_sys = get_terms_from_label_list_tok(p_seq[:seq_len], token_seq, label_beg, label_in)
        terms_true = terms_list[i]
        sys_cnt += len(terms_sys)
        true_cnt += len(terms_true)

        new_hit_cnt = count_hit(terms_true, terms_sys)
        hit_cnt += new_hit_cnt

    p, r, f1 = prf1(true_cnt, sys_cnt, hit_cnt)
    return p, r, f1


def prf1_for_terms(preds, token_seqs, aspect_terms_true_list, opinion_terms_true_list):
    aspect_p, aspect_r, aspect_f1 = prf1_for_single_term_type(preds, token_seqs, aspect_terms_true_list, 1, 2)
    opinion_p, opinion_r, opinion_f1 = prf1_for_single_term_type(preds, token_seqs, opinion_terms_true_list, 3, 4)
    return aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1


def count_hit(terms_true, terms_pred):
    terms_true, terms_pred = terms_true.copy(), terms_pred.copy()
    terms_true.sort()
    terms_pred.sort()
    idx_pred = 0
    cnt_hit = 0
    for t in terms_true:
        while idx_pred < len(terms_pred) and terms_pred[idx_pred] < t:
            idx_pred += 1
        if idx_pred == len(terms_pred):
            continue
        if terms_pred[idx_pred] == t:
            cnt_hit += 1
            idx_pred += 1
    return cnt_hit


def set_evaluate(set_true, set_pred):
    if not set_pred:
        return 0, 0, 0

    cnt_hit = 0
    for v in set_true:
        if v in set_pred:
            cnt_hit += 1
    p = cnt_hit / len(set_pred)
    r = cnt_hit / len(set_true)
    f1 = 0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f1


def get_max_len(sequences):
    max_len = 0
    for s in sequences:
        max_len = max(max_len, len(s))
    return max_len


def pad_embed_sequences(seqs, pad_embed):
    max_len = get_max_len(seqs)
    padded_seqs, seq_lens = list(), list()
    for seq in seqs:
        new_seq = [embed for embed in seq]
        for _ in range(max_len - len(seq)):
            new_seq.append(pad_embed)
        padded_seqs.append(new_seq)
        seq_lens.append(len(seq))
    return padded_seqs, seq_lens


def pad_sequences(sequences, pad_token, fixed_len=False):
    max_len = get_max_len(sequences)

    padded_seqs, seq_lens = list(), list()
    for seq in sequences:
        padded_seq = seq + [pad_token for _ in range(max_len - len(seq))]
        padded_seqs.append(padded_seq)
        if fixed_len:
            seq_lens.append(max_len)
        else:
            seq_lens.append(len(seq))
    return padded_seqs, seq_lens


def pad_feat_sequence(sequences, feat_dim):
    max_len = get_max_len(sequences)
    padded_seqs, seq_lens = list(), list()
    for seq in sequences:
        padded_seq = np.zeros([max_len, feat_dim], np.float32)
        padded_seq[:len(seq), :] = seq
        padded_seqs.append(padded_seq)
        seq_lens.append(len(seq))
    return padded_seqs, seq_lens


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


def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    return lines


def load_json_objs(filename):
    f = open(filename, encoding='utf-8')
    objs = list()
    for line in f:
        objs.append(json.loads(line))
    f.close()
    return objs


def save_json_objs(objs, dst_file):
    with open(dst_file, 'w', encoding='utf-8') as fout:
        for obj in objs:
            fout.write('{}\n'.format(json.dumps(obj, ensure_ascii=False)))


def next_sent_pos(fin):
    pos_tags = list()
    for line in fin:
        line = line.strip()
        if not line:
            break
        pos_tags.append(line)
    return pos_tags


def get_indexed_word(w, dec=True):
    p = w.rfind('-')
    s = w[:p]
    idx = int(w[p + 1:])
    if dec:
        idx -= 1
    return s, idx


def next_sent_dependency(fin):
    dep_list = list()
    for line in fin:
        line = line.strip()
        if not line:
            break
        dep_tup = line.split(' ')
        wgov, idx_gov = get_indexed_word(dep_tup[0])
        wdep, idx_dep = get_indexed_word(dep_tup[1])
        dep_tup = (dep_tup[2], (idx_gov, wgov), (idx_dep, wdep))
        dep_list.append(dep_tup)
    return dep_list


def read_sent_dep_tups_rbsep(fin):
    tups = list()
    for line in fin:
        line = line.strip()
        if not line:
            return tups
        line = line[:-1]
        line = line.replace('(', ' ')
        line = line.replace(', ', ' ')
        rel, gov, dep = line.split(' ')
        w_gov, idx_gov = get_indexed_word(gov, False)
        w_dep, idx_dep = get_indexed_word(dep, False)
        tups.append((rel, (idx_gov, w_gov), (idx_dep, w_dep)))
        # tups.append(line.split(' '))
    return tups


def load_dep_tags_list(filename, space_sep=True):
    f = open(filename, encoding='utf-8')
    sent_dep_tags_list = list()
    while True:
        if space_sep:
            dep_tags = next_sent_dependency(f)
        else:
            dep_tags = read_sent_dep_tups_rbsep(f)
        if not dep_tags:
            break
        sent_dep_tags_list.append(dep_tags)
    f.close()
    return sent_dep_tags_list


def load_pos_tags(filename):
    f = open(filename, encoding='utf-8')
    sent_pos_tags_list = list()
    while True:
        sent_pos_tags = next_sent_pos(f)
        if not sent_pos_tags:
            break
        sent_pos_tags_list.append(sent_pos_tags)
    f.close()
    return sent_pos_tags_list


def save_dep_tags(dep_tags_list, filename, sep_by_space):
    fout = open(filename, 'w', encoding='utf-8', newline='\n')
    for dep_tags in dep_tags_list:
        for dep_tup in dep_tags:
            rel, (igov, wgov), (idep, wdep) = dep_tup
            if sep_by_space:
                fout.write('{}-{} {}-{} {}\n'.format(wgov, igov, wdep, idep, rel))
            else:
                fout.write('{}({}-{}, {}-{})\n'.format(rel, wgov, igov, wdep, idep))
        fout.write('\n')
    fout.close()


def roll_params(params, rel_list):
    (rel_dict, Wv, Wc, b, b_c, We) = params
    rels = np.concatenate([rel_dict[key].ravel() for key in rel_list])
    return np.concatenate((rels, Wv.ravel(), Wc.ravel(), b.ravel(), b_c.ravel(), We.ravel()))


def unroll_params(arr, d, c, len_voc, rel_list):
    mat_size = d * d
    # classification
    mat_class_size = c * d
    rel_dict = {}
    ind = 0

    for r in rel_list:
        rel_dict[r] = arr[ind: ind + mat_size].reshape((d, d))
        ind += mat_size

    Wv = arr[ind: ind + mat_size].reshape((d, d))
    ind += mat_size

    Wc = arr[ind: ind + mat_class_size].reshape((c, d))
    ind += mat_class_size

    b = arr[ind: ind + d].reshape((d, 1))
    ind += d

    b_c = arr[ind: ind + c].reshape((c, 1))
    ind += c

    We = arr[ind: ind + len_voc * d].reshape((d, len_voc))

    return [rel_dict, Wv, Wc, b, b_c, We]


def init_crfrnn_grads(rel_list, d, c, len_voc):
    rel_grads = dict()
    for rel in rel_list:
        rel_grads[rel] = np.zeros((d, d))

    return [rel_grads, np.zeros((d, d)), np.zeros((d, 1)), np.zeros((d, len_voc))]


def roll_params_noWcrf(params, rel_list):
    rel_dict, Wv, b, We = params
    rels = np.concatenate([rel_dict[key].ravel() for key in rel_list])
    return np.concatenate((rels, Wv.ravel(), b.ravel(), We.ravel()))


def unroll_params_noWcrf(arr, d, c, len_voc, rel_list):
    mat_size = d * d
    rel_dict = {}
    ind = 0

    for r in rel_list:
        rel_dict[r] = arr[ind: ind + mat_size].reshape((d, d))
        ind += mat_size

    Wv = arr[ind : ind + mat_size].reshape((d, d))
    ind += mat_size

    b = arr[ind: ind + d].reshape((d, 1))
    ind += d

    We = arr[ind: ind + len_voc * d].reshape( (d, len_voc))

    return [rel_dict, Wv, b, We]


def aspect_terms_from_labeled(sent_tree, y_pred):
    y_pred = [str(yi) for yi in y_pred]
    words = [n.word for n in sent_tree.nodes[1:] if n.is_word]
    phrases = list()
    i = 0
    while i < len(y_pred):
        yi = y_pred[i]
        if yi != '1':
            i += 1
            continue
        beg = i
        while i + 1 < len(y_pred) and y_pred[i + 1] == '2':
            i += 1
        # phrases.append(' '.join(words[beg:i + 1]).lower())
        phrases.append(' '.join(words[beg:i + 1]))
        i += 1
    return phrases


def trim_word_vecs_file(text_files, origin_word_vec_file, dst_word_vec_file, dst_file_type='pkl', val_sep_dst=','):
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

    if dst_file_type == 'pkl':
        with open(dst_word_vec_file, 'wb') as fout:
            pickle.dump((vocab, word_vec_matrix), fout, protocol=pickle.HIGHEST_PROTOCOL)
        return

    with open(dst_word_vec_file, 'w', encoding='utf-8', newline='\n') as fout:
        for w, vec in zip(vocab, word_vec_matrix):
            vec_str = val_sep_dst.join([str(v) for v in vec])
            fout.write('{}{}{}\n'.format(w, val_sep_dst, vec_str))


def get_apects_true(sents, to_lower=False):
    aspect_terms = set()
    for s in sents:
        terms = s.get('terms', None)
        if terms is not None:
            for t in terms:
                if to_lower:
                    aspect_terms.add(t['term'].lower())
                else:
                    aspect_terms.add(t['term'])
    return aspect_terms


def bin_word_vec_file_to_txt(bin_word_vec_file, dst_file):
    from gensim.models.keyedvectors import KeyedVectors
    model = KeyedVectors.load_word2vec_format(bin_word_vec_file, binary=True)
    model.save_word2vec_format(dst_file, binary=False)


def get_terms_from_label_list_tok(labels, words, label_beg, label_in):
    terms = list()
    # words = tok_text.split(' ')
    # print(labels_pred)
    # print(len(words), len(labels_pred))
    assert len(words) == len(labels)

    p = 0
    while p < len(words):
        yi = labels[p]
        if yi == label_beg:
            pright = p
            while pright + 1 < len(words) and labels[pright + 1] == label_in:
                pright += 1
            terms.append(' '.join(words[p: pright + 1]))
            p = pright + 1
        else:
            p += 1
    return terms


def get_terms_from_label_list(labels, tok_text, label_beg, label_in):
    words = tok_text.split(' ')
    return get_terms_from_label_list_tok(labels, words, label_beg, label_in)


def get_filename(filepath: str):
    pb = filepath.rfind('/')
    pe = filepath.rfind('.')
    print(filepath[pb + 1:pe])
