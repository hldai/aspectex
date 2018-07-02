import json
import numpy as np


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


def get_apects_true(sents):
    aspect_terms = set()
    for s in sents:
        terms = s.get('terms', None)
        if terms is not None:
            for t in terms:
                # aspect_terms.add(t['term'].lower())
                aspect_terms.add(t['term'])
    return aspect_terms
