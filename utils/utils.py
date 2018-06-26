import json
import numpy as np


def load_word_vec_file(filename, vocab):
    word_vecs = dict()
    f = open(filename, encoding='utf-8')
    for line in f:
        vals = line.strip().split(' ')
        word = vals[0]
        if word not in vocab:
            continue

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


def next_sent_dependency(fin):
    dep_list = list()
    for line in fin:
        line = line.strip()
        if not line:
            break
        dep_tup = line.split(' ')
        dep_list.append(dep_tup)
    return dep_list


def load_dep_tags_list(filename):
    f = open(filename, encoding='utf-8')
    sent_dep_tags_list = list()
    while True:
        dep_tags = next_sent_dependency(f)
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

    Wv = arr[ind : ind + mat_size].reshape((d, d))
    ind += mat_size

    Wc = arr[ind : ind + mat_class_size].reshape((c, d))
    ind += mat_class_size

    b = arr[ind : ind + d].reshape((d, 1))
    ind += d

    b_c = arr[ind : ind + c].reshape((c, 1))
    ind += c

    We = arr[ind : ind + len_voc * d].reshape( (d, len_voc))

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

    b = arr[ind : ind + d].reshape((d, 1))
    ind += d

    We = arr[ind : ind + len_voc * d].reshape( (d, len_voc))

    return [rel_dict, Wv, b, We]
