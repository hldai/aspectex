import pickle
import random
import time
import numpy as np
import rnn.propagation as prop
from rnn.adagrad import Adagrad
from utils import utils
from utils.modelutils import filter_incorrect_dep_trees


def __init_dtrnn_params(word_vec_dim, n_classes, rels):
    r = np.sqrt(6) / np.sqrt(2 * word_vec_dim + 1)
    r_Wc = 1.0 / np.sqrt(word_vec_dim)
    rel_dict = dict()
    np.random.seed(3)
    for rel in rels:
        rel_dict[rel] = np.random.rand(word_vec_dim, word_vec_dim) * 2 * r - r

    Wv = np.random.rand(word_vec_dim, word_vec_dim) * 2 * r - r
    Wc = np.random.rand(n_classes, word_vec_dim) * 2 * r_Wc - r_Wc
    b = np.zeros((word_vec_dim, 1))
    b_c = np.random.rand(n_classes, 1)
    return rel_dict, Wv, Wc, b, b_c


# returns list of zero gradients which backprop modifies, used for pretraining of RNN
def init_dtrnn_grads(rel_list, d, c, len_voc):
    rel_grads = dict()
    for rel in rel_list:
        rel_grads[rel] = np.zeros((d, d))

    return [rel_grads, np.zeros((d, d)), np.zeros((c, d)), np.zeros((d, 1)), np.zeros((c, 1)), np.zeros((d, len_voc))]


# this function computes the objective / grad for each minibatch
def objective_and_grad(par_data):
    params, d, c, len_voc, rel_list = par_data[0]
    data = par_data[1]

    # returns list of initialized zero gradients which backprop modifies
    grads = init_dtrnn_grads(rel_list, d, c, len_voc)
    (rel_dict, Wv, Wc, b, b_c, L) = params

    error_sum = 0.0
    tree_size = 0

    # compute error and gradient for each tree in minibatch
    # also keep track of total number of nodes in minibatch
    for index, tree in enumerate(data):

        nodes = tree.get_word_nodes()
        for node in nodes:
            node.vec = L[:, node.ind].reshape((d, 1))

        prop.forward_prop(params, tree, d, c)
        error_sum += tree.error()
        tree_size += len(nodes)

        prop.backprop(params[:-1], tree, d, c, len_voc, grads)

    return error_sum, grads, tree_size


def __get_grads(trees_batch, rel_dict, Wv, Wc, b, b_c, We, d, n_classes, vocab_size, rel_list, lambs):
    # non-data params
    params = (rel_dict, Wv, Wc, b, b_c, We)
    oparams = [params, d, n_classes, vocab_size, rel_list]

    param_data = [oparams, trees_batch]
    # param_data.append(oparams)
    # param_data.append(trees_batch)

    # gradient and error
    result = objective_and_grad(param_data)
    [total_err, grads, all_nodes] = result

    # add L2 regularization
    lambda_W, lambda_We, lambda_C = lambs

    reg_cost = 0.0
    # regularization for relation matrices
    for key in rel_list:
        reg_cost += 0.5 * lambda_W * np.sum(rel_dict[key] ** 2)
        grads[0][key] = grads[0][key] / all_nodes
        grads[0][key] += lambda_W * rel_dict[key]

    # regularization for transformation matrix Wv
    reg_cost += 0.5 * lambda_W * np.sum(Wv ** 2)
    grads[1] = grads[1] / all_nodes
    grads[1] += lambda_W * Wv

    # regularization for classification matrix Wc
    reg_cost += 0.5 * lambda_C * np.sum(Wc ** 2)
    grads[2] = grads[2] / all_nodes
    grads[2] += lambda_C * Wc

    # regularization for bias b
    grads[3] = grads[3] / all_nodes

    # regularization for bias b_c
    grads[4] = grads[4] / all_nodes

    # print(reg_cost.shape, We.shape, total_err)
    reg_cost += 0.5 * lambda_We * np.sum(We ** 2)

    # regularization for word embedding matrix
    grads[5] = grads[5] / all_nodes
    grads[5] += lambda_We * We

    cost = total_err / all_nodes + reg_cost

    return cost, grads


def __proc_batch(trees_batch, use_mixed_word_vec, rel_dict, word_vec_dim, vec_len_mixed,
                 n_classes, vocab_size, rel_list, lambs, Wv, Wc, b, b_c, We, ada):
    # return cost, grad
    if use_mixed_word_vec:
        err, grads = __get_grads(trees_batch, rel_dict, Wv, Wc, b, b_c, We, word_vec_dim + vec_len_mixed,
                                 n_classes, vocab_size, rel_list, lambs)
    else:
        err, grads = __get_grads(trees_batch, rel_dict, Wv, Wc, b, b_c, We, word_vec_dim,
                                 n_classes, vocab_size, rel_list, lambs)

    grad = utils.roll_params(grads, rel_list)
    update = ada.rescale_update(grad)
    updates = utils.unroll_params(update, word_vec_dim, n_classes, vocab_size, rel_list)
    for rel in rel_list:
        rel_dict[rel] -= updates[0][rel]
    Wv -= updates[1]
    Wc -= updates[2]
    b -= updates[3]
    b_c -= updates[4]
    We -= updates[5]

    return err


def __train():
    labeled_input_file = 'd:/data/aspect/rncrf/labeled_input.pkl'
    word_vecs_file = 'd:/data/aspect/rncrf/word_vecs.pkl'
    dst_params_file = 'd:/data/aspect/rncrf/deprnn-params.pkl'

    seed_i = 12
    n_classes = 5
    batch_size = 5
    n_epochs = 10
    vec_len_mixed = 50
    use_mixed_word_vec = False
    lamb_W, lamb_We, lamb_Wc = 0.001, 0.001, 0.001
    lambs = (lamb_W, lamb_We, lamb_Wc)

    with open(labeled_input_file, 'rb') as f:
        vocab, rel_list, trees = pickle.load(f)

    trees_train, trees_test = trees[:75], trees[75:]
    print(len(trees_train), 'train samples', len(trees_test), 'test samples')

    with open(word_vecs_file, 'rb') as f:
        We_origin = pickle.load(f)
    word_vec_dim = We_origin.shape[0]

    rel_list.remove('root')

    trees_train = filter_incorrect_dep_trees(trees_train)
    rel_dict, Wv, Wc, b, b_c = __init_dtrnn_params(word_vec_dim, n_classes, rel_list)
    params = (rel_dict, Wv, Wc, b, b_c, We_origin)

    r = utils.roll_params(params, rel_list)
    ada = Adagrad(r.shape)

    min_error = float('inf')
    for epoch in range(n_epochs):
        random.seed(seed_i)
        random.shuffle(trees_train)

        n_train = len(trees_train)
        batches = [trees_train[x: x + batch_size] for x in range(0, n_train, batch_size)]

        t = time.time()
        epoch_err = 0
        for batch_idx, batch in enumerate(batches):
            err = __proc_batch(
                batch, use_mixed_word_vec, rel_dict, word_vec_dim, vec_len_mixed, n_classes, len(vocab), rel_list,
                lambs, Wv, Wc, b, b_c, We_origin, ada)
            epoch_err += err
            # log_str = 'epoch: {}, batch_idx={}, err={}, time={}'.format(
            #     epoch, batch_idx, err, time.time() - t
            # )
            # print(log_str)

        print('epoch={}, err={}, time={}'.format(epoch, epoch_err, time.time() - t))

    with open(dst_params_file, 'wb') as fout:
        pickle.dump((params, vocab, rel_list), fout)


if __name__ == '__main__':
    __train()
