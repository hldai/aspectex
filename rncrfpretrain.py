import pickle
import random
import time
import numpy as np
import rnn.propagation as prop
from rnn.adagrad import Adagrad
from utils import utils
from utils.modelutils import filter_incorrect_dep_trees
import config


def evaluate(trees_test, rel_dict, Wv, b, We, rel_list, d, c, aspect_terms_true, mixed=False):
    vocab_test = __get_vocab_from_dep_trees(trees_test)
    word_vecs_dict = utils.load_word_vec_file(config.GNEWS_LIGHT_WORD_VEC_FILE, vocab_test)

    aspect_words = set()
    all_y_true, all_y_pred = list(), list()
    count = 0
    for ind, tree in enumerate(trees_test):
        nodes = tree.get_word_nodes()
        sent = []
        h_input = np.zeros((len(tree.nodes) - 1, d))
        y_label = np.zeros((len(tree.nodes) - 1,), dtype=int)

        for index, node in enumerate(nodes):
            if node.word.lower() in vocab:
                node.vec = We[:, node.ind].reshape((d, 1))
            elif node.word.lower() in word_vecs_dict.keys():
                if mixed:
                    node.vec = (word_vecs_dict[node.word.lower()].append(2 * np.random.rand(50) - 1)).reshape((d, 1))
                else:
                    node.vec = word_vecs_dict[node.word.lower()].reshape(d, 1)
            else:
                node.vec = np.random.rand(d, 1)
                count += 1

        prop.forward_prop([rel_dict, Wv, b, We], tree, d, c, labels=False)

        for idx, node in enumerate(tree.nodes):
            if idx == 0:
                continue

            if tree.get_node(idx).is_word == 0:
                y_label[idx - 1] = 0
                sent.append(None)

                for i in range(d):
                    h_input[idx - 1][i] = 0
            else:
                y_label[idx - 1] = node.true_label
                sent.append(node.word)

                for i in range(d):
                    h_input[idx - 1][i] = node.p[i]

        crf_sent_features = sent2features(d, sent, h_input)
        for item in y_label:
            all_y_true.append(str(item))

        prediction = tagger.tag(crf_sent_features)
        print(tree.disp())
        print(prediction)
        cur_aspect_terms = __get_aspect_terms_from_labeled(tree, prediction)
        for t in cur_aspect_terms:
            aspect_words.add(t)
        # tree.disp()
        # print(prediction)
        # print(aspect_terms)
        for label in prediction:
            all_y_pred.append(label)

    p, r, f1 = utils.set_evaluate(aspect_terms_true, aspect_words)
    print(p, r, f1)


def __init_dtrnn_params(word_vec_dim, n_classes, rels):
    r = np.sqrt(6) / np.sqrt(2 * word_vec_dim + 1)
    r_Wc = 1.0 / np.sqrt(word_vec_dim)
    rel_Wr_dict = dict()
    np.random.seed(3)
    for rel in rels:
        rel_Wr_dict[rel] = np.random.rand(word_vec_dim, word_vec_dim) * 2 * r - r

    Wv = np.random.rand(word_vec_dim, word_vec_dim) * 2 * r - r
    Wc = np.random.rand(n_classes, word_vec_dim) * 2 * r_Wc - r_Wc
    b = np.zeros((word_vec_dim, 1))
    b_c = np.random.rand(n_classes, 1)
    return rel_Wr_dict, Wv, Wc, b, b_c


# returns list of zero gradients which backprop modifies, used for pretraining of RNN
def init_dtrnn_grads(rel_list, d, c, len_voc):
    rel_Wr_grads = dict()
    for rel in rel_list:
        rel_Wr_grads[rel] = np.zeros((d, d))

    return [rel_Wr_grads, np.zeros((d, d)), np.zeros((c, d)), np.zeros((d, 1)), np.zeros((c, 1)),
            np.zeros((d, len_voc))]


# this function computes the objective / grad for each minibatch
def objective_and_grad(all_params, trees_batch):
    params_train, d, n_classes, len_voc, rel_list = all_params

    # returns list of initialized zero gradients which backprop modifies
    # rel_Wr, Wv, Wc, b, b_c, We
    grads = init_dtrnn_grads(rel_list, d, n_classes, len_voc)
    rel_Wr_dict, Wv, Wc, b, b_c, We = params_train

    error_sum = 0.0
    tree_size = 0

    # compute error and gradient for each tree in minibatch
    # also keep track of total number of nodes in minibatch
    for idx, tree in enumerate(trees_batch):
        nodes = tree.get_word_nodes()
        for node in nodes:
            node.vec = We[:, node.ind].reshape((d, 1))

        prop.forward_prop(params_train, tree, d, n_classes)
        error_sum += tree.error()
        tree_size += len(nodes)

        prop.backprop(params_train[:-1], tree, d, n_classes, len_voc, grads)

    return error_sum, grads, tree_size


def __get_grads(trees_batch, rel_Wr_dict, Wv, Wc, b, b_c, We, d, n_classes, vocab_size, rel_list, lambs):
    # non-data params
    params_train = (rel_Wr_dict, Wv, Wc, b, b_c, We)
    all_params = [params_train, d, n_classes, vocab_size, rel_list]

    # gradient and error
    total_err, grads, tree_size = objective_and_grad(all_params, trees_batch)

    # add L2 regularization
    lambda_W, lambda_We, lambda_C = lambs

    reg_cost = 0.0
    # regularization for relation matrices
    for key in rel_list:
        reg_cost += 0.5 * lambda_W * np.sum(rel_Wr_dict[key] ** 2)
        grads[0][key] = grads[0][key] / tree_size
        grads[0][key] += lambda_W * rel_Wr_dict[key]

    # regularization for transformation matrix Wv
    reg_cost += 0.5 * lambda_W * np.sum(Wv ** 2)
    grads[1] = grads[1] / tree_size
    grads[1] += lambda_W * Wv

    # regularization for classification matrix Wc
    reg_cost += 0.5 * lambda_C * np.sum(Wc ** 2)
    grads[2] = grads[2] / tree_size
    grads[2] += lambda_C * Wc

    # regularization for bias b
    grads[3] = grads[3] / tree_size

    # regularization for bias b_c
    grads[4] = grads[4] / tree_size

    # print(reg_cost.shape, We.shape, total_err)
    reg_cost += 0.5 * lambda_We * np.sum(We ** 2)

    # regularization for word embedding matrix
    grads[5] = grads[5] / tree_size
    grads[5] += lambda_We * We

    cost = total_err / tree_size + reg_cost

    return cost, grads


def __proc_batch(trees_batch, use_mixed_word_vec, rel_Wr_dict, word_vec_dim, vec_len_mixed,
                 n_classes, vocab_size, rel_list, lambs, Wv, Wc, b, b_c, We, ada):
    # return cost, grad
    if use_mixed_word_vec:
        err, grads = __get_grads(trees_batch, rel_Wr_dict, Wv, Wc, b, b_c, We, word_vec_dim + vec_len_mixed,
                                 n_classes, vocab_size, rel_list, lambs)
    else:
        err, grads = __get_grads(trees_batch, rel_Wr_dict, Wv, Wc, b, b_c, We, word_vec_dim,
                                 n_classes, vocab_size, rel_list, lambs)

    grad = utils.roll_params(grads, rel_list)
    update = ada.rescale_update(grad)
    updates = utils.unroll_params(update, word_vec_dim, n_classes, vocab_size, rel_list)
    for rel in rel_list:
        rel_Wr_dict[rel] -= updates[0][rel]
    Wv -= updates[1]
    Wc -= updates[2]
    b -= updates[3]
    b_c -= updates[4]
    We -= updates[5]

    return err


def __train(train_data_file, word_vecs_file, test_data_file, dst_model_file):
    seed_i = 12
    n_classes = 5
    batch_size = 25
    n_epochs = 5
    vec_len_mixed = 50
    adagrad_reset = 30
    use_mixed_word_vec = False
    lamb_W, lamb_We, lamb_Wc = 0.001, 0.001, 0.001
    lambs = (lamb_W, lamb_We, lamb_Wc)

    with open(train_data_file, 'rb') as f:
        vocab, rel_list, trees_train = pickle.load(f)
    with open(test_data_file, 'rb') as f:
        _, _, trees_test = pickle.load(f)

    # trees_train, trees_test = trees[:75], trees[75:]
    print(len(trees_train), 'train samples')

    with open(word_vecs_file, 'rb') as f:
        We_origin = pickle.load(f)
    word_vec_dim = We_origin.shape[0]

    rel_list.remove('root')

    trees_train = filter_incorrect_dep_trees(trees_train)
    rel_Wr_dict, Wv, Wc, b, b_c = __init_dtrnn_params(word_vec_dim, n_classes, rel_list)
    params_train = (rel_Wr_dict, Wv, Wc, b, b_c, We_origin)

    r = utils.roll_params(params_train, rel_list)
    ada = Adagrad(r.shape[0])

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
                batch, use_mixed_word_vec, rel_Wr_dict, word_vec_dim, vec_len_mixed, n_classes, len(vocab), rel_list,
                lambs, Wv, Wc, b, b_c, We_origin, ada)
            epoch_err += err
            # log_str = 'epoch: {}, batch_idx={}, err={}, time={}'.format(
            #     epoch, batch_idx, err, time.time() - t
            # )
            # print(log_str)

        # reset adagrad weights
        if epoch % adagrad_reset == 0 and epoch != 0:
            ada.reset_weights()

        print('epoch={}, err={}, time={}'.format(epoch, epoch_err, time.time() - t))

    with open(dst_model_file, 'wb') as fout:
        pickle.dump((params_train, vocab, rel_list), fout)


if __name__ == '__main__':
    # data_file = 'd:/data/aspect/rncrf/labeled_input.pkl'
    # word_vecs_file = 'd:/data/aspect/rncrf/word_vecs.pkl'
    # dst_params_file = 'd:/data/aspect/rncrf/deprnn-params.pkl'
    __train(config.SE14_LAPTOP_TRAIN_RNCRF_DATA_FILE, config.SE14_LAPTOP_TRAIN_WORD_VECS_FILE,
            config.SE14_LAPTOP_TEST_RNCRF_DATA_FILE, config.SE14_LAPTOP_PRE_MODEL_FILE)
