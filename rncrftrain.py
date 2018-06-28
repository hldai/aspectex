# -*- coding: utf-8 -*-
import numpy as np
from utils.mathutils import *
from rnn.adagrad import Adagrad
import rnn.crfpropagation as prop
# from classify.learn_classifiers import validate
import pickle
import time
import config
from utils import utils
from utils.modelutils import filter_incorrect_dep_trees

import pycrfsuite

# for ordered dictionary
from collections import OrderedDict


def __get_vocab_from_dep_trees(dep_trees):
    vocab_test = set()
    trees_test = filter_incorrect_dep_trees(dep_trees)
    for ind, tree in enumerate(trees_test):
        nodes = tree.get_word_nodes()
        for index, node in enumerate(nodes):
            vocab_test.add(node.word.lower())
    return vocab_test


def __get_aspect_terms_from_labeled(sent_tree, y_pred):
    words = [n.word for n in sent_tree.nodes[1:] if n.is_word]
    phrases = list()
    i = 0
    while i < len(y_pred):
        yi = y_pred[i]
        if yi != '1':
            i += 1
            continue
        while i + 1 < len(y_pred) and y_pred[i + 1] == '2':
            i += 1
        phrases.append(' '.join(words[i:i + 1]))
        i += 1
    return phrases


def evaluate(inst_ind, trees_test, rel_dict, Wv, b, We, vocab, rel_list, d, c, aspect_terms_true, mixed=False):
    # output labels
    tagger = pycrfsuite.Tagger()
    # tagger.open(str(epoch) + str(inst_ind) + 'crf.model')
    tagger.open('crfmodel.bin')

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


# convert pos tag to one-hot vector
def pos2vec(pos):
    pos_list = ['PUNCT', 'SYM', 'CONJ', 'NUM', 'DET', 'ADV', 'X', 'ADP', 'ADJ', 'VERB',
                'NOUN', 'PROPN', 'PART', 'PRON', 'INTJ']

    ind = pos_list.index(pos)
    vec = np.zeros(15)
    vec[ind] = 1

    return vec


# convert word to its namelist feature
def name2vec(sent, i, name_term, name_word):
    word = sent[i]
    name_vec = [0., 0.]

    if word is not None:
        for term in name_term:
            if word == term:
                name_vec[0] = 1.
            elif i == 0 and len(sent) > 1 and sent[i + 1] is not None:
                if word + ' ' + sent[i + 1] in term:
                    name_vec[0] = 1.
            elif i == len(sent) - 1 and len(sent) > 1 and sent[i - 1] is not None:
                if sent[i - 1] + ' ' + word in term:
                    name_vec[0] = 1.
            # elif i > 0 and i < len(sent) - 1:
            elif 0 < i < len(sent) - 1:
                if (sent[i + 1] is not None and word + ' ' + sent[i + 1] in term) \
                        or (sent[i - 1] is not None and sent[i - 1] + ' ' + word in term):
                    name_vec[0] = 1.

        if word in name_word:
            name_vec[1] = 1.

    return name_vec


# for constructing crf word features by taking rnn hidden representation
def word2features(d, sent, h_input, i):
    # for ordered dictionary
    word_features = OrderedDict()

    word_features['bias'] = 1.

    # if it is punctuation
    if sent[i] == None:
        word_features['punkt'] = 1.

    else:
        for n in range(d):
            word_features['worde=%d' % n] = h_input[i, n]
    ''' 
    # you can append human-engineered features here by providing corresponding lists
    #add pos features
    for n in range(15):
        word_features['pos=%d' % n] = pos_mat[i, n]


    #add namelist features
    name_vec = name2vec(sent, i, name_term, name_word)
    word_features['namelist1'] = name_vec[0]
    word_features['namelist2'] = name_vec[1]

    # add opinion word lexicon
    if sent[i] in senti_list:
        word_features['sentiment'] = 1.
    else:
        word_features['sentiment'] = 0.
    '''
    if i > 0 and sent[i - 1] is None:
        word_features['-1punkt'] = 1.

    elif i > 0:
        for n in range(d):
            word_features['-1worde=%d' % n] = h_input[i - 1, n]
        '''
        #add pos features
        for n in range(15):
            word_features['-1pos=%d' % n] = pos_mat[i - 1, n]

        #add namelist features
        name_vec = name2vec(sent, i - 1, name_term, name_word)
        word_features['-1namelist1'] = name_vec[0]
        word_features['-1namelist2'] = name_vec[1]

        # add opinion word lexicon
        if sent[i-1] in senti_list:
            word_features['-1sentiment'] = 1.
        else:
            word_features['-1sentiment'] = 0.
        '''

    else:
        word_features['BOS'] = 1.

    if i < len(sent) - 1 and sent[i + 1] is None:
        word_features['+1punkt'] = 1.

    elif i < len(sent) - 1:
        for n in range(d):
            word_features['+1worde=%d' % n] = h_input[i + 1, n]
        '''
        #add pos features
        for n in range(15):
            word_features['+1pos=%d' % n] = pos_mat[i + 1, n]

        #add namelist features
        name_vec = name2vec(sent, i + 1, name_term, name_word)
        word_features['+1namelist1'] = name_vec[0]
        word_features['+1namelist2'] = name_vec[1]

        # add opinion word lexicon
        if sent[i+1] in senti_list:
            word_features['+1sentiment'] = 1.
        else:
            word_features['+1sentiment'] = 0.
        '''

    else:
        word_features['EOS'] = 1.

    return word_features


# for constructing crf feature mapping for a sentence
def sent2features(d, sent, h_input):
    return pycrfsuite.ItemSequence([word2features(d, sent, h_input, i) for i in range(len(sent))])


# compute gradients and updates
# boolean determines whether to save the crf model
def par_objective(epoch, data, rel_dict, Wv, b, L, Wcrf, dim, n_classes, len_voc,
                  rel_list, lambdas, trainer, num, eta, dec, save_model):
    error_sum = np.zeros(1)
    num_nodes = 0
    tree_size = 0

    # compute for one instance
    tree = data
    nodes = tree.get_word_nodes()

    for node in nodes:
        node.vec = L[:, node.ind].reshape((dim, 1))

    prop.forward_prop([rel_dict, Wv, b, L], tree, dim, n_classes)
    tree_size += len(nodes)

    # after a rnn forward pass, compute crf
    sent = []
    # input matrix composed of hidden vector from RNN
    h_input = np.zeros((len(tree.nodes) - 1, dim))
    y_label = np.zeros((len(tree.nodes) - 1,), dtype=int)

    for ind, node in enumerate(tree.nodes):
        if ind != 0:

            # if current token is punctuation
            if tree.get_node(ind).is_word == 0:
                y_label[ind - 1] = 0
                sent.append(None)

                for i in range(dim):
                    h_input[ind - 1][i] = 0
            # if current token is a word
            else:
                y_label[ind - 1] = node.true_label
                sent.append(node.word)

                for i in range(dim):
                    h_input[ind - 1][i] = node.p[i]

    crf_sent_features = sent2features(dim, sent, h_input)
    # when parameters are updated, hidden vectors are also updated for crf input
    # this is for updating CRF input features, num is the index of the instance
    trainer.modify(crf_sent_features, num)
    # crf feature dimension
    attr_size = 3 * (dim + 1) + 3
    d_size = (len(tree.nodes) - 1) * attr_size
    # delta for hidden matrix from crf
    delta_features = np.zeros(d_size)

    # check if we need to store the model
    if save_model:
        # trainer.train(model=str(epoch) + str(num) + 'crf.model', weight=Wcrf, delta=delta_features, inst=num, eta=eta,
        #               decay=dec, loss=error_sum, check=1)
        trainer.train('crfmodel.bin', weight=Wcrf, delta=delta_features, inst=num, eta=eta,
                      decay=dec, loss=error_sum, check=1)
    else:
        trainer.train(model='', weight=Wcrf, delta=delta_features, inst=num, eta=eta, decay=dec, loss=error_sum,
                      check=1)

    grad_h = []
    start = 0
    # pass delta h to separate feature vectors to backpropagate to rnn
    for ind, node in enumerate(tree.nodes):
        if ind != 0:
            grad_h.append(-delta_features[start: start + attr_size])
            start += attr_size

    for ind, node in enumerate(tree.nodes):
        if ind != 0:
            if tree.get_node(ind).is_word != 0:
                node.grad_h = grad_h[ind - 1][1: dim + 1].reshape(dim, 1)
                # check if the sentence only contains one word
                if len(tree.nodes) > 2:
                    if ind == 1:
                        if tree.get_node(ind + 1).is_word != 0:
                            node.grad_h += grad_h[ind][2 * dim + 2: 3 * dim + 2].reshape(dim, 1)

                    elif ind < len(sent) - 1:
                        if tree.get_node(ind + 1).is_word != 0:
                            node.grad_h += grad_h[ind][2 * dim + 2: 3 * dim + 2].reshape(dim, 1)
                        if tree.get_node(ind - 1).is_word != 0:
                            node.grad_h += grad_h[ind - 2][dim + 2: 2 * dim + 2].reshape(dim, 1)
                    else:
                        if tree.get_node(ind - 1).is_word != 0:
                            node.grad_h += grad_h[ind - 2][dim + 2: 2 * dim + 2].reshape(dim, 1)

    # initialize gradients
    grads = utils.init_crfrnn_grads(rel_list, dim, n_classes, len_voc)
    prop.backprop([rel_dict, Wv, b], tree, dim, n_classes, len_voc, grads)
    lambda_W, lambda_L = lambdas

    reg_cost = 0.0
    # regularization for relation matrices
    for key in rel_list:
        reg_cost += 0.5 * lambda_W * sum(rel_dict[key] ** 2)
        grads[0][key] = grads[0][key] / tree_size
        grads[0][key] += lambda_W * rel_dict[key]
    # regularization for transformation matrix and bias
    reg_cost += 0.5 * lambda_W * sum(Wv ** 2)
    grads[1] = grads[1] / tree_size
    grads[1] += lambda_W * Wv
    grads[2] = grads[2] / tree_size
    # regularization for word embedding
    reg_cost += 0.5 * lambda_L * sum(L ** 2)
    grads[3] = grads[3] / tree_size
    grads[3] += lambda_L * L

    cost = error_sum[0] + reg_cost

    return cost, grads, Wcrf


# create new function for initializating trainer for CRF (feature map)
def trainer_initialization(m_trainer, trees, params, d, c, len_voc, rel_list):
    param_list = utils.unroll_params_noWcrf(params, d, c, len_voc, rel_list)
    rel_dict, Wv, b, L = param_list

    for tree in trees:
        nodes = tree.get_word_nodes()

        for node in nodes:
            node.vec = L[:, node.ind].reshape((d, 1))

        prop.forward_prop(param_list, tree, d, c)

        sent = []
        # input feature matrix to crf for the sentence
        h_input = np.ones((len(tree.nodes) - 1, d))
        y_label = np.zeros((len(tree.nodes) - 1,), dtype=int)

        for ind, node in enumerate(tree.nodes):
            if ind != 0:

                if tree.get_node(ind).is_word == 0:
                    y_label[ind - 1] = 0
                    sent.append(None)

                    for i in range(d):
                        h_input[ind - 1][i] = 0
                else:
                    y_label[ind - 1] = node.true_label
                    sent.append(node.word)

                    for i in range(d):
                        h_input[ind - 1][i] = node.p[i]

        y_label = np.asarray(y_label)

        crf_sent_features = sent2features(d, sent, h_input)
        crf_sent_labels = [str(item) for item in y_label]
        m_trainer.append(crf_sent_features, crf_sent_labels)

    return m_trainer


def __training_epoch(t, Wv, We, Wcrf, b, trees_test, aspect_terms_true):
    decay = 1.
    epoch_error = 0.0

    for inst_ind, inst in enumerate(tdata):
        now = time.time()

        t += 1.
        eta = 1 / (lamb * (t_0 + t))
        decay *= (1.0 - eta * lamb)

        # check if it is the end and need to store the model
        if inst_ind % 1000 == 0:
            if use_mixed_word_vec:
                err, gradient, Wcrf = par_objective(
                    epoch, inst, rel_Wr_dict, Wv, b, We, Wcrf, word_vec_dim + train_vec_len, n_classes, len(vocab),
                    rel_list, lambdas, trainer, inst_ind, eta, decay, True)
            else:
                err, gradient, Wcrf = par_objective(
                    epoch, inst, rel_Wr_dict, Wv, b, We, Wcrf, word_vec_dim, n_classes, len(vocab),
                    rel_list, lambdas, trainer, inst_ind, eta, decay, True)
        else:
            if use_mixed_word_vec:
                err, gradient, Wcrf = par_objective(epoch, inst, rel_Wr_dict, Wv, b, We, Wcrf,
                                                    word_vec_dim + train_vec_len, n_classes, len(vocab),
                                                    rel_list, lambdas, trainer, inst_ind, eta, decay, False)
            else:
                err, gradient, Wcrf = par_objective(
                    epoch, inst, rel_Wr_dict, Wv, b, We, Wcrf, word_vec_dim, n_classes, len(vocab), rel_list,
                    lambdas, trainer, inst_ind, eta, decay, False)
                # gc.collect()

        grad_vec = utils.roll_params_noWcrf(gradient, rel_list)
        update = ag.rescale_update(grad_vec)
        gradient = utils.unroll_params_noWcrf(update, word_vec_dim, n_classes, len(vocab), rel_list)

        for rel in rel_list:
            rel_Wr_dict[rel] -= gradient[0][rel]
        Wv -= gradient[1]
        b -= gradient[2]
        We -= gradient[3]

        lstring = 'epoch: ' + str(epoch) + ' inst_ind: ' + str(inst_ind) \
                  + ' error, ' + str(err) + ' time = ' + str(time.time() - now) + ' sec'
        # print(lstring)

        epoch_error += err

        if inst_ind % 1000 == 0:
            evaluate(inst_ind, trees_test, rel_Wr_dict, Wv, b, We, vocab, rel_list,
                     word_vec_dim, n_classes, aspect_terms_true, mixed=False)

    Wcrf *= decay
    # done with epoch
    print('done with epoch ', epoch, ' epoch error = ', epoch_error, ' min error = ', min_error)
    lstring = 'done with epoch ' + str(epoch) + ' epoch error = ' + str(epoch_error) \
              + ' min error = ' + str(min_error) + '\n\n'

    '''
    # save parameters if the current model is better than previous best model
    if epoch_error < min_error:
        min_error = epoch_error
        print 'saving model...'
        #params = unroll_params(r, args['d'], len(vocab), rel_list)

        if (args['op']):
            params = unroll_params_crf(r, args['d'] + args['len'], args['c'], len(vocab), rel_list)
        else:
            params = unroll_params_crf(r, args['d'], args['c'], len(vocab), rel_list)
        cPickle.dump( ( params, vocab, rel_list), paramfile)

        cPickle.dump( ( [rel_dict, Wv, b, We], vocab, rel_list), paramfile)

    else:
        os.remove(str(epoch)+'crf.model')
    '''


def __get_apects_true(sents):
    aspect_terms = set()
    for s in sents:
        terms = s.get('terms', None)
        if terms is not None:
            for t in terms:
                aspect_terms.add(t['term'])
    return aspect_terms


# train and save model
if __name__ == '__main__':
    # parser.add_argument('-agr', '--adagrad_reset', help='reset sum of squared gradients after this many\
    #                      epochs', type=int, default=50)

    # sents_file = 'd:/data/aspect/rncrf/sample_sents.json'
    # train_data_file = 'd:/data/aspect/rncrf/labeled_input.pkl'
    # word_vecs_file = 'd:/data/aspect/rncrf/word_vecs.pkl'
    # deprnn_params_file = 'd:/data/aspect/rncrf/deprnn-params.pkl'
    # dst_params_file = 'd:/data/aspect/rncrf/rncrf-params.pkl'

    sents_file = config.SE14_LAPTOP_TEST_SENTS_FILE
    train_data_file = config.SE14_LAPTOP_TRAIN_RNCRF_DATA_FILE
    test_data_file = config.SE14_LAPTOP_TEST_RNCRF_DATA_FILE
    word_vecs_file = config.SE14_LAPTOP_TRAIN_WORD_VECS_FILE
    pretrain_model_file = config.SE14_LAPTOP_PRE_MODEL_FILE
    dst_model_file = config.SE14_LAPTOP_MODEL_FILE

    n_classes = 5
    lamb_W = 0.0001
    lamb_We = 0.0001
    n_epoch = 10
    use_mixed_word_vec = False
    train_vec_len = 50

    with open(train_data_file, 'rb') as f:
        _, _, trees_train = pickle.load(f)
    with open(test_data_file, 'rb') as f:
        _, _, trees_test = pickle.load(f)

    n_train = 75
    # trees_train, trees_test = trees[:n_train], trees[n_train:]
    # sents = utils.load_json_objs(sents_file)
    # aspect_terms_true = __get_apects_true(sents[n_train:])

    sents = utils.load_json_objs(sents_file)
    aspect_terms_true = __get_apects_true(sents)

    # import pre-trained model parameters
    with open(pretrain_model_file, 'rb') as f:
        params_train, vocab, rel_list = pickle.load(f)
    rel_Wr_dict, Wv, Wc, b, b_c, We = params_train

    word_vec_dim, n_words = We.shape
    lambdas = [lamb_W, lamb_We]

    print('number of training sentences:', len(trees_train))
    # print('number of validation sentences:', len(val_trees))
    print('number of dependency relations:', len(rel_list))
    print('number of classes:', n_classes)

    trees_train = filter_incorrect_dep_trees(trees_train)

    # add train_size
    n_train = len(trees_train)

    # crf weight matrix
    Wcrf = np.zeros(n_classes * (3 * (word_vec_dim + 1 + 1)) + 18)  # dimension
    # store delta for CRF backpropagation
    delta_features = np.zeros(3000)

    # r is 1-D param vector
    r_noWcrf = utils.roll_params_noWcrf((rel_Wr_dict, Wv, b, We), rel_list)

    dim = r_noWcrf.shape[0]
    print('parameter vector dimensionality:', dim)

    crf_loss = np.zeros(1)

    # minibatch adagrad training
    ag = Adagrad(r_noWcrf.shape[0])

    # initialize trainer object for CRF
    trainer = pycrfsuite.Trainer(algorithm='l2sgd', verbose=False)
    trainer.set_params({
        'c2': 1.,
        'max_iterations': 1  # stop earlier
    })

    trainer = trainer_initialization(trainer, trees_train, r_noWcrf, word_vec_dim, n_classes, len(vocab), rel_list)
    trainer.train(model='', weight=Wcrf, delta=delta_features, inst=0, eta=0, decay=0, loss=crf_loss, check=0)

    # learning rate decay
    t = -1.
    lamb = 2. * 1. / n_train
    t_0 = 1. / (lamb * 0.02)

    for tdata in [trees_train]:
        min_error = float('inf')
        for epoch in range(0, n_epoch):
            __training_epoch(t, Wv, We, Wcrf, b, trees_test, aspect_terms_true)
