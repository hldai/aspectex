import numpy as np
import subprocess
import pickle
from obj.deptree import DepTree
from utils import utils
import config


def __dependency_parse(text_file, dst_file):
    proc = subprocess.Popen([
        'java', '-cp', stanford_nlp_lib_file,
        'edu.stanford.nlp.parser.lexparser.LexicalizedParser',
        '-nthreads', '2',
        '-sentences', 'newline',
        '-retainTmpSubcategories',
        '-outputFormat', 'typedDependencies',
        '-outputFormatOptions', 'basicDependencies',
        stanford_nlp_en_parse_file,
        text_file],
        stdout=subprocess.PIPE)
    output, err = proc.communicate()

    with open(dst_file, 'wb') as fout:
        fout.write(output)


# word-idx to (idx, word)
def __get_word_tup(dep_word):
    p = dep_word.rfind('-')
    s = dep_word[:p]
    # idx = int(dep_word[p + 1:]) - 1
    idx = int(dep_word[p + 1:])
    return s, idx


def __read_sent_dep_tups(fin):
    tups = list()
    for line in fin:
        line = line.strip()
        if not line:
            return tups
        line = line[:-1]
        line = line.replace('(', ' ')
        line = line.replace(', ', ' ')
        rel, gov, dep = line.split(' ')
        w_gov, idx_gov = __get_word_tup(gov)
        w_dep, idx_dep = __get_word_tup(dep)
        tups.append((rel, (idx_gov, w_gov), (idx_dep, w_dep)))
        # tups.append(line.split(' '))
    return tups


def __get_max_word_index(dep_tups):
    max_ind = -1
    for rel, gov, dep in dep_tups:
        midx = max(gov[0], dep[0])
        if midx > max_ind:
            max_ind = midx
    return max_ind


def __label_nodes_with_opinions_sample(labeled_sent, nodes, tree):
    if '##' not in labeled_sent:
        return

    opinions = labeled_sent.split('##')[1].strip()
    opinions = opinions.split(',')
    # print(opinions)
    for opinion in opinions:
        opinion_words = opinion.split()[:-1]
        # print(op_words)
        if len(opinion_words) == 1:
            for ind, term in enumerate(nodes):
                if term is not None and term == opinion_words[0] and tree.get_node(ind).true_label == 0:
                    tree.get_node(ind).true_label = 3
        elif len(opinion_words) > 1:
            for ind, term in enumerate(nodes):
                if term is None:
                    continue
                if term == opinion_words[0] and ind < len(
                        nodes) - 1 and nodes[ind + 1] is not None and nodes[ind + 1] == opinion_words[1]:
                    tree.get_node(ind).true_label = 3
                    for i in range(len(opinion_words) - 1):
                        if nodes[ind + i + 1] is not None and nodes[ind + i + 1] == opinion_words[i + 1]:
                            tree.get_node(ind + i + 1).true_label = 4


def __label_nodes_with_opinions(nodes, tree, opinion_terms):
    # print(opinions)
    for opinion_term in opinion_terms:
        opinion_words = opinion_term.split()
        # print(op_words)
        if len(opinion_words) == 1:
            for ind, term in enumerate(nodes):
                if term is not None and term == opinion_words[0] and tree.get_node(ind).true_label == 0:
                    tree.get_node(ind).true_label = 3
        elif len(opinion_words) > 1:
            for ind, term in enumerate(nodes):
                if term is None:
                    continue
                if term == opinion_words[0] and ind < len(
                        nodes) - 1 and nodes[ind + 1] is not None and nodes[ind + 1] == opinion_words[1]:
                    tree.get_node(ind).true_label = 3
                    for i in range(len(opinion_words) - 1):
                        if nodes[ind + i + 1] is not None and nodes[ind + i + 1] == opinion_words[i + 1]:
                            tree.get_node(ind + i + 1).true_label = 4


def __label_aspect_nodes(aspect_terms, nodes, tree):
    # if aspect_term == 'NIL':
    #     return
    #
    # aspects = aspect_term.split(',')

    # deals with same word but different labels
    for aspect in aspect_terms:
        aspect = aspect.strip()
        # aspect is a phrase
        if ' ' in aspect:
            aspect_list = aspect.split()
            for ind, term in enumerate(nodes):
                if term == aspect_list[0] and ind < len(nodes) - 1 and nodes[ind + 1] == aspect_list[1]:
                    tree.get_node(ind).true_label = 1

                    for i in range(len(aspect_list) - 1):
                        if ind + i + 1 < len(nodes):
                            if nodes[ind + i + 1] == aspect_list[i + 1]:
                                tree.get_node(ind + i + 1).true_label = 2
                    break
        # aspect is a single word
        else:
            for ind, term in enumerate(nodes):
                if term == aspect and tree.get_node(ind).true_label == 0:
                    tree.get_node(ind).true_label = 1


def __build_tree_obj(dep_parse_file, sents_file, vocab=None):
    trees = list()
    rel_list = list()

    sents = utils.load_json_objs(sents_file)

    make_vocab = False
    if vocab is None:
        make_vocab = True
        vocab = list()

    sent_idx = 0
    f = open(dep_parse_file, 'r', encoding='utf-8')
    while True:
        dep_tups = __read_sent_dep_tups(f)
        if not dep_tups:
            break
        # print(dep_tups)

        max_ind = __get_max_word_index(dep_tups)

        nodes = [None for _ in range(max_ind + 1)]
        for rel, gov, dep in dep_tups:
            nodes[gov[0]] = gov[1]
            nodes[dep[0]] = dep[1]

        tree = DepTree(nodes)

        # labeled_sent = next(f_opinion_sents).strip()
        # __label_opinion_nodes(labeled_sent, nodes, tree)
        sent = sents[sent_idx]
        sent_opinions = sent.get('opinions', None)
        if sent_opinions:
            __label_nodes_with_opinions(nodes, tree, sent_opinions)
        sent_aspects = sent.get('terms', None)
        if sent_aspects is not None:
            terms = [a['term'] for a in sent_aspects]
            __label_aspect_nodes(terms, nodes, tree)
        # __label_aspect_nodes(aspect_words[sent_idx], nodes, tree)

        for rel, gov, dep in dep_tups:
            tree.add_edge(gov[0], dep[0], rel)

        trees.append(tree)

        for node in tree.get_word_nodes():
            w = node.word.lower()
            if make_vocab:
                if w not in vocab:
                    vocab.append(w)
                node.ind = vocab.index(w)
            else:
                if w in vocab:
                    node.ind = vocab.index(w)
                else:
                    node.ind = -1

            for ind, rel in node.kids:
                if rel not in rel_list:
                    rel_list.append(rel)

        sent_idx += 1

    f.close()

    print(len(rel_list), 'relations')
    # print(len(vocab), 'words in vocab')
    return vocab, rel_list, trees


def __get_pretrained_word_vecs():
    from gensim.models.keyedvectors import KeyedVectors
    model = KeyedVectors.load_word2vec_format(config.RAW_WORD_VEC_FILE, binary=True)
    model.save_word2vec_format(config.GOOGLE_NEWS_WORD_VEC_FILE, binary=False)


def __gen_word_vec_matrix_file(dep_parse_files, dst_file):
    word_vec_dict = utils.load_word_vec_file(config.GNEWS_LIGHT_WORD_VEC_FILE)

    # word_cnt_dict = dict()
    vocab = set()
    for dep_parse_file in dep_parse_files:
        f = open(dep_parse_file, 'r', encoding='utf-8')
        while True:
            dep_tups = __read_sent_dep_tups(f)
            if not dep_tups:
                break

            for rel, gov, dep in dep_tups:
                # print(rel, gov, dep)
                word = dep[1].lower()
                if word in word_vec_dict:
                    vocab.add(word)
            # exit()
        f.close()

    print(len(vocab), 'words in word vec vocab')
    vocab = list(vocab)
    vec_dim = len(next(iter(word_vec_dict.values())))
    word_vec_matrix = np.zeros((vec_dim, len(vocab)), np.float32)
    for idx, word in enumerate(vocab):
        vec = word_vec_dict.get(word, None)
        if vec is not None:
            word_vec_matrix[:, idx] = vec

    with open(dst_file, 'wb') as fout:
        pickle.dump((vocab, word_vec_matrix), fout, pickle.HIGHEST_PROTOCOL)


def __proc_word_vecs(data_file, dst_file):
    with open(data_file, 'rb') as f:
        vocab, _, _ = pickle.load(f)

    vocab_set = set(vocab)
    word_vec_dict = utils.load_word_vec_file(config.GNEWS_LIGHT_WORD_VEC_FILE, vocab_set)

    vec_dim = len(next(iter(word_vec_dict.values())))
    word_vec_matrix = np.zeros((vec_dim, len(vocab)), np.float32)
    for idx, word in enumerate(vocab):
        vec = word_vec_dict.get(word, None)
        if vec is not None:
            word_vec_matrix[:, idx] = vec

    return word_vec_matrix
    # with open(dst_file, 'wb') as fout:
    #     pickle.dump(word_vec_matrix, fout, pickle.HIGHEST_PROTOCOL)


def __word_legal(w):
    if not w.islower():
        return False
    for ch in w:
        if ch in {'_', '#', '/', '.', '@'}:
            return False
    return True


def __filter_word_vecs():
    wcnt = 0
    f = open(config.GOOGLE_NEWS_WORD_VEC_FILE, encoding='utf-8')
    fout = open(config.GNEWS_LIGHT_WORD_VEC_FILE, 'w', encoding='utf-8')
    for i, line in enumerate(f):
        if i % 100000 == 0:
            print(i, wcnt)
        # if i > 100000:
        #     break

        vals = line.split(' ')
        word = vals[0]
        if not __word_legal(word):
            # print(word)
            continue
        fout.write(line)

        wcnt += 1
    f.close()
    fout.close()
    print(wcnt)


def __gen_data_for_train():
    # with open(config.SE14_LAPTOP_WORD_VECS_FILE, 'rb') as f:
    #     vocab, _ = pickle.load(f)

    # __dependency_parse(config.SE14_LAPTOP_TRAIN_SENT_TEXTS_FILE, config.SE14_LAPTOP_TRAIN_DEP_PARSE_FILE)
    vocab, rel_list, trees = __build_tree_obj(
        config.SE14_LAPTOP_TRAIN_DEP_PARSE_FILE, config.SE14_LAPTOP_TRAIN_SENTS_FILE)

    We = __proc_word_vecs(config.SE14_LAPTOP_TRAIN_RNCRF_DATA_FILE, config.SE14_LAPTOP_TRAIN_WORD_VECS_FILE)
    with open(config.SE14_LAPTOP_TRAIN_RNCRF_DATA_FILE, 'wb') as fout:
        pickle.dump((vocab, We, rel_list, trees), fout, pickle.HIGHEST_PROTOCOL)


def __gen_data_for_test():
    # with open(config.SE14_LAPTOP_WORD_VECS_FILE, 'rb') as f:
    #     vocab, _ = pickle.load(f)
    with open(config.SE14_LAPTOP_TRAIN_RNCRF_DATA_FILE, 'rb') as f:
        vocab, _, _, _ = pickle.load(f)

    # __dependency_parse(config.SE14_LAPTOP_TEST_SENT_TEXTS_FILE, config.SE14_LAPTOP_TEST_DEP_PARSE_FILE)
    vocab, rel_list, trees = __build_tree_obj(
        config.SE14_LAPTOP_TEST_DEP_PARSE_FILE, config.SE14_LAPTOP_TEST_SENTS_FILE, vocab)
    with open(config.SE14_LAPTOP_TEST_RNCRF_DATA_FILE, 'wb') as fout:
        pickle.dump((rel_list, trees), fout, pickle.HIGHEST_PROTOCOL)


stanford_nlp_lib_file = 'd:/lib/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar'
stanford_nlp_en_parse_file = 'd:/lib/stanford-models/englishPCFG.ser.gz'

# text_file = 'd:/data/aspect/rncrf/sample.txt'
# aspect_word_file = 'd:/data/aspect/rncrf/aspectTerm_sample.txt'
# sent_opinion_file = 'd:/data/aspect/rncrf/opinion_sample.txt'
# dep_parse_file = 'd:/data/aspect/rncrf/raw_parses_sample.txt'
# labeled_input_file = 'd:/data/aspect/rncrf/labeled_input.pkl'
# word_vecs_file = 'd:/data/aspect/rncrf/word_vecs.pkl'

sents_file = 'd:/data/aspect/semeval14/Laptops_Train.json'
sent_texts_file = 'd:/data/aspect/semeval14/Laptops_Train_text.txt'
aspect_word_file = 'd:/data/aspect/rncrf/aspectTerm_sample.txt'
sent_opinion_file = 'd:/data/aspect/rncrf/opinion_sample.txt'
dep_parse_file = 'd:/data/aspect/rncrf/raw_parses_sample.txt'
labeled_input_file = 'd:/data/aspect/rncrf/labeled_input.pkl'
word_vecs_file = 'd:/data/aspect/rncrf/word_vecs.pkl'

# __get_pretrained_word_vecs()
# __filter_word_vecs()

# __dependency_parse(sent_texts_file, dep_parse_file)
# vocab, rel_list, trees = __build_tree_obj(dep_parse_file)
# with open(labeled_input_file, 'wb') as fout:
#     pickle.dump((vocab, rel_list, trees), fout, pickle.HIGHEST_PROTOCOL)
# __proc_word_vecs()

# __gen_word_vec_matrix_file([config.SE14_LAPTOP_TRAIN_DEP_PARSE_FILE, config.SE14_LAPTOP_TEST_DEP_PARSE_FILE],
#                            config.SE14_LAPTOP_WORD_VECS_FILE)
# __gen_data_for_train()
# __gen_data_for_test()
