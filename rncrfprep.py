import numpy as np
import subprocess
import pickle
from obj.deptree import DepTree
import utils
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


def __label_opinion_nodes(labeled_sent, nodes, tree):
    if '##' not in labeled_sent:
        return

    opinions = labeled_sent.split('##')[1].strip()
    opinions = opinions.split(',')
    # print(opinions)
    for opinion in opinions:
        op_words = opinion.split()[:-1]
        # print(op_words)
        if len(op_words) == 1:
            for ind, term in enumerate(nodes):
                if term is not None and term == op_words[0] and tree.get_node(ind).true_label == 0:
                    tree.get_node(ind).true_label = 3
        elif len(op_words) > 1:
            for ind, term in enumerate(nodes):
                if term is None:
                    continue
                if term == op_words[0] and ind < len(
                        nodes) - 1 and nodes[ind + 1] is not None and nodes[ind + 1] == op_words[1]:
                    tree.get_node(ind).true_label = 3
                    for i in range(len(op_words) - 1):
                        if nodes[ind + i + 1] is not None and nodes[ind + i + 1] == op_words[i + 1]:
                            tree.get_node(ind + i + 1).true_label = 4


def __label_aspect_nodes(aspect_term, nodes, tree):
    if aspect_term == 'NIL':
        return

    aspects = aspect_term.split(',')

    # deals with same word but different labels
    for aspect in aspects:
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


def __build_tree_obj(dep_parse_file):
    vocab = list()
    trees = list()
    rel_list = list()

    aspect_words = utils.read_lines(aspect_word_file)

    sent_idx = 0
    f = open(dep_parse_file, 'r', encoding='utf-8')
    f_opinion_sents = open(sent_opinion_file, encoding='utf-8')
    while True:
        dep_tups = __read_sent_dep_tups(f)
        if not dep_tups:
            break

        max_ind = __get_max_word_index(dep_tups)

        nodes = [None for _ in range(max_ind + 1)]
        for rel, gov, dep in dep_tups:
            nodes[gov[0]] = gov[1]
            nodes[dep[0]] = dep[1]

        tree = DepTree(nodes)

        labeled_sent = next(f_opinion_sents).strip()
        __label_opinion_nodes(labeled_sent, nodes, tree)
        __label_aspect_nodes(aspect_words[sent_idx], nodes, tree)

        for rel, gov, dep in dep_tups:
            tree.add_edge(gov[0], dep[0], rel)

        trees.append(tree)

        for node in tree.get_word_nodes():
            if node.word.lower() not in vocab:
                vocab.append(node.word.lower())
            node.ind = vocab.index(node.word.lower())

            for ind, rel in node.kids:
                if rel not in rel_list:
                    rel_list.append(rel)

        sent_idx += 1

    f.close()
    f_opinion_sents.close()

    print(len(rel_list), 'relations')
    print(len(vocab), 'words in vocab')
    return vocab, rel_list, trees


def __get_pretrained_word_vecs():
    from gensim.models.keyedvectors import KeyedVectors
    model = KeyedVectors.load_word2vec_format(config.RAW_WORD_VEC_FILE, binary=True)
    model.save_word2vec_format(config.GOOGLE_NEWS_WORD_VEC_FILE, binary=False)


def __proc_word_vecs():
    with open(labeled_input_file, 'rb') as f:
        vocab, _, _ = pickle.load(f)

    vocab_set = set(vocab)
    word_vec_dict = dict()
    f = open(config.GOOGLE_NEWS_WORD_VEC_FILE, encoding='utf-8')
    for i, line in enumerate(f):
        if i % 10000 == 0:
            print(i)

        vals = line.split(' ')
        word = vals[0]
        if word not in vocab_set:
            continue

        word_vec_dict[word] = np.asarray([float(v) for v in vals[1:]], np.float32)
    f.close()

    vec_dim = len(next(iter(word_vec_dict.values())))
    word_vec_matrix = np.zeros((vec_dim, len(vocab)), np.float32)
    for idx, word in enumerate(vocab):
        vec = word_vec_dict.get(word, None)
        if vec is not None:
            word_vec_matrix[:, idx] = vec

    with open(word_vecs_file, 'wb') as fout:
        pickle.dump(word_vec_matrix, fout, pickle.HIGHEST_PROTOCOL)


stanford_nlp_lib_file = 'd:/lib/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar'
stanford_nlp_en_parse_file = 'd:/lib/stanford-models/englishPCFG.ser.gz'

text_file = 'd:/data/aspect/rncrf/sample.txt'
aspect_word_file = 'd:/data/aspect/rncrf/aspectTerm_sample.txt'
sent_opinion_file = 'd:/data/aspect/rncrf/opinion_sample.txt'
dep_parse_file = 'd:/data/aspect/rncrf/raw_parses_sample.txt'
labeled_input_file = 'd:/data/aspect/rncrf/labeled_input.pkl'
word_vecs_file = 'd:/data/aspect/rncrf/word_vecs.pkl'

# __dependency_parse(text_file, dep_parse_file)

# vocab, rel_list, trees = __build_tree_obj(dep_parse_file)
# with open(labeled_input_file, 'wb') as fout:
#     pickle.dump((vocab, rel_list, trees), fout, pickle.HIGHEST_PROTOCOL)

# __get_pretrained_word_vecs()

__proc_word_vecs()
