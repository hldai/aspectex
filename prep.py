import os
import re
import pickle
import random
import json
import config
from utils import utils

N_HEADER_LINES = 11
TAG_STRS = ['u', 'p', 's', 'cs', 'cc']


def __process_huliu04_file(filename, review_id_beg):
    filepath = os.path.join(config.DATA_DIR_HL04, filename)
    reviews, sents, sents_text = list(), list(), list()
    f = open(filepath)
    for _ in zip(range(N_HEADER_LINES), f):
        pass

    cur_rev_id = review_id_beg
    for line in f:
        line = line.strip()
        if line.startswith('[t]'):
            cur_rev_id += 1
            reviews.append({'review_id': cur_rev_id, 'title': line[3:].strip(), 'file': filename})
        elif line.startswith('##'):
            sents.append({'text': line[2:], 'review_id': cur_rev_id})
        else:
            p = line.find('##')
            sent = {'text': line[p + 2:], 'review_id': cur_rev_id}
            aspects = list()
            vals = line[:p].split(',')
            for val in vals:
                val = val.strip()
                m = re.match('(.*?)\[([+-]\d)\]', val)
                if m is None:
                    continue
                aspect = {'target': m.group(1), 'rate': m.group(2)}
                tags = list()
                for ts in TAG_STRS:
                    if '[{}]'.format(ts) in val:
                        tags.append(ts)
                if tags:
                    aspect['tags'] = tags
                aspects.append(aspect)

            if aspects:
                sent['aspects'] = aspects
            sents.append(sent)
            # print(line[:p])
    f.close()

    return reviews, sents


def __process_hl04():
    filenames = utils.read_lines(config.DATA_FILE_LIST_FILE_HL04)
    reviews, sents, sents_text = list(), list(), list()
    for filename in filenames:
        tmp_revs, tmp_sents = __process_huliu04_file(filename, len(reviews))
        reviews += tmp_revs
        sents += tmp_sents

    with open(config.SENT_TEXT_FILE_HL04, 'w', encoding='utf-8', newline='\n') as fout:
        for s in sents:
            assert '\n' not in s['text']
            fout.write('{}\n'.format(s['text']))

    fout = open(config.REVIEWS_FILE_HL04, 'w', encoding='utf-8', newline='\n')
    for r in reviews:
        fout.write('{}\n'.format(json.dumps(r, ensure_ascii=False)))
    fout.close()

    fout = open(config.SENTS_FILE_HL04, 'w', encoding='utf-8', newline='\n')
    for s in sents:
        fout.write('{}\n'.format(json.dumps(s, ensure_ascii=False)))
    fout.close()


def __read_opinions_file(filename):
    opinions_sents = list()
    f = open(filename, encoding='utf-8')
    for line in f:
        line = line.strip()
        if line == 'NIL':
            opinions_sents.append(None)
            continue

        vals = line.split(',')
        terms = list()
        for v in vals:
            v = v.strip()
            term = v[:-2].strip()
            polarity = v[-2:]
            terms.append(term)
        opinions_sents.append(terms)
    f.close()
    return opinions_sents


def __process_raw_sem_eval_data(xml_file, opinions_file, dst_sents_file, dst_sents_text_file):
    opinions_sents = __read_opinions_file(opinions_file)

    f = open(xml_file, encoding='utf-8')
    text_all = f.read()
    sents = list()
    sent_pattern = '<sentence id="(.*?)">\s*<text>(.*?)</text>\s*(.*?)</sentence>'
    miter = re.finditer(sent_pattern, text_all, re.DOTALL)
    for i, m in enumerate(miter):
        sent = {'id': m.group(1), 'text': m.group(2)}
        aspect_terms = list()
        aspect_term_pattern = '<aspectTerm\s*term="(.*)"\s*polarity="(.*)"\s*from="(\d*)"\s*to="(\d*)"/>'
        miter_terms = re.finditer(aspect_term_pattern, m.group(3))
        for m_terms in miter_terms:
            # print(m_terms.group(1), m_terms.group(2), m_terms.group(3))
            # aspect_terms.append(
            #     {'term': m_terms.group(1), 'polarity': m_terms.group(2), 'from': int(m_terms.group(3)),
            #      'to': int(m_terms.group(4))})
            aspect_terms.append(
                {'term': m_terms.group(1), 'polarity': m_terms.group(2), 'span': (
                    int(m_terms.group(3)), int(m_terms.group(4)))})
        if aspect_terms:
            sent['terms'] = aspect_terms
        if opinions_sents[i] is not None:
            sent['opinions'] = opinions_sents[i]
        sents.append(sent)
    f.close()

    utils.save_json_objs(sents, dst_sents_file)
    with open(dst_sents_text_file, 'w', encoding='utf-8') as fout:
        for sent in sents:
            fout.write('{}\n'.format(sent['text']))


def __rncrf_sample_to_json():
    sent_text_file = 'd:/data/aspect/rncrf/sample.txt'
    aspect_terms_file = 'd:/data/aspect/rncrf/aspectTerm_sample.txt'
    dst_file = 'd:/data/aspect/rncrf/sample_sents.json'

    with open(sent_text_file, encoding='utf-8') as f:
        sent_texts = [line.strip() for line in f]

    sents = list()
    f = open(aspect_terms_file, encoding='utf-8')
    for i, line in enumerate(f):
        line = line.strip()
        terms = None
        if line != 'NULL':
            terms = line.split(',')
            terms = [t.lower() for t in terms]
        sent = {'text': sent_texts[i]}
        if terms is not None:
            sent['terms'] = terms
        sents.append(sent)
    f.close()

    utils.save_json_objs(sents, dst_file)


def __gen_judge_train_data():
    judge_train_sents_file = 'd:/data/aspect/semeval14/judge_data/laptops_jtrain_sents.json'
    judge_train_dep_file = 'd:/data/aspect/semeval14/judge_data/laptops_jtrain_dep.txt'
    judge_test_sents_file = 'd:/data/aspect/semeval14/judge_data/laptops_jtest_sents.json'
    judge_test_dep_file = 'd:/data/aspect/semeval14/judge_data/laptops_jtest_dep.txt'
    judge_test_text_file = 'd:/data/aspect/semeval14/judge_data/laptops_jtest_texts.txt'

    sents = utils.load_json_objs(config.SE14_LAPTOP_TRAIN_SENTS_FILE)
    dep_tags_list = utils.load_dep_tags_list(config.SE14_LAPTOP_TRAIN_DEP_PARSE_FILE, space_sep=False)
    n_sents = len(sents)
    n_train = 2000
    assert n_sents == len(dep_tags_list)

    import numpy as np
    perm = np.random.permutation(n_sents)
    idxs_train, idxs_test = perm[:n_train], perm[n_train:]
    sents_train = [sents[idx] for idx in idxs_train]
    sents_test = [sents[idx] for idx in idxs_test]
    dep_tags_train = [dep_tags_list[idx] for idx in idxs_train]
    dep_tags_test = [dep_tags_list[idx] for idx in idxs_test]

    utils.save_json_objs(sents_train, judge_train_sents_file)
    utils.save_json_objs(sents_test, judge_test_sents_file)
    utils.save_dep_tags(dep_tags_train, judge_train_dep_file, False)
    utils.save_dep_tags(dep_tags_test, judge_test_dep_file, False)

    with open(judge_test_text_file, 'w', encoding='utf-8', newline='\n') as fout:
        for sent in sents_test:
            fout.write('{}\n'.format(sent['text']))


def __trim_word_vecs_file(text_files, origin_word_vec_file, dst_word_vec_file):
    import numpy as np
    import pickle

    word_vecs_dict = utils.load_word_vec_file(origin_word_vec_file)
    print('{} words in word vec file'.format(len(word_vecs_dict)))
    vocab = set()
    for text_file in text_files:
        f = open(text_file, encoding='utf-8')
        for line in f:
            words = line.strip().split(' ')
            for w in words:
                if w in word_vecs_dict:
                    vocab.add(w)
        f.close()
    print(len(vocab), 'words')

    dim = next(iter(word_vecs_dict.values())).shape[0]
    print(dim)
    vocab = list(vocab)
    word_vec_matrix = np.zeros((len(vocab) + 1, dim), np.float32)
    word_vec_matrix[0] = np.random.normal(size=dim)
    for i, word in enumerate(vocab):
        word_vec_matrix[i + 1] = word_vecs_dict[word]

    with open(dst_word_vec_file, 'wb') as fout:
        pickle.dump((vocab, word_vec_matrix), fout, protocol=pickle.HIGHEST_PROTOCOL)


def __gen_yelp_review_sents(yelp_review_file, dst_file):
    import nltk
    f = open(yelp_review_file, encoding='utf-8')
    fout = open(dst_file, 'w', encoding='utf-8', newline='\n')
    for i, line in enumerate(f):
        review = json.loads(line)
        review_text = review['text']
        sents = nltk.sent_tokenize(review_text)
        # print(sents)
        for sent in sents:
            sent = sent.strip()
            if not sent:
                continue
            sent = re.sub('\s+', ' ', sent)
            fout.write('{}\n'.format(sent))
        if i % 10000 == 0:
            print(i)
        # if i > 10000:
        #     break
    f.close()
    fout.close()


def __select_random_yelp_review_sents(sents_file, dst_file):
    f = open(sents_file, encoding='utf-8')
    fout = open(dst_file, 'w', encoding='utf-8', newline='\n')
    for line in f:
        v = random.uniform(0, 1)
        if v < 0.01:
            fout.write(line)
    f.close()
    fout.close()


def __filter_non_english_sents(tok_sents_file, dst_file):
    with open(config.SE14_REST_GLOVE_WORD_VEC_FILE, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
    vocab = set(vocab)
    f = open(tok_sents_file, encoding='utf-8')
    fout = open(dst_file, 'w', encoding='utf-8', newline='\n')
    for line in f:
        words = line.strip().split(' ')
        hit_cnt = 0
        for w in words:
            if w in vocab:
                hit_cnt += 1
        # print(hit_cnt / len(words))
        r = hit_cnt / len(words)
        if r > 0.6:
            fout.write(line)
    fout.close()
    f.close()


def __split_training_set(train_sents_file):
    valid_data_percent = 0.2


# test_file_json = 'd:/data/aspect/semeval14/Laptops_Test_Gold.json'
# train_file_xml = 'd:/data/aspect/semeval14/Laptops_Train.xml'
# train_file_json = 'd:/data/aspect/semeval14/Laptops_Train.json'

# __process_hl04()
# __rncrf_sample_to_json()

# __process_raw_sem_eval_data(config.SE14_LAPTOP_TRAIN_XML_FILE, config.SE14_LAPTOP_TRAIN_OPINIONS_FILE,
#                             config.SE14_LAPTOP_TRAIN_SENTS_FILE, config.SE14_LAPTOP_TRAIN_SENT_TEXTS_FILE)
# __process_raw_sem_eval_data(config.SE14_LAPTOP_TEST_XML_FILE, config.SE14_LAPTOP_TEST_OPINIONS_FILE,
#                             config.SE14_LAPTOP_TEST_SENTS_FILE, config.SE14_LAPTOP_TEST_SENT_TEXTS_FILE)
# __gen_judge_train_data()

# __trim_word_vecs_file(
#     [config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE, config.SE14_LAPTOP_TEST_TOK_TEXTS_FILE],
#     config.GLOVE_WORD_VEC_FILE, config.SE14_LAPTOP_GLOVE_WORD_VEC_FILE
# )

# __process_raw_sem_eval_data(config.SE14_REST_TRAIN_XML_FILE, config.SE14_REST_TRAIN_OPINIONS_FILE,
#                             config.SE14_REST_TRAIN_SENTS_FILE, config.SE14_REST_TRAIN_SENT_TEXTS_FILE)
# __process_raw_sem_eval_data(config.SE14_REST_TEST_XML_FILE, config.SE14_REST_TEST_OPINIONS_FILE,
#                             config.SE14_REST_TEST_SENTS_FILE, config.SE14_REST_TEST_SENT_TEXTS_FILE)

# __trim_word_vecs_file(
#     [config.SE14_REST_TRAIN_TOK_TEXTS_FILE, config.SE14_REST_TEST_TOK_TEXTS_FILE],
#     config.GLOVE_WORD_VEC_FILE, config.SE14_REST_GLOVE_WORD_VEC_FILE
# )

yelp_rest_review_sents_file = 'd:/data/res/yelp-review-sents-round-9.txt'
eng_yelp_rest_review_sents_file = 'd:/data/res/yelp-review-eng-tok-sents-round-9.txt'
# __gen_yelp_review_sents('d:/data/yelp/srcdata/yelp_academic_dataset_review.json',
#                         yelp_rest_review_sents_file)
# __select_random_yelp_review_sents(yelp_rest_review_sents_file,
#                                   'd:/data/res/yelp-review-sents-round-9-rand-part.txt')
# __filter_non_english_sents('d:/data/res/yelp-review-tok-sents-round-9.txt', eng_yelp_rest_review_sents_file)
