import os
import numpy as np
import re
import pandas as pd
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


def __gen_word_cnts_file(tok_texts_file, dst_file):
    texts = utils.read_lines(tok_texts_file)
    word_cnts_dict = dict()
    total_word_cnt = 0
    for sent_text in texts:
        words = sent_text.split()
        total_word_cnt += len(words)
        for w in words:
            cnt = word_cnts_dict.get(w, 0)
            word_cnts_dict[w] = cnt + 1

    word_cnt_tups = list(word_cnts_dict.items())
    word_cnt_tups.sort(key=lambda x: -x[1])

    word_cnt_rate_tups = list()
    for w, cnt in word_cnt_tups:
        word_cnt_rate_tups.append((w, cnt, cnt / total_word_cnt))
    df = pd.DataFrame(word_cnt_rate_tups, columns=['word', 'cnt', 'p'])
    with open(dst_file, 'w', encoding='utf-8', newline='\n') as fout:
        df.to_csv(fout, index=False, float_format='%.5f')
    print(total_word_cnt)


def __split_training_set(train_sents_file, dst_file):
    valid_data_percent = 0.2
    sents = utils.load_json_objs(train_sents_file)
    n_sents = len(sents)
    perm = np.random.permutation(n_sents)
    valid_idxs = set(perm[:int(n_sents * valid_data_percent)])
    with open(dst_file, 'w', encoding='utf-8', newline='\n') as fout:
        train_valid_labels = ['1' if i in valid_idxs else '0' for i in range(n_sents)]
        fout.write(' '.join(train_valid_labels))
        fout.write('\n')


# test_file_json = 'd:/data/aspect/semeval14/Laptops_Test_Gold.json'
# train_file_xml = 'd:/data/aspect/semeval14/Laptops_Train.xml'
# train_file_json = 'd:/data/aspect/semeval14/Laptops_Train.json'
txt_yelp_word_vecs_file = 'd:/data/res/yelp-word-vecs.txt'

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

# utils.trim_word_vecs_file(
#     [config.SE14_REST_TRAIN_TOK_TEXTS_FILE, config.SE14_REST_TEST_TOK_TEXTS_FILE],
#     config.GLOVE_WORD_VEC_FILE, config.SE14_REST_GLOVE_WORD_VEC_FILE
# )
utils.trim_word_vecs_file(
    [config.SE14_REST_TRAIN_TOK_TEXTS_FILE, config.SE14_REST_TEST_TOK_TEXTS_FILE],
    txt_yelp_word_vecs_file, config.SE14_REST_YELP_WORD_VEC_FILE
)

yelp_rest_review_sents_file = 'd:/data/res/yelp-review-sents-round-9.txt'
eng_yelp_rest_review_sents_file = 'd:/data/res/yelp-review-eng-tok-sents-round-9.txt'
# __gen_yelp_review_sents('d:/data/yelp/srcdata/yelp_academic_dataset_review.json',
#                         yelp_rest_review_sents_file)
# __select_random_yelp_review_sents(yelp_rest_review_sents_file,
#                                   'd:/data/res/yelp-review-sents-round-9-rand-part.txt')
# __filter_non_english_sents('d:/data/res/yelp-review-tok-sents-round-9.txt', eng_yelp_rest_review_sents_file)

# laptops_train_word_cnts_file = 'd:/data/aspect/semeval14/laptops/word_cnts.txt'
# __gen_word_cnts_file(config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE, laptops_train_word_cnts_file)
restaurants_train_word_cnts_file = 'd:/data/aspect/semeval14/restaurant/word_cnts.txt'
# __gen_word_cnts_file(config.SE14_REST_TRAIN_TOK_TEXTS_FILE, restaurants_train_word_cnts_file)
# __split_training_set(config.SE14_LAPTOP_TRAIN_SENTS_FILE, config.SE14_LAPTOP_TRAIN_VALID_SPLIT_FILE)
# __split_training_set(config.SE14_REST_TRAIN_SENTS_FILE, config.SE14_REST_TRAIN_VALID_SPLIT_FILE)
