import os
import numpy as np
import re
import pickle
import random
import json
import config
from utils import utils, datautils
from platform import platform

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
            # v = v.strip()
            # term = v[:-2].strip()
            # polarity = v[-2:]
            if v.strip():
                terms.append(v.strip())
        opinions_sents.append(terms)
    f.close()
    return opinions_sents


def __get_sent_objs_se14(file_text, opinions_sents):
    sents = list()
    sent_pattern = '<sentence id="(.*?)">\s*<text>(.*?)</text>\s*(.*?)</sentence>'
    miter = re.finditer(sent_pattern, file_text, re.DOTALL)
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
    return sents


def __get_sent_objs_se15(file_text, opinions_sents):
    sents = list()
    sent_pattern = '<sentence id="(.*?)">\s*<text>(.*?)</text>\s*(.*?)</sentence>'
    miter = re.finditer(sent_pattern, file_text, re.DOTALL)
    for i, m in enumerate(miter):
        sent = {'id': m.group(1), 'text': m.group(2)}
        aspect_terms = list()
        aspect_term_pattern = '<Opinion\s*target="(.*)"\s*category=.*?polarity="(.*)"\s*from="(\d*)"\s*to="(\d*)"/>'
        miter_terms = re.finditer(aspect_term_pattern, m.group(3))
        for m_terms in miter_terms:
            # print(m_terms.group(1), m_terms.group(2), m_terms.group(3))
            # aspect_terms.append(
            #     {'term': m_terms.group(1), 'polarity': m_terms.group(2), 'from': int(m_terms.group(3)),
            #      'to': int(m_terms.group(4))})
            if m_terms.group(1) == 'NULL':
                continue
            aspect_terms.append(
                {'term': m_terms.group(1), 'polarity': m_terms.group(2), 'span': (
                    int(m_terms.group(3)), int(m_terms.group(4)))})
        if aspect_terms:
            sent['terms'] = aspect_terms
        if opinions_sents[i] is not None:
            sent['opinions'] = opinions_sents[i]
        sents.append(sent)
    return sents


def __process_raw_sem_eval_data(xml_file, opinions_file, dst_sents_file, dst_sents_text_file, fn_get_sent_objs):
    opinions_sents = __read_opinions_file(opinions_file)

    f = open(xml_file, encoding='utf-8')
    text_all = f.read()
    sents = fn_get_sent_objs(text_all, opinions_sents)
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


def __split_to_sents(txt_file, dst_file):
    import nltk
    f = open(txt_file, encoding='utf-8')
    fout = open(dst_file, 'w', encoding='utf-8', newline='\n')
    for i, line in enumerate(f):
        sents = nltk.sent_tokenize(line.strip())
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


def __select_random_yelp_review_sents(sents_file, ratio, dst_file):
    f = open(sents_file, encoding='utf-8')
    fout = open(dst_file, 'w', encoding='utf-8', newline='\n')
    for line in f:
        v = random.uniform(0, 1)
        if v < ratio:
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

        if len(words) < 2 and not utils.has_alphabet(line):
            continue

        if r > 0.6:
            fout.write(line)
    fout.close()
    f.close()


def __gen_word_cnts_file(tok_texts_file, dst_file):
    import pandas as pd

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


def __gen_se15_opinion_file(sent_text_opinion_file, dst_file):
    f = open(sent_text_opinion_file, encoding='utf-8')
    fout = open(dst_file, 'w', encoding='utf-8')
    for line in f:
        if '##' not in line:
            fout.write('\n')
            continue
        sent_text, opinion_terms_str = line.strip().split('##')
        term_strs = opinion_terms_str.split(',')
        terms = list()
        for s in term_strs:
            term = s[:-2].strip()
            polarity = s[-2:]
            terms.append(term)
        fout.write('{}\n'.format(','.join(terms)))

    f.close()
    fout.close()


def __get_yelp_review_texts_file():
    reviews_file = 'd:/data/yelp/srcdata/yelp_academic_dataset_review.json'
    dst_file = 'd:/data/res/yelp-reivew-texts.txt'
    f = open(reviews_file, encoding='utf-8')
    fout = open(dst_file, 'w', encoding='utf-8')
    for i, line in enumerate(f):
        r = json.loads(line)
        fout.write('{}\n'.format(re.sub('\s+', ' ', r['text'])))
        if i % 100000 == 0:
            print(i)
        # if i > 10:
        #     break
    f.close()
    fout.close()


env = 'Windows' if platform().startswith('Windows') else 'Linux'

if env == 'Windows':
    # txt_yelp_word_vecs_file = 'd:/data/res/yelp-word-vecs-sg-100-n10-i20-w5.txt'
    # se14_rest_wv_file = 'd:/data/aspect/semeval14/model-data/yelp-word-vecs-sg-100-n10-i20-w5.pkl'
    # se15_rest_wv_file = 'd:/data/aspect/semeval15/model-data/yelp-word-vecs-sg-100-n10-i20-w5.pkl'
    txt_yelp_word_vecs_file = 'd:/data/res/yelp-w2v-sg-100-n10-i30-w5.txt'
    se14_rest_wv_file = 'd:/data/aspect/semeval14/model-data/yelp-w2v-sg-100-n10-i30-w5.pkl'
    se15_rest_wv_file = 'd:/data/aspect/semeval15/model-data/yelp-w2v-sg-100-n10-i30-w5.pkl'
    # txt_yelp_word_vecs_file = 'd:/data/res/yelp-word-vecs-sg-100-n10-i20.txt'
    # txt_amazon_word_vecs_file = 'd:/data/res/electronics-word-vecs-100.txt'
    # txt_amazon_word_vecs_file = 'd:/data/amazon/elec-w2v-nr-100-sg-n10-w8-i30.txt'
    # txt_amazon_word_vecs_file = 'd:/data/amazon/elec-w2v-100-sg-n10-w8-i30.txt'
    # se14_laptop_wv_file = 'd:/data/aspect/semeval14/model-data/amazon-wv-100-sg-n10-w8-i30.pkl'
    txt_amazon_word_vecs_file = 'd:/data/amazon/elec-w2v-nr-100-sg-n10-w8-i30.txt'
    se14_laptop_wv_file = 'd:/data/aspect/semeval14/model-data/amazon-wv-nr-100-sg-n10-w8-i30.pkl'
else:
    txt_yelp_word_vecs_file = '/home/hldai/data/yelp/yelp-word-vecs-sg-100-n10-i20-w5.txt'
    se14_rest_wv_file = '/home/hldai/data/aspect/semeval14/model-data/yelp-word-vecs-sg-100-n10-i20-w5.pkl'
    txt_amazon_word_vecs_file = '/home/hldai/data/amazon/elec-w2v-300-sg-n10-w8-i30.txt'
    se14_laptop_wv_file = '/home/hldai/data/aspect/semeval14/model-data/amazon-wv-300-sg-n10-w8-i30.pkl'
    se15_rest_wv_file = '/home/hldai/data/aspect/semeval15/model-data/yelp-word-vecs-sg-100-n10-i20-w5.pkl'
    # txt_amazon_word_vecs_file = '/home/hldai/data/amazon/elec-w2v-nr-100-sg-n10-w8-i30.txt'
    # se14_laptop_wv_file = '/home/hldai/data/aspect/semeval14/model-data/amazon-wv-nr-100-sg-n10-w8-i30.pkl'

yelp_review_sents_file = 'd:/data/res/yelp/yelp-review-sents-round-9.txt'
yelp_rest_sents_file = 'd:/data/res/yelp/yelp-rest-sents-r9.txt'
yelp_rest_sents_tok_file = 'd:/data/res/yelp/yelp-rest-sents-r9-tok.txt'
yelp_rest_sents_tok_eng_file = 'd:/data/res/yelp/yelp-rest-sents-r9-tok-eng.txt'
yelp_rest_sents_tok_eng_part_file = 'd:/data/res/yelp/eng-part/yelp-rest-sents-r9-tok-eng-part0_04.txt'
# yelp_sents_part_eng_file = 'd:/data/res/yelp/yelp-review-eng-tok-sents-round-9.txt'
yelp_tok_sents_eng_file = 'd:/data/res/yelp/yelp-eng-tok-sents-r9.txt'
yelp_tok_sents_part_eng_file = 'd:/data/res/yelp/eng-part/yelp-eng-tok-sents-r9-rand-0_02.txt'

se15_rest_sent_opinions_train_file = '/home/hldai/data/aspect/semeval15/restaurants/sentence_res15_op'
se15_rest_opinions_train_file = '/home/hldai/data/aspect/semeval15/restaurants/opinions_train.txt'
se15_rest_sent_opinions_test_file = '/home/hldai/data/aspect/semeval15/restaurants/sentence_restest15_op'
se15_rest_opinions_test_file = '/home/hldai/data/aspect/semeval15/restaurants/opinions_test.txt'

restaurants_train_word_cnts_file = 'd:/data/aspect/semeval14/restaurants/word_cnts.txt'
rest15_train_word_cnts_file = 'd:/data/aspect/semeval15/restaurants/word_cnts.txt'

# test_file_json = 'd:/data/aspect/semeval14/Laptops_Test_Gold.json'
# train_file_xml = 'd:/data/aspect/semeval14/Laptops_Train.xml'
# train_file_json = 'd:/data/aspect/semeval14/Laptops_Train.json'

# __process_hl04()
# __rncrf_sample_to_json()

# __process_raw_sem_eval_data(
#     config.SE14_LAPTOP_TRAIN_XML_FILE, config.SE14_LAPTOP_TRAIN_OPINIONS_FILE,
#     config.SE14_LAPTOP_TRAIN_SENTS_FILE, config.SE14_LAPTOP_TRAIN_SENT_TEXTS_FILE, __get_sent_objs_se14)
# __process_raw_sem_eval_data(
#     config.SE14_LAPTOP_TEST_XML_FILE, config.SE14_LAPTOP_TEST_OPINIONS_FILE,
#     config.SE14_LAPTOP_TEST_SENTS_FILE, config.SE14_LAPTOP_TEST_SENT_TEXTS_FILE, __get_sent_objs_se14)
# __gen_judge_train_data()

# __trim_word_vecs_file(
#     [config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE, config.SE14_LAPTOP_TEST_TOK_TEXTS_FILE],
#     config.GLOVE_WORD_VEC_FILE, config.SE14_LAPTOP_GLOVE_WORD_VEC_FILE
# )
# utils.trim_word_vecs_file(
#     [config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE, config.SE14_LAPTOP_TEST_TOK_TEXTS_FILE],
#     txt_amazon_word_vecs_file, se14_laptop_wv_file
# )

# __process_raw_sem_eval_data(
#     config.SE14_REST_TRAIN_XML_FILE, config.SE14_REST_TRAIN_OPINIONS_FILE,
#     config.SE14_REST_TRAIN_SENTS_FILE, config.SE14_REST_TRAIN_SENT_TEXTS_FILE, __get_sent_objs_se14)
# __process_raw_sem_eval_data(config.SE14_REST_TEST_XML_FILE, config.SE14_REST_TEST_OPINIONS_FILE,
#                             config.SE14_REST_TEST_SENTS_FILE, config.SE14_REST_TEST_SENT_TEXTS_FILE)

# utils.trim_word_vecs_file(
#     [config.SE14_REST_TRAIN_TOK_TEXTS_FILE, config.SE14_REST_TEST_TOK_TEXTS_FILE],
#     config.GLOVE_WORD_VEC_FILE, config.SE14_REST_GLOVE_WORD_VEC_FILE
# )
# utils.trim_word_vecs_file(
#     [config.SE14_REST_TRAIN_TOK_TEXTS_FILE, config.SE14_REST_TEST_TOK_TEXTS_FILE],
#     txt_yelp_word_vecs_file, se14_rest_wv_file
# )

# __gen_se15_opinion_file(se15_rest_sent_opinions_train_file, se15_rest_opinions_train_file)
# __gen_se15_opinion_file(se15_rest_sent_opinions_test_file, se15_rest_opinions_test_file)
# __process_raw_sem_eval_data(
#     config.SE15_REST_TRAIN_XML_FILE, se15_rest_opinions_train_file,
#     config.SE15_REST_TRAIN_SENTS_FILE, config.SE15_REST_TRAIN_SENT_TEXTS_FILE, __get_sent_objs_se15)
# __process_raw_sem_eval_data(
#     config.SE15_REST_TEST_XML_FILE, se15_rest_opinions_test_file,
#     config.SE15_REST_TEST_SENTS_FILE, config.SE15_REST_TEST_SENT_TEXTS_FILE, __get_sent_objs_se15)
# utils.trim_word_vecs_file(
#     [config.SE15_REST_TRAIN_TOK_TEXTS_FILE, config.SE15_REST_TEST_TOK_TEXTS_FILE,
#      config.SE14_REST_TRAIN_TOK_TEXTS_FILE, config.SE14_REST_TEST_TOK_TEXTS_FILE],
#     txt_yelp_word_vecs_file, se15_rest_wv_file
# )

# datautils.get_yelp_restaurant_reviews('d:/data/yelp/srcdata/yelp_academic_dataset_review.json',
#                                       'd:/data/yelp/srcdata/yelp_academic_dataset_business.json',
#                                       yelp_rest_sents_file)
# __filter_non_english_sents('d:/data/res/yelp-review-tok-texts.txt',
#                            'd:/data/res/yelp-review-eng-tok-texts.txt')
# __filter_non_english_sents('d:/data/res/yelp/yelp-review-tok-sents-round-9-full.txt',
#                            yelp_tok_sents_eng_file)
# __filter_non_english_sents(yelp_rest_sents_tok_file, yelp_rest_sents_tok_eng_file)
# datautils.gen_yelp_review_sents('d:/data/yelp/srcdata/yelp_academic_dataset_review.json',
#                         yelp_rest_review_sents_file)
# __select_random_yelp_review_sents(yelp_tok_sents_eng_file, 0.02, yelp_tok_sents_part_eng_file)
# __select_random_yelp_review_sents(yelp_rest_sents_tok_eng_file, 0.04, yelp_rest_sents_tok_eng_part_file)
# __get_yelp_review_texts_file()

# laptops_train_word_cnts_file = 'd:/data/aspect/semeval14/laptops/word_cnts.txt'
# __gen_word_cnts_file(config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE, laptops_train_word_cnts_file)
# __gen_word_cnts_file(config.SE14_REST_TRAIN_TOK_TEXTS_FILE, restaurants_train_word_cnts_file)
# __gen_word_cnts_file(config.SE15_REST_TRAIN_TOK_TEXTS_FILE, rest15_train_word_cnts_file)

# __split_training_set(config.SE14_LAPTOP_TRAIN_SENTS_FILE, config.SE14_LAPTOP_TRAIN_VALID_SPLIT_FILE)
# __split_training_set(config.SE14_REST_TRAIN_SENTS_FILE, config.SE14_REST_TRAIN_VALID_SPLIT_FILE)
# __split_training_set(config.SE15_REST_TRAIN_SENTS_FILE, config.SE15_REST_TRAIN_VALID_SPLIT_FILE)
# utils.bin_word_vec_file_to_txt(
#     'd:/data/res/yelp-w2v-sg-200-n10-i30-w5.bin',
#     'd:/data/res/yelp-w2v-sg-200-n10-i30-w5.txt'
# )
# utils.bin_word_vec_file_to_txt(
#     '/home/hldai/data/amazon/electronics-word-vecs-100-sg-n10-i20-w5.bin',
#     '/home/hldai/data/amazon/electronics-word-vecs-100-sg-n10-i20-w5.txt'
# )
# utils.bin_word_vec_file_to_txt(
#     '/home/hldai/data/amazon/elec-w2v-nr-100-sg-n10-w8-i30.bin',
#     '/home/hldai/data/amazon/elec-w2v-nr-100-sg-n10-w8-i30.txt'
# )

# __split_to_sents('/home/hldai/data/amazon/electronics_5_text.txt',
#                  '/home/hldai/data/amazon/electronics_5_tok_sent_texts.txt')
