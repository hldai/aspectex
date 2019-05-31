import json
from utils import utils
import config


def __posible_alt(w, wt):
    if w + 's' == wt or wt + 's' == w:
        return True

    if len(w) <= 3 or len(wt) <= 3:
        return False

    minl = min(len(w), len(wt))
    return w[:minl - 3] == wt[:minl - 3]


def __count_adj_phrases():
    sents = utils.load_json_objs('d:/data/aspect/huliu04/sents.json')
    pos_tags_list = utils.load_pos_tags('d:/data/aspect/huliu04/sents-text-pos.txt')

    all_aspects = set()
    max_sys_aspects = set()
    aspects_with_jj = set()
    for i, sent in enumerate(sents):
        sent_words = sent['text'].split(' ')
        pos_tags = pos_tags_list[i]
        assert len(sent_words) == len(pos_tags)

        aspect_objs = sent.get('aspects', None)
        if aspect_objs is None:
            continue

        for a in aspect_objs:
            aw = a['target']
            all_aspects.add(aw)

            possible = False
            words = aw.split(' ')
            nw = len(words)
            has_jj = False
            for p in range(len(sent_words) - nw + 1):
                curw = ' '.join(sent_words[p:p + nw])
                if aw != curw and not __posible_alt(curw, aw):
                    continue

                has_jj = False
                for j in range(p, p + nw):
                    # if pos_tags[j] == 'JJ':
                    if pos_tags[j] == 'JJ' or pos_tags[j] == 'VBG':
                        has_jj = True
                        break
                if not has_jj:
                    possible = True
                else:
                    aspects_with_jj.add(curw)

            if possible:
                max_sys_aspects.add(aw)
            # else:
            #     # print(aw)
            #     if not has_jj:
            #         print(aw)
            #         print(sent)
            #         print()
    print(len(all_aspects), 'aspects')
    print(len(max_sys_aspects), len(max_sys_aspects) / len(all_aspects), (len(max_sys_aspects) + 7) / len(all_aspects))
    for w in aspects_with_jj:
        print(w)

    exit()
    for i, sent in enumerate(sents):
        sent_words = sent['text'].split(' ')
        pos_tags = pos_tags_list[i]
        assert len(sent_words) == len(pos_tags)

        aspect_objs = sent.get('aspects', None)
        if aspect_objs is None:
            continue

        for a in aspect_objs:
            aw = a['target']

            possible = False
            words = aw.split(' ')
            nw = len(words)
            has_jj = False
            for p in range(len(sent_words) - nw + 1):
                curw = ' '.join(sent_words[p:p + nw])
                if aw != curw and not __posible_alt(curw, aw):
                    continue

                has_jj = False
                for j in range(p, p + nw):
                    if pos_tags[j] == 'JJ' or pos_tags[j] == 'VBG':
                    # if pos_tags[j] not in {'NN', 'NNS', 'NNP'}:
                        has_jj = True
                        break
                if not has_jj:
                    possible = True

            if not possible:
                if not has_jj and aw not in max_sys_aspects:
                    print(aw)
                    print(sent)
                    print()


def __semeval_rule_insight():
    train_file = 'd:/data/aspect/semeval14/Laptops_Train.json'
    test_file = 'd:/data/aspect/semeval14/Laptops_Test_Gold.json'
    sents_train = utils.load_json_objs(train_file)
    sents_test = utils.load_json_objs(test_file)

    def __count_terms(sents):
        cnt_dict = dict()
        for sent in sents:
            aspect_terms = sent.get('terms', None)
            if aspect_terms is not None:
                for term in aspect_terms:
                    s = term['term']
                    cnt = cnt_dict.get(s, 0)
                    cnt_dict[s] = cnt + 1
        return cnt_dict

    term_cnts_train = __count_terms(sents_train)
    term_cnts_test = __count_terms(sents_test)
    term_cnt_tups = [(t, cnt) for t, cnt in term_cnts_test.items()]
    term_cnt_tups.sort(key=lambda x: -x[1])
    for t, cnt in term_cnt_tups:
        if t not in term_cnts_train:
            print(t, cnt)


def __rule_result_differ():
    idxs_rule = utils.read_lines('d:/data/aspect/semeval14/rules-correct.txt')
    idxs_neu = utils.read_lines('d:/data/aspect/semeval14/lstmcrf-correct.txt')
    idxs_rule = [int(idx) for idx in idxs_rule]
    idxs_neu = [int(idx) for idx in idxs_neu]
    print(idxs_rule)
    print(idxs_neu)
    idxs_rule_only = list()
    for i in idxs_rule:
        if i not in idxs_neu:
            idxs_rule_only.append(i)
    idxs_neu_only = list()
    for i in idxs_neu:
        if i not in idxs_rule:
            idxs_neu_only.append(i)
    print(idxs_rule_only)
    print(len(idxs_rule_only))
    print(idxs_neu_only)
    print(len(idxs_neu_only))


def __dataset_statistics():
    # sents_file = 'd:/data/aspect/semeval14/laptops/laptops_test_sents.json'
    # sents_file = 'd:/data/aspect/semeval14/restaurants/restaurants_test_sents.json'
    # sents_file = 'd:/data/aspect/semeval14/restaurants/restaurants_train_sents.json'
    # sents_file = 'd:/data/aspect/semeval15/restaurants/restaurants_train_sents.json'
    sents_file = 'd:/data/aspect/semeval15/restaurants/restaurants_test_sents.json'
    sents = utils.load_json_objs(sents_file)
    print(len(sents), 'sentences')
    at_cnt, ot_cnt = 0, 0
    for s in sents:
        at_cnt += len(s.get('terms', list()))
        ot_cnt += len(s.get('opinions', list()))
    print(at_cnt, 'aspect terms')
    print(ot_cnt, 'opinion terms')


def __amazon_statistics():
    reviews_file = 'd:/data/amazon/Electronics_5.json'
    f = open(reviews_file, encoding='utf-8')
    rcnt = 0
    asin_set = set()
    for line in f:
        r = json.loads(line)
        asin_set.add(r['asin'])
        # print(r)
        rcnt += 1
        # if rcnt > 5:
        #     break
    f.close()
    print(len(asin_set), 'asins')


def __is_correct(terms_sys, terms_true):
    for t in terms_true:
        if t not in terms_sys:
            return False
    for t in terms_sys:
        if t not in terms_true:
            return False
    return True


def __check_errors():
    sents_file = 'd:/data/aspect/semeval14/laptops/laptops_test_sents.json'
    lstmcrf_aspects_file = 'd:/data/aspect/semeval14/lstmcrf-aspects.txt'
    lstmcrf_opinions_file = 'd:/data/aspect/semeval14/lstmcrf-opinions.txt'
    nrdj_aspects_file = 'd:/data/aspect/semeval14/nrdj-aspects.txt'
    nrdj_opinions_file = 'd:/data/aspect/semeval14/nrdj-opinions.txt'
    rule_aspects_file = 'd:/data/aspect/semeval14/laptops/laptops-test-aspect-rule-result.txt'

    sents = utils.load_json_objs(sents_file)
    lc_aspects_list = utils.load_json_objs(lstmcrf_aspects_file)
    nrdj_aspects_list = utils.load_json_objs(nrdj_aspects_file)
    rule_aspects_list = utils.load_json_objs(rule_aspects_file)
    for sent, lc_aspects, nrdj_aspects, rule_aspects in zip(
            sents, lc_aspects_list, nrdj_aspects_list, rule_aspects_list):
        terms = [t['term'].lower() for t in sent.get('terms', list())]
        lc_correct = __is_correct(lc_aspects, terms)
        nrdj_correct = __is_correct(nrdj_aspects, terms)
        rule_correct = __is_correct(rule_aspects, terms)
        if not lc_correct and not rule_correct and nrdj_correct:
            print(sent['text'])
            print(terms)
            print(lc_aspects)
            print(rule_aspects)
            print(nrdj_aspects)
            print()


def __count_rule_extracted_terms():
    # aspect_terms_file = 'd:/data/aspect/semeval14/laptops/laptops-test-aspect-rule-result.txt'
    # opinion_terms_file = 'd:/data/aspect/semeval14/laptops/laptops-test-opinion-rule-result.txt'
    aspect_terms_file = 'd:/data/aspect/semeval15/restaurants/restaurants-test-aspect-rule-result.txt'
    opinion_terms_file = 'd:/data/aspect/semeval15/restaurants/restaurants-test-opinion-rule-result.txt'

    aspect_terms_list = utils.load_json_objs(aspect_terms_file)
    opinion_terms_list = utils.load_json_objs(opinion_terms_file)

    num_aspect_terms = sum([len(terms) for terms in aspect_terms_list])
    print(num_aspect_terms)

    num_opinion_terms = sum([len(terms) for terms in opinion_terms_list])
    print(num_opinion_terms)


def __count_words(tok_texts_file):
    f = open(tok_texts_file, encoding='utf-8')
    n_min, n_max = 1e9, 0
    for line in f:
        words = line.strip().split(' ')
        # print(len(words))
        if len(words) == 1:
            print(line)
        n_min = min(n_min, len(words))
        n_max = max(n_max, len(words))
    f.close()
    print(n_min, n_max)


def __missing_terms():
    opinion_terms_file = 'd:/data/aspect/semeval14/opinion-terms-full.txt'
    opinion_terms_vocab = set(utils.read_lines(opinion_terms_file))
    train_sents = utils.load_json_objs(config.SE15R_FILES['train_sents_file'])
    test_sents = utils.load_json_objs(config.SE15R_FILES['test_sents_file'])
    train_terms = set()
    test_terms = dict()
    for s in train_sents:
        for t in s['opinions']:
            train_terms.add(t.lower())
    for s in test_sents:
        for t in s['opinions']:
            cnt = test_terms.get(t.lower(), 0)
            test_terms[t.lower()] = cnt + 1
            # test_terms.add(t.lower())
    for t, cnt in test_terms.items():
        if t not in train_terms:
            print(t, cnt, t in opinion_terms_vocab)


def __count_hits(terms_true, terms_sys):
    cnt = 0
    for t in terms_true:
        if t in terms_sys:
            cnt += 1
    return cnt


def __check_opinion_errors():
    terms_sys_list = utils.load_json_objs('d:/onedrive/opinion_terms_bert_output_r.txt')
    terms_sys_nr_list = utils.load_json_objs('d:/onedrive/opinion_terms_bert_output.txt')
    test_sents = utils.load_json_objs(config.SE15R_FILES['test_sents_file'])
    terms_true_list = [s['opinions'] for s in test_sents]
    for s, terms_true, terms_sys, terms_sys_nr in zip(
            test_sents, terms_true_list, terms_sys_list, terms_sys_nr_list):
        if not terms_true and not terms_sys:
            continue
        terms_true = [t.lower() for t in terms_true]
        if len(terms_true) == len(terms_sys) and __count_hits(terms_true, terms_sys) == len(terms_true):
            continue
        print(s['text'])
        print(terms_true, terms_sys, terms_sys_nr)
        print()


def check_unseen_terms():
    sents_file = 'd:/data/aspect/semeval14/laptops/laptops_test_sents.json'
    lstmcrf_aspects_file = 'd:/data/aspect/semeval14/lstmcrf-aspects.txt'
    lstmcrf_opinions_file = 'd:/data/aspect/semeval14/lstmcrf-opinions.txt'
    nrdj_aspects_file = 'd:/data/aspect/semeval14/nrdj-aspects.txt'
    nrdj_opinions_file = 'd:/data/aspect/semeval14/nrdj-opinions.txt'
    rule_aspects_file = 'd:/data/aspect/semeval14/laptops/laptops-test-aspect-rule-result.txt'

    sents = utils.load_json_objs(sents_file)
    lc_aspects_list = utils.load_json_objs(lstmcrf_aspects_file)
    nrdj_aspects_list = utils.load_json_objs(nrdj_aspects_file)
    rule_aspects_list = utils.load_json_objs(rule_aspects_file)
    terms_true_list, terms_nrdj_list = list(), list()
    n_true, n_nrdj, n_hit = 0, 0, 0
    for sent, lc_aspects, nrdj_aspects, rule_aspects in zip(
            sents, lc_aspects_list, nrdj_aspects_list, rule_aspects_list):
        terms = [t['term'].lower() for t in sent.get('terms', list())]
        print(terms, nrdj_aspects)
        terms_true_list.append(terms)
        terms_nrdj_list.append(nrdj_aspects)
        n_true += len(terms)
        n_nrdj += len(nrdj_aspects)
        n_hit += utils.count_hit(terms, nrdj_aspects)
        # lc_correct = __is_correct(lc_aspects, terms)
        # nrdj_correct = __is_correct(nrdj_aspects, terms)
        # rule_correct = __is_correct(rule_aspects, terms)
        # if not lc_correct and not rule_correct and nrdj_correct:
        #     print(sent['text'])
        #     print(terms)
        #     print(lc_aspects)
        #     print(rule_aspects)
        #     print(nrdj_aspects)
        #     print()

    p, r, f1 = utils.prf1(n_true, n_nrdj, n_hit)
    print(p, r, f1)


# __count_adj_phrases()
# __semeval_rule_insight()
# __dataset_statistics()
# __amazon_statistics()
# __check_errors()
# __count_rule_extracted_terms()
# __count_words('d:/data/aspect/semeval14/laptops/laptops_train_texts_tok.txt')
# __count_words('d:/data/aspect/semeval14/laptops/laptops_test_texts_tok.txt')
# __count_words('d:/data/aspect/semeval14/restaurants/restaurants_test_texts_tok.txt')
# __count_words('d:/data/aspect/semeval14/restaurants/restaurants_train_texts_tok.txt')
# __count_words('d:/data/aspect/semeval15/restaurants/restaurants_train_texts_tok.txt')
# __count_words('d:/data/aspect/semeval15/restaurants/restaurants_test_texts_tok.txt')
# __missing_terms()
# __check_opinion_errors()
check_unseen_terms()
