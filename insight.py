from utils import utils


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


# __count_adj_phrases()
# __semeval_rule_insight()

texts = utils.read_lines('d:/data/aspect/semeval14/judge_data/laptops_jtest_texts_tok.txt')
y = utils.read_lines('d:/data/aspect/semeval14/judge_data/test_correctness.txt')
y = [int(yi) for yi in y]
pos_len, neg_len = 0, 0
for text, yi in zip(texts, y):
    if yi == 0:
        neg_len += len(text)
    else:
        pos_len += len(text)
n_pos = sum(y)
n_neg = len(y) - n_pos
print(neg_len / n_neg, pos_len / n_pos)
