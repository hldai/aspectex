import config
from utils import utils


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


def __subj_noun_rule(sents, dep_parse_file):
    f = open(dep_parse_file, encoding='utf-8')
    for i, sent in enumerate(sents):
        dep_tups = __read_sent_dep_tups(f)
        has_nsubj = False
        for ix, (rel, gov, dep) in enumerate(dep_tups):
            if rel == 'nmod':
                print(rel, gov, dep)
                print(sent['text'])
                print(sent['terms'])
                print()
            if rel != 'nsubj':
                continue
            # idx_gov, w_gov = gov
            # for jx in range(len(dep_tups)):
            #     rel2, gov2, dep2 = dep_tups[jx]
            #     # if dep2[0] == idx_gov and rel2 not in {'root', 'conj', 'dep', 'parataxis', 'ccomp', 'advcl'}:
            #     if ix != jx and gov2[0] == idx_gov and rel2 == 'advmod':
            #         print(sent)
            #         print(rel, gov, dep)
            #         print(dep_tups[jx])
            #         print()
            # has_nsubj = True

        # if has_nsubj:
        #     print(sent)
        #     print(dep_tups)
        #     print()
    f.close()


def __load_terms_in_train():
    sents_train = utils.load_json_objs(config.SE14_LAPTOP_TRAIN_SENTS_FILE)
    terms_train = set()
    for sent in sents_train:
        terms = sent.get('terms', None)
        if terms is None:
            continue
        for t in terms:
            terms_train.add(t['term'].lower())
    terms_train = list(terms_train)
    terms_train.sort(key=lambda x: -len(x))
    return terms_train


def __get_phrase_new(dep_tags, pos_tags, base_word_idxs, opinion_terms):
    words = [tup[2][1] for tup in dep_tags]
    phrase_word_idxs = set(base_word_idxs)
    find_compound = True
    while find_compound:
        find_compound = False
        for rel, (igov, wgov), (idep, wdep) in dep_tags:
            if rel != 'compound':
                continue
            if igov in phrase_word_idxs and idep not in phrase_word_idxs:
                find_compound = True
                phrase_word_idxs.add(idep)
            if idep in phrase_word_idxs and igov not in phrase_word_idxs:
                find_compound = True
                phrase_word_idxs.add(igov)

    ileft = min(phrase_word_idxs)
    iright = max(phrase_word_idxs)
    ileft_new, iright_new = ileft, iright
    while ileft_new > 0:
        if pos_tags[ileft_new - 1] in {'NN', 'NNP', 'NNS'}:
            ileft_new -= 1
        else:
            break
    while iright_new < len(pos_tags) - 1:
        if pos_tags[iright_new + 1] in {'NN', 'NNP', 'NNS', 'CD'}:
            iright_new += 1
        else:
            break

    # print(ileft, iright, ileft_new, iright_new)
    # print(pos_tags)
    phrase = ' '.join([words[widx] for widx in range(ileft_new, iright_new + 1)])
    return phrase


def __get_phrase(dep_tags, pos_tags, base_word_idx):
    words = [tup[2][1] for tup in dep_tags]
    dl = 1
    while base_word_idx - dl > -1:
        if dep_tags[base_word_idx - dl][0] != 'compound':
            break
        dl += 1
    dr = 1
    while base_word_idx + dr < len(dep_tags):
        if dep_tags[base_word_idx + dr][0] != 'compound':
            break
        dr += 1
    return ' '.join(words[base_word_idx - dl + 1: base_word_idx + dr])


def __rule1(dep_tags, pos_tags, sent_text, opinion_terms, nouns_filter, terms_true):
    aspect_terms = set()
    for dep_tup in dep_tags:
        rel, gov, dep = dep_tup
        if rel != 'nsubj':
            continue

        igov, wgov = gov
        idep, wdep = dep
        if wdep in nouns_filter:
            continue
        if wgov not in opinion_terms:
            continue

        phrase = __get_phrase(dep_tags, pos_tags, idep)
        # if phrase in terms_true and wgov not in opinion_terms:
        #     print(dep_tup)
        aspect_terms.add(phrase)
        # print(rel, wgov, pos_tags[igov], wdep, pos_tags[idep], phrase)
    return aspect_terms


def __rule2(dep_tags, pos_tags, sent_text, opinion_terms, nouns_filter, terms_true):
    aspect_terms = set()
    for dep_tup in dep_tags:
        rel, gov, dep = dep_tup
        if rel != 'amod':
            continue

        igov, wgov = gov
        idep, wdep = dep
        if wgov in nouns_filter:
            continue
        if wdep not in opinion_terms:
            continue

        # print(dep_tup)
        phrase = __get_phrase(dep_tags, pos_tags, igov)
        aspect_terms.add(phrase)
    return aspect_terms


tmp_tup_list = list()
def __rule3(dep_tags, pos_tags, sent_text, opinion_terms, nouns_filter, terms_true):
    aspect_terms = set()
    for dep_tup in dep_tags:
        rel, gov, dep = dep_tup
        if rel != 'dobj':
            continue

        igov, wgov = gov
        idep, wdep = dep
        if wdep in nouns_filter:
            continue
        # if wdep not in opinion_terms:
        #     continue

        phrase = __get_phrase(dep_tags, pos_tags, idep)
        hit = False
        for j in range(len(dep_tags)):
            if dep_tags[j][1][0] == igov and dep_tags[j][0] == 'nsubj':
                hit = True
                break

        if hit and wgov in {'has', 'have', 'had', 'got', 'offers', 'enjoy', 'like', 'love'}:
            aspect_terms.add(phrase)
        else:
            # print(dep_tup)
            tmp_tup_list.append(dep_tup)
    return aspect_terms


def __remove_embeded(matched_tups):
    matched_tups_new = list()
    for i, t0 in enumerate(matched_tups):
        exist = False
        for j, t1 in enumerate(matched_tups):
            if i != j and t1[0] <= t0[0] and t1[1] >= t0[1]:
                exist = True
                break
        if not exist:
            matched_tups_new.append(t0)
    return matched_tups_new


def __find_word_spans(text_lower, words):
    p = 0
    word_spans = list()
    for w in words:
        wp = text_lower[p:].find(w)
        if wp < 0:
            word_spans.append((-1, -1))
            continue
        word_spans.append((p + wp, p + wp + len(w)))
        p += wp + len(w)
    return word_spans


def __pharse_for_span(span, sent_text_lower, words, pos_tags, dep_tags, opinion_terms):
    word_spans = __find_word_spans(sent_text_lower, words)
    widxs = list()
    for i, wspan in enumerate(word_spans):
        if (wspan[0] <= span[0] < wspan[1]) or (wspan[1] >= span[1] > span[1]):
            widxs.append(i)

    phrase = __get_phrase_new(dep_tags, pos_tags, widxs, opinion_terms)
    return phrase


def __rule4(dep_tags, pos_tags, sent_text, opinion_terms, nouns_filter, terms_train, terms_true):
    sent_text_lower = sent_text.lower()
    matched_tups = list()
    for t in terms_train:
        pbeg = sent_text_lower.find(t)
        if pbeg < 0:
            continue
        if pbeg != 0 and sent_text_lower[pbeg - 1].isalpha():
            continue
        pend = pbeg + len(t)
        if pend != len(sent_text_lower) and sent_text_lower[pend].isalpha():
            continue
        matched_tups.append((pbeg, pend))
        # break

    matched_tups = __remove_embeded(matched_tups)
    sent_words = [tup[2][1] for tup in dep_tags]
    aspect_terms = set()
    for matched_span in matched_tups:
        phrase = __pharse_for_span(matched_span, sent_text_lower, sent_words, pos_tags, dep_tags, opinion_terms)
        aspect_terms.add(phrase)

    return aspect_terms


def __count_hit(y_true, y_sys):
    hit_cnt = 0
    for yi in y_true:
        if yi in y_sys:
            hit_cnt += 1
    return hit_cnt


def __rule_insight():
    # print(terms_train)
    terms_train = __load_terms_in_train()
    opinion_terms = utils.read_lines('d:/data/aspect/semeval14/opinion-terms.txt')
    opinion_terms = set(opinion_terms)
    nouns_filter = utils.read_lines('d:/data/aspect/semeval14/nouns-filter.txt')
    nouns_filter = set(nouns_filter)

    dep_tags_list = utils.load_dep_tags_list('d:/data/aspect/semeval14/laptops-test-rule-dep.txt')
    pos_tags_list = utils.load_pos_tags('d:/data/aspect/semeval14/laptops-test-rule-pos.txt')

    correct_sent_idxs = list()

    sents_test = utils.load_json_objs(config.SE14_LAPTOP_TEST_SENTS_FILE)
    hit_cnt, true_cnt, sys_cnt = 0, 0, 0
    for sent_idx, sent in enumerate(sents_test):
        term_objs = sent.get('terms', list())
        terms_true = [t['term'].lower() for t in term_objs]
        true_cnt += len(terms_true)
        sent_text = sent['text']
        dep_tags = dep_tags_list[sent_idx]
        pos_tags = pos_tags_list[sent_idx]
        assert len(dep_tags) == len(pos_tags)

        aspect_terms = __rule1(dep_tags, pos_tags, sent_text, opinion_terms, nouns_filter, terms_true)
        aspect_terms_new = __rule2(dep_tags, pos_tags, sent_text, opinion_terms, nouns_filter, terms_true)
        aspect_terms = aspect_terms.union(aspect_terms_new)
        aspect_terms_new = __rule3(dep_tags, pos_tags, sent_text, opinion_terms, nouns_filter, terms_true)
        aspect_terms = aspect_terms.union(aspect_terms_new)
        aspect_terms_new = __rule4(dep_tags, pos_tags, sent_text, opinion_terms, nouns_filter, terms_train, terms_true)
        aspect_terms = aspect_terms.union(aspect_terms_new)

        sys_cnt += len(aspect_terms)
        new_hit_cnt = __count_hit(terms_true, aspect_terms)
        if new_hit_cnt == len(terms_true) and new_hit_cnt == len(aspect_terms):
            correct_sent_idxs.append(sent_idx)
        hit_cnt += new_hit_cnt
        if len(terms_true) and not new_hit_cnt:
            print(terms_true)
            print(aspect_terms)
            print(sent_text)
            print(pos_tags)
            print(dep_tags)
            print()

    print(hit_cnt, true_cnt, sys_cnt)
    p = hit_cnt / (sys_cnt + 1e-8)
    r = hit_cnt / (true_cnt + 1e-8)
    print(p, r, 2 * p * r / (p + r + 1e-8))

    with open('d:/data/aspect/semeval14/rules-correct.txt', 'w', encoding='utf-8') as fout:
        fout.write('\n'.join([str(i) for i in correct_sent_idxs]))


def __differ():
    idxs_rule = utils.read_lines('d:/data/aspect/semeval14/rules-correct.txt')
    idxs_neu = utils.read_lines('d:/data/aspect/semeval14/deprnn-correct.txt')
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


# sents = utils.load_json_objs(config.SE14_LAPTOP_TEST_SENTS_FILE)
# __subj_noun_rule(sents, config.SE14_LAPTOP_TEST_DEP_PARSE_FILE)
__rule_insight()
# __differ()
