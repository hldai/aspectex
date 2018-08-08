NOUN_POS_TAGS = {'NN', 'NNP', 'NNS', 'NNPS'}
VB_POS_TAGS = {'VB', 'VBN', 'VBP', 'VBZ', 'VBG', 'VBD'}


def load_rule_patterns_file(filename):
    l1_rules, l2_rules = list(), list()
    f = open(filename, encoding='utf-8')
    for line in f:
        vals = line.strip().split()
        assert len(vals) == 3 or len(vals) == 8
        if len(vals) == 3:
            l1_rules.append(vals)
        else:
            l2_rules.append(
                (((vals[0], vals[1], vals[2]), int(vals[3])), ((vals[4], vals[5], vals[6]), int(vals[7]))))
    f.close()
    return l1_rules, l2_rules


def get_noun_phrase_from_seed(dep_tags, pos_tags, base_word_idxs):
    words = [tup[2][1] for tup in dep_tags]
    phrase_word_idxs = set(base_word_idxs)

    ileft = min(phrase_word_idxs)
    iright = max(phrase_word_idxs)
    ileft_new, iright_new = ileft, iright
    while ileft_new > 0:
        if pos_tags[ileft_new - 1] in NOUN_POS_TAGS:
            ileft_new -= 1
        else:
            break
    while iright_new < len(pos_tags) - 1:
        if pos_tags[iright_new + 1] in {'NN', 'NNP', 'NNS', 'CD'}:
            iright_new += 1
        else:
            break

    phrase = ' '.join([words[widx] for widx in range(ileft_new, iright_new + 1)])
    return phrase


def __match_pattern_word(pw, w, pos_tag, opinion_terms_vocab):
    if pw == '_AV' and pos_tag in VB_POS_TAGS:
        return True
    if pw == '_AN' and pos_tag in NOUN_POS_TAGS:
        return True
    if pw == '_OP' and w in opinion_terms_vocab:
        return True
    if pw.isupper() and pos_tag == pw:
        return True
    return pw == w


def __match_l1_pattern(pattern, dep_tag, pos_tags, opinion_terms_vocab):
    prel, pgov, pdep = pattern
    rel, (igov, wgov), (idep, wdep) = dep_tag
    if rel != prel:
        return False
    return __match_pattern_word(pgov, wgov, pos_tags[igov], opinion_terms_vocab) and __match_pattern_word(
        pdep, wdep, pos_tags[idep], opinion_terms_vocab)


def __get_l1_pattern_matched_dep_tags(pattern, dep_tags, pos_tags, opinion_terms_vocab):
    matched_idxs = list()
    for i, dep_tag in enumerate(dep_tags):
        if __match_l1_pattern(pattern, dep_tag, pos_tags, opinion_terms_vocab):
            matched_idxs.append(i)
    return matched_idxs


def __get_aspect_term_from_matched_pattern(pattern, dep_tags, pos_tags, matched_dep_tag_idx):
    if pattern[1].startswith('_A'):
        aspect_position = 1
    elif pattern[2].startswith('_A'):
        aspect_position = 2
    else:
        return None

    dep_tag = dep_tags[matched_dep_tag_idx]
    widx, w = dep_tag[aspect_position]
    if pattern[aspect_position] == '_AV':
        return w
    else:
        return get_noun_phrase_from_seed(dep_tags, pos_tags, [widx])


def find_terms_by_l1_pattern(pattern, dep_tags, pos_tags, opinion_terms_vocab, filter_terms_vocab):
    terms = list()
    matched_dep_tag_idxs = __get_l1_pattern_matched_dep_tags(pattern, dep_tags, pos_tags, opinion_terms_vocab)
    for idx in matched_dep_tag_idxs:
        term = __get_aspect_term_from_matched_pattern(pattern, dep_tags, pos_tags, idx)

        if term in filter_terms_vocab:
            continue
        terms.append(term)
    return terms


def find_terms_by_l2_pattern(pattern, dep_tags, pos_tags, opinion_terms_vocab, filter_terms_vocab):
    (pl, ipl), (pr, ipr) = pattern
    terms = list()
    matched_dep_tag_idxs = __get_l1_pattern_matched_dep_tags(pl, dep_tags, pos_tags, opinion_terms_vocab)
    for idx in matched_dep_tag_idxs:
        dep_tag_l = dep_tags[idx]
        sw_idx = dep_tag_l[ipl][0]  # index of the shared word

        for j, dep_tag_r in enumerate(dep_tags):
            if dep_tag_r[ipr][0] != sw_idx:
                continue
            if not __match_l1_pattern(pr, dep_tag_r, pos_tags, opinion_terms_vocab):
                continue

            term = __get_aspect_term_from_matched_pattern(pl, dep_tags, pos_tags, idx)
            if term is None:
                term = __get_aspect_term_from_matched_pattern(pr, dep_tags, pos_tags, j)
            if term is None or term in filter_terms_vocab:
                # print(p, 'term not found')
                continue

            terms.append(term)
    return terms
