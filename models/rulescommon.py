NOUN_POS_TAGS = {'NN', 'NNP', 'NNS', 'NNPS'}
VB_POS_TAGS = {'VB', 'VBN', 'VBP', 'VBZ', 'VBG', 'VBD'}


def load_rule_patterns_file(filename):
    f = open(filename, encoding='utf-8')
    f.close()


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
