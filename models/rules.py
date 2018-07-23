def __get_noun_phrase(dep_tags, pos_tags, base_word_idxs, opinion_terms):
    words = [tup[2][1] for tup in dep_tags]
    phrase_word_idxs = set(base_word_idxs)

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

    phrase = ' '.join([words[widx] for widx in range(ileft_new, iright_new + 1)])
    return phrase


def rule1(dep_tags, pos_tags, opinion_terms, nouns_filter):
    aspect_terms = set()
    sent_words = [dep_tup[2][1] for dep_tup in dep_tags]
    for dep_tup in dep_tags:
        rel, gov, dep = dep_tup
        if rel not in {'nsubj', 'csubj', 'nmod'}:
            continue

        igov, wgov = gov
        idep, wdep = dep
        if wdep in nouns_filter:
            continue
        if wgov not in opinion_terms:
            continue

        # print(rel, wgov, wdep)
        # phrase = __get_phrase(dep_tags, pos_tags, idep)
        phrase = __get_noun_phrase(dep_tags, pos_tags, [idep], opinion_terms)
        # phrase1 = __get_phrase_for_rule1(dep_tags, pos_tags, sent_words, idep, igov)
        # if phrase in terms_true and wgov not in opinion_terms:
        #     print(dep_tup)
        aspect_terms.add(phrase)
        # print(rel, wgov, pos_tags[igov], wdep, pos_tags[idep], phrase)
    return aspect_terms


def rule2(dep_tags, pos_tags, opinion_terms, nouns_filter):
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
        # phrase = __get_phrase(dep_tags, pos_tags, igov)
        phrase = __get_noun_phrase(dep_tags, pos_tags, [igov], opinion_terms)
        aspect_terms.add(phrase)
    return aspect_terms


def rule3(dep_tags, pos_tags, opinion_terms, nouns_filter):
    aspect_terms = set()
    for dep_tup in dep_tags:
        rel, gov, dep = dep_tup
        if rel not in {'dobj', 'xcomp'}:
            continue

        igov, wgov = gov
        idep, wdep = dep
        if wdep in nouns_filter:
            continue
        # if wdep not in opinion_terms:
        #     continue

        phrase = __get_noun_phrase(dep_tags, pos_tags, [idep], opinion_terms)
        # phrase = __get_phrase(dep_tags, pos_tags, idep)
        hit = False
        for j in range(len(dep_tags)):
            if dep_tags[j][1][0] == igov and dep_tags[j][0] == 'nsubj':
                hit = True
                break

        aspect_terms.add(phrase)
        # if hit and wgov in {'has', 'have', 'had', 'got', 'offers', 'enjoy', 'like', 'love'}:
        #     aspect_terms.add(phrase)
        # else:
        #     print(dep_tup)
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
        if (wspan[0] <= span[0] < wspan[1]) or (wspan[0] < span[1] <= wspan[1]):
            widxs.append(i)

    if not widxs:
        # print(span)
        # print(sent_text_lower[span[0]: span[1]])
        # print(sent_text_lower)
        # print(words)
        # print(word_spans)
        # exit()
        return None

    phrase = __get_noun_phrase(dep_tags, pos_tags, widxs, opinion_terms)
    return phrase


def rule4(dep_tags, pos_tags, sent_text, opinion_terms, nouns_filter, terms_train):
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
        if phrase is not None:
            aspect_terms.add(phrase)

    return aspect_terms


def __word_in_terms(word, terms):
    for t in terms:
        if word in t:
            return True
    return False


def rule5(dep_tags, pos_tags, opinion_terms, nouns_filter):
    aspect_terms = set()
    for dep_tup in dep_tags:
        rel, gov, dep = dep_tup
        if rel != 'advmod':
            continue

        igov, wgov = gov
        idep, wdep = dep
        if pos_tags[igov] in {'JJ'}:
            continue
        if wgov in nouns_filter:
            continue
        if wdep not in opinion_terms:
            continue

        # print(dep_tup)
        # phrase = __get_phrase(dep_tags, pos_tags, igov)
        aspect_terms.add(wgov)
        # print(wgov, wdep)
    return aspect_terms


def conj_rule(dep_tags, pos_tags, opinion_terms, nouns_filter, terms_extracted):
    aspect_terms = set()
    for dep_tup in dep_tags:
        rel, gov, dep = dep_tup
        if rel != 'conj':
            continue

        igov, wgov = gov
        idep, wdep = dep

        if __word_in_terms(wgov, terms_extracted) and not __word_in_terms(
                wdep, terms_extracted) and wdep not in nouns_filter:
            phrase = __get_noun_phrase(dep_tags, pos_tags, [idep], opinion_terms)
            # phrase = __get_phrase(dep_tags, pos_tags, idep)
            aspect_terms.add(phrase)
        elif __word_in_terms(wdep, terms_extracted) and not __word_in_terms(
                wgov, terms_extracted) and wgov not in nouns_filter:
            phrase = __get_noun_phrase(dep_tags, pos_tags, [idep], opinion_terms)
            # phrase = __get_phrase(dep_tags, pos_tags, idep)
            aspect_terms.add(phrase)

    return aspect_terms


def rec_rule1(dep_tags, pos_tags, nouns_filter, opinion_terms):
    words = [dep_tag[2][1] for dep_tag in dep_tags]
    assert len(words) == len(pos_tags)

    noun_phrases = list()
    pleft = 0
    while pleft < len(words):
        if pos_tags[pleft] not in {'NN', 'NNS', 'NNP'}:
            pleft += 1
            continue
        pright = pleft + 1
        while pright < len(words) and pos_tags[pright] in {'NN', 'NNS', 'NNP', 'CD'}:
            pright += 1

        # if pleft > 0 and pos_tags[pleft - 1] == 'JJ' and words[pleft - 1] not in opinion_terms:
        #     pleft -= 1

        phrase = ' '.join(words[pleft: pright])
        if phrase not in nouns_filter:
            noun_phrases.append(phrase)
        pleft = pright
    # print(' '.join(words))
    # print(noun_phrases)
    return noun_phrases
