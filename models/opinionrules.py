def rule1(dep_tags, pos_tags):
    words = [dep_tag[2][1] for dep_tag in dep_tags]
    assert len(words) == len(pos_tags)
    # print(words)
    # print(pos_tags)

    opinion_terms = list()
    for i, w in enumerate(words):
        pos = pos_tags[i]
        if pos in {'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'VB', 'VBN', 'VBD', 'VBP', 'VBZ', 'VBG'} and w not in {
            "n't", 'not', 'so', 'also'}:
            opinion_terms.append(w)
    # print(' '.join(words))
    # print(noun_phrases)
    return opinion_terms


def rule2(dep_tags, pos_tags):
    words = [dep_tag[2][1] for dep_tag in dep_tags]
    assert len(words) == len(pos_tags)
    # print(words)
    # print(pos_tags)

    opinion_terms = list()
    for dep_tup in dep_tags:
        rel, gov, dep = dep_tup
        if rel not in {'nsubj'}:
            continue

        igov, wgov = gov
        idep, wdep = dep

        # if wdep == 'i':
        #     opinion_terms.append(wgov)
        if wdep == 'it' and pos_tags[igov] == 'JJ':
            opinion_terms.append(wgov)

    # print(' '.join(words))
    # print(noun_phrases)
    # print(opinion_terms)
    return opinion_terms


def rule3(sent_text: str, terms_dict):
    terms = list()
    for t in terms_dict:
        pbeg = sent_text.find(t)
        pend = pbeg + len(t)
        if pbeg > 0 and sent_text[pbeg - 1].isalpha():
            continue
        if pend < len(sent_text) and sent_text[pend].isalpha():
            continue
        terms.append(t)
    return terms


def rule4(dep_tags, pos_tags):
    words = [dep_tag[2][1] for dep_tag in dep_tags]
    assert len(words) == len(pos_tags)
    # print(words)
    # print(pos_tags)

    opinion_terms = list()
    for dep_tup in dep_tags:
        rel, gov, dep = dep_tup
        if rel not in {'amod'}:
            continue

        igov, wgov = gov
        idep, wdep = dep
        print(wgov, wdep)
        opinion_terms.append(wdep)

    return opinion_terms
