import utils


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


__count_adj_phrases()
