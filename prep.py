import os
import re
import json
import config
import utils

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


__process_hl04()
