import os
import re
import json
import config
from utils import utils

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


def __sem_eval_to_json(src_file, dst_file):
    f = open(src_file, encoding='utf-8')
    text_all = f.read()
    sents = list()
    sent_pattern = '<sentence id="(.*?)">\s*<text>(.*?)</text>\s*(.*?)</sentence>'
    miter = re.finditer(sent_pattern, text_all, re.DOTALL)
    for m in miter:
        sent = {'id': m.group(1), 'text': m.group(2)}
        aspect_terms = list()
        aspect_term_pattern = '<aspectTerm\s*term="(.*)"\s*polarity="(.*)"\s*from="(\d*)"\s*to="(\d*)"/>'
        miter_terms = re.finditer(aspect_term_pattern, m.group(3))
        for m_terms in miter_terms:
            # print(m_terms.group(1), m_terms.group(2), m_terms.group(3))
            aspect_terms.append(
                {'term': m_terms.group(1), 'polarity': m_terms.group(2), 'from': int(m_terms.group(3)),
                 'to': int(m_terms.group(4))})
        if aspect_terms:
            sent['terms'] = aspect_terms
        sents.append(sent)
    f.close()

    utils.save_json_objs(sents, dst_file)


test_file_xml = 'd:/data/aspect/semeval14/Laptops_Test_Gold.xml'
test_file_json = 'd:/data/aspect/semeval14/Laptops_Test_Gold.json'
train_file_xml = 'd:/data/aspect/semeval14/Laptops_Train.xml'
train_file_json = 'd:/data/aspect/semeval14/Laptops_Train.json'

# __process_hl04()
# __sem_eval_to_json(test_file_xml, test_file_json)
# __sem_eval_to_json(train_file_xml, train_file_json)
