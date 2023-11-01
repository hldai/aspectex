import os
import json


def save_json_objs(objs, dst_file):
    with open(dst_file, 'w', encoding='utf-8') as fout:
        for obj in objs:
            fout.write('{}\n'.format(json.dumps(obj, ensure_ascii=False)))


def __read_opinions_file(filename, has_polarity=True):
    opinions_sents = list()
    f = open(filename, encoding='utf-8')
    for line in f:
        line = line.strip()
        if line == 'NIL':
            opinions_sents.append(None)
            continue

        vals = line.split(',')
        terms = list()
        for v in vals:
            v = v.strip()
            if not v:
                continue
            if has_polarity:
                if not (v[-2:] == '-1' or v[-2:] == '+1'):
                    print(line, v)
                assert v[-2:] == '-1' or v[-2:] == '+1'
                term = v[:-2].strip()
                # polarity = v[-2:]
                terms.append(term)
            else:
                terms.append(v)
        opinions_sents.append(terms)
    f.close()
    return opinions_sents


def __process_raw_sem_eval_data(xml_file, opinions_file, dst_sents_file, dst_sents_text_file, fn_get_sent_objs):
    opinions_sents = __read_opinions_file(opinions_file)

    # f = open(xml_file, encoding='utf-8')
    # text_all = f.read()
    # sents = fn_get_sent_objs(text_all, opinions_sents)
    # f.close()
    sents = fn_get_sent_objs(xml_file, opinions_sents)

    save_json_objs(sents, dst_sents_file)
    if dst_sents_text_file is not None:
        with open(dst_sents_text_file, 'w', encoding='utf-8') as fout:
            for sent in sents:
                fout.write('{}\n'.format(sent['text']))


def __get_sent_objs_se14_xml(filename, opinions_sents):
    import xml.etree.ElementTree as ET
    dom = ET.parse(filename)
    root = dom.getroot()

    sents = list()
    for i, sent in enumerate(root.iter('sentence')):
        sent_obj = {'id': sent.attrib['id'], 'text': sent.find('text').text}
        aspect_terms = list()
        for term_elem in sent.iter('aspectTerm'):
            # print(term_elem.attrib['term'])
            aspect_terms.append(
                {'term': term_elem.attrib['term'], 'polarity': term_elem.attrib['polarity'], 'span': (
                    int(term_elem.attrib['from']), int(term_elem.attrib['to']))})
            # aspect_terms.append(
            #     {'term': term_elem.attrib['term'], 'span': (
            #         int(term_elem.attrib['from']), int(term_elem.attrib['to']))})

        if aspect_terms:
            sent_obj['terms'] = aspect_terms
        if opinions_sents[i] is not None:
            sent_obj['opinions'] = opinions_sents[i]
        sents.append(sent_obj)
    return sents


SE14_LAPTOP_TRAIN_XML_FILE = 'rinante-data/semeval14/laptops/Laptops_Train.xml'
SE14_LAPTOP_TRAIN_OPINIONS_FILE = 'rinante-data/semeval14/laptops/train_laptop_opinions.txt'
output_sents_file = 'rinante-data/semeval14/laptops/laptops_train_sents.json'
output_sent_texts_file = 'rinante-data/semeval14/laptops/laptops_train_texts.txt'

__process_raw_sem_eval_data(
    SE14_LAPTOP_TRAIN_XML_FILE, SE14_LAPTOP_TRAIN_OPINIONS_FILE,
    output_sents_file, output_sent_texts_file, __get_sent_objs_se14_xml)
