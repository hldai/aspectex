import json
from utils import utils
import re


def __get_category_names(cat_str):
    pleft, pright = 0, 0
    cur_char = ''
    slen = len(cat_str)
    names = list()
    while pleft < slen:
        # print(pleft, pright, slen)
        while pleft < slen:
            ch = cat_str[pleft]
            if ch == "'" or ch == '"':
                cur_char = ch
                break
            pleft += 1

        pright = pleft + 1
        while pright < slen and cat_str[pright] != cur_char:
            pright += 1
        if pright >= slen:
            break
        names.append(cat_str[pleft + 1: pright])
        pleft = pright + 1
    return names


def __get_prod_categories(prod_str):
    m = re.search("'categories': \\[(.*?)\\]\\]", prod_str)
    if m is None:
        return list()
    cat_str = m.group(1)
    assert '\\' not in cat_str

    cat_str = cat_str + ']'
    miter = re.finditer('\[(.*?)\]', cat_str)
    cat_str_list = list()
    for m in miter:
        # print(m.group(1))
        cat_str_list.append(m.group(1))
    categories_list = list()
    for cat_str in cat_str_list:
        cur_category_names = __get_category_names(cat_str)
        categories_list.append(cur_category_names)
    return categories_list


def __gen_category_name_file(metadata_file, dst_file):
    category_names = set()
    f = open(metadata_file, encoding='utf-8')
    for i, line in enumerate(f):
        if i % 1000000 == 0:
            print(i)
        # if i > 10000:
        #     break
        categories_list = __get_prod_categories(line)
        for categories in categories_list:
            category_names.update(categories)

        # prod = json.loads(line)
        # prod_category_dict[prod['asin']] = prod['categories']
        # break
    f.close()
    category_names = list(category_names)
    category_names.sort()
    with open(dst_file, 'w', encoding='utf-8', newline='\n') as fout:
        for n in category_names:
            fout.write('{}\n'.format(n))


def __get_prod_by_category(metadata_file, category_name, dst_file):
    asins = set()
    f = open(metadata_file, encoding='utf-8')
    for i, line in enumerate(f):
        if i % 1000000 == 0:
            print(i)
        # if i > 1000000:
        #     break
        # line = line.replace("'asin'", '"asin"')
        m = re.search("'asin': '(.+?)'", line)
        asin = m.group(1)
        categories_list = __get_prod_categories(line)
        hit = False
        for categories in categories_list:
            for cat in categories:
                if cat == category_name:
                    hit = True
                    asins.add(asin)
                    break
            if hit:
                break
        # if hit:
        #     print(line)
        # prod = json.loads(line)
        # prod_category_dict[prod['asin']] = prod['categories']
        # break
    f.close()

    with open(dst_file, 'w', encoding='utf-8', newline='\n') as fout:
        for asin in asins:
            fout.write('{}\n'.format(asin))


def __select_reviews_file(src_reviws_file, asins_file, dst_file):
    asins = utils.read_lines(asins_file)
    asins = set(asins)
    print(len(asins), 'asins')

    f = open(src_reviws_file, encoding='utf-8')
    fout = open(dst_file, 'w', encoding='utf-8', newline='\n')
    for i, line in enumerate(f):
        # if i > 10000:
        #     break
        review_obj = json.loads(line)
        if review_obj['asin'] not in asins:
            continue
        # print(review_obj)
        fout.write('{}\n'.format(json.dumps(review_obj, ensure_ascii=False)))
    f.close()
    fout.close()


def __gen_review_text_file(reviews_file, dst_file):
    f = open(reviews_file, encoding='utf-8')
    fout = open(dst_file, 'w', encoding='utf-8', newline='\n')
    for line in f:
        review_obj = json.loads(line)
        text = review_obj['reviewText']
        assert '\n' not in text
        fout.write('{}\n'.format(text))
    f.close()
    fout.close()


def __to_sents(text_file, dst_file):
    import nltk
    f = open(text_file, encoding='utf-8')
    fout = open(dst_file, 'w', encoding='utf-8')
    max_n_spaces = 0
    sent_cnt = 0
    for line in f:
        sents = nltk.sent_tokenize(line)
        for sent in sents:
            n_spaces = sent.count(' ')
            max_n_spaces = max(max_n_spaces, n_spaces)
            if n_spaces < 5 or n_spaces > 60:
                continue
            if '---' in sent or '===' in sent:
                continue
            fout.write('{}\n'.format(sent))
            sent_cnt += 1
        # break
    f.close()
    fout.close()
    print(sent_cnt, 'sents', max_n_spaces)


metadata_file = 'd:/data/amazon/metadata.json'
category_name_file = 'd:/data/amazon/categories.txt'
laptop_asins_file = 'd:/data/amazon/laptop-asins.txt'
electronics_reviews_file = 'd:/data/amazon/Electronics_5.json'
laptop_reivews_file = 'd:/data/amazon/laptops-reivews.json'
laptop_reivew_text_file = 'd:/data/amazon/laptops-reivews-text.txt'
laptop_reivew_sent_text_file = 'd:/data/amazon/laptops-reivews-sent-text.txt'
# __gen_category_name_file(metadata_file, category_name_file)
# __get_prod_by_category(metadata_file, 'Laptops', laptop_asins_file)
# __select_reviews_file(electronics_reviews_file, laptop_asins_file, laptop_reivews_file)
# __gen_review_text_file(laptop_reivews_file, laptop_reivew_text_file)
# __to_sents(laptop_reivew_text_file, laptop_reivew_sent_text_file)
