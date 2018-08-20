import os
from utils import utils
import config


def __gen_cmla_word_vecs_file():
    tok_texts_file_train = 'd:/projects/ext/Coupled-Multi-layer-Attentions/util/data_semEval/sentence_res15'
    tok_texts_file_test = 'd:/projects/ext/Coupled-Multi-layer-Attentions/util/data_semEval/sentence_restest15'
    se15_cmla_word_vecs_file = 'd:/data/res/glove-vecs-se15-rest-cmla.txt'
    utils.trim_word_vecs_file(
        [tok_texts_file_train, tok_texts_file_test],
        config.GLOVE_WORD_VEC_FILE, se15_cmla_word_vecs_file, 'txt'
    )


def __gen_aspect_opinion_file(sents_file, dst_aspect_file, dst_opinion_file):
    sents = utils.load_json_objs(sents_file)
    aspects_list, opinions_list = list(), list()
    for sent in sents:
        aspects_list.append([t['term'] for t in sent.get('terms', list())])
        opinions_list.append(sent.get('opinions', list()))

    def __write_terms_file(terms_list, dst_file):
        with open(dst_file, 'w', encoding='utf-8') as fout:
            for terms in terms_list:
                fout.write('{}\n'.format(','.join(terms)))

    __write_terms_file(aspects_list, dst_aspect_file)
    __write_terms_file(opinions_list, dst_opinion_file)


def __merge_train_test(train_sents_file, test_sents_file, train_valid_split_file, dst_sents_file, dst_datasplit_file):
    train_sents = utils.load_json_objs(train_sents_file)
    test_sents = utils.load_json_objs(test_sents_file)
    all_sents = train_sents + test_sents
    utils.save_json_objs(all_sents, dst_sents_file)

    train_valid_split_labels = utils.read_lines(train_valid_split_file)[0]
    train_valid_split_labels = [int(v) for v in train_valid_split_labels.split(' ')]
    all_data_split_labels = train_valid_split_labels + [2 for _ in range(len(test_sents))]
    with open(dst_datasplit_file, 'w', encoding='utf-8') as fout:
        fout.write('{}\n'.format(' '.join([str(v) for v in all_data_split_labels])))


def __texts_file_from_sents(sents_file, dst_texts_file):
    sents = utils.load_json_objs(sents_file)
    with open(dst_texts_file, 'w', encoding='utf-8') as fout:
        for sent in sents:
            sent_text = sent['text']
            assert '\n' not in sent_text
            fout.write('{}\n'.format(sent['text']))


def __gen_aspect_terms_file(sents_file, dst_terms_file):
    sents = utils.load_json_objs(sents_file)
    fout = open(dst_terms_file, 'w', encoding='utf-8')
    for sent in sents:
        terms = [t['term'] for t in sent.get('terms', list())]
        if not terms:
            fout.write('\n')
        else:
            fout.write('{}\n'.format(','.join(terms)))
    fout.close()


def __gen_opinion_terms_file(sents_file, dst_terms_file):
    sents = utils.load_json_objs(sents_file)
    fout = open(dst_terms_file, 'w', encoding='utf-8')
    for sent in sents:
        terms = [t for t in sent.get('opinions', list())]
        if not terms:
            fout.write('\n')
        else:
            fout.write('{}\n'.format(','.join(terms)))
    fout.close()


def __has_digit(w):
    for ch in w:
        if ch.isdigit():
            return True
    return False


def __replace_digits(src_text_file, dst_text_file):
    f = open(src_text_file, encoding='utf-8')
    fout = open(dst_text_file, 'w', encoding='utf-8')
    for i, line in enumerate(f):
        words = line.strip().split(' ')
        words_new = list()
        for w in words:
            if __has_digit(w):
                words_new.append('NUM')
            else:
                words_new.append(w)
        fout.write('{}\n'.format(' '.join(words_new)))
        if i % 100000 == 0:
            print(i)
        # if i > 1000:
        #     break
    f.close()
    fout.close()


def __replace_digits_in_terms_file(src_terms_file, dst_file):
    f = open(src_terms_file, encoding='utf-8')
    fout = open(dst_file, 'w', encoding='utf-8')
    for line in f:
        line = line.strip()
        terms = line.split(',')
        terms_new = list()
        for t in terms:
            words = t.split(' ')
            words_new = list()
            for w in words:
                if __has_digit(w):
                    words_new.append('NUM')
                else:
                    words_new.append(w)
            terms_new.append(' '.join(words_new))
        fout.write('{}\n'.format(','.join(terms_new)))

    fout.close()
    f.close()


se14_laptops_train_sents_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_train_sents.json'
se14_laptops_test_sents_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_test_sents.json'
se14_laptops_all_sents_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_all_sents.json'
se14_laptops_train_valid_split_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_train_valid_split.txt'
se14_laptops_all_sent_texts_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_all_sent_texts.txt'
se14_laptops_all_sent_nrtexts_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_all_sent_nrtexts.txt'

se14_laptops_train_aspect_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_train_aspects.txt'
se14_laptops_train_nr_aspect_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_train_nr_aspects.txt'
se14_laptops_test_aspect_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_test_aspects.txt'
se14_laptops_test_nr_aspect_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_test_nr_aspects.txt'
se14_laptops_test_opinion_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_test_opinions.txt'
se14_laptops_all_aspect_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_all_aspects.txt'
se14_laptops_all_nr_aspect_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_all_nr_aspects.txt'
se14_laptops_all_opinion_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_all_opinions.txt'
se14_laptops_datasplit_file = '/home/hldai/data/aspect/semeval14/laptops/datasplit-full.txt'

se15_restaurants_train_sents_file = '/home/hldai/data/aspect/semeval15/restaurants/restaurants_train_sents.json'
se15_restaurants_test_sents_file = '/home/hldai/data/aspect/semeval15/restaurants/restaurants_test_sents.json'
se15_rest_train_valid_split_file = '/home/hldai/data/aspect/semeval15/restaurants/restaurants_train_valid_split.txt'
se15_rest_all_sents_file = '/home/hldai/data/aspect/semeval15/restaurants/rest_all_sents.json'
se15_restaurants_datasplit_file = '/home/hldai/data/aspect/semeval15/restaurants/rest-data-split-full.txt'
se15_rest_all_aspects_file = '/home/hldai/data/aspect/semeval15/restaurants/rest_all_aspects.txt'
se15_rest_all_opinions_file = '/home/hldai/data/aspect/semeval15/restaurants/rest_all_opinions.txt'
se15_rest_all_sent_texts_file = '/home/hldai/data/aspect/semeval15/restaurants/rest_all_sent_texts.txt'

se14_rest_all_sents_file = os.path.join(config.DATA_DIR_SE14, 'restaurants/rest-sents-all.json')
se14_rest_all_data_split_file = os.path.join(config.DATA_DIR_SE14, 'restaurants/rest-data-split-full.txt')
se14_rest_all_aspects_file = os.path.join(config.DATA_DIR_SE14, 'restaurants/rest-all-aspects.txt')
se14_rest_all_opinions_file = os.path.join(config.DATA_DIR_SE14, 'restaurants/rest-all-opinions.txt')
se14_rest_train_aspects_file = os.path.join(config.DATA_DIR_SE14, 'restaurants/rest-train-aspects.txt')
se14_rest_train_opinions_file = os.path.join(config.DATA_DIR_SE14, 'restaurants/rest-train-opinions.txt')
se14_rest_test_aspects_file = os.path.join(config.DATA_DIR_SE14, 'restaurants/rest-test-aspects.txt')
se14_rest_test_opinions_file = os.path.join(config.DATA_DIR_SE14, 'restaurants/rest-test-opinions.txt')
se14_rest_all_sent_texts_file = os.path.join(config.DATA_DIR_SE14, 'restaurants/rest-all-sent-texts.txt')

# __merge_train_test(config.SE14_REST_TRAIN_SENTS_FILE, config.SE14_REST_TEST_SENTS_FILE,
#                    config.SE14_REST_TRAIN_VALID_SPLIT_FILE, se14_rest_all_sents_file,
#                    se14_rest_all_data_split_file)

# __merge_train_test(config.SE14_LAPTOP_TRAIN_SENTS_FILE, config.SE14_LAPTOP_TEST_SENTS_FILE,
#                    se14_laptops_train_valid_split_file, se14_laptops_all_sents_file, se14_laptops_datasplit_file)
# __gen_aspect_opinion_file(se14_laptops_all_sents_file, se14_laptops_all_aspect_file, se14_laptops_all_opinion_file)
# __texts_file_from_sents(se14_laptops_all_sents_file, se14_laptops_all_sent_texts_file)

# __merge_train_test(se15_restaurants_train_sents_file, se15_restaurants_test_sents_file,
#                    se15_rest_train_valid_split_file, se15_rest_all_sents_file,
#                    se15_restaurants_datasplit_file)
# __gen_aspect_opinion_file(se15_rest_all_sents_file, se15_rest_all_aspects_file, se15_rest_all_opinions_file)
# __gen_aspect_opinion_file(se14_rest_all_sents_file, se14_rest_all_aspects_file, se14_rest_all_opinions_file)
__gen_aspect_opinion_file(
    config.SE14_REST_TRAIN_SENTS_FILE, se14_rest_train_aspects_file, se14_rest_train_opinions_file)
__gen_aspect_opinion_file(
    config.SE14_REST_TEST_SENTS_FILE, se14_rest_test_aspects_file, se14_rest_test_opinions_file)
# __texts_file_from_sents(se15_rest_all_sents_file, se15_rest_all_sent_texts_file)
# __texts_file_from_sents(se14_rest_all_sents_file, se14_rest_all_sent_texts_file)

# __gen_aspect_terms_file(se14_laptops_train_sents_file, se14_laptops_train_aspect_file)
# __gen_aspect_terms_file(se14_laptops_test_sents_file, se14_laptops_test_aspect_file)
# __gen_opinion_terms_file(se14_laptops_test_sents_file, se14_laptops_test_opinion_file)

# __replace_digits(se14_laptops_all_sent_texts_file, se14_laptops_all_sent_nrtexts_file)
# __replace_digits('/home/hldai/data/amazon/electronics_5_text.txt',
#                  '/home/hldai/data/amazon/electronics_5_nr_text.txt')
# __replace_digits('/home/hldai/data/aspect/semeval14/laptops/laptops_train_texts.txt',
#                  '/home/hldai/data/aspect/semeval14/laptops/laptops_train_nr_texts.txt')
# __replace_digits('/home/hldai/data/aspect/semeval14/laptops/laptops_test_texts.txt',
#                  '/home/hldai/data/aspect/semeval14/laptops/laptops_test_nr_texts.txt')

# __replace_digits_in_terms_file(se14_laptops_train_aspect_file, se14_laptops_train_nr_aspect_file)
# __replace_digits_in_terms_file(se14_laptops_test_aspect_file, se14_laptops_test_nr_aspect_file)
# __replace_digits_in_terms_file(se14_laptops_all_aspect_file, se14_laptops_all_nr_aspect_file)
