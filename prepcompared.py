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


def __gen_rncrf_aspect_opinion_file(sents_file, dst_aspect_file, dst_opinion_file):
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


se14_laptops_train_sents_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_train_sents.json'
se14_laptops_test_sents_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_test_sents.json'
se14_laptops_all_sents_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_all_sents.json'
se14_laptops_train_valid_split_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_train_valid_split.txt'
se14_laptops_all_sent_texts_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_all_sent_texts.txt'

se14_laptops_all_aspect_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_all_aspects.txt'
se14_laptops_all_opinion_file = '/home/hldai/data/aspect/semeval14/laptops/laptops_all_opinions.txt'
se14_laptops_datasplit_file = '/home/hldai/data/aspect/semeval14/laptops/datasplit-full.txt'

# __merge_train_test(se14_laptops_train_sents_file, se14_laptops_test_sents_file, se14_laptops_train_valid_split_file,
#                    se14_laptops_all_sents_file, se14_laptops_datasplit_file)
# __gen_rncrf_aspect_opinion_file(se14_laptops_all_sents_file, se14_laptops_all_aspect_file,
#                                 se14_laptops_all_opinion_file)
__texts_file_from_sents(se14_laptops_all_sents_file, se14_laptops_all_sent_texts_file)
