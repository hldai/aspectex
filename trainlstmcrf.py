import datetime
import pickle
from utils.loggingutils import init_logging
from utils import datautils
import config
from models.lstmcrf import LSTMCRF


def __train_lstmcrf(word_vecs_file, train_tok_texts_file, train_sents_file, train_valid_split_file,
                    test_tok_texts_file, test_sents_file, task, load_model_file=None, error_file=None):
    init_logging('log/lstmcrf-{}.log'.format(str_today), mode='a', to_stdout=True)

    dst_aspects_file = 'd:/data/aspect/semeval14/lstmcrf-aspects.txt'
    dst_opinions_file = 'd:/data/aspect/semeval14/lstmcrf-opinions.txt'

    n_tags = 5 if task == 'both' else 3
    n_epochs = 100
    lr = 0.001

    print('loading data ...')
    with open(word_vecs_file, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)

    save_model_file = None
    train_data, valid_data, test_data = datautils.get_data_semeval(
        train_sents_file, train_tok_texts_file, train_valid_split_file,
        test_sents_file, test_tok_texts_file,
        vocab, -1, task)

    # train_data, valid_data = __get_data_semeval(vocab, -1)
    # train_data, valid_data = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE1_FILE)
    # train_data, valid_data = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE2_FILE)
    print('done')

    # lstmcrf = LSTMCRF(n_tags, word_vecs_matrix, hidden_size_lstm=hidden_size_lstm)
    lstmcrf = LSTMCRF(n_tags, word_vecs_matrix, hidden_size_lstm=hidden_size_lstm, model_file=load_model_file)
    # print(valid_data.aspects_true_list)
    # lstmcrf.train(train_data.word_idxs_list, train_data.labels_list, valid_data.word_idxs_list,
    #               valid_data.labels_list, vocab, valid_data.tok_texts, valid_data.aspects_true_list,
    #               valid_data.opinions_true_list,
    #               n_epochs=n_epochs, save_file=save_model_file, error_file=error_file)
    lstmcrf.train(train_data, valid_data, test_data, n_epochs=n_epochs, lr=lr, dst_aspects_file=dst_aspects_file,
                  dst_opinions_file=dst_opinions_file)


str_today = datetime.date.today().strftime('%y-%m-%d')

# dm = 'semeval15'
dm = 'semeval14'
# dataset_name = 'restaurant'
dataset_name = 'laptops'
hidden_size_lstm = 100
n_epochs = 200

if dataset_name == 'laptops':
    # word_vecs_file = config.SE14_LAPTOP_GLOVE_WORD_VEC_FILE
    word_vecs_file = config.SE14_LAPTOP_AMAZON_WORD_VEC_FILE
    pre_tok_texts_file = config.AMAZON_TOK_TEXTS_FILE
    pre_aspect_terms_file = config.AMAZON_RM_TERMS_FILE
    pre_opinion_terms_file = config.AMAZON_TERMS_TRUE4_FILE

    train_valid_split_file = config.SE14_LAPTOP_TRAIN_VALID_SPLIT_FILE
    train_tok_texts_file = config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE
    train_sents_file = config.SE14_LAPTOP_TRAIN_SENTS_FILE
    test_tok_texts_file = config.SE14_LAPTOP_TEST_TOK_TEXTS_FILE
    test_sents_file = config.SE14_LAPTOP_TEST_SENTS_FILE
else:
    # word_vecs_file = config.SE14_REST_GLOVE_WORD_VEC_FILE
    # pre_aspect_terms_file = 'd:/data/aspect/semeval14/restaurants/yelp-aspect-rule-result-r.txt'
    # aspect_terms_file = 'd:/data/aspect/semeval14/restaurant/yelp-aspect-rule-result-r1.txt'
    pre_aspect_terms_file = 'd:/data/aspect/{}/restaurants/yelp-aspect-rm-rule-result.txt'.format(dm)
    pre_opinion_terms_file = 'd:/data/aspect/{}/restaurants/yelp-opinion-rule-result.txt'.format(dm)
    pre_tok_texts_file = 'd:/data/res/yelp-review-eng-tok-sents-round-9.txt'
    # rule_model_file = 'd:/data/aspect/semeval14/tf-model/drest/yelp-nrdj.ckpl'

    if dm == 'semeval14':
        train_valid_split_file = config.SE14_REST_TRAIN_VALID_SPLIT_FILE
        train_tok_texts_file = config.SE14_REST_TRAIN_TOK_TEXTS_FILE
        train_sents_file = config.SE14_REST_TRAIN_SENTS_FILE
        test_tok_texts_file = config.SE14_REST_TEST_TOK_TEXTS_FILE
        test_sents_file = config.SE14_REST_TEST_SENTS_FILE
        word_vecs_file = config.SE14_REST_YELP_WORD_VEC_FILE
    else:
        train_valid_split_file = config.SE15_REST_TRAIN_VALID_SPLIT_FILE
        train_tok_texts_file = config.SE15_REST_TRAIN_TOK_TEXTS_FILE
        train_sents_file = config.SE15_REST_TRAIN_SENTS_FILE
        test_tok_texts_file = config.SE15_REST_TEST_TOK_TEXTS_FILE
        test_sents_file = config.SE15_REST_TEST_SENTS_FILE
        word_vecs_file = config.SE15_REST_YELP_WORD_VEC_FILE


__train_lstmcrf(word_vecs_file, train_tok_texts_file, train_sents_file, train_valid_split_file, test_tok_texts_file,
                test_sents_file, 'both')
