from models.dslstmcrf import DSLSTMCRF
import datetime
import pickle
import logging
import config
from utils.loggingutils import init_logging
from utils import datautils


def __pre_train_dlc(word_vecs_file, tok_texts_file, aspect_terms_file, opinion_terms_file,
                    dst_model_file, task):
    init_logging('log/dslc-pre-{}.log'.format(str_today), mode='a', to_stdout=True)

    # n_train = 1000
    n_train = -1
    label_opinions = True
    # label_opinions = False
    n_tags = 5 if label_opinions else 3
    # n_tags = 5 if task == 'train' else 3
    batch_size = 20
    lr = 0.001
    share_lstm = False

    print('loading data ...')
    with open(word_vecs_file, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)

    # train_data_src1, valid_data_src1 = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE1_FILE)
    # train_data_src1, valid_data_src1 = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE3_FILE)
    train_data_src1, valid_data_src1 = datautils.get_data_amazon(
        vocab, aspect_terms_file, tok_texts_file, 'aspect')
    train_data_src2, valid_data_src2 = datautils.get_data_amazon(
        vocab, opinion_terms_file, tok_texts_file, 'opinion')
    print('done')

    nrdj = DSLSTMCRF(word_vecs_matrix, hidden_size_lstm=hidden_size_lstm,
                     model_file=None, batch_size=batch_size)

    nrj_train_data_src1 = nrj_train_data_src2 = None
    # if train_mode != 'target-only':
    # nrj_train_data_src1 = NeuRuleDoubleJoint.TrainData(
    #     train_data_src1.word_idxs_list, train_data_src1.labels_list, valid_data_src1.word_idxs_list,
    #     valid_data_src1.labels_list, valid_data_src1.tok_texts, valid_data_src1.aspects_true_list, None
    # )
    # nrj_train_data_src2 = NeuRuleDoubleJoint.TrainData(
    #     train_data_src2.word_idxs_list, train_data_src2.labels_list, valid_data_src2.word_idxs_list,
    #     valid_data_src2.labels_list, valid_data_src2.tok_texts, None,
    #     valid_data_src2.opinions_true_list
    # )

    nrdj.pre_train(train_data_src1, valid_data_src1, train_data_src2, valid_data_src2, vocab,
                   n_epochs=5, lr=lr, save_file=dst_model_file)


def __train_dlc(word_vecs_file, train_tok_texts_file, train_sents_file, train_valid_split_file, test_tok_texts_file,
                test_sents_file, load_model_file, task):
    init_logging('log/dslc-train-{}.log'.format(str_today), mode='a', to_stdout=True)

    # n_train = 1000
    n_train = -1
    label_opinions = True
    # label_opinions = False
    n_tags = 5 if label_opinions else 3
    # n_tags = 5 if task == 'train' else 3
    batch_size = 5
    lr = 0.001

    print('loading data ...')
    with open(word_vecs_file, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
    logging.info('word vec dim: {}, n_words={}'.format(word_vecs_matrix.shape[1], word_vecs_matrix.shape[0]))
    train_data, valid_data, test_data = datautils.get_data_semeval(
        train_sents_file, train_tok_texts_file, train_valid_split_file, test_sents_file, test_tok_texts_file,
        vocab, n_train, label_opinions)
    print('done')

    dlc = DSLSTMCRF(word_vecs_matrix, hidden_size_lstm=hidden_size_lstm,
                     model_file=load_model_file, batch_size=batch_size)

    dlc.train(train_data, valid_data, test_data, vocab, n_epochs=n_epochs, lr=lr)


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
    # rule_model_file = config.LAPTOP_RULE_MODEL2_FILE
    rule_model_file = 'd:/data/aspect/{}/tf-model/dlclap/amazon-dlc.ckpl'.format(dm)

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
    rule_model_file = 'd:/data/aspect/{}/tf-model/dlc/yelp-dlc.ckpl'.format(dm)
    # rule_model_file = 'd:/data/aspect/semeval14/tf-model/dlc/yelp-dlc.ckpl'
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

__pre_train_dlc(word_vecs_file, pre_tok_texts_file, pre_aspect_terms_file,
                pre_opinion_terms_file, rule_model_file, 'both')
# __train_dlc(word_vecs_file, train_tok_texts_file, train_sents_file, train_valid_split_file,
#             test_tok_texts_file, test_sents_file, rule_model_file, 'both')
