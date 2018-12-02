import os
import logging
import datetime
from models import robert, bertmodel
from models.bertlstmcrf import BertLSTMCRF
from utils.loggingutils import init_logging
from utils import bldatautils
import config


def __train_bert():
    str_today = datetime.date.today().strftime('%y-%m-%d')
    init_logging('log/bertlstmcrf-{}.log'.format(str_today), mode='a', to_stdout=True)

    # dataset = 'se14r'
    dataset = 'se15r'

    if dataset == 'se14l':
        bert_embed_file_train = os.path.join(config.DATA_DIR_SE14, 'laptops/laptops_train_texts_tok_bert.txt')
        bert_embed_file_test = os.path.join(config.DATA_DIR_SE14, 'laptops/laptops_test_texts_tok_bert.txt')
        train_valid_split_file = config.SE14_LAPTOP_TRAIN_VALID_SPLIT_FILE
        train_sents_file = config.SE14_LAPTOP_TRAIN_SENTS_FILE
        test_sents_file = config.SE14_LAPTOP_TEST_SENTS_FILE
        # dst_aspects_file = 'd:/data/aspect/semeval14/lstmcrf-aspects.txt'
        # dst_opinions_file = 'd:/data/aspect/semeval14/lstmcrf-opinions.txt'
    elif dataset == 'se14r':
        bert_embed_file_train = os.path.join(
            config.DATA_DIR_SE14, 'restaurants/restaurants_train_texts_tok_bert.txt')
        bert_embed_file_test = os.path.join(
            config.DATA_DIR_SE14, 'restaurants/restaurants_test_texts_tok_bert.txt')
        train_valid_split_file = config.SE14_REST_TRAIN_VALID_SPLIT_FILE
        train_sents_file = config.SE14_REST_TRAIN_SENTS_FILE
        test_sents_file = config.SE14_REST_TEST_SENTS_FILE
    else:
        bert_embed_file_train = os.path.join(
            config.SE15_DIR, 'restaurants/restaurants_train_texts_tok_bert.txt')
        bert_embed_file_test = os.path.join(
            config.SE15_DIR, 'restaurants/restaurants_test_texts_tok_bert.txt')
        train_valid_split_file = config.SE15_REST_TRAIN_VALID_SPLIT_FILE
        train_sents_file = config.SE15_REST_TRAIN_SENTS_FILE
        test_sents_file = config.SE15_REST_TEST_SENTS_FILE

    print('loading data ...')
    data_train, data_valid = bldatautils.load_train_data_bert(
        bert_embed_file_train, train_sents_file, train_valid_split_file)
    data_test = bldatautils.load_valid_data_bert(bert_embed_file_test, test_sents_file)
    print('done')

    word_embed_dim = len(data_train.word_embed_seqs[0][0])
    n_tags = 5
    n_epochs = 100
    lr = 0.001

    # with open(word_vecs_file, 'rb') as f:
    #     vocab, word_vecs_matrix = pickle.load(f)

    logging.info(test_sents_file)
    logging.info('token_embed_dim={}'.format(word_embed_dim))

    save_model_file = None

    print(data_train.word_embed_seqs[0])
    exit()
    lstmcrf = BertLSTMCRF(n_tags, word_embed_dim, hidden_size_lstm=500, batch_size=5)
    lstmcrf.train(data_train, data_valid, data_test, n_epochs=n_epochs, lr=lr)


def __train_bert_ol():
    str_today = datetime.date.today().strftime('%y-%m-%d')
    init_logging('log/bertlstmcrfol-{}.log'.format(str_today), mode='a', to_stdout=True)

    # dataset = 'se14r'
    dataset = 'se15r'
    n_labels = 5

    dataset_files = config.DATA_FILES[dataset]

    n_train, data_valid = bldatautils.load_train_data_bert_ol(
        dataset_files['train_sents_file'], dataset_files['train_valid_split_file'],
        dataset_files['bert_valid_tokens_file'])
    data_test = bldatautils.load_test_data_bert_ol(
        dataset_files['test_sents_file'], dataset_files['bert_test_tokens_file'])

    bert_config = bertmodel.BertConfig.from_json_file(config.BERT_CONFIG_FILE)
    bm = robert.Robert(
        bert_config, n_labels=n_labels, seq_length=config.BERT_SEQ_LEN, is_train=False,
        init_checkpoint=dataset_files['bert_init_checkpoint']
    )

    lstmcrf = BertLSTMCRF(n_labels, config.BERT_EMBED_DIM, hidden_size_lstm=500, batch_size=5)
    lstmcrf.train_ol(
        robert_model=bm, train_tfrec_file=dataset_files['train_tfrecord_file'],
        valid_tfrec_file=dataset_files['valid_tfrecord_file'], test_tfrec_file=dataset_files['test_tfrecord_file'],
        seq_length=config.BERT_SEQ_LEN, n_train=n_train, data_valid=data_valid, data_test=data_test
    )


if __name__ == '__main__':
    # __train_bert()
    __train_bert_ol()
