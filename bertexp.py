import os
import logging
import datetime
from models.bertlstmcrf import BertLSTMCRF
from utils.loggingutils import init_logging
from utils import bldatautils
import config


def __train_bert():
    str_today = datetime.date.today().strftime('%y-%m-%d')
    init_logging('log/bertlstmcrf-{}.log'.format(str_today), mode='a', to_stdout=True)

    bert_embed_file_train = os.path.join(config.DATA_DIR_SE14, 'laptops/laptops_train_texts_tok_bert.txt')
    bert_embed_file_test = os.path.join(config.DATA_DIR_SE14, 'laptops/laptops_test_texts_tok_bert.txt')
    # dst_aspects_file = 'd:/data/aspect/semeval14/lstmcrf-aspects.txt'
    # dst_opinions_file = 'd:/data/aspect/semeval14/lstmcrf-opinions.txt'

    print('loading data ...')
    data_train = bldatautils.load_train_data_bert(bert_embed_file_train, config.SE14_LAPTOP_TRAIN_SENTS_FILE)
    data_test = bldatautils.load_valid_data_bert(bert_embed_file_test, config.SE14_LAPTOP_TEST_SENTS_FILE)
    print('done')

    word_embed_dim = len(data_train.word_embed_seqs[0][0])
    n_tags = 5
    n_epochs = 100
    lr = 0.001

    # with open(word_vecs_file, 'rb') as f:
    #     vocab, word_vecs_matrix = pickle.load(f)

    logging.info('token_embed_dim={}'.format(word_embed_dim))

    save_model_file = None

    lstmcrf = BertLSTMCRF(n_tags, word_embed_dim)
    lstmcrf.train(data_train, data_test, data_test, n_epochs=n_epochs, lr=lr)


if __name__ == '__main__':
    __train_bert()
