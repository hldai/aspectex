import datetime
import pickle
from utils.loggingutils import init_logging
from utils import datautils
import os
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
    lstmcrf = LSTMCRF(n_tags, word_vecs_matrix, hidden_size_lstm=hidden_size_lstm, model_file=load_model_file,
                      train_word_embeddings=False)
    # print(valid_data.aspects_true_list)
    # lstmcrf.train(train_data.word_idxs_list, train_data.labels_list, valid_data.word_idxs_list,
    #               valid_data.labels_list, vocab, valid_data.tok_texts, valid_data.aspects_true_list,
    #               valid_data.opinions_true_list,
    #               n_epochs=n_epochs, save_file=save_model_file, error_file=error_file)
    lstmcrf.train(train_data, valid_data, test_data, n_epochs=n_epochs, lr=lr, dst_aspects_file=dst_aspects_file,
                  dst_opinions_file=dst_opinions_file)


str_today = datetime.date.today().strftime('%y-%m-%d')

hidden_size_lstm = 100
n_epochs = 200

# dataset = 'se15r'
# dataset = 'se14r'
dataset = 'se14l'

if dataset == 'se14l':
    # word_vecs_file = os.path.join(config.SE14_DIR, 'model-data/amazon-wv-300-sg-n10-w8-i30.pkl')
    # word_vecs_file = os.path.join(config.SE14_DIR, 'model-data/amazon-wv-300-sg-n10-w8-i30.pkl')
    word_vecs_file = os.path.join(config.SE14_DIR, 'model-data/laptops-amazon-word-vecs.pkl')
elif dataset == 'se14r':
    word_vecs_file = os.path.join(config.SE14_DIR, 'model-data/yelp-w2v-sg-100-n10-i30-w5.pkl')
else:
    word_vecs_file = os.path.join(config.SE15_DIR, 'model-data/yelp-w2v-sg-100-n10-i30-w5.pkl')

dataset_files = config.DATA_FILES[dataset]
__train_lstmcrf(word_vecs_file, dataset_files['train_tok_texts_file'], dataset_files['train_sents_file'],
                dataset_files['train_valid_split_file'], dataset_files['test_tok_texts_file'],
                dataset_files['test_sents_file'], 'both')
